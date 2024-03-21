from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import make_scorer
import warnings
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils.classification_metrics import f1, signal_significance
from utils.plot import class_frac, sig_frac, sg_bg_hist, roc


def run_output(path, grid_search=False):
    output = ModelOutput(path, grid_search)
    output.config()
    output.images()
    cut = output.cut
    prob = output.out_prob
    return prob, cut


def run_model(df, train_col, model_name='gbdt', grid_search=False, scoring='f1'):
    model = CutEngine(df, train_col, model_name, grid_search, scoring)
    model.prepare_data()

    class_frac(df)
    sig_frac(df)

    plt.hist(model.y_train, label='train_sig', histtype='step')
    plt.hist(model.y_test, label='test_sig', histtype='step')
    plt.xlabel('Signal/background label')
    plt.ylabel('Count')
    plt.title('Signal/background counts used for testing and training')
    plt.legend()
    plt.show()
    plt.clf()

    sg_bg_hist(df, 'h5_momentum', bin_number=100,  true_sig=True)
    sg_bg_hist(df, 'h5_dwall', bin_number=100, true_sig=True)
    sg_bg_hist(df, 'h5_towall', bin_number=100, true_sig=True)

    if grid_search:
        model.grid_search()
    model.train()
    model.make_calibration_curve()
    model.plot_probs(y_log=True)

    if scoring is None:
        test_acc = model.test()
        train_acc = model.test_on_train()
        print('\n')
        print(f'The train accuracy is : {train_acc}')
        print('\n')
        print(f'The test accuracy is : {test_acc}')
    else:
        test_acc, test_metric = model.test()
        train_acc, train_metric = model.test_on_train()
        print('\n')
        print(f'The train accuracy is : {train_acc}')
        print(f'The value for train metric is : {train_metric}')
        print('\n')
        print(f'The test accuracy is : {test_acc}')
        print(f'The test metric is : {test_metric}')

    print('\n')
    print(f' The best threshold is : {model.best_thresh}\n')

    feature_imp = model.get_features_importance()
    for key in feature_imp:
        print(f'{key} : {feature_imp[key]}')

    roc(df['true_sig'], model.y_prob, mode='rejection', algorithms=model_name, cut=model.cut)
    roc(df['true_sig'], model.y_prob, mode='efficiency', algorithms=model_name, cut=model.cut)

    df[model_name + '_sig'] = model.cut

    sg_bg_hist(df, 'h5_momentum', bin_number=100, true_sig=False, cut=df[model_name + '_sig'])
    sg_bg_hist(df, 'h5_dwall', bin_number=100, true_sig=False, cut=df[model_name + '_sig'])
    sg_bg_hist(df, 'h5_towall', bin_number=100, true_sig=False, cut=df[model_name + '_sig'])


class CutEngine:
    def __init__(self, df, train_col, model_name='gbdt', grid_search=False, scoring=None):

        self.model_name = model_name
        self.training_col = train_col
        self.df = df
        self.features = [key for key in train_col if key != 'true_sig']
        self.scoring = scoring
        self.custom_scorer = None

        if self.scoring == 'f1':
            def extract_f1(y_true, y_pred):
                precision, recall, f1_score = f1(y_true, y_pred)
                return f1_score

            self.custom_scorer = make_scorer(extract_f1)

        if self.scoring == 'signal_significance':
            self.custom_scorer = make_scorer(signal_significance)

        if self.scoring is None:
            self.custom_scorer = 'accuracy'

        self.X = None
        self.X_scaled = None
        self.y = None
        self.X_train = None
        self.X_train_scaled = None
        self.X_test = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

        self.y_prob = None
        self.test_prob = None
        self.train_prob = None
        self.test_prediction = None
        self.train_prediction = None
        self.prediction = None
        self.best_thresh = None
        self.cut = None

        if grid_search:
            self.param_grid_gbdt = {
                'n_estimators': [250, 300, 350],
                'subsample': [0.1, 0.4, 0.6],
                'tol': [10 ** -6, 10 ** -8, 10 ** -10],
                'max_depth': [8, 10, 15]

            }

            self.param_grid_adaboost = {
                'n_estimators': [100, 150, 200, 250, 300],
            }

            self.param_grid_xgboost = {
                'n_estimators': [250, 300, 350],
                'subsample': [0.1, 0.4, 0.6],
                'max_depth': [8, 10, 15]
            }

            if self.model_name == 'gbdt':
                self.param_grid = self.param_grid_gbdt
                self.rf_model = GradientBoostingClassifier()

            elif self.model_name == 'adaboost':
                self.param_grid = self.param_grid_adaboost
                self.rf_model = AdaBoostClassifier()

            elif self.model_name == 'xgboost':
                self.param_grid = self.param_grid_xgboost
                self.rf_model = XGBClassifier()
        else:
            if self.model_name == 'gbdt':
                self.rf_model = GradientBoostingClassifier(n_estimators=300, subsample=0.1, tol=2.51*10**-7,
                                                           learning_rate=0.01, max_depth=13, random_state=159)

            elif self.model_name == 'adaboost':
                self.rf_model = AdaBoostClassifier(n_estimators=300, learning_rate=0.01, random_state=159)

            elif self.model_name == 'xgboost':
                self.rf_model = XGBClassifier(n_estimators=300, subsample=0.1, learning_rate=0.01, max_depth=13,
                                              random_state=159)

        self.rf_model_cal = CalibratedClassifierCV(self.rf_model, cv=3, method='isotonic')

    def prepare_data(self):

        df_chosen = self.df[self.training_col]

        self.X = df_chosen.drop(columns=['true_sig'])
        self.y = df_chosen['true_sig']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=159)

        scaler = StandardScaler()

        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.X_scaled = scaler.transform(self.X)

    def grid_search(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FitFailedWarning)

            grid = GridSearchCV(self.rf_model, self.param_grid, cv=3, error_score=0, refit=True,
                                scoring=self.custom_scorer, verbose=0, n_jobs=-1)
            grid.fit(self.X_train, self.y_train)

        self.rf_model = grid.best_estimator_
        print(f"Best Parameters: {grid.best_params_}")
        return grid.best_params_

    def train(self):
        print(f"Using {self.model_name} model")

        self.rf_model.fit(self.X_train, self.y_train)
        self.rf_model_cal.fit(self.X_train, self.y_train)

        self.test_prob = self.rf_model.predict_proba(self.X_test)[:, 1]
        self.train_prob = self.rf_model.predict_proba(self.X_train)[:, 1]
        self.y_prob = self.rf_model.predict_proba(self.X)[:, 1]

        self.test_prediction = self.rf_model.predict(self.X_test)
        self.train_prediction = self.rf_model.predict(self.X_train)
        self.prediction = self.rf_model.predict(self.X)

    def test(self, save_path=None):

        threshold = np.round(np.linspace(0, 1, 100), 2)
        predictions = (self.test_prob[:, np.newaxis] >= threshold).astype(int)

        if self.scoring == 'f1':
            array = np.array([f1(self.y_test, prediction) for prediction in predictions.T])

            precision_array = array[:, 0]
            recall_array = array[:, 1]
            f1_array = array[:, 2]

            best_index = np.nanargmax(f1_array)
            self.best_thresh = threshold[best_index]

            plt.plot(threshold, precision_array, label=self.model_name + ' precision')
            plt.plot(threshold, recall_array, label=self.model_name + ' recall')
            plt.plot(threshold, f1_array, label=self.model_name + ' f1_score')
            plt.axvline(x=float(self.best_thresh), color='black', linestyle='--', label='Selected Threshold')
            plt.xlabel('Threshold')
            plt.title('Precision, recall and f1_score for different thresholds')
            plt.legend()
            if save_path is not None:
                path = os.path.join(save_path, 'images')
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(os.path.join(path, 'precision_recall_f1.png'))
            plt.show()
            plt.clf()

            plt.plot(recall_array, precision_array)
            plt.scatter(recall_array[best_index], precision_array[best_index], color='black', marker='o', s=40,
                        label='Recall and Precision for selected threshold')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('Precision, recall for different thresholds')
            if save_path is not None:
                path = os.path.join(save_path, 'images')
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(os.path.join(path, 'precision_recall.png'))
            plt.show()
            plt.clf()

            self.cut = np.array(self.y_prob >= self.best_thresh)
            test_predictions = (self.test_prob >= self.best_thresh).astype(int)
            test_accuracy = accuracy_score(self.y_test, test_predictions)
            return test_accuracy, f1_array[best_index]

        if self.scoring == 'signal_significance':
            significance_array = np.array([signal_significance(self.y_test, prediction) for prediction
                                           in predictions.T])
            best_index = np.nanargmax(significance_array)
            self.best_thresh = threshold[best_index]

            plt.plot(threshold, significance_array)
            plt.axvline(x=float(self.best_thresh), color='black', linestyle='--', label='Selected Threshold')
            plt.xlabel('threshold')
            plt.ylabel('Signal Significance')
            plt.title('Signal Significance metric for different thresholds')
            if save_path is not None:
                path = os.path.join(save_path, 'images')
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(os.path.join(path, 'Signal_significance.png'))
            plt.show()
            plt.clf()

            self.cut = np.array(self.y_prob >= self.best_thresh)
            test_predictions = (self.test_prob >= self.best_thresh).astype(int)
            test_accuracy = accuracy_score(self.y_test, test_predictions)
            return test_accuracy, significance_array[best_index]

        if self.scoring is None:
            self.cut = np.array(self.prediction.astype(bool))
            test_predictions = self.test_prediction
            test_accuracy = accuracy_score(self.y_test, test_predictions)
            return test_accuracy

    def test_on_train(self):
        train_predictions = (self.train_prob >= self.best_thresh).astype(int)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        if self.scoring is None:
            train_accuracy = accuracy_score(self.y_train, self.train_prediction)
            return train_accuracy
        elif self.scoring == 'f1':
            train_f1 = f1(self.y_train, train_predictions)[2]
            return train_accuracy, train_f1
        elif self.scoring == 'signal_significance':
            train_significance = signal_significance(self.y_train, train_predictions)
            return train_accuracy, train_significance

    def plot_probs(self, y_log=True, save_path=None):
        plt.hist([self.train_prob[self.y_train == i] for i in range(2)], histtype="step", bins=50, range=(0, 1),
                 label=["train_Background", "train_Signal"])
        plt.hist([self.test_prob[self.y_test == i] for i in range(2)], histtype="step", bins=50, range=(0, 1),
                 label=["test_Background", "test_Signal"])
        plt.axvline(x=float(self.best_thresh), color='black', linestyle='--', label='Selected Threshold')
        plt.title(f'Probability Distribution for test and train events')
        plt.xlabel("Computed probability")
        plt.ylabel("Number of events")
        if y_log:
            plt.yscale('log')
        plt.legend()
        if save_path is not None:
            path = os.path.join(save_path, 'images')
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, f'probability_distribution.png'))
        plt.show()
        plt.clf()

    def make_calibration_curve(self, save_path=None):
        prob_true, prob_pred = calibration_curve(self.y_test, self.rf_model_cal.predict_proba(self.X_test)[:, 1],
                                                 n_bins=10, strategy='uniform')

        plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='black')

        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend()
        if save_path is not None:
            path = os.path.join(save_path, 'images')
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, 'Calibration_curve.png'))
        plt.show()
        plt.clf()

    def get_features_importance(self):
        imp = self.rf_model.feature_importances_
        dic_imp = {}
        for i in range(len(self.features)):
            dic_imp[self.features[i]] = imp[i]
        return dic_imp


class ModelOutput:
    def __init__(self, path, grid_search=False):
        self.path = path
        self.grid = grid_search

        self.cut = np.load(f'{path}cut.npy', allow_pickle=True)
        self.out_prob = np.load(f'{path}probability_output.npy', allow_pickle=True)
        self.prediction = np.load(f'{path}prediction_output.npy', allow_pickle=True)
        self.threshold = np.load(f'{path}threshold.npy', allow_pickle=True)

    def images(self):
        images_dir = os.path.join(self.path, 'images')
        for filename in os.listdir(images_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, filename)
                with Image.open(image_path) as img:
                    img.show()

    def config(self):

        with open(f'{self.path}config.txt', 'r') as file:
            line_number = 0
            for line in file:
                line_number += 1
                if line_number > 8:
                    line = line.strip()
                    print(line)

        if self.grid:
            with open(f'{self.path}best_params.txt', 'r') as file:
                for line in file:
                    line = line.strip()
                    print(line)
