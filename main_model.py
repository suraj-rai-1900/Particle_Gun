import create_file
import Classification_model
from utils.plot import roc
import numpy as np
import os

# df = create_file.relevant_df(apply_presel=False, select_labels=[1, 2])
df = create_file.read_folder(select_labels=[1, 2])

save_path = '/home/surajrai1900/neut/outputs/gbdt/fq_emu'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Features: e/mu_likelihood ratio, pi0/e_likelihood ratio, e_likelihood, mu_likelihood, pi0_likelihood,
#           reco_electron_mom, reco_electron_dwall, reco_electron_towall, pe, pmu, ppi0, pi0_mass

# Models : gbdt, xgboost, adaboost

# Scoring : f1, signal_significance, None

train_col = ['elikelihood', 'mulikelihood', 'mom', 'towall', 'dwall', 'sig']
train_labels = [1, 2]
grid_search = False
model_name = 'gbdt'
scoring = 'f1'

print(f'train features: {train_col}')
print(f'train labels: {train_labels}')
print(f'grid search: {grid_search}')
print(f'model name: {model_name}')
print(f'Scoring: {scoring}')
print(f'Save path: {save_path}')
if not grid_search:
    print('The hyperparameters are: n_estimators = 300, max_depth = 13, learning rate = 0.01')

model = Classification_model.CutEngine(df, train_col, model_name, grid_search, scoring)
model.prepare_data()

if grid_search:
    best_params = model.grid_search()
    bestPara_output = os.path.join(save_path, 'best_params.txt')
    with open(bestPara_output, 'w') as file:
        for key, value in best_params.items():
            file.write('%s: %s\n' % (key, value))

model.train()

if scoring is None:
    test_acc = model.test()
    train_acc = model.test_on_train()
    print(f'The train accuracy is:{train_acc}')
    print('\n')
    print(f'The test accuracy is:{test_acc}')
elif scoring == 'f1':
    test_acc, test_f1 = model.test(save_path)
    train_acc, train_f1 = model.test_on_train()
    print(f'The train accuracy is:{train_acc}')
    print(f'The train f1 score is: {train_f1}')
    print('\n')
    print(f'The test accuracy is:{test_acc}')
    print(f'The test f1 score is: {test_f1}')
elif scoring == 'signal_significance':
    test_acc, test_significance = model.test(save_path)
    train_acc, train_significance = model.test_on_train()
    print(f'The train accuracy is:{train_acc}')
    print(f'The train signal significance score is: {train_significance}')
    print('\n')
    print(f'The test accuracy is:{test_acc}')
    print(f'The test signal significance score is: {test_significance}')

print(f'The best threshold is : {model.best_thresh}')
model_output = np.array(model.y_prob)
model_prediction = np.array(model.prediction)
cut = np.array(model.cut)
threshold = np.array(model.best_thresh)

model.plot_probs(save_path=save_path)
model.plot_probs(save_path=save_path)

probability_output = os.path.join(save_path, 'probability_output.npy')
prediction_output = os.path.join(save_path, 'prediction_output.npy')
threshold_output = os.path.join(save_path, 'threshold.npy')
cut_output = os.path.join(save_path, 'cut.npy')

np.save(probability_output, model_output)
np.save(prediction_output, model_prediction)
np.save(threshold_output, threshold)
np.save(cut_output, cut)

print('\n')

feature_imp = model.get_features_importance()
for key in feature_imp:
    print(f'{key} : {feature_imp[key]}')

fig_rejection, _ = roc(df['true_sig'], model.y_prob, mode='rejection', algorithms=model_name, cut=model.cut)
fig_rejection.savefig(os.path.join(save_path, 'images', 'rejection_roc_plot.png'))
fig_efficiency, _ = roc(df['true_sig'], model.y_prob, mode='efficiency', algorithms=model_name, cut=model.cut)
fig_efficiency.savefig(os.path.join(save_path, 'images', 'efficiency_roc_plot.png'))


