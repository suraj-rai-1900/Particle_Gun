import itertools
import numpy as np
import cuts
from sklearn.metrics import accuracy_score
from utils.classification_metrics import f1, signal_significance


def tuner_output(path):
    cut_coeff = np.load(f'{path}coefficients.npy', allow_pickle=True)
    metric = np.load(f'{path}metric.npy', allow_pickle=True)
    with open(f'{path}config.txt', 'r') as file:
        line_number = 0
        for line in file:
            line_number += 1
            if line_number > 8:
                line = line.strip()
                print(line)
    return cut_coeff, metric


def optimum_vertical_cut(df, variable1, variable2, scoring=None):
    x = np.arange(0, np.nanmax(df[variable2]), 1)
    vertical_cut_1 = np.array((np.array(df[variable1]).reshape(1, -1) > x.reshape(-1, 1)))
    vertical_cut_2 = np.array((np.array(df[variable1]).reshape(1, -1) < x.reshape(-1, 1)))

    vertical_metric_1 = np.array([])
    vertical_metric_2 = np.array([])
    vertical_metric = 0
    for row in vertical_cut_1:
        if scoring is None:
            vertical_metric = accuracy_score(df['true_sig'], row)
        elif scoring == 'f1':
            vertical_metric = f1(df['true_sig'], row)[2]
        elif scoring == 'signal_significance':
            vertical_metric = signal_significance(df['true_sig'], row)

        vertical_metric_1 = np.append(vertical_metric_1, vertical_metric)

    vertical_metric = 0
    for row in vertical_cut_2:
        if scoring is None:
            vertical_metric = accuracy_score(df['true_sig'], row)
        elif scoring == 'f1':
            vertical_metric = f1(df['true_sig'], row)[2]
        elif scoring == 'signal_significance':
            vertical_metric = signal_significance(df['true_sig'], row)

        vertical_metric_2 = np.append(vertical_metric_2, vertical_metric)

    max_1 = np.nanargmax(vertical_metric_1)
    max_2 = np.nanargmax(vertical_metric_2)
    if vertical_metric_1[max_1] > vertical_metric_2[max_2]:
        return [x[max_1]], vertical_cut_1[max_1]
    else:
        return [x[max_2]], vertical_cut_2[max_2]


def finetune_linear(df, variable1, variable2, guess, algorithm, scoring=None, greater=False):
    if algorithm == 'fitqun':
        a = np.repeat(np.arange(guess[0] - 0.1, guess[0] + 0.1, 0.01), 20)
        b = np.tile(np.arange(guess[1] - 10, guess[1] + 10, 1), 20)
        cut_value = np.dot(np.array(df[variable2]).reshape(-1, 1), a.reshape(1, -1)) + b.reshape(1, -1)

        if greater:
            cut = (np.array(df[variable1]).reshape(-1, 1) > cut_value)
        else:
            cut = (np.array(df[variable1]).reshape(-1, 1) < cut_value)
    else:
        a = np.repeat(np.arange(guess[0] - 0.01, guess[0] + 0.01, 0.001), 20)
        b = np.tile(np.arange(guess[1] - 0.1, guess[1] + 0.1, 0.01), 20)
        cut_value = np.dot(np.array(df[variable2]).reshape(-1, 1), a.reshape(1, -1)) + b.reshape(1, -1)

        if greater:
            cut = (np.array(df[variable1]).reshape(-1, 1) > 10 ** cut_value)
        else:
            cut = (np.array(df[variable1]).reshape(-1, 1) < 10 ** cut_value)

    metric_value = 0
    metric_array = np.array([])
    for row in cut.T:
        if scoring is None:
            metric_value = accuracy_score(df['true_sig'], row)
        elif scoring == 'f1':
            metric_value = f1(df['true_sig'], row)[2]
        elif scoring == 'signal_significance':
            metric_value = signal_significance(df['true_sig'], row)

        metric_array = np.append(metric_array, metric_value)

    index = np.nanargmax(metric_array)

    return [a[index], b[index]]


def optimum_cut_linear(df, variable1, variable2, algorithm, scoring=None, greater=False):
    metric_array = np.array([])
    best_a_array = np.array([])
    best_b_array = np.array([])
    if algorithm == 'fitqun':
        for i in np.arange(-10, 10, 1):
            a = np.repeat(np.arange(i, i + 1, 0.1), 10)
            metric = np.array([])
            a_array = np.array([])
            b_array = np.array([])
            for j in np.arange(-1500, 1500, 100):
                b = np.tile(np.arange(j, j + 100, 10), 10)
                cut_value = np.dot(np.array(df[variable2]).reshape(-1, 1), a.reshape(1, -1)) + b.reshape(1, -1)
                if greater:
                    cut = (np.array(df[variable1]).reshape(-1, 1) > cut_value)
                else:
                    cut = (np.array(df[variable1]).reshape(-1, 1) < cut_value)

                a_array = np.append(a_array, a)
                b_array = np.append(b_array, b)

                metric_value = 0
                for row in cut.T:
                    if scoring is None:
                        metric_value = accuracy_score(df['true_sig'], row)
                    elif scoring == 'f1':
                        metric_value = f1(df['true_sig'], row)[2]
                    elif scoring == 'signal_significance':
                        metric_value = signal_significance(df['true_sig'], row)

                    if np.isnan(metric_value):
                        metric = np.append(metric, 0)
                    else:
                        metric = np.append(metric, metric_value)

            index = np.nanargmax(metric)
            best_a_array = np.append(best_a_array, a_array[index])
            best_b_array = np.append(best_b_array, b_array[index])
            metric_array = np.append(metric_array, metric[index])
    elif algorithm == 'softmax':
        for i in np.arange(-1, 1, 0.1):
            a = np.repeat(np.arange(i, i + 0.1, 0.01), 10)
            metric = np.array([])
            a_array = np.array([])
            b_array = np.array([])
            for j in np.arange(-12, 0, 1):
                b = np.tile(np.arange(j, j + 1, 0.1), 10)
                cut_value = np.dot(np.array(df[variable2]).reshape(-1, 1), a.reshape(1, -1)) + b.reshape(1, -1)
                if greater:
                    cut = (np.array(df[variable1]).reshape(-1, 1) > 10 ** cut_value)
                else:
                    cut = (np.array(df[variable1]).reshape(-1, 1) < 10 ** cut_value)

                a_array = np.append(a_array, a)
                b_array = np.append(b_array, b)

                metric_value = 0
                for row in cut.T:
                    if scoring is None:
                        metric_value = accuracy_score(df['true_sig'], row)
                    elif scoring == 'f1':
                        metric_value = f1(df['true_sig'], row)[2]
                    elif scoring == 'signal_significance':
                        metric_value = signal_significance(df['true_sig'], row)

                    if np.isnan(metric_value):
                        metric = np.append(metric, 0)
                    else:
                        metric = np.append(metric, metric_value)

            index = np.nanargmax(metric)
            best_a_array = np.append(best_a_array, a_array[index])
            best_b_array = np.append(best_b_array, b_array[index])
            metric_array = np.append(metric_array, metric[index])
    best_index = np.nanargmax(metric_array)

    return [best_a_array[best_index], best_b_array[best_index]]


def optimum_cut_quadratic(df, variable1, variable2, algorithm, scoring=None, greater=False):
    metric_array = np.array([])
    best_a_array = np.array([])
    best_b_array = np.array([])
    best_c_array = np.array([])
    if algorithm == 'fitqun':
        for k in np.arange(-1, 1, 0.1):
            a = np.arange(k, k + 0.1, 0.001)
            metric = np.array([])
            a_array = np.array([])
            b_array = np.array([])
            c_array = np.array([])
            for i in np.arange(-10, 10, 1):
                b = np.repeat(np.arange(i, i + 1, 0.1), 10)
                for j in np.arange(-1500, 1500, 100):
                    c = np.tile(np.arange(j, j + 100, 10), 10)
                    cut_value = (np.dot(np.array(df[variable2] ** 2).reshape(-1, 1), a.reshape(1, -1)) +
                                 np.dot(np.array(df[variable2]).reshape(-1, 1), b.reshape(1, -1)) + c.reshape(1, -1))
                    if greater:
                        cut = (np.array(df[variable1]).reshape(-1, 1) > cut_value)
                    else:
                        cut = (np.array(df[variable1]).reshape(-1, 1) < cut_value)

                    a_array = np.append(a_array, a)
                    b_array = np.append(b_array, b)
                    c_array = np.append(c_array, c)

                    metric_value = 0
                    for row in cut.T:
                        if scoring is None:
                            metric_value = accuracy_score(df['true_sig'], row)
                        elif scoring == 'f1':
                            metric_value = f1(df['true_sig'], row)[2]
                        elif scoring == 'signal_significance':
                            metric_value = signal_significance(df['true_sig'], row)

                        if np.isnan(metric_value):
                            metric = np.append(metric, 0)
                        else:
                            metric = np.append(metric, metric_value)

                index = np.nanargmax(metric)
                best_a_array = np.append(best_a_array, a_array[index])
                best_b_array = np.append(best_b_array, b_array[index])
                best_c_array = np.append(best_c_array, c_array[index])
                metric_array = np.append(metric_array, metric[index])
    elif algorithm == 'softmax':
        for k in np.arange(-0.1, 0.1, 0.01):
            a = np.arange(k, k + 0.01, 0.0001)
            metric = np.array([])
            a_array = np.array([])
            b_array = np.array([])
            c_array = np.array([])
            for i in np.arange(-1, 1, 0.1):
                b = np.repeat(np.arange(i, i + 0.1, 0.01), 10)
                for j in np.arange(-12, 0, 1):
                    c = np.tile(np.arange(j, j + 1, 0.1), 10)
                    cut_value = (np.dot(np.array(df[variable2] ** 2).reshape(-1, 1), a.reshape(1, -1)) +
                                 np.dot(np.array(df[variable2]).reshape(-1, 1), b.reshape(1, -1)) + c.reshape(1, -1))
                    if greater:
                        cut = (np.array(df[variable1]).reshape(-1, 1) > 10 ** cut_value)
                    else:
                        cut = (np.array(df[variable1]).reshape(-1, 1) < 10 ** cut_value)

                    a_array = np.append(a_array, a)
                    b_array = np.append(b_array, b)
                    c_array = np.append(c_array, c)

                    metric_value = 0
                    for row in cut.T:
                        if scoring is None:
                            metric_value = accuracy_score(df['true_sig'], row)
                        elif scoring == 'f1':
                            metric_value = f1(df['true_sig'], row)[2]
                        elif scoring == 'signal_significance':
                            metric_value = signal_significance(df['true_sig'], row)

                        if np.isnan(metric_value):
                            metric = np.append(metric, 0)
                        else:
                            metric = np.append(metric, metric_value)

                index = np.nanargmax(metric)
                best_a_array = np.append(best_a_array, a_array[index])
                best_b_array = np.append(best_b_array, b_array[index])
                best_c_array = np.append(best_c_array, c_array[index])
                metric_array = np.append(metric_array, metric[index])
    best_index = np.nanargmax(metric_array)

    return [best_a_array[best_index], best_b_array[best_index], best_c_array[best_index]]


def finetune_quadratic(df, variable1, variable2, guess, algorithm, scoring=None, greater=False):
    if algorithm == 'fitqun':
        a = np.arange(guess[0] - 0.001, guess[0] + 0.001, 0.000005)
        b = np.repeat(np.arange(guess[1] - 0.1, guess[1] + 0.1, 0.01), 20)
        c = np.tile(np.arange(guess[2] - 10, guess[2] + 10, 1), 20)

        cut_value = (np.dot(np.array(df[variable2] ** 2).reshape(-1, 1), a.reshape(1, -1)) +
                     np.dot(np.array(df[variable2]).reshape(-1, 1), b.reshape(1, -1))) + c.reshape(1, -1)

        if greater:
            cut = (np.array(df[variable1]).reshape(-1, 1) > cut_value)
        else:
            cut = (np.array(df[variable1]).reshape(-1, 1) < cut_value)
    else:
        a = np.arange(guess[0] - 0.0001, guess[0] + 0.0001, 0.0000005)
        b = np.repeat(np.arange(guess[1] - 0.01, guess[1] + 0.01, 0.001), 20)
        c = np.tile(np.arange(guess[2] - 0.1, guess[2] + 0.1, 0.01), 20)

        cut_value = (np.dot(np.array(df[variable2] ** 2).reshape(-1, 1), a.reshape(1, -1)) +
                     np.dot(np.array(df[variable2]).reshape(-1, 1), b.reshape(1, -1))) + c.reshape(1, -1)

        if greater:
            cut = (np.array(df[variable1]).reshape(-1, 1) > 10 ** cut_value)
        else:
            cut = (np.array(df[variable1]).reshape(-1, 1) < 10 ** cut_value)

    metric_value = 0
    metric_array = np.array([])
    for row in cut.T:
        if scoring is None:
            metric_value = accuracy_score(df['true_sig'], row)
        elif scoring == 'f1':
            metric_value = f1(df['true_sig'], row)[2]
        elif scoring == 'signal_significance':
            metric_value = signal_significance(df['true_sig'], row)

        metric_array = np.append(metric_array, metric_value)

    index = np.nanargmax(metric_array)
    return [a[index], b[index], c[index]]


class CutTuner:
    def __init__(self, df, variables, cut_algorithm, scoring=None):
        self.variables = variables
        self.df = df
        self.scoring = scoring
        self.cut_algorithm = cut_algorithm
        self.y = None
        self.X = None
        self.relation = None
        self.cut_coefficients = None
        self.vertical_cut = None

    def relevant_variables(self):
        if self.cut_algorithm == 'fitqun':
            if cuts.check_labels(self.df, [1, 2]):
                self.y = ['e/mu_likelihood ratio']
                self.relation = [True]
            elif cuts.check_labels(self.df, [1, 3]):
                self.y = ['pi0/e_likelihood ratio']
                self.relation = [False]
            elif cuts.check_labels(self.df, [1, 2, 3]) or cuts.check_labels(self.df, [0, 1, 2, 3]):
                if 'e/mu_likelihood ratio' in self.variables and 'pi0/e_likelihood ratio' in self.variables:
                    self.y = ['e/mu_likelihood ratio', 'pi0/e_likelihood ratio']
                    self.relation = [True, False]
                elif 'e/mu_likelihood ratio' in self.variables and 'pi0/e_likelihood ratio' not in self.variables:
                    self.y = ['e/mu_likelihood ratio']
                    self.relation = [True]
                elif 'e/mu_likelihood ratio' not in self.variables and 'pi0/e_likelihood ratio' in self.variables:
                    self.y = ['pi0/e_likelihood ratio']
                    self.relation = [False]

        elif self.cut_algorithm == 'softmax':
            if cuts.check_labels(self.df, [1, 2]):
                if 'pe' in self.variables and 'pmu' in self.variables:
                    self.y = ['pe', 'pmu']
                    self.relation = [True, False]
                elif 'pe' in self.variables and 'pmu' not in self.variables:
                    self.y = ['pe']
                    self.relation = [True]
                elif 'pe' not in self.variables and 'pmu' in self.variables:
                    self.y = ['pmu']
                    self.relation = [False]

            elif cuts.check_labels(self.df, [1, 3]):
                if 'pe' in self.variables and 'ppi0' in self.variables:
                    self.y = ['pe', 'ppi0']
                    self.relation = [True, False]
                elif 'pe' in self.variables and 'ppi0' not in self.variables:
                    self.y = ['pe']
                    self.relation = [True]
                elif 'pe' not in self.variables and 'ppi0' in self.variables:
                    self.y = ['ppi0']
                    self.relation = [False]
            elif cuts.check_labels(self.df, [1, 2, 3]):
                self.y = ['pe', 'pmu', 'ppi0']
                self.relation = [True, False, False]
            elif cuts.check_labels(self.df, [1, 2, 3]):
                self.y = ['pe', 'pmu', 'ppi0', 'pgamma']
                self.relation = [True, False, False, False]

        self.X = [variable for variable, y_value in zip(self.variables, itertools.cycle(self.y))
                  if variable != y_value]

    def optimize_linear_cut(self):
        self.cut_coefficients = {}
        df = self.df
        for i in range(len(self.y)):
            for j in range(len(self.X)):
                linear_metric = 0
                guess = optimum_cut_linear(df, self.y[i], self.X[j], scoring=self.scoring, greater=self.relation[i],
                                           algorithm=self.cut_algorithm)
                if self.relation[i]:
                    linear_cut = (df[self.y[i]] > guess[0] * df[self.X[j]] + guess[1])
                else:
                    linear_cut = (df[self.y[i]] < guess[0] * df[self.X[j]] + guess[1])

                x, self.vertical_cut = optimum_vertical_cut(df, self.y[i], self.X[j], scoring=self.scoring)
                if self.scoring is None:
                    linear_metric = [accuracy_score(df['true_sig'], self.vertical_cut), accuracy_score(df['true_sig'],
                                                                                                       linear_cut)]
                elif self.scoring == 'f1':
                    linear_metric = f1(df['true_sig'], [self.vertical_cut, linear_cut])[2]
                elif self.scoring == 'signal_significance':
                    linear_metric = signal_significance(df['true_sig'], [self.vertical_cut, linear_cut])

                if linear_metric[0] < linear_metric[1]:
                    self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'] = finetune_linear(df, self.y[i], self.X[j],
                                                                                           guess=guess,
                                                                                           scoring=self.scoring,
                                                                                           greater=self.relation[i],
                                                                                           algorithm=self.cut_algorithm)
                    if self.cut_algorithm == 'fitqun':
                        if self.relation[i]:
                            df = df[
                                df[self.y[i]] > self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]]
                                + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1]]
                        else:
                            df = df[
                                df[self.y[i]] < self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]]
                                + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1]]
                    else:
                        if self.relation[i]:
                            df = df[df[self.y[i]] > 10 ** (
                                        self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]]
                                        + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1])]
                        else:
                            df = df[df[self.y[i]] < 10 ** (
                                        self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]]
                                        + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1])]
                else:
                    self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'] = x
                    df = df[self.vertical_cut]

    def optimize_quadratic_cut(self):
        self.cut_coefficients = {}
        df = self.df
        for i in range(len(self.y)):
            for j in range(len(self.X)):
                guess = optimum_cut_quadratic(df, self.y[i], self.X[j], scoring=self.scoring, greater=self.relation[i],
                                              algorithm=self.cut_algorithm)
                self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'] = finetune_quadratic(df, self.y[i], self.X[j],
                                                                                          guess=guess,
                                                                                          scoring=self.scoring,
                                                                                          greater=self.relation[i],
                                                                                          algorithm=self.cut_algorithm)
                if self.cut_algorithm == 'fitqun':
                    if self.relation[i]:
                        df = df[df[self.y[i]] >
                                self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]] * df[self.X[j]]
                                + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * df[self.X[j]]
                                + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2]]
                    else:
                        df = df[df[self.y[i]] >
                                self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]] * df[self.X[j]]
                                + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * df[self.X[j]]
                                + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2]]
                else:
                    if self.relation[i]:
                        df = df[df[self.y[i]] >
                                10 ** (self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]] * df[
                                       self.X[j]]
                                       + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * df[self.X[j]]
                                       + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2])]
                    else:
                        df = df[df[self.y[i]] >
                                10 ** (self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * df[self.X[j]] * df[
                                       self.X[j]]
                                       + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * df[self.X[j]]
                                       + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2])]

    def print_metric(self, cut_type):
        metric = {}
        for i in range(len(self.y)):
            for j in range(len(self.X)):
                if self.cut_algorithm == 'fitqun':
                    if cut_type == 'linear':
                        if len(self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}']) == 2:
                            if self.relation[i]:
                                cut = (self.df[self.y[i]] > self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] *
                                       self.df[self.X[j]] +
                                       self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1])
                            else:
                                cut = (self.df[self.y[i]] < self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] *
                                       self.df[self.X[j]] +
                                       self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1])
                        else:
                            cut = self.vertical_cut
                    elif cut_type == 'quadratic':
                        if self.relation[i]:
                            cut = (self.df[self.y[i]] >
                                   self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * self.df[self.X[j]] *
                                   self.df[self.X[j]]
                                   + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * self.df[self.X[j]]
                                   + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2])
                        else:
                            cut = (self.df[self.y[i]] <
                                   self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * self.df[self.X[j]] *
                                   self.df[self.X[j]]
                                   + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * self.df[self.X[j]]
                                   + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2])
                else:
                    if cut_type == 'linear':
                        if len(self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}']) == 2:
                            if self.relation[i]:
                                cut = (self.df[self.y[i]] > 10 ** (
                                            self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] *
                                            self.df[self.X[j]] +
                                            self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1]))
                            else:
                                cut = (self.df[self.y[i]] < 10 ** (
                                            self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] *
                                            self.df[self.X[j]] +
                                            self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1]))
                        else:
                            cut = self.vertical_cut
                    elif cut_type == 'quadratic':
                        if self.relation[i]:
                            cut = (self.df[self.y[i]] >
                                   10 ** (self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * self.df[self.X[j]] *
                                          self.df[self.X[j]]
                                          + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * self.df[self.X[j]]
                                          + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2]))
                        else:
                            cut = (self.df[self.y[i]] <
                                   10 ** (self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][0] * self.df[self.X[j]] *
                                          self.df[self.X[j]]
                                          + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][1] * self.df[self.X[j]]
                                          + self.cut_coefficients[f'{self.y[i]} Vs {self.X[j]}'][2]))
                if self.scoring is None:
                    metric[f'{self.y[i]} Vs {self.X[j]}'] = accuracy_score(self.df['true_sig'], cut)
                elif self.scoring == 'f1':
                    metric[f'{self.y[i]} Vs {self.X[j]}'] = f1(self.df['true_sig'], cut)[2]
                elif self.scoring == 'signal_significance':
                    metric[f'{self.y[i]} Vs {self.X[j]}'] = signal_significance(self.df['true_sig'], cut)

        return metric
