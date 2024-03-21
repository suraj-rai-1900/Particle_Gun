import create_file
import cut_tuning
import numpy as np
import os

df = create_file.relevant_df(apply_presel=False, select_labels=[1, 2])

save_path = '/home/surajrai1900/outputs/fq_cut/e_muf1/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Features: e/mu_likelihood ratio, pi0/e_likelihood ratio reco_electron_mom, reco_electron_dwall, reco_electron_towall,
# pe, pmu, ppi0, pi0_mass

# Scoring : f1, signal_significance, None

variables = ['e/mu_likelihood ratio', 'reco_electron_mom']
algorithm = 'fitqun'
scoring = 'f1'
cut_type = 'linear'

print(f'Variables: {variables}')
print(f'Algorithm: {algorithm}')
print(f'Scoring: {scoring}')
print(f'Cut type: {cut_type}')

tuner = cut_tuning.CutTuner(df, variables, algorithm, scoring=scoring)
tuner.relevant_variables()
if cut_type == 'linear':
    tuner.optimize_linear_cut()
elif cut_type == 'quadratic':
    tuner.optimize_quadratic_cut()

metric_value = tuner.print_metric(cut_type=cut_type)

coefficient = np.array(tuner.cut_coefficients)
coefficient_output = os.path.join(save_path, 'coefficients.npy')
np.save(coefficient_output, coefficient)

metric = np.array(metric_value)
metric_output = os.path.join(save_path, 'metric.npy')
np.save(metric_output, metric)
