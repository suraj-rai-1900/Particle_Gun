import numpy as np
import pandas as pd
from watchmal_dependencies import binning as bins
from sklearn.metrics import confusion_matrix


def calculate_tpr_fpr(true_signal, prediction):
    if not isinstance(np.array(prediction)[0], (list, np.ndarray, pd.Series)):
        prediction = [prediction]

    tpr_array = []
    fpr_array = []
    for i in range(len(prediction)):
        tn, fp, fn, tp = confusion_matrix(np.array(true_signal), np.array(prediction[i])).ravel()
        tpr_array.append(tp / (tp + fn))
        fpr_array.append(fp / (fp + tn))

    if len(tpr_array) == 1:
        return tpr_array[0], fpr_array[0]
    else:
        return tpr_array, fpr_array


def signal_significance(true_sig, cut_array, w=1):

    """
    Uses the ratio of signal events passing the cuts to the statistical error , i.e. sqrt(S+B)  as
    a metric to tune a particular cut. The background has been given a weight (to be kept greater than 1)
    for efficient background removal, default value 1.
    """
    if not isinstance(np.array(cut_array)[0], (list, np.ndarray, pd.Series)):
        cut_array = [cut_array]
    significance_array = []
    for cut in cut_array:
        true_sig_cut = true_sig[cut.astype(bool)]

        sig = np.sum(true_sig_cut == 1)
        bg = np.sum(true_sig_cut == 0)
        sg_significance = sig / ((sig + w*bg) ** 0.5)
        significance_array.append(sg_significance)
    if len(significance_array) == 1:
        return significance_array[0]
    else:
        return significance_array


def f1(true_sig, cut_array, w=1):
    """
    Uses f1 score defined as the harmonic mean of precision and recall as a metric to tune the cuts.
    Recall has been given a weight 'w', a greater value of w results in a bigger value of recall.
    Default value of w is 1.
    """
    if not isinstance(np.array(cut_array)[0], (list, np.ndarray, pd.Series)):
        cut_array = [cut_array]
    f1_array = []
    recall_array = []
    precision_array = []
    for cut in cut_array:
        # Calculate precision, recall, and F1-score
        true_positives = np.sum((true_sig == 1) & (cut.astype(int) == 1))
        false_positives = np.sum((true_sig == 0) & (cut.astype(int) == 1))
        false_negatives = np.sum((true_sig == 1) & (cut.astype(int) == 0))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = (1 + w) * (precision * recall) / (w*precision + recall)

        f1_array.append(f1_score)
        precision_array.append(precision)
        recall_array.append(recall)

    if len(f1_array) == 1:
        return precision_array[0], recall_array[0], f1_array[0]
    else:
        return precision_array, recall_array, f1_array


def sg_eff(true_sig, cut_array):
    """
    Gives the signal efficiency for a cut or list of cuts
    """
    if not isinstance(np.array(cut_array)[0], (list, np.ndarray, pd.Series)):
        cut_array = [cut_array]

    true_sig = true_sig.astype(bool)
    eff_array = []
    for cut in cut_array:
        true_sig_cut = true_sig[cut]
        total_sig_before_cut = np.sum(true_sig)
        selected_as_signal = np.sum(true_sig_cut)
        sig_eff = selected_as_signal / total_sig_before_cut
        eff_array.append(sig_eff)

    if len(eff_array) == 1:
        return eff_array[0]
    else:
        return eff_array


def bg_rej(true_sig, cut_array):
    """
    Gives the background rejection for a specific cut or list of cuts
    """
    if not isinstance(np.array(cut_array)[0], (list, np.ndarray, pd.Series)):
        cut_array = [cut_array]

    true_sig = true_sig.astype(bool)
    rej_array = []
    for cut in cut_array:
        true_sig_cut = true_sig[cut]
        total_bg_before_cut = np.sum(~true_sig)
        selected_as_signal = np.sum(~true_sig_cut)
        bg_re = 1 - selected_as_signal / total_bg_before_cut
        rej_array.append(bg_re)

    if len(rej_array) == 1:
        return rej_array[0]
    else:
        return rej_array


def cut_with_fixed_efficiency(df, probabilities, particle_label, efficiency):

    if not isinstance(particle_label, (list, np.ndarray, pd.Series)):
        particle_label = [particle_label]
    if not isinstance(np.array(probabilities)[0], (list, np.ndarray, pd.Series)):
        probabilities = [probabilities]

    cuts = []
    thresholds = []
    for i in range(len(probabilities)):
        try:
            threshold = np.quantile(probabilities[i][df['h5_labels'].isin(particle_label)], 1 - efficiency)
            thresholds.append(threshold)
        except IndexError as ex:
            raise ValueError("There are zero selected events so cannot calculate a cut with any efficiency.") from ex
        cuts.append(np.array(np.array(probabilities[i]) > threshold))
    if len(thresholds) == 1:
        return cuts[0], thresholds[0]
    else:
        return cuts, thresholds


def cut_with_fixed_rejection(df, probabilities, particle_label, rejection):

    if not isinstance(particle_label, (list, np.ndarray, pd.Series)):
        particle_label = [particle_label]
    if not isinstance(np.array(probabilities)[0], (list, np.ndarray, pd.Series)):
        probabilities = [probabilities]

    cuts = []
    thresholds = []
    for i in range(len(probabilities)):
        try:
            threshold = np.quantile(probabilities[i][df['h5_labels'].isin(particle_label)], rejection)
            thresholds.append(threshold)
        except IndexError as ex:
            raise ValueError("There are zero selected events so cannot calculate a cut.") from ex
        cuts.append(np.array(np.array(probabilities[i]) > threshold))
    if len(thresholds) == 1:
        return cuts[0], thresholds[0]
    else:
        return cuts, thresholds


def cut_with_constant_binned_efficiency(df, probabilities, particle_label, efficiency, binning):

    if not isinstance(particle_label, (list, np.ndarray, pd.Series)):
        particle_label = [particle_label]
    if not isinstance(np.array(probabilities)[0], (list, np.ndarray, pd.Series)):
        probabilities = [probabilities]

    cuts = []
    thresholds_array = []
    for i in range(len(probabilities)):
        binned_probabilities = bins.apply_binning(probabilities[i], binning,
                                                  selection=(df['h5_labels'].isin(particle_label)))
        thresholds = bins.binned_quantiles(binned_probabilities, 1 - efficiency)
        # put inf as first and last threshold for overflow bins
        padded_thresholds = np.concatenate(([np.inf], thresholds, [np.inf]))
        thresholds_array.append(padded_thresholds)
        cuts.append(np.array(np.array(probabilities[i]) > padded_thresholds[binning[1]]))
    if len(thresholds_array) == 1:
        return cuts[0], thresholds_array[0]
    else:
        return cuts, thresholds_array


def cut_with_constant_binned_rejection(df, probabilities, particle_label, rejection, binning):
    if not isinstance(particle_label, (list, np.ndarray, pd.Series)):
        particle_label = [particle_label]
    if not isinstance(np.array(probabilities)[0], (list, np.ndarray, pd.Series)):
        probabilities = [probabilities]

    cuts = []
    thresholds_array = []
    for i in range(len(probabilities)):
        binned_probabilities = bins.apply_binning(probabilities[i], binning,
                                                  selection=(df['h5_labels'].isin(particle_label)))
        thresholds = bins.binned_quantiles(binned_probabilities, rejection)
        # put inf as first and last threshold for overflow bins
        padded_thresholds = np.concatenate(([np.inf], thresholds, [np.inf]))
        thresholds_array.append(padded_thresholds)
        cuts.append(np.array(np.array(probabilities[i]) > padded_thresholds[binning[1]]))
    if len(thresholds_array) == 1:
        return cuts[0], thresholds_array[0]
    else:
        return cuts, thresholds_array
