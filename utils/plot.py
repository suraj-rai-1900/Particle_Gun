import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn import metrics
from watchmal_dependencies import binning as bins
from utils.classification_metrics import calculate_tpr_fpr

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 28}
mpl.rc('font', **font)
mpl.rcParams['figure.figsize'] = (12, 9)
mpl.rcParams["figure.autolayout"] = True


def sig_frac(df):

    label_list = [0, 1]
    legends = ['Background', 'Signal']
    counts = [len(df[df['true_sig'] == label]) / len(df) for label in label_list]
    plt.bar(legends, counts)
    plt.xlabel("Class")
    plt.ylabel("Class Fraction [a.u.]")
    plt.title("Signal and Background Fraction Distribution")
    plt.xticks(rotation=30, ha='right')
    plt.show()
    plt.clf()


def class_frac(df):

    label_list = [0, 1, 2, 3]
    legends = ['gamma', 'e', 'mu', 'pi0']
    counts = [len(df[df["h5_labels"] == label]) / len(df) for label in label_list]
    plt.bar(legends, counts)
    plt.xlabel("Particle Type")
    plt.ylabel("Class Fraction [a.u.]")
    plt.title("Class Fraction Distribution")
    plt.xticks(rotation=30, ha='right')
    plt.show()
    plt.clf()


def sg_bg_hist2d(df, key1, key2, x_range, y_range, bin_x, bin_y, logbin_y=False, true_sig=True, cut=None):

    if true_sig:
        sig_cut = (df['true_sig'] == 1)
        bg_cut = (df['true_sig'] == 0)
    else:
        sig_cut = (cut == 1)
        bg_cut = (cut == 0)

    df_sg = df[sig_cut]
    df_bg = df[bg_cut]

    if logbin_y:
        bins_y = [10 ** x for x in np.linspace(-12, 0, 50)]
        yscale = "log"
    else:
        bins_y = bin_y
        yscale = "linear"

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    hist, x_edges, y_edges, im = ax.hist2d(df_sg[key1], df_sg[key2], bins=[bin_x, bins_y], range=[x_range, y_range],
                                           norm=mpl.colors.LogNorm())
    ax.set_xlabel(key1, fontsize=15)
    ax.set_ylabel(key2, fontsize=20)
    ax.set_yscale(yscale)
    ax.set_title('Signal', fontsize=20)
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    hist, x_edges, y_edges, im = ax.hist2d(df_bg[key1], df_bg[key2], bins=[bin_x, bins_y], range=[x_range, y_range],
                                           norm=mpl.colors.LogNorm())
    ax.set_xlabel(key1, fontsize=15)
    ax.set_ylabel(key2, fontsize=20)
    ax.set_yscale(yscale)
    ax.set_title('Background', fontsize=20)
    plt.colorbar(im, ax=ax)

    return fig, axes


def sg_bg_hist(df, variable, bin_number, true_sig=True, cut=None):

    fig, ax = plt.subplots(1, 1)
    if true_sig:
        sig_cut = (df['true_sig'] == 1)
        bg_cut = (df['true_sig'] == 0)
    else:
        sig_cut = (cut == 1)
        bg_cut = (cut == 0)
    ax.hist(df[sig_cut][variable], bins=bin_number, label='Signal' + variable, histtype='step')
    ax.hist(df[bg_cut][variable], bins=bin_number, label='Background' + variable, histtype='step')
    ax.set_xlabel(variable)
    ax.set_ylabel('Count')
    ax.legend()
    plt.show()
    return fig, ax


def sel_comp(df, sg_label, bg_label, cut_array, algorithms):

    label_map = ['gamma', 'e', 'mu', 'pi0']
    color_map = ['cyan', 'red', 'blue', 'green']
    lines = ['--', '-', '-.', ':']

    if not isinstance(sg_label, (list, np.ndarray)):
        sg_label = [sg_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]
    if not isinstance(np.array(cut_array)[0], (list, np.ndarray, pd.Series)):
        cut_array = [cut_array]
    if not isinstance(algorithms, (list, np.ndarray)):
        algorithms = [algorithms]

    df_cut = []
    for cut in cut_array:
        df_cut.append(df[cut])

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    for i in range(len(cut_array)):
        for label in sg_label:
            ax.hist(df_cut[i][df_cut[i]['h5_labels'] == label]['reco_electron_mom'], bins=15, range=(0, 1500),
                    label='sig' + ' ' + label_map[label] + ' ' + algorithms[i], histtype='step', linewidth=1.5,
                    color=color_map[label], linestyle=lines[i])
        for label in bg_label:
            ax.hist(df_cut[i][df_cut[i]['h5_labels'] == label]['reco_electron_mom'], bins=15, range=(0, 1500),
                    label='bg' + ' ' + label_map[label] + ' ' + algorithms[i], histtype='step', linewidth=1.5,
                    color=color_map[label], linestyle=lines[i])
    ax.set_xlabel('Reco Momentum', fontsize=20)
    ax.set_ylabel('Event count', fontsize=20)
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1]
    count = 0
    for i in range(len(cut_array)):
        for k in range(len(cut_array)):
            if k > i:

                for label in sg_label:
                    hist_cut1_sg, sig_x_edges = np.histogram(df_cut[i][df_cut[i]['h5_labels'] ==
                                                             label]['reco_electron_mom'], range=(0, 1500), bins=10)
                    hist_cut2_sg, sig_x_edges = np.histogram(df_cut[k][df_cut[k]['h5_labels'] ==
                                                             label]['reco_electron_mom'], range=(0, 1500), bins=10)
                    vals_x_sig = np.array([])
                    vals_y_sig = np.array([])
                    for j in range(10):
                        vals_x_sig = np.concatenate((vals_x_sig, np.linspace(sig_x_edges[j], sig_x_edges[j + 1], 10)))
                        vals_y_sig = np.concatenate((vals_y_sig, np.array([hist_cut2_sg[j] / hist_cut1_sg[j] for
                                                                           i in range(10)])))
                    ax.plot(vals_x_sig, vals_y_sig, label='Signal ' + label_map[label] +
                            f' {algorithms[k]}/{algorithms[i]}', linewidth=1.5, linestyle=lines[count],
                            color=color_map[label])

                for label in bg_label:
                    hist_cut1_bg, bg_x_edges = np.histogram(df_cut[i][df_cut[i]['h5_labels'] ==
                                                            label]['reco_electron_mom'], range=(0, 1500), bins=10)
                    hist_cut2_bg, bg_x_edges = np.histogram(df_cut[k][df_cut[k]['h5_labels'] ==
                                                            label]['reco_electron_mom'], range=(0, 1500), bins=10)
                    vals_x_bg = np.array([])
                    vals_y_bg = np.array([])
                    for j in range(10):
                        vals_x_bg = np.concatenate((vals_x_bg, np.linspace(bg_x_edges[j], bg_x_edges[j + 1], 10)))
                        vals_y_bg = np.concatenate((vals_y_bg, np.array([hist_cut2_bg[j] / hist_cut1_bg[j] for
                                                                         i in range(10)])))
                    ax.plot(vals_x_bg, vals_y_bg, label='Background' + ' ' + label_map[label] + ' ' +
                            f' {algorithms[k]}/{algorithms[i]}', linewidth=1.5, linestyle=lines[count],
                            color=color_map[label])
                count += 1
    ax.set_xlabel("Reco Momentum", fontsize=20)
    ax.set_ylabel('Cut Ratios', fontsize=20)
    ax.hlines(1, 0, 1500, colors='black', linewidth=2)
    ax.set_ylim(0, 3)
    ax.legend()

    plt.show()

    return fig, axes


def roc(true_sig, sig_prob_array, algorithms, cut=None, ax=None, x_lim=None, y_lim=None, y_log=None,
        x_log=None, legend='best', mode='rejection'):

    color_map = ['blue', 'orange', 'green']

    if not isinstance(np.array(sig_prob_array)[0], (list, np.ndarray, pd.Series)):
        sig_prob_array = [sig_prob_array]

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    for i in range(len(sig_prob_array)):
        fpr, tpr, _ = metrics.roc_curve(true_sig, sig_prob_array[i])
        auc = metrics.auc(fpr, tpr)
        label = f"{algorithms[i]} (AUC={auc:.4f})"
        if mode == 'rejection':
            if y_log is None:
                y_log = True
            with np.errstate(divide='ignore'):
                ax.plot(tpr, 1 / fpr, label=label)
            x_label = "Signal Efficiency"
            y_label = "Background Rejection"
        elif mode == 'efficiency':
            ax.plot(fpr, tpr, label=label)
            x_label = "Background_Mis_ID rate"
            y_label = "Signal Efficiency"
        else:
            raise ValueError(f"Unknown ROC curve mode '{mode}'.")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    if cut is not None:
        tpr_array, fpr_array = calculate_tpr_fpr(true_sig, cut)
        if not isinstance(np.array(cut)[0], (list, np.ndarray, pd.Series)):
            cut = [cut]
        if not isinstance(tpr_array, (list, np.ndarray)):
            tpr_array = [tpr_array]
            fpr_array = [fpr_array]
        for i in range(len(cut)):
            if cut[i] is not None:
                if mode == 'rejection':
                    ax.scatter(tpr_array[i], 1/fpr_array[i], color=color_map[i], marker='o', s=40)
                else:
                    ax.scatter(fpr_array[i], tpr_array[i], color=color_map[i], marker='o', s=40)

    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    if y_lim:
        ax.set_ylim(y_lim)
    if x_lim:
        ax.set_xlim(x_lim)
    if legend:
        ax.legend(loc=legend)
    return fig, ax


def efficiency_profile(df, particle_label, binning, algorithms, threshold_array=None,
                       signal_probability_array=None, x_label="", y_label="", ax=None, legend='best',
                       y_lim=None, errors=True, cut_array=None, reverse=False, **plot_args):

    if not isinstance(particle_label, (list, np.ndarray, pd.Series)):
        particle_label = [particle_label]

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    plot_args.setdefault('lw', 2)

    cuts = []
    if cut_array is None:
        if not isinstance(np.array(signal_probability_array)[0], (list, np.ndarray, pd.Series)):
            signal_probability_array = [signal_probability_array]
        if not isinstance(threshold_array, (int, float)):
            if len(signal_probability_array) == 1:
                threshold_array = [threshold_array]
            for i in range(len(signal_probability_array)):
                if isinstance(threshold_array[i], (list, np.ndarray, pd.Series)):
                    cuts.append(np.array(np.array(signal_probability_array[i]) > threshold_array[i][binning[1]]))
                else:
                    cuts.append(np.array(np.array(signal_probability_array[i]) > threshold_array[i]))
        else:
            for i in range(len(signal_probability_array)):
                cuts.append(np.array(np.array(signal_probability_array[i]) > threshold_array))

    else:
        if not isinstance(np.array(cut_array)[0], (list, np.ndarray, pd.Series)):
            cut_array = [cut_array]
        for cut in cut_array:
            cuts.append(cut)

    for i in range(len(cuts)):
        binned_values = bins.apply_binning(cuts[i], binning, selection=(df['h5_labels'].isin(particle_label)))
        x = bins.bin_centres(binning[0])
        if errors:
            y_values, y_errors = bins.binned_efficiencies(binned_values, errors, reverse=reverse)
            x_errors = bins.bin_halfwidths(binning[0])
            plot_args.setdefault('marker', '')
            plot_args.setdefault('capsize', 4)
            plot_args.setdefault('capthick', 2)
            ax.errorbar(x, y_values, yerr=y_errors, xerr=x_errors, label=algorithms[i], **plot_args)
        else:
            y = bins.binned_efficiencies(binned_values, errors, reverse=reverse)
            plot_args.setdefault('marker', 'o')
            ax.plot(x, y, label=algorithms[i], **plot_args)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax


def tp_fp_fn_tn_hist2d(df, variable1, variable2, x_range, y_range, bin_x, bin_y, cut, logbin_y=False):
    tp_cut = np.array(((df['true_sig'] == 1) & (cut == 1)))
    fp_cut = np.array(((df['true_sig'] == 0) & (cut == 1)))
    tn_cut = np.array(((df['true_sig'] == 0) & (cut == 0)))
    fn_cut = np.array(((df['true_sig'] == 1) & (cut == 0)))

    cut_array = [tp_cut, fp_cut, tn_cut, fn_cut]
    description = ['True Positives', 'False Positives', 'True Negatives', 'False Negatives']

    if logbin_y:
        bins_y = [10 ** x for x in np.linspace(-12, 0, 50)]
        yscale = "log"
    else:
        bins_y = bin_y
        yscale = "linear"

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for i in range(4):
        ax = axes.flatten()[i]
        df_cut = df[cut_array[i]]
        hist, x_edges, y_edges, im = ax.hist2d(df_cut[variable1], df_cut[variable2], bins=[bin_x, bins_y],
                                               range=[x_range, y_range], norm=mpl.colors.LogNorm())
        if i in [2, 3]:
            ax.set_xlabel(variable1, fontsize=20)
        if i in [0, 2]:
            ax.set_ylabel(variable2, fontsize=20)
        ax.set_yscale(yscale)
        ax.set_title(description[i], fontsize=20)
        plt.colorbar(im, ax=ax)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return fig, axes
