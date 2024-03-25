import numpy as np
import pandas as pd
import cuts
import h5py
from watchmal_dependencies import math, fq_output
import warnings

warnings.filterwarnings("ignore")


def normalize_softmax(df, sig_label, bg_label):
    if not isinstance(sig_label, (list, np.ndarray)):
        sig_label = [sig_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]
    prob_sig = 0
    prob_bg = 0

    label_map = ['pgamma', 'pe', 'pmu', 'ppi0']
    for label in sig_label:
        prob_sig += df[label_map[label]]
    for label in bg_label:
        prob_bg += df[label_map[label]]

    norm_prob_sig = prob_sig/(prob_sig + prob_bg)
    norm_prob_bg = prob_bg/(prob_sig + prob_bg)
    return norm_prob_sig, norm_prob_bg


def get_h5_fq_offsets(h5_true_file='/home/pdeperio/machine_learning/data/IWCD_mPMT_Short/'
                      'IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5'):

    idxs_path = ('/home/pdeperio/machine_learning/data/IWCD_mPMT_Short'
                 '/index_lists/4class_e_mu_gamma_pi0/IWCD_mPMT_Short_4_class_3M_emgp0_idxs.npz')

    test_idxs = np.load(idxs_path, allow_pickle=True)['test_idxs']
    h5_file = h5py.File(h5_true_file, 'r')

    h5_root_files = np.array(h5_file['root_files'])[test_idxs].squeeze()
    h5_event_ids = np.array(h5_file['event_ids'])[test_idxs].squeeze()

    root_file_index = dict.fromkeys(h5_root_files)
    root_file_index.update((k, i) for i, k in enumerate(root_file_index))
    root_file_indices = np.vectorize(root_file_index.__getitem__)(h5_root_files)
    h5_fq_offsets = 3000 * root_file_indices + h5_event_ids

    return h5_fq_offsets


def create_true_df(file='/home/pdeperio/machine_learning/data/'
                   'IWCD_mPMT_Short/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5'):

    idxs_path = ('/home/pdeperio/machine_learning/data/IWCD_mPMT_Short/index_lists/'
                 '4class_e_mu_gamma_pi0/IWCD_mPMT_Short_4_class_3M_emgp0_idxs.npz')

    test_idxs = np.load(idxs_path, allow_pickle=True)['test_idxs']
    h5_file = h5py.File(file, 'r')

    events_hits_index = np.append(h5_file['event_hits_index'], h5_file['hit_pmt'].shape[0])
    h5_nhits = (events_hits_index[test_idxs + 1] - events_hits_index[test_idxs]).squeeze()

    df = pd.DataFrame()
    df['h5_labels'] = np.array(h5_file['labels'])[test_idxs].squeeze()
    df['h5_dwall'] = math.dwall(np.array(h5_file['positions'])[test_idxs].squeeze())
    df['h5_towall'] = math.towall(np.array(h5_file['positions'])[test_idxs].squeeze(),
                                  np.array(h5_file['angles'])[test_idxs])
    df['h5_momentum'] = math.momentum_from_energy(np.array(h5_file['energies'])[test_idxs].squeeze(),
                                                  np.array(h5_file['labels'])[test_idxs].squeeze())
    df['h5_vetos'] = np.array(h5_file['veto'])[test_idxs].squeeze()
    df['h5_nhits'] = h5_nhits
    df['h5_angles'] = np.array(h5_file['angles'])[test_idxs].squeeze()
    return df


def create_softmax(file_path='/home/surajrai1900/WatChMaL/outputs/2023-09-19/03-59-17/outputs/'):
    softmax = file_path + 'softmax.npy'
    predictions = file_path + 'predictions.npy'

    df = pd.DataFrame()
    df['softmax_predictions'] = np.load(predictions)

    data_softmax = np.load(softmax)
    df['pgamma'] = data_softmax[:, 0]
    df['pe'] = data_softmax[:, 1]
    df['pmu'] = data_softmax[:, 2]
    df['ppi0'] = data_softmax[:, 3]
    return df


def create_fq_df(file_path='/home/pdeperio/machine_learning/data/IWCD_mPMT_Short/fiTQun/'):
    particle_names = ['gamma', 'e-', 'mu-', 'pi0']
    fq_files = [file_path + f'IWCD_mPMT_Short_{i}_E0to1000MeV_unif-pos-R400-y300cm_4pi-dir.fiTQun.root' for i in
                particle_names]
    fq = fq_output.FiTQunOutput(fq_files)

    offsets = get_h5_fq_offsets()

    reco_e_pos = np.array(fq.electron_position)
    reco_e_angles = math.angles_from_direction(np.array(fq.electron_direction))

    df = pd.DataFrame()
    df['reco_electron_mom'] = np.array(fq.electron_momentum)[offsets]
    df['reco_muon_mom'] = np.array(fq.muon_momentum)[offsets]
    df['reco_pi0_mom'] = np.array(fq.pi0_momentum)[offsets]
    df['reco_electron_dwall'] = math.dwall(reco_e_pos)[offsets]
    df['reco_electron_towall'] = math.towall(reco_e_pos, reco_e_angles)[offsets]
    df['e_likelihood'] = np.array(fq.electron_nll)[offsets]
    df['mu_likelihood'] = np.array(fq.muon_nll)[offsets]
    df['pi0_likelihood'] = np.array(fq.pi0_nll)[offsets]
    df['pi0_mass'] = np.array(fq.pi0_mass)[offsets]
    df['e/mu_likelihood ratio'] = df['mu_likelihood'] - df['e_likelihood']
    df['pi0/e_likelihood ratio'] = df['e_likelihood'] - df['pi0_likelihood']

    return df


def relevant_df(true_variables=None, reco_variables=None, softmax_variables=None,
                true_sig=1, apply_presel=False, select_labels=None, discriminator=False):

    if select_labels is None:
        select_labels = [0, 1, 2, 3]

    if softmax_variables is None:
        softmax_variables = ['pgamma', 'pe', 'pmu', 'ppi0']

    if reco_variables is None:
        reco_variables = ['e/mu_likelihood ratio', 'pi0/e_likelihood ratio', 'e_likelihood', 'mu_likelihood',
                          'pi0_likelihood', 'reco_electron_mom', 'reco_electron_dwall', 'reco_electron_towall',
                          'pi0_mass']

    if true_variables is None:
        true_variables = ['h5_labels', 'h5_momentum', 'h5_towall', 'h5_dwall', 'h5_angles']

    if not isinstance(true_sig, (list, np.ndarray)):
        true_sig = [true_sig]

    df_true = create_true_df()
    df_fq = create_fq_df()
    df_softmax = create_softmax()

    df = pd.DataFrame({item: df_softmax[item] for item in softmax_variables})
    df[true_variables] = df_true[true_variables]
    df[reco_variables] = df_fq[reco_variables]
    df['true_sig'] = (df['h5_labels'].isin(true_sig)).astype(int)

    df = df.dropna()
    if apply_presel:
        df = df[cuts.basic(df)]

    if not isinstance(select_labels, (list, np.ndarray)):
        select_labels = [select_labels]
    df = df[df['h5_labels'].isin(select_labels)]

    if select_labels == [1, 2]:
        df['pe'], df['pmu'] = normalize_softmax(df, [0, 1, 3], [2])
        df.drop(columns=['pgamma', 'ppi0'], inplace=True)

    elif select_labels == [1, 3]:
        df['pe'], df['ppi0'] = normalize_softmax(df, [0, 1], [3])
        df.drop(columns=['pmu', 'pgamma'], inplace=True)

    elif select_labels == [1, 4]:
        df['pe'], df['pgamma'] = normalize_softmax(df, [1], [0])
        df.drop(columns=['pmu', 'ppi0'], inplace=True)

    elif select_labels == [1, 2, 3]:
        df['pe'] = df['pe'] + df['pgamma']
        df.drop(columns=['pgamma'], inplace=True)

    if discriminator:
        df['fq_discriminator'] = cuts.fq_discriminator(df)
        df['softmax_discriminator'] = cuts.softmax_discriminator(df)

    return df
