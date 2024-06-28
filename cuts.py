import numpy as np


def check_labels(df, labels):
    if 'h5_labels' in df.columns:
        label_map = [0, 1, 2, 3]
        label_count = []
        for i in range(4):
            label_count.append(np.sum(df['h5_labels'] == i))
        label_array = []
        for i in range(len(label_count)):
            if label_count[i] > 0:
                label_array.append(label_map[i])
        if label_array == labels:
            return True
        else:
            return False
    else:
        return False


def basic(df):
    basic_cuts = np.array(
            (df['reco_electron_dwall'] > 50)
            & (df['reco_electron_towall'] > 100)
            & (df['reco_electron_mom'] > 100)
            & ~((df['h5_labels'] == 2) & ~(df['h5_momentum'] > 2 * df['h5_dwall']))
    )
    return basic_cuts


def fq_cuts(df):

    if df.shape[0] == np.sum(basic(df)):
        preselection = True
    else:
        preselection = False

    if check_labels(df, [1, 2]):
        if preselection:
            cuts = np.array(
                    (df['e/mu_likelihood ratio'] > -31 + 0.09 * df['reco_electron_mom'])
                    # & (df['e/mu_likelihood ratio'] > 45.6 - 0.14 * df['reco_electron_dwall'])
                    # & (df['e/mu_likelihood ratio'] > 149.1 - 0.69 * df['reco_electron_towall'])
            )
        else:
            cuts = np.array(
                    (df['e/mu_likelihood ratio'] > 9 - 0.06 * df['reco_electron_mom'])
                    # & (df['e/mu_likelihood ratio'] > 44.5 - 0.25 * df['reco_electron_dwall'])
                    # & (df['e/mu_likelihood ratio'] > 148.6 - 0.74 * df['reco_electron_towall'])
            )
        return cuts

    elif check_labels(df, [1, 3]):
        if preselection:
            cuts = np.array(
                    (df['pi0/e_likelihood ratio'] < 287 - 0.1 * df['reco_electron_mom'])
                    & (df['pi0/e_likelihood ratio'] < 445 - 4.59 * df['pi0_mass'])
                    # & (df['pi0/e_likelihood ratio'] < 66.2 + 1.92 * df['reco_electron_dwall'])
                    # & (df['pi0/e_likelihood ratio'] < 56.5 + 0.95 * df['reco_electron_towall'])
            )
        else:
            cuts = np.array(
                    (df['pi0/e_likelihood ratio'] < 371 - 0.18 * df['reco_electron_mom'])
                    & (df['pi0/e_likelihood ratio'] < 483 - 4.64 * df['pi0_mass'])
                    # & (df['pi0/e_likelihood ratio'] < 227.9 + 2.79 * df['reco_electron_dwall'])
                    # & (df['pi0/e_likelihood ratio'] < 70.9 + 2.79 * df['reco_electron_towall'])
            )
        return cuts

    elif check_labels(df, [1, 2, 3]):
        if preselection:
            cuts_emu = np.array(
                    (df['e/mu_likelihood ratio'] > 119.8 - 0.32 * df['reco_electron_mom'])
                    & (df['e/mu_likelihood ratio'] > 45.6 - 0.14 * df['reco_electron_dwall'])
                    & (df['e/mu_likelihood ratio'] > 149.1 - 0.69 * df['reco_electron_towall'])
            )
            cuts_epi0 = np.array(
                    (df['pi0/e_likelihood ratio'] < 32.2 + 0.42 * df['reco_electron_mom'])
                    & (df['pi0/e_likelihood ratio'] < 165 - 0.1 * df['pi0_mass'])
                    & (df['pi0/e_likelihood ratio'] < 66.2 + 1.92 * df['reco_electron_dwall'])
                    & (df['pi0/e_likelihood ratio'] < 56.5 + 0.95 * df['reco_electron_towall'])
            )

        else:
            cuts_emu = np.array(
                    (df['e/mu_likelihood ratio'] > 119.2 - 0.38 * df['reco_electron_mom'])
                    & (df['e/mu_likelihood ratio'] > 44.5 - 0.25 * df['reco_electron_dwall'])
                    & (df['e/mu_likelihood ratio'] > 148.6 - 0.74 * df['reco_electron_towall'])
            )

            cuts_epi0 = np.array(
                    (df['pi0/e_likelihood ratio'] < 36.1 + 0.81 * df['reco_electron_mom'])
                    & (df['pi0/e_likelihood ratio'] < 322 - 1.8 * df['pi0_mass'])
                    & (df['pi0/e_likelihood ratio'] < 227.9 + 2.79 * df['reco_electron_dwall'])
                    & (df['pi0/e_likelihood ratio'] < 70.9 + 2.79 * df['reco_electron_towall'])
            )

        cuts = cuts_emu & cuts_epi0
        return cuts

    else:
        print('The fitqun cuts for the labels have not been defined yet')


def ml_cuts(df):
    if df.shape[0] == np.sum(basic(df)):
        preselection = True
    else:
        preselection = False

    if check_labels(df, [1, 2]):
        if preselection:
            cuts = np.array(
                (df['pe'] > 10 ** (-0.2))
            )
        else:
            cuts = np.array(
                # (df['pmu'] < 10 ** (0.0197 * df['reco_electron_mom'] - 0.66))
                (df['pe'] > 10 ** (-0.19))
            )
        return cuts

    elif check_labels(df, [1, 3]):
        if preselection:
            cuts = np.array(
                (df['pe'] > 10 ** (-0.15))
                & (df['ppi0'] < 10 ** (-0.006 * df['pi0_mass'] - 0.19))
            )
        else:
            cuts = np.array(
                (df['pe'] > 10 ** (-0.16))
                # & (df['pe'] > 10 ** (-1.01 * df['pi0_mass'] - 0.13))
                # & (df['ppi0'] < 10 ** (0.1039 * df['reco_electron_mom'] - 0.69))
                & (df['ppi0'] < 10 ** (-0.002 * df['pi0_mass'] - 0.439))
            )
        return cuts

    elif check_labels(df, [1, 2, 3]):
        if preselection:
            cuts_emu = np.array(
                (df['pmu'] < 10 ** (-0.6))
            )
            cuts_epi0 = np.array(
                    (df['ppi0'] < 10 ** (-0.6))
                    & (df['pe'] > 10 ** (-1.2))
            )
        else:
            cuts_emu = np.array(
                (df['pmu'] < 10 ** (-0.6))
            )
            cuts_epi0 = np.array(
                    (df['ppi0'] < 10 ** (-0.6))
                    & (df['pe'] > 10 ** (-1.2))
            )

        cuts = cuts_emu & cuts_epi0
        return cuts

    else:
        print('The softmax cuts for the labels have not been defined yet')


def fq_discriminator(df):
    if df.shape[0] == np.sum(basic(df)):
        preselection = True
    else:
        preselection = False

    if check_labels(df, [1, 2]):
        if preselection:
            fq_dis = np.array(
                (df['e/mu_likelihood ratio'] + 31 - 0.09 * df['reco_electron_mom'])
                # + (df['e/mu_likelihood ratio'] - 45.6 + 0.14 * df['reco_electron_dwall'])
                # + (df['e/mu_likelihood ratio'] - 149.1 + 0.69 * df['reco_electron_towall'])
            )
        else:
            fq_dis = np.array(
                (df['e/mu_likelihood ratio'] - 9 + 0.06 * df['reco_electron_mom'])
                # + (df['e/mu_likelihood ratio'] - 44.5 + 0.25 * df['reco_electron_dwall'])
                # + df['e/mu_likelihood ratio'] - 148.6 + 0.74 * df['reco_electron_towall']
            )
        return fq_dis
    elif check_labels(df, [1, 3]):
        if preselection:
            fq_dis = np.array(
                (-df['pi0/e_likelihood ratio'] + 287 - 0.1 * df['reco_electron_mom'])
                + (-df['pi0/e_likelihood ratio'] + 445 - 4.59 * df['pi0_mass'])
                # + (-df['pi0/e_likelihood ratio'] + 66.2 + 1.92 * df['reco_electron_dwall'])
                # + (-df['pi0/e_likelihood ratio'] + 56.5 + 0.95 * df['reco_electron_towall'])
            )
        else:
            fq_dis = np.array(
                (-df['pi0/e_likelihood ratio'] + 371 - 0.18 * df['reco_electron_mom'])
                + (-df['pi0/e_likelihood ratio'] + 483 - 4.64 * df['pi0_mass'])
                # + (-df['pi0/e_likelihood ratio'] + 227.9 + 2.79 * df['reco_electron_dwall'])
                # + (-df['pi0/e_likelihood ratio'] + 70.9 + 2.79 * df['reco_electron_towall'])
            )
        return fq_dis
    elif check_labels(df, [1, 2, 3]):
        if preselection:
            fq_dis = np.array(
                (df['e/mu_likelihood ratio'] - 119.8 + 0.32 * df['reco_electron_mom'])
                + (df['e/mu_likelihood ratio'] - 45.6 + 0.14 * df['reco_electron_dwall'])
                + (df['e/mu_likelihood ratio'] - 149.1 + 0.69 * df['reco_electron_towall'])
                + (-df['pi0/e_likelihood ratio'] + 32.2 + 0.42 * df['reco_electron_mom'])
                + (-df['pi0/e_likelihood ratio'] + 165 - 0.1 * df['pi0_mass'])
                + (-df['pi0/e_likelihood ratio'] + 66.2 + 1.92 * df['reco_electron_dwall'])
                + (-df['pi0/e_likelihood ratio'] + 56.5 + 0.95 * df['reco_electron_towall'])
            )

        else:
            fq_dis = np.array(
                (df['e/mu_likelihood ratio'] - 119.2 + 0.38 * df['reco_electron_mom'])
                + (df['e/mu_likelihood ratio'] - 44.5 + 0.25 * df['reco_electron_dwall'])
                + (df['e/mu_likelihood ratio'] - 148.6 + 0.74 * df['reco_electron_towall'])
                + (-df['pi0/e_likelihood ratio'] + 36.1 + 0.81 * df['reco_electron_mom'])
                + (-df['pi0/e_likelihood ratio'] + 322 - 1.8 * df['pi0_mass'])
                + (-df['pi0/e_likelihood ratio'] + 227.9 + 2.79 * df['reco_electron_dwall'])
                + (-df['pi0/e_likelihood ratio'] + 70.9 + 2.79 * df['reco_electron_towall'])
            )
        return fq_dis
    elif check_labels(df, [0, 1, 2, 4]):
        return np.array(df['e_likelihood'])


def softmax_discriminator(df):
    if df.shape[0] == np.sum(basic(df)):
        preselection = True
    else:
        preselection = False

    if check_labels(df, [1, 2]):
        if preselection:
            softmax_dis = np.array(
                 df['pe'] - 10 ** (-0.2)
            )
        else:
            softmax_dis = np.array(
                # (- df['pmu'] + 10 ** (0.0179 * df['reco_electron_mom'] - 0.66))
                (df['pe'] - 10 ** (-0.19))
            )
        return softmax_dis

    elif check_labels(df, [1, 3]):
        if preselection:
            softmax_dis = np.array(
                (df['pe'] - 10 ** (-0.15))
                + (- df['ppi0'] + 10 ** (-0.006 * df['pi0_mass'] - 0.19))
            )
        else:
            softmax_dis = np.array(
                (df['pe'] - 10 ** (-0.16))
                # + (df['pe'] - 10 ** (-1.01 * df['pi0_mass'] - 0.13))
                # + (- df['ppi0'] + 10 ** (0.1039 * df['reco_electron_mom'] - 0.69))
                + (- df['ppi0'] + 10 ** (-0.002 * df['pi0_mass'] - 0.439))
            )
        return softmax_dis

    elif check_labels(df, [1, 2, 3]):
        if preselection:
            softmax_dis = np.array(
                (- df['pmu'] + 10 ** (-0.6))
                + (- df['ppi0'] + 10 ** (-0.6))
                + (df['pe'] - 10 ** (-1.2))
            )
        else:
            softmax_dis = np.array(
                (- df['pmu'] + 10 ** (-0.6))
                + (- df['ppi0'] + 10 ** (-0.6))
                + (df['pe'] - 10 ** (-1.2))
            )
        return softmax_dis
    elif check_labels(df, [0, 1, 2, 4]):
        return np.array(df['pe'])


def discriminator(df, algorithm):

    if algorithm == 'fitqun':
        df['fq_discriminator'] = fq_discriminator(df)
    elif algorithm == 'softmax':
        df['softmax_discriminator'] = softmax_discriminator(df)
