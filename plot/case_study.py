import argparse, matplotlib.pyplot as plt, numpy as np, os, pandas as pd, torch

from plot.pmort_vs_time import pmort_vs_time
from utils.helpers import array


def case_study(model, idx):
    case_study_dir = os.path.join('results', f'case_study_{idx}')

    # Plot
    x, model = pmort_vs_time(model, idx, show_plot=False)

    # Reverse dict
    token2index = np.load(os.path.join('data', 'token2index_48_20.npy')).item()
    index2token = {v: k for k, v in token2index.items()}
    index2token[0] = ''

    # Hourly tokens and weights
    weight = np.exp(array(model.embedder.embedW.weights)[:, 0])
    token_inds = []
    tokens = []
    W = []
    for t in np.arange(0, 48):
        x_t = x[0, (t < x[0, :, 0]) & (x[0, :, 0] < (t+1)), 1]

        w = weight[x_t.astype(np.int64)]
        inds = np.argsort(-w)

        token_inds += [inds]
        tokens += [np.array([index2token[i] for i in x_t])[inds]]
        W += [w[inds]]

    if not os.path.exists(case_study_dir):
        os.makedirs(case_study_dir)

    # Table
    pd.DataFrame(token_inds).transpose().to_csv(
        os.path.join(case_study_dir, 'input_inds.csv'), index=None)
    pd.DataFrame(tokens).transpose().to_csv(
        os.path.join(case_study_dir, 'inputs.csv'), index=None)
    pd.DataFrame(W).transpose().to_csv(
        os.path.join(case_study_dir, 'weights.csv'), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot dynamic AUROC.')
    parser.add_argument('model', type=str,
                        help='Path to model prefix.')
    parser.add_argument('idx', type=int,
                        help='Index of patient case in arrays.')
    args = parser.parse_args()

    case_study(args.model, args.idx)
