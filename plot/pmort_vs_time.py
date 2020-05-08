import argparse, matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, os, torch
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rc('font', family='serif')

from .bootstrap import bootstrap2d
from flexehr.utils.modelIO import load_metadata, load_model
from utils.helpers import array


def pmort_vs_time(model, idx, show_plot=True):
    # Load data
    data = np.load(os.path.join('data', 'arrs_48_20.npy')).item()
    models = [f for f in os.listdir('results') if f.startswith(model)]

    # Print info
    print(idx, data['paths_train'][idx], data['Y_train'][idx])

    preds = np.zeros((len(models), 48))
    for i, model in enumerate(models):
        # Load model
        model_dir = os.path.join('results', model)
        model = load_model(model_dir)

        # Predict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = data['X_train'][idx:idx+1]
        x = torch.tensor(x).to(device)
        preds[i] = array(model(x))

    # Bootstrap values
    mu, lb, ub = bootstrap2d(preds, low=0.025, high=0.975, n_samples=10000)

    # Plot
    plt.fill_between(range(1, len(mu)+1), lb, ub, color='r', alpha=0.5, label='95% CI')
    plt.plot(range(1, len(mu)+1), mu, ':ko', label='Mean')

    plt.xlim(0, 48)
    plt.xticks([0, 12, 24, 36, 48], fontsize=12)
    plt.xlabel('In-Hospital Hours After Admission', fontsize=15)

    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11), fontsize=12)
    plt.ylabel('p$_{mortality}$', fontsize=15)
    plt.legend()

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join('figs', f'pmortvstime_48_20bins_idx{idx}.pdf'))
        return array(x), model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot dynamic mortity probability.')
    parser.add_argument('model', type=str,
                        help='Model prefix.')
    args = parser.parse_args()

    idx = np.random.randint(17500)

    pmort_vs_time(args.model, idx)
