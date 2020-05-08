import argparse, matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, os, torch
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rc('font', family='serif')
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

from .bootstrap import bootstrap2d
from flexehr.utils.modelIO import load_metadata, load_model
from utils.helpers import array


def predict(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    with torch.no_grad():
        for X in test_loader:
            X = X[0].to(device)
            preds += [model(X)]

    return array(torch.cat(preds))


def auc_vs_time(model, show_plot=False):
    # Load data
    data = np.load(os.path.join('data', 'arrs_48_20.npy')).item()
    models = [f for f in os.listdir('results') if f.startswith(model)]

    # Test set
    test_dataset = TensorDataset(torch.tensor(data['X_test']))
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)

    aucs = np.zeros((len(models), 48))
    for i, model in enumerate(models):        
        # Load model
        model_dir = os.path.join('results', model)
        model = load_model(model_dir)
        metadata = load_metadata(model_dir)

        # Predict
        preds = predict(test_loader, model)
        for j in np.arange(48):
            fpr, tpr, _ = metrics.roc_curve(data['Y_test'], preds[:, j])
            aucs[i, j] = metrics.auc(fpr, tpr)

    # Bootstrap values
    mu, lb, ub = bootstrap2d(aucs, low=0.025, high=0.975, n_samples=10000)
    print(lb)
    print(mu)
    print(ub)

    # Plot
    # OASIS & SAPS II
    plt.plot([0, 48], [0.6631, 0.6631], '--b', markersize=5, lw=1)
    p1 = plt.scatter(24, 0.6631, c='b', marker='^', label='OASIS')
    plt.plot([0, 48], [0.7048, 0.7048], '--g', markersize=5, lw=1)
    p2 = plt.scatter(24, 0.7048, c='g', marker='s', label='SAPS II')

    p3 = plt.fill_between(range(1, len(mu)+1), lb, ub, color='r', alpha=0.5, label='95% CI')
    p4 = plt.plot(range(1, len(mu)+1), mu, 'k', lw=1, label='Ours')

    plt.xlim(0, 48)
    plt.xticks([0, 12, 24, 36, 48], fontsize=12)
    plt.xlabel('In-Hospital Hours After Admission', fontsize=15)

    plt.ylim(0.6, 0.9)
    plt.yticks(np.linspace(0.6, 0.9, 7), fontsize=12)
    plt.ylabel('AUROC', fontsize=15)

    leg = plt.legend()
    leg.get_lines()[0].set_linewidth(2)

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join('figs', f'aucvstime_48_20bins.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot dynamic AUROC.')
    parser.add_argument('model', type=str,
                        help='Model prefix.')
    args = parser.parse_args()

    auc_vs_time(args.model)
