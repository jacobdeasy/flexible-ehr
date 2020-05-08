import argparse, matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, os, torch
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
from scipy import interp
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

from .bootstrap import bootstrap
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


def auroc(model, show_plot=False):
    # Load data
    data = np.load(os.path.join('data', 'arrs_48_20.npy')).item()
    models = [f for f in os.listdir('results') if f.startswith(model)]

    # Test set
    test_dataset = TensorDataset(torch.tensor(data['X_test']))
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)

    base_fpr = np.linspace(0, 1, 101)
    tprs = np.zeros((len(models), 101))
    aucs = np.zeros((len(models)))
    for i, model in enumerate(models):
        # Load model
        model_dir = os.path.join('results', model)
        model = load_model(model_dir)
        metadata = load_metadata(model_dir)

        # Predict
        preds = predict(test_loader, model)
        fpr, tpr, _ = metrics.roc_curve(data['Y_test'], preds[:, -1])
        aucs[i] = metrics.auc(fpr, tpr)

        # Interpolate for bootstrap
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs[i] = tpr

    # Plot
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + 2 * std_tprs, 1)
    tprs_lower = mean_tprs - 2 * std_tprs

    plt.plot(base_fpr, mean_tprs, 'k', label=f'Ours: {np.mean(aucs):.4f}')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper,
                     color='red', alpha=0.5, label='95% CI')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc='lower right')

    if show_plot:
        plt.show()
    else:
        np.save(os.path.join('figs', 'auroc_info'),
                np.stack((base_fpr, tprs_lower, mean_tprs, tprs_upper)))
        plt.savefig(os.path.join('figs', f'auroc_48_20bins.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot dynamic AUROC.')
    parser.add_argument('model', type=str,
                        help='Model prefix.')
    parser.add_argument('-s', '--show-plot', type=bool, default=False)
    args = parser.parse_args()

    auroc(args.model, show_plot=args.show_plot)
