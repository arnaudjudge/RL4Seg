from pathlib import Path

import h5py
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
from utils.test_metrics import dice
from vital.data.camus.config import Label
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence

path = 'reward_eval_uncertainty_tempscaled_fulltrain.h5'

output = Path('reward_uncertainty_ts_ft')

output.mkdir(exist_ok=True)


def correlation(metric, metric_key, uncertainty, uncertainty_key, ids=None):
    f, ax = plt.subplots(1, 1)

    data = {metric_key: metric, uncertainty_key: uncertainty}

    corr, _ = pearsonr(uncertainty, metric)

    f = f'{metric_key} ~ {uncertainty_key}'
    model = ols(formula=f, data=data).fit()
    cook_distance = OLSInfluence(model).cooks_distance
    (distance, p_value) = cook_distance

    threshold = 4 / len(metric)
    # the observations with Cook's distances higher than the threshold value are labeled in the plot
    # influencial_data = distance[distance > threshold]
    # print(influencial_data.keys())
    # print(influencial_data[list(influencial_data.keys())])
    # print(influencial_data[list(influencial_data.keys())])
    if ids is not None:
        # print(f"{metric_key}, {uncertainty_key} influencial data {ids[list(influencial_data.keys())]}")
        indices = (-distance).argsort()[:8]
        for idx in indices:
            ax.text(metric[idx], uncertainty[idx], str(ids[idx]), fontsize=5, c='r')

    sns.regplot(x=metric_key, y=uncertainty_key, data=data, ax=ax)
    sns.scatterplot(x=metric_key, y=uncertainty_key, data=data, ax=ax, hue=distance, size=distance,
                    sizes=(50, 200), edgecolor='black', linewidth=1, legend=None)
    # sns.scatterplot(data[metric_key], data[uncertainty_key], hue=distance, size=distance, sizes=(50, 200),
    #                 edgecolor='black', linewidth=1, ax=ax[j, i])

    ax.set_xlabel(metric_key, fontsize=20)
    ax.set_ylabel(uncertainty_key, fontsize=20)
    ax.set_title(f"R={corr:.3f}", fontsize=20)

    plt.savefig(output / f'corr-{metric_key}-{uncertainty_key}.png')



def compute_mi(error, uncertainty, norm=True):
    """Computes mutual information between error and uncertainty.

    Args:
        error: numpy binary array indicating error.
        uncertainty: numpy float array indicating uncertainty.

    Returns:
        mutual_information
    """
    hist_2d, x_edges, y_edges = np.histogram2d(error.ravel(), uncertainty.ravel())

    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    if norm:
        hx = -np.sum(np.multiply(px[px > 0], np.log(px[px > 0])))
        hy = -np.sum(np.multiply(py[py > 0], np.log(py[py > 0])))
        mi = 2 * mi / (hx + hy)

    return mi.item()


def histogram(conf, acc, nb_bins=21):
    plt.figure()
    plt.hist(
        conf[np.where(acc == 1)],
        bins=np.linspace(0, 1, num=nb_bins),
        density=True,
        color="green",
        label="Successes",
    )
    plt.hist(
        conf[np.where(acc == 0)],
        bins=np.linspace(0, 1, num=nb_bins),
        density=True,
        alpha=0.5,
        color="red",
        label="Errors",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Relative density")
    plt.xlim(left=0, right=1)
    plt.legend()
    plt.savefig(output / f'histogram.png')



def ece(confidences, accuracies, nb_bins=20):
    nb_bins = nb_bins
    bin_boundaries = np.linspace(0, 1, nb_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = np.zeros(1)
    bins_avg_conf = []
    bins_avg_acc = []
    prob_in_bins = []
    bins_size = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin.item() > 0:
            prob_in_bins.append(prop_in_bin)
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bins_avg_conf.append(avg_confidence_in_bin)
            bins_avg_acc.append(accuracy_in_bin)
            bins_size.append(in_bin.sum())

    # print(bins_avg_conf)
    # print(bins_avg_acc)
    # print(prob_in_bins)

    bins_avg_conf = np.array(bins_avg_conf)
    bins_avg_acc = np.array(bins_avg_acc)

    print(bins_avg_conf)
    print(bins_avg_acc)
    np.save(f"{output}/bins_conf.npy", np.stack((bins_avg_conf, bins_avg_acc)))

    mce = np.max(np.abs(bins_avg_conf - bins_avg_acc))

    f, ax = plt.subplots(1, 1)
    ax.plot(bins_avg_conf, bins_avg_acc)
    ax.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")

    ax2 = ax.twinx()
    try:
        ax2.bar(bins_avg_conf, bins_size, alpha=0.7, width=np.min(np.diff(bins_avg_conf)) / 2)
    except:
        ax2.bar(bins_avg_conf, bins_size, alpha=0.7)

    print("ECE: ", ece)
    print("MCE: ", mce)

    plt.savefig(output / f'reliability_diagram.png')


    return ece, mce



def ace(confidences, accuracies, nb_bins=20):

    idx = np.argsort(confidences)
    confidences = confidences[idx]
    accuracies = accuracies[idx]

    confidences = np.array_split(confidences, nb_bins)
    accuracies = np.array_split(accuracies, nb_bins)

    ece = np.zeros(1)
    bins_avg_conf = []
    bins_avg_acc = []
    prob_in_bins = []
    for c, a in zip(confidences, accuracies):
        # Calculated |confidence - accuracy| in each bin
        prop_in_bin = len(c) / nb_bins

        if prop_in_bin > 0:
            prob_in_bins.append(prop_in_bin)
            accuracy_in_bin = a.mean()
            avg_confidence_in_bin = c.mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bins_avg_conf.append(avg_confidence_in_bin)
            bins_avg_acc.append(accuracy_in_bin)

    # print(bins_avg_conf)
    # print(bins_avg_acc)
    # print(prob_in_bins)

    bins_avg_conf = np.array(bins_avg_conf)
    bins_avg_acc = np.array(bins_avg_acc)

    mce = np.max(np.abs(bins_avg_conf - bins_avg_acc))


    plt.figure()
    plt.plot(bins_avg_conf, bins_avg_acc)
    plt.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.savefig(output / f'reliability_diagram_ACE.png')

    print("ACE: ", ece)
    print("AMCE: ", mce)


    return ece, mce


confidences, accuracies, preds, gts = [], [], [], []

dices = []
average_reward = []
weighted_reward = []
mis, norm_mis = [], []

ids = []

with h5py.File(path, "r") as h5:
    print(len(h5.keys()))

    for key in h5.keys():
        ids.append(str(key))
        img = np.array(h5[key]['img']).T
        pred = np.array(h5[key]['pred']).T
        gt = np.array(h5[key]['gt']).T
        reward_map = np.array(h5[key]['reward_map']).T
        uncertainty_map = 1 - reward_map
        accuracy = np.array(h5[key]['accuracy_map']).T
        error = 1 - accuracy

        # if key == "di-D743-7A76-069A_ES":
        #     err = nib.Nifti1Image(error, affine=np.diag(np.asarray([1, 1, 1, 0])), header=nib.Nifti1Header())
        #     err.to_filename(output / f'{key}_error.nii.gz')
        #     reward = nib.Nifti1Image(uncertainty_map, affine=np.diag(np.asarray([1, 1, 1, 0])), header=nib.Nifti1Header())
        #     reward.to_filename(output / f'{key}_unc.nii.gz')
        np.save(f"{output}/{key}_error.npy", error)
        np.save(f"{output}/{key}_unc.npy", uncertainty_map)

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(24, 6))
        ax1.imshow(img.squeeze(), cmap='gray')
        ax2.imshow(gt.squeeze(), cmap='gray')
        ax3.imshow(pred.squeeze(), cmap='gray')
        ax4.imshow(error.squeeze(), cmap='gray')
        ax5.imshow(uncertainty_map.squeeze(), cmap='gray')

        ax1.set_title("Image")
        ax2.set_title("GT")
        ax3.set_title("Pred")
        ax4.set_title("Error")
        ax5.set_title("Uncertainty")

        plt.savefig(output / f'{key}.jpg')
        plt.close()

        mi_norm = compute_mi(accuracy, reward_map, norm=True)
        mi = compute_mi(accuracy, reward_map, norm=False)

        correct_map = np.equal(pred, gt).astype(int)
        confidences.append(reward_map.flatten())
        accuracies.append(correct_map.flatten())
        preds.append(pred.flatten())
        gts.append(gt.flatten())

        norm_mis.append(mi_norm)
        mis.append(mi)

        dices.append(dice(pred, gt, labels=(Label.LV, Label.MYO)))

        mask = pred != Label.BG



        # IF mask sum is 0, use prediction sum to avoid inf uncertainty
        mean = np.mean(uncertainty_map.squeeze())
        weighted_mean = np.sum(uncertainty_map.squeeze(), axis=(-2, -1)) / np.sum(mask, axis=(-2, -1))

        # print(mean)
        # print(weighted_mean)
        # print(np.sum(reward_map.squeeze(), axis=(-2, -1)))
        # print(np.sum(mask, axis=(-2, -1)))
        #
        # plt.figure()
        # plt.imshow(mask)
        # plt.show()


        average_reward.append(mean)
        weighted_reward.append(weighted_mean)

confidences = np.concatenate(confidences)
accuracies = np.concatenate(accuracies)
preds = np.concatenate(preds)
gts = np.concatenate(gts)

average_reward = np.array(average_reward)
weighted_reward = np.array(weighted_reward)

norm_mis = np.array(norm_mis)
mis = np.array(mis)

dices = np.array(dices)

fg = preds + gts
not_bg = fg != 0  # Background class is always 0
confidences = confidences[not_bg]
accuracies = accuracies[not_bg]

ece(confidences, accuracies)
ace(confidences, accuracies)
histogram(confidences, accuracies)

print("Normalized MI: ", np.mean(norm_mis))
print("MI: ", np.mean(mis))

correlation(dices, 'Dice', average_reward, 'ImgReward', ids=ids)
correlation(dices, 'Dice', weighted_reward, 'ImgWReward', ids=ids)

correlation(dices, 'Dice', mis, 'MI', ids=ids)
correlation(dices, 'Dice', norm_mis, 'NormMI', ids=ids)

# corr, _ = pearsonr(average_reward, dices)
# weighted_corr, _ = pearsonr(weighted_reward, dices)

# f, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(average_reward, dices)
# ax2.scatter(weighted_reward, dices)
# ax1.set_title(f"Average Reward vs Dice (Pearson Correlation: {corr:.3f})")
# ax2.set_title(f"Average (weighted) Reward vs Dice (Pearson Correlation: {weighted_corr:.3f})")
# ax1.set_ylabel("Dice")
# ax2.set_ylabel("Dice")
# ax1.set_xlabel("Image Reward")
# ax2.set_xlabel("Image Reward")


# corr, _ = pearsonr(dices, mis)
# norm_corr, _ = pearsonr(dices, norm_mis)
#
# f, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(dices, mis)
# ax2.scatter(dices, norm_mis)
# ax1.set_title(f"Pearson Correlation: {corr:.3f})")
# ax2.set_title(f"Pearson Correlation: {norm_corr:.3f})")
# ax1.set_ylabel("Dice")
# ax2.set_ylabel("Dice")
# ax1.set_xlabel("MI")
# ax2.set_xlabel("NORM MI")


plt.show()
