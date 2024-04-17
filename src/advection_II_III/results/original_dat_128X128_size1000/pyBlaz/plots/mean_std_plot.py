import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main():
    trials = 5

    # Initialize lists to store error metrics for each trial
    metrics_clean = []
    metrics_case1 = []
    metrics_case2 = []
    metrics_case3 = []
    metrics_case4 = []

    metrics_case1_2 = []
    metrics_case2_2 = []
    metrics_case3_2 = []
    metrics_case4_2 = []

    metrics_case1_3 = []
    metrics_case2_3 = []
    metrics_case3_3 = []
    metrics_case4_3 = []

    metrics_case1_4 = []
    metrics_case2_4 = []
    metrics_case3_4 = []
    metrics_case4_4 = []

    metrics_case1_5 = []
    metrics_case2_5 = []
    metrics_case3_5 = []
    metrics_case4_5 = []

    # -------------------------Perturb train----------------------------------
    # Load error metrics for each trial
    for i in range(trials):
        with open(f"../raw/trail_{i}/y_error_deeponet_clean.dat") as file:
            metrics_clean.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_8/trail_{i}/y_error_deeponet_case1.dat") as file:
            metrics_case1.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_16/trail_{i}/y_error_deeponet_case1.dat") as file:
            metrics_case2.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_8/trail_{i}/y_error_deeponet_case1.dat") as file:
            metrics_case3.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_16/trail_{i}/y_error_deeponet_case1.dat") as file:
            metrics_case4.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_8/trail_{i}/y_error_deeponet_case2.dat") as file:
            metrics_case1_2.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_16/trail_{i}/y_error_deeponet_case2.dat") as file:
            metrics_case2_2.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_8/trail_{i}/y_error_deeponet_case2.dat") as file:
            metrics_case3_2.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_16/trail_{i}/y_error_deeponet_case2.dat") as file:
            metrics_case4_2.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_8/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_case1_3.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_16/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_case2_3.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_8/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_case3_3.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_16/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_case4_3.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_8/trail_{i}/y_error_deeponet_case4.dat") as file:
            metrics_case1_4.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_16/trail_{i}/y_error_deeponet_case4.dat") as file:
            metrics_case2_4.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_8/trail_{i}/y_error_deeponet_case4.dat") as file:
            metrics_case3_4.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_16/trail_{i}/y_error_deeponet_case4.dat") as file:
            metrics_case4_4.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_8/trail_{i}/y_error_deeponet_case5.dat") as file:
            metrics_case1_5.append(np.loadtxt(file))

        with open(f"../blocksize_4_index_16/trail_{i}/y_error_deeponet_case5.dat") as file:
            metrics_case2_5.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_8/trail_{i}/y_error_deeponet_case5.dat") as file:
            metrics_case3_5.append(np.loadtxt(file))

        with open(f"../blocksize_8_index_16/trail_{i}/y_error_deeponet_case5.dat") as file:
            metrics_case4_5.append(np.loadtxt(file))

    clean_l2_errors = [np.linalg.norm(metrics, 2) for metrics in metrics_clean]
    clean_linf_errors = [np.linalg.norm(metrics, np.inf) for metrics in metrics_clean]
    clean_l2_mean = np.mean(clean_l2_errors)
    clean_linf_mean = np.mean(clean_linf_errors)
    clean_l2_std = np.std(clean_l2_errors, ddof=1)
    clean_linf_std = np.std(clean_linf_errors, ddof=1)

    other_metrics_names = [
        "metrics_case1",
        "metrics_case2",
        "metrics_case3",
        "metrics_case4",
        "metrics_case1_2",
        "metrics_case2_2",
        "metrics_case3_2",
        "metrics_case4_2",
        "metrics_case1_3",
        "metrics_case2_3",
        "metrics_case3_3",
        "metrics_case4_3",
        "metrics_case1_5",
        "metrics_case2_5",
        "metrics_case3_5",
        "metrics_case4_5",
    ]

    l2_means = [clean_l2_mean]
    l2_stddevs = [clean_l2_std]
    linf_means = [clean_linf_mean]
    linf_stddevs = [clean_linf_std]

    for metrics_name in other_metrics_names:
        metrics = eval(metrics_name)
        l2_errors = [np.linalg.norm(metrics, 2) for metrics in metrics]
        linf_errors = [np.linalg.norm(metrics, np.inf) for metrics in metrics]
        l2_test_result = stats.ttest_ind(clean_l2_errors, l2_errors, equal_var=False)
        linf_test_result = stats.ttest_ind(clean_linf_errors, linf_errors, equal_var=False)
        print(f"{metrics_name} l2 test p-value: {l2_test_result.pvalue}, linf test p-value: {linf_test_result.pvalue}")
        l2_means.append(np.mean(l2_errors))

        l2_stddevs.append(np.std(l2_errors, ddof=1))
        linf_means.append(np.mean(linf_errors))
        linf_stddevs.append(np.std(linf_errors, ddof=1))

    plot_error_bar(l2_means, l2_stddevs, linf_means, linf_stddevs)


def plot_error_bar(l2_means, l2_stddevs, linf_means, linf_stddevs):
    cases = [
        "Clean",
        r"all  BS4 int8 train",
        r"all  BS4 int16 train",
        r"all  BS8 int8 train",
        r"all  BS8 int16 train",
        r"all  BS4 int8 test",
        r"all  BS4 int16 test",
        r"all  BS8 int8 test",
        r"all  BS8 int16 test",
        r"all  BS4 int8 both",
        r"all  BS4 int16 both",
        r"all  BS8 int8 both",
        r"all  BS8 int16 both",
        r"half  BS4 int8 train",
        r"half  BS4 int16 train",
        r"half  BS8 int8 train",
        r"half  BS8 int16 train",
    ]

    fig, ax = plt.subplots()
    fig.set_size_inches(len(cases) / 1.5, 4)
    horizontal_positions = np.arange(len(cases))
    offset = 0.2

    ax.errorbar(
        horizontal_positions - offset / 2,
        l2_means,
        yerr=l2_stddevs,
        fmt="o",
        color="tab:blue",
    )

    # Add L_inf axis on the right side
    ax2 = ax.twinx()

    ax2.errorbar(
        horizontal_positions + offset / 2,
        linf_means,
        yerr=linf_stddevs,
        fmt="o",
        color="tab:orange",
    )
    ax.axhline(y=l2_means[0], linestyle="--", color="tab:blue", alpha=0.25)
    ax2.axhline(y=linf_means[0], linestyle="--", color="tab:orange", alpha=0.25)

    # ax2.axhline(y=mean_clean_inf, linestyle='--', color='tab:orange')
    ax2.set_ylabel(r"$L_\infty$ error", color="tab:orange")

    ax.set_xlabel("scheme")
    ax.set_ylabel("$L_2$ error", color="tab:blue")
    ax.set_xticks(horizontal_positions)
    ax.set_xticklabels(
        cases,
        rotation=-30,
        # ha="center",
        ha="left",
    )

    # ax.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    plt.title("SZ error per perturbation scheme")
    plt.tight_layout()
    plt.savefig("mean_plot_pyBlaz.pdf")


if __name__ == "__main__":
    main()
