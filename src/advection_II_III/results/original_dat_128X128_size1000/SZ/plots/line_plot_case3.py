import matplotlib.pyplot as plt
import numpy as np


def main():
    # ------------------clean--------------------------------
    metrics_error = []
    for i in range(5):
        with open(f"../../ZFP/raw/trail_{i}/y_error_deeponet_clean.dat") as file:
            metrics_error.append([[float(x) for x in line.split()] for line in file.readlines()])

    # Plotting error metrics for each trial
    dist = []
    for i, error in enumerate(metrics_error):
        dist.append(np.linalg.norm(error, axis=1))
        plt.plot(dist[i], color="tab:blue", alpha=(i + 1) / 10)

    # Calculating average error metrics
    # average_error = np.mean(metrics_error, axis=0)
    average_dist = np.mean(np.array(dist), axis=0)

    # Plotting average error metrics
    plt.plot(average_dist, label="clean", color="tab:blue", linewidth=2)

    # Add legend, labels, and title

    # Save plot

    # ------------------eb0.1--------------------------------
    metrics_error = []
    for i in range(5):
        with open(f"../eb0.1/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_error.append([[float(x) for x in line.split()] for line in file.readlines()])

    # Plotting error metrics for each trial
    dist = []
    for i, error in enumerate(metrics_error):
        dist.append(np.linalg.norm(error, axis=1))
        plt.plot(dist[i], color="tab:orange", alpha=(i + 1) / 10)

    # Calculating average error metrics
    # average_error = np.mean(metrics_error, axis=0)
    average_dist = np.mean(np.array(dist), axis=0)

    # Plotting average error metrics
    plt.plot(average_dist, label=r"$0.1 \times \sigma$", color="tab:orange", linewidth=2)

    # Add legend, labels, and title

    # ------------------eb0.01--------------------------------
    metrics_error = []
    for i in range(5):
        with open(f"../eb0.01/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_error.append([[float(x) for x in line.split()] for line in file.readlines()])

    # Plotting error metrics for each trial
    dist = []
    for i, error in enumerate(metrics_error):
        dist.append(np.linalg.norm(error, axis=1))
        plt.plot(dist[i], color="tab:green", alpha=(i + 1) / 10)

    # Calculating average error metrics
    # average_error = np.mean(metrics_error, axis=0)
    average_dist = np.mean(np.array(dist), axis=0)

    # Plotting average error metrics
    plt.plot(average_dist, label=r"$0.01 \times \sigma$", color="tab:green", linewidth=2)

    # Add legend, labels, and title

    # ------------------eb0.01--------------------------------
    metrics_error = []
    for i in range(5):
        with open(f"../eb0.001/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_error.append([[float(x) for x in line.split()] for line in file.readlines()])

    # Plotting error metrics for each trial
    dist = []
    for i, error in enumerate(metrics_error):
        dist.append(np.linalg.norm(error, axis=1))
        plt.plot(dist[i], color="tab:red", alpha=(i + 1) / 10)

    # Calculating average error metrics
    # average_error = np.mean(metrics_error, axis=0)
    average_dist = np.mean(np.array(dist), axis=0)

    # Plotting average error metrics
    plt.plot(average_dist, label=r"$0.001 \times \sigma$", color="tab:red", linewidth=2)

    # Add legend, labels, and title

    # ------------------eb0.0001--------------------------------
    metrics_error = []
    for i in range(5):
        with open(f"../eb0.0001/trail_{i}/y_error_deeponet_case3.dat") as file:
            metrics_error.append([[float(x) for x in line.split()] for line in file.readlines()])

    # Plotting error metrics for each trial
    dist = []
    for i, error in enumerate(metrics_error):
        dist.append(np.linalg.norm(error, axis=1))
        plt.plot(dist[i], color="tab:purple", alpha=(i + 1) / 10)

    # Calculating average error metrics
    # average_error = np.mean(metrics_error, axis=0)
    average_dist = np.mean(np.array(dist), axis=0)

    # Plotting average error metrics
    plt.plot(
        average_dist,
        label=r"$0.0001 \times \sigma$",
        color="purple",
        linewidth=2,
    )

    # Add legend, labels, and title
    plt.legend()
    plt.xlabel("Time", size=13)
    plt.ylabel(r"$L_2 \text{ error}$", size=13)
    plt.yscale("log")
    plt.title("Perturb both train and test data", size=13)
    # Save plot
    plt.tight_layout()
    plt.show()
    plt.savefig("error_plot_case3.pdf")


if __name__ == "__main__":
    main()
