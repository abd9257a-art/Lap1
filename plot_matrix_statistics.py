"""Plot row and column histograms for matrix statistics."""

import matplotlib.pyplot as plt
from numpy_tasks import matrix_statistics
from grader_contracts.numpy_tasks import RandomMatrixInput


def plot_matrix_statistics(data: RandomMatrixInput) -> None:
    stats = matrix_statistics(data)
    matrix = stats.matrix

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, row in enumerate(matrix):
        axes[0].hist(row, alpha=0.5, label=f"Row {i}")

    axes[0].set_title("Row value distributions")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    for j in range(matrix.shape[1]):
        axes[1].hist(matrix[:, j], alpha=0.5, label=f"Column {j}")

    axes[1].set_title("Column value distributions")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_matrix_statistics(
        RandomMatrixInput(rows=5, columns=4, mean=0, std=1, seed=42)
    )