"""Решения задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    if len(matrices) != len(vectors):
        raise ValueError("The number of matrices and vectors must match")
    if len(matrices) == 0:
        raise ValueError("At least one matrix-vector pair is required")
    products = [np.asarray(matrix) @ np.asarray(vector)
                for matrix, vector in zip(matrices, vectors)]
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [list(np.unique(row)) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [list(np.unique(matrix[:, col])) for col in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    parity = np.indices((rows, columns)).sum(axis=0) % 2
    return np.where(parity == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    x_start = (image_width - width) // 2
    y_start = (image_height - height) // 2
    x_end, y_end = x_start + width, y_start + height
    x0, x1 = max(0, x_start), min(image_width, x_end)
    y0, y1 = max(0, y_start), min(image_height, y_end)
    if x0 < x1 and y0 < y1:
        image[y0:y1, x0:x1] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    y, x = np.ogrid[:image_height, :image_width]
    x0, y0 = (image_width - 1) / 2, (image_height - 1) / 2
    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values, dtype=float), data.window
    if values.ndim != 1:
        raise ValueError("Time series must be one-dimensional")
    if window < 1 or window > len(values):
        raise ValueError("Window size must be between 1 and the series length")
    maxima = [i for i in range(1, len(values) - 1)
              if values[i] > values[i - 1] and values[i] > values[i + 1]]
    minima = [i for i in range(1, len(values) - 1)
              if values[i] < values[i - 1] and values[i] < values[i + 1]]
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.asarray(data.labels, dtype=int), data.class_count
    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional vector")
    if labels.size == 0:
        return np.zeros((0, 0 if class_count is None else class_count), dtype=int)
    if np.any(labels < 0):
        raise ValueError("Class labels must be non-negative")
    if class_count is None:
        class_count = int(labels.max()) + 1
    if class_count < 0 or np.any(labels >= class_count):
        raise ValueError("class_count must be greater than every class label")
    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result
