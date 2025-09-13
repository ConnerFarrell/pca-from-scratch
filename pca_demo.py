# -*- coding: utf-8 -*-

"""
PCA from Scratch + Visualization Demo

- Implements PCA via covariance eigen-decomposition (NumPy).
- Compares to scikit-learn PCA on the same standardized data.
- Visualizes 2D and 3D embeddings for high-dimensional datasets (Iris by default).

"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA as SKPCA


# -----------------------------
# PCA (from scratch) utilities
# -----------------------------

def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero-mean and unit-variance.

    Returns:
        X_std: standardized data
        mean_: per-feature mean
        scale_: per-feature standard deviation (with zeros protected)
    """
    mean_ = np.mean(X, axis=0)
    scale_ = np.std(X, axis=0, ddof=0)
    # Protect against zero std to avoid divide-by-zero
    scale_safe = np.where(scale_ == 0, 1.0, scale_)
    X_std = (X - mean_) / scale_safe
    return X_std, mean_, scale_safe


@dataclass
class PCAResult:
    components_: np.ndarray        # shape (n_components, n_features)
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    mean_: np.ndarray
    scores_: np.ndarray            # projected data: shape (n_samples, n_components)


def pca_fit_transform(
    X: np.ndarray,
    n_components: Optional[int] = None,
    standardize_input: bool = True
) -> PCAResult:
    """
    Fit PCA using covariance eigen-decomposition and return the projected data.

    Steps:
    1) (Optional) Standardize X to zero-mean, unit-variance.
    2) Compute covariance matrix S = (X^T X) / (n-1).
    3) Eigen-decompose S (symmetric) with eigh.
    4) Sort eigenvalues/eigenvectors in descending order.
    5) Choose top k components.
    6) Project X onto components to obtain scores.

    Notes:
    - Using eigh because covariance matrix is symmetric positive semi-definite.
    """
    if standardize_input:
        X_proc, mean_, scale_ = standardize(X)
    else:
        mean_ = np.mean(X, axis=0)
        X_proc = X - mean_
        scale_ = np.ones_like(mean_)

    n_samples, n_features = X_proc.shape
    if n_components is None:
        n_components = n_features

    # Covariance matrix
    S = (X_proc.T @ X_proc) / (n_samples - 1)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(S)  # ascending order
    order = np.argsort(eigvals)[::-1]     # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Select components
    k = int(n_components)
    components = eigvecs[:, :k].T  # shape (k, n_features)

    # Scores (projections)
    scores = X_proc @ components.T  # (n_samples, k)

    # Explained variance (eigenvalues of covariance)
    explained_variance = eigvals[:k]

    # Total variance is sum of all eigenvalues
    total_var = np.sum(eigvals) if np.sum(eigvals) > 0 else 1.0
    explained_variance_ratio = explained_variance / total_var

    # Singular values relate to sqrt of eigenvalues * sqrt(n_samples - 1)
    singular_values = np.sqrt(explained_variance * (n_samples - 1))

    return PCAResult(
        components_=components,
        explained_variance_=explained_variance,
        explained_variance_ratio_=explained_variance_ratio,
        singular_values_=singular_values,
        mean_=mean_,
        scores_=scores
    )


# -----------------------------
# Comparison utilities
# -----------------------------

def compare_with_sklearn(X: np.ndarray, n_components: int = 2) -> Tuple[PCAResult, SKPCA]:
    """
    Fit our PCA (on standardized data) and sklearn PCA (on the same standardized data),
    then return both and print a brief comparison.
    """
    # Standardize first to match our default behavior
    X_std, _, _ = standardize(X)

    ours = pca_fit_transform(X_std, n_components=n_components, standardize_input=False)

    sk = SKPCA(n_components=n_components, svd_solver="full")
    sk.fit(X_std)
    sk_scores = sk.transform(X_std)

    # Compare explained variance ratios
    ratio_diff = np.abs(ours.explained_variance_ratio_ - sk.explained_variance_ratio_)

    # Components can differ by sign (and minor rotations if eigenvalues are equal).
    # We'll compute absolute alignment between our components and sklearn's.
    # Alignment matrix: ours.components_ @ sk.components_.T -> ideally close to a signed permutation.
    alignment = np.abs(ours.components_ @ sk.components_.T)

    print("=== PCA Comparison (ours vs. scikit-learn on standardized data) ===")
    print(f"Explained variance ratio (ours):      {np.round(ours.explained_variance_ratio_, 6)}")
    print(f"Explained variance ratio (sklearn):  {np.round(sk.explained_variance_ratio_, 6)}")
    print(f"Absolute difference:                 {np.round(ratio_diff, 6)}")
    print("Component alignment (abs dot products):")
    print(np.round(alignment, 6))

    # Return sklearn object with added scores for convenience
    sk.scores_ = sk_scores
    return ours, sk


# -----------------------------
# Visualization utilities
# -----------------------------

def scatter_2d(scores: np.ndarray, y: np.ndarray, title: str = "PCA 2D Projection", target_names=None):
    """
    2D scatter plot of PCA scores.
    - No explicit colors set; matplotlib will handle default mapping.
    """
    plt.figure()
    if target_names is None:
        plt.scatter(scores[:, 0], scores[:, 1], c=y)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(title)
    else:
        # Plot by class to include legend without specifying colors
        for t in np.unique(y):
            idx = y == t
            plt.scatter(scores[idx, 0], scores[idx, 1], label=str(target_names[t]))
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_3d(scores: np.ndarray, y: np.ndarray, title: str = "PCA 3D Projection", target_names=None):
    """
    3D scatter plot of PCA scores.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if target_names is None:
        ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=y)
    else:
        for t in np.unique(y):
            idx = y == t
            ax.scatter(scores[idx, 0], scores[idx, 1], scores[idx, 2], label=str(target_names[t]))
        ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Demo runner
# -----------------------------

def demo(dataset: str = "iris", n_components_2d: int = 2, n_components_3d: int = 3):
    """
    Run a full demo:
      - Load dataset (iris or wine)
      - Compare scratch PCA vs sklearn PCA on standardized data
      - Show 2D and 3D projections using our PCA implementation
    """
    if dataset.lower() == "iris":
        data = datasets.load_iris()
        X = data.data
        y = data.target
        names = data.target_names
    elif dataset.lower() == "wine":
        data = datasets.load_wine()
        X = data.data
        y = data.target
        names = data.target_names
    else:
        raise ValueError("Unsupported dataset. Choose 'iris' or 'wine'.")

    # Compare (uses standardized data internally)
    ours2, sk2 = compare_with_sklearn(X, n_components=n_components_2d)

    # Visualize (2D)
    scatter_2d(ours2.scores_, y, title=f"{dataset.title()} PCA (Scratch) - 2D", target_names=names)

    # 3D using our PCA
    ours3 = pca_fit_transform(X, n_components=n_components_3d, standardize_input=True)
    scatter_3d(ours3.scores_, y, title=f"{dataset.title()} PCA (Scratch) - 3D", target_names=names)


if __name__ == "__main__":
    # Default: run Iris demo
    demo(dataset="iris")