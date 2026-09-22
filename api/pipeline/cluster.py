"""Layout and clustering.

Two computations, in this order:

1. **UMAP** projects the embeddings to 2D. A fixed ``random_state`` makes the
   projection reproducible, which matters because the layout is persisted: two
   runs over the same corpus must not produce two different maps.
2. **HDBSCAN** finds dense groups *in the 2D projection*, not in the original
   high-dimensional space. Clustering the projection rather than the source means
   the clusters correspond to the blobs the user can actually see. The tradeoff
   is accepted deliberately: the clusters are slightly less natural than
   high-dimensional ones, in exchange for a map that does not lie about where
   its groups are.

Two traps this module avoids:

* **Noise is not a topic.** HDBSCAN labels outliers ``-1``. Treating ``-1`` as a
  cluster produces one giant meaningless group; it is kept separate and rendered
  as its own muted category.
* **``umap.transform()`` must not be used for growth.** It places new points
  using an approximation that is inconsistent with the fitted embedding, so
  existing nodes visibly jump. On growth the whole projection is recomputed with
  the same seed, and ``landscapes.generation`` is bumped so the UI can animate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from config import Settings
from models import UNCLUSTERED_LABEL

logger = logging.getLogger(__name__)

#: Below this many papers UMAP and HDBSCAN have nothing to work with, so the
#: points are placed on a circle and the clustering is skipped.
MIN_FOR_UMAP = 5


class ReducerLike(Protocol):
    def fit_transform(self, matrix: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Layout:
    """Coordinates and cluster labels, aligned to the caller's id order."""

    coords: np.ndarray  # (n, 2) float32
    labels: np.ndarray  # (n,) int; -1 means unclustered

    @property
    def cluster_count(self) -> int:
        """Number of real clusters. Noise is not counted."""
        return count_clusters(self.labels)

    @property
    def unclustered_count(self) -> int:
        return int(np.sum(self.labels == UNCLUSTERED_LABEL))


def circle_positions(count: int) -> np.ndarray:
    """Fallback placement for inputs too small to project."""
    if count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def project(
    matrix: np.ndarray,
    settings: Settings,
    *,
    reducer: ReducerLike | None = None,
) -> np.ndarray:
    """Project an (n, dim) matrix to (n, 2). Deterministic for a fixed seed."""
    count = matrix.shape[0]
    if count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if count < MIN_FOR_UMAP:
        logger.info("Only %d paper(s); using a circular layout instead of UMAP", count)
        return circle_positions(count)

    if reducer is not None:
        return np.asarray(reducer.fit_transform(matrix), dtype=np.float32)

    import umap

    model = umap.UMAP(
        n_neighbors=min(settings.umap_n_neighbors, count - 1),
        min_dist=settings.umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=settings.umap_random_state,
        # Single-threaded so the projection is reproducible run to run.
        n_jobs=1,
    )
    return np.asarray(model.fit_transform(matrix), dtype=np.float32)


def cluster_labels(
    coords: np.ndarray,
    settings: Settings,
    *,
    clusterer: Any | None = None,
) -> np.ndarray:
    """Cluster 2D coordinates with HDBSCAN. ``-1`` marks unclustered papers."""
    count = coords.shape[0]
    if count == 0:
        return np.zeros(0, dtype=int)

    # An explicit min_cluster_size is an instruction; the derived one is a
    # heuristic, and heuristics need a safety net (see the fallback below).
    auto_sized = settings.hdbscan_min_cluster_size <= 0
    min_cluster_size = settings.hdbscan_min_cluster_size or max(3, count // 12)
    if count < min_cluster_size:
        logger.info(
            "%d paper(s) is below min_cluster_size=%d; everything is unclustered",
            count,
            min_cluster_size,
        )
        return np.full(count, UNCLUSTERED_LABEL, dtype=int)

    injected = clusterer is not None
    if clusterer is None:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            # Without this, sklearn leaves ``min_samples`` at ``min_cluster_size``
            # and HDBSCAN rejects boundary points as noise: measured on a real
            # 60-paper landscape, 24 of them. Lowering it to 2 fixes the noise
            # but over-splits clean structure, so 3 is the default.
            min_samples=max(1, min(settings.hdbscan_min_samples, min_cluster_size)),
            metric="euclidean",
        )

    try:
        labels = np.asarray(clusterer.fit_predict(coords), dtype=int)
    except Exception as exc:  # noqa: BLE001 - clustering is a nicety, not a gate
        logger.warning("Clustering failed (%s); treating every paper as unclustered", exc)
        return np.full(count, UNCLUSTERED_LABEL, dtype=int)

    if auto_sized and not injected and count_clusters(labels) == 0:
        # HDBSCAN found no density structure at all. On small corpora this is
        # near-arbitrary rather than meaningful: a real 19-paper landscape gave
        # zero clusters at min_samples=3 and three clusters plus 37% noise at
        # min_samples=2. Rendering that as nineteen grey dots communicates
        # nothing, so treat the set as the single region it evidently is and let
        # the naming stage describe it. Only for the derived threshold -- an
        # explicit HDBSCAN_MIN_CLUSTER_SIZE means the caller wants silence.
        logger.info(
            "No density structure in %d papers; treating the set as one region", count
        )
        labels = np.zeros(count, dtype=int)

    noise = int(np.sum(labels == UNCLUSTERED_LABEL))
    logger.info(
        "Clustered %d papers into %d cluster(s); %d unclustered",
        count,
        count_clusters(labels),
        noise,
    )
    if noise and noise > count // 3:
        # A map that is a third grey dots has failed at the one thing it is for.
        # Worth a warning because the usual cause is a settings change.
        logger.warning(
            "%d of %d papers (%.0f%%) are unclustered; consider lowering "
            "HDBSCAN_MIN_SAMPLES",
            noise,
            count,
            100.0 * noise / count,
        )
    return labels


def layout(
    matrix: np.ndarray,
    settings: Settings,
    *,
    reducer: ReducerLike | None = None,
    clusterer: Any | None = None,
) -> Layout:
    """Project and cluster in one call."""
    coords = project(matrix, settings, reducer=reducer)
    labels = cluster_labels(coords, settings, clusterer=clusterer)
    return Layout(coords=coords, labels=labels)


def count_clusters(labels: np.ndarray) -> int:
    """Number of real clusters. ``-1`` is noise and is excluded."""
    present = {int(label) for label in labels if int(label) != UNCLUSTERED_LABEL}
    return len(present)


def group_labels(labels: np.ndarray) -> dict[int, list[int]]:
    """Map each label (including ``-1``) to the row indices carrying it."""
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return groups


def cluster_centroids(coords: np.ndarray, labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """Mean position of each label's members, for placing cluster labels."""
    centroids: dict[int, tuple[float, float]] = {}
    for label, indices in group_labels(labels).items():
        points = coords[indices]
        centroids[label] = (float(points[:, 0].mean()), float(points[:, 1].mean()))
    return centroids


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Scale coordinates into roughly [-1, 1] so the frontend has a stable range.

    The absolute scale of a UMAP projection is arbitrary, so the stored values
    are normalized once here rather than leaving every consumer to guess a zoom
    level. Degenerate spreads (all points identical) collapse to the origin.
    """
    if coords.shape[0] == 0:
        return coords
    centered = coords - coords.mean(axis=0, keepdims=True)
    span = np.abs(centered).max()
    if span <= 0:
        return np.zeros_like(coords)
    return (centered / span).astype(np.float32)


#: Distinguishable palette for cluster coloring, assigned in label order.
CLUSTER_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#4f46e5",
    "#0d9488",
    "#b45309",
    "#9333ea",
)

#: Deliberately desaturated so the unclustered bucket never reads as a topic.
UNCLUSTERED_COLOR = "#94a3b8"


def color_for_label(label: int) -> str:
    if label == UNCLUSTERED_LABEL:
        return UNCLUSTERED_COLOR
    return CLUSTER_COLORS[label % len(CLUSTER_COLORS)]


__all__ = [
    "CLUSTER_COLORS",
    "MIN_FOR_UMAP",
    "UNCLUSTERED_COLOR",
    "Layout",
    "circle_positions",
    "cluster_centroids",
    "cluster_labels",
    "color_for_label",
    "count_clusters",
    "group_labels",
    "layout",
    "normalize_coords",
    "project",
]
