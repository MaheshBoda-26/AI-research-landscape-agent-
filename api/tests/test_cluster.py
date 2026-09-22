"""Tests for embeddings, projection, and clustering.

Clustering runs against real scikit-learn HDBSCAN on synthetic blobs, which is
fast and needs no network. UMAP is injected as a fake except in one test that
verifies the real library is wired up correctly, because numba's first-run
compilation is the only reason to keep it out of the hot path.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import store
from config import Settings
from conftest import make_paper
from pipeline.cluster import (
    MIN_FOR_UMAP,
    UNCLUSTERED_COLOR,
    circle_positions,
    cluster_centroids,
    cluster_labels,
    color_for_label,
    count_clusters,
    group_labels,
    layout,
    normalize_coords,
    project,
)
from pipeline.embed import embed_papers, embedding_matrix, embedding_text


class FakeReducer:
    def __init__(self) -> None:
        self.calls = 0

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.stack([matrix[:, 0], matrix[:, 1]], axis=1)


class FakeEmbedder:
    def __init__(self, dim: int = 8, model_name: str = "fake-embed") -> None:
        self._dim = dim
        self._model_name = model_name
        self.encode_calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts) -> np.ndarray:
        self.encode_calls += 1
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), self._dim)).astype(np.float32)


def blobs(sizes=(12, 12, 12), spread: float = 0.25) -> np.ndarray:
    """Well-separated 2D clusters, deterministic."""
    rng = np.random.default_rng(7)
    parts = [
        rng.normal(loc=index * 9.0, scale=spread, size=(count, 2))
        for index, count in enumerate(sizes)
    ]
    return np.vstack(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def test_circle_positions_are_evenly_spaced():
    coords = circle_positions(4)
    assert coords.shape == (4, 2)
    radii = np.linalg.norm(coords, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-6)


def test_circle_positions_of_nothing_is_empty():
    assert circle_positions(0).shape == (0, 2)


def test_projection_of_nothing_is_empty(settings: Settings):
    assert project(np.zeros((0, 8), dtype=np.float32), settings).shape == (0, 2)


def test_tiny_input_skips_umap_and_uses_the_circle(settings: Settings):
    """Below the threshold UMAP has nothing to work with, so it must not run."""
    reducer = FakeReducer()
    matrix = np.zeros((MIN_FOR_UMAP - 1, 8), dtype=np.float32)
    coords = project(matrix, settings, reducer=reducer)
    assert reducer.calls == 0
    assert coords.shape == (MIN_FOR_UMAP - 1, 2)


def test_projection_is_deterministic_for_a_fixed_seed(settings: Settings):
    matrix = np.random.default_rng(1).normal(size=(30, 12)).astype(np.float32)
    first = project(matrix, settings, reducer=FakeReducer())
    second = project(matrix, settings, reducer=FakeReducer())
    assert np.array_equal(first, second)


def test_real_umap_is_wired_up_and_reproducible(settings: Settings):
    """Guards against a misconfigured or missing umap-learn install."""
    settings = replace(settings, umap_n_neighbors=5)
    matrix = np.random.default_rng(3).normal(size=(25, 10)).astype(np.float32)
    first = project(matrix, settings)
    second = project(matrix, settings)
    assert first.shape == (25, 2)
    # The persisted layout must be reproducible run to run.
    assert np.allclose(first, second, atol=1e-4)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def test_three_blobs_cluster_into_three(settings: Settings):
    coords = blobs()
    labels = cluster_labels(coords, settings)
    assert count_clusters(labels) == 3


def test_noise_is_labelled_minus_one_and_is_not_a_cluster(settings: Settings):
    coords = np.vstack([blobs(), np.array([[200.0, 200.0]], dtype=np.float32)])
    labels = cluster_labels(coords, settings)
    assert labels[-1] == -1
    assert count_clusters(labels) == 3


def test_a_single_outlier_never_becomes_its_own_cluster(settings: Settings):
    """The single most common way this kind of map looks broken."""
    coords = np.vstack([blobs(sizes=(10, 10)), np.array([[500.0, -500.0]], dtype=np.float32)])
    labels = cluster_labels(coords, settings)
    groups = group_labels(labels)
    assert count_clusters(labels) == 2
    assert len(groups[-1]) == 1


#: The literal 2D coordinates of a real 19-paper landscape ("mixture-of-experts
#: routing") that HDBSCAN placed entirely in the noise bucket. Real data, kept
#: verbatim, because a synthetic cloud does not reproduce the failure:
#: UMAP output this small has no density gradient for HDBSCAN to find.
REAL_DEGENERATE_LAYOUT = np.array(
    [
        [-0.1703, 0.3824], [0.1922, -0.1518], [0.2029, 0.2865],
        [-0.3924, -0.1449], [-0.6449, 0.2188], [0.5322, 0.0292],
        [0.2535, -0.4524], [0.4682, 0.6668], [-0.0669, -0.788],
        [0.7371, 0.2924], [0.1128, 0.8744], [-0.602, -0.7999],
        [0.7185, -0.6007], [-0.748, -0.4877], [-0.2646, 0.0365],
        [-0.228, -0.4006], [1.0, -0.3857], [-0.4291, 0.8272],
        [-0.6713, 0.5973],
    ],
    dtype=np.float32,
)


def test_the_real_degenerate_layout_really_does_yield_no_clusters():
    """Pins the premise of the fallback test below.

    If a future settings change makes HDBSCAN find structure here, the fallback
    stops being exercised and this test says so.
    """
    from sklearn.cluster import HDBSCAN

    labels = HDBSCAN(min_cluster_size=3, min_samples=3).fit_predict(
        REAL_DEGENERATE_LAYOUT
    )
    assert count_clusters(labels) == 0


def test_a_small_coherent_corpus_is_one_region_not_nineteen_grey_dots(
    settings: Settings,
):
    """Nineteen anonymous dots is not a map.

    When HDBSCAN finds nothing at all, the honest reading is that the corpus is
    one region -- and the naming stage can then describe it.
    """
    labels = cluster_labels(REAL_DEGENERATE_LAYOUT, settings)
    assert count_clusters(labels) == 1
    assert int(np.sum(labels == -1)) == 0


def test_the_fallback_does_not_apply_to_an_explicit_min_cluster_size(
    settings: Settings,
):
    """An explicit threshold is an instruction to stay quiet, not a hint."""
    labels = cluster_labels(
        REAL_DEGENERATE_LAYOUT, replace(settings, hdbscan_min_cluster_size=25)
    )
    assert count_clusters(labels) == 0


def test_the_fallback_does_not_override_an_injected_clusterer(
    settings: Settings,
):
    """A caller-supplied clusterer's verdict is final, noise included."""

    class AllNoise:
        def fit_predict(self, coords: np.ndarray) -> np.ndarray:
            return np.full(coords.shape[0], -1, dtype=int)

    labels = cluster_labels(
        REAL_DEGENERATE_LAYOUT, settings, clusterer=AllNoise()
    )
    assert set(labels) == {-1}


def test_the_fallback_never_hides_a_healthy_landscape(settings: Settings):
    """The safety net must not fire when real structure exists."""
    labels = cluster_labels(blobs(sizes=(20, 20, 20)), settings)
    assert count_clusters(labels) == 3


def test_group_labels_keeps_noise_as_its_own_key():
    labels = np.array([0, 0, -1, 1, 1, 1])
    groups = group_labels(labels)
    assert set(groups) == {0, 1, -1}
    assert groups[-1] == [2]


def test_unclustered_is_excluded_from_the_cluster_count():
    assert count_clusters(np.array([0, 0, 0, -1, -1])) == 1
    assert count_clusters(np.array([-1, -1, -1])) == 0


def test_everything_unclustered_is_handled(settings: Settings):
    coords = np.random.default_rng(5).normal(scale=50.0, size=(30, 2)).astype(np.float32)
    labels = cluster_labels(coords, settings)
    assert len(labels) == 30  # never crashes, even with no structure


def test_input_below_min_cluster_size_is_all_unclustered(settings: Settings):
    labels = cluster_labels(np.zeros((2, 2), dtype=np.float32), settings)
    assert list(labels) == [-1, -1]


def test_clustering_of_nothing_is_empty(settings: Settings):
    assert cluster_labels(np.zeros((0, 2), dtype=np.float32), settings).shape == (0,)


def test_a_failing_clusterer_degrades_to_all_unclustered(settings: Settings):
    class Exploding:
        def fit_predict(self, *_args, **_kwargs):
            raise RuntimeError("hdbscan exploded")

    labels = cluster_labels(blobs(), settings, clusterer=Exploding())
    assert set(labels) == {-1}


def test_explicit_min_cluster_size_overrides_the_auto_rule(settings: Settings):
    tuned = replace(settings, hdbscan_min_cluster_size=25)
    labels = cluster_labels(blobs(sizes=(12, 12, 12)), tuned)
    # No group reaches 25 members, so nothing is a cluster.
    assert count_clusters(labels) == 0


def capture_clusterer(monkeypatch) -> dict[str, object]:
    """Replace sklearn's HDBSCAN with a stub that records its constructor kwargs.

    The constructor is imported lazily inside ``cluster_labels``, so the patch
    has to land on ``sklearn.cluster`` rather than on ``pipeline.cluster``.
    """
    seen: dict[str, object] = {}

    class Recording:
        def fit_predict(self, coords: np.ndarray) -> np.ndarray:
            return np.zeros(coords.shape[0], dtype=int)

    def capture(**kwargs):
        seen.update(kwargs)
        return Recording()

    monkeypatch.setattr("sklearn.cluster.HDBSCAN", capture)
    return seen


def test_min_samples_is_threaded_through_from_settings(settings: Settings, monkeypatch):
    """Regression guard.

    sklearn leaves ``min_samples`` at ``min_cluster_size`` unless it is passed
    explicitly, which on a real landscape marked 40% of papers as noise. If the
    setting ever stops reaching the clusterer, this fails.
    """
    seen = capture_clusterer(monkeypatch)
    cluster_labels(blobs(), replace(settings, hdbscan_min_cluster_size=9))
    assert seen["min_samples"] == settings.hdbscan_min_samples
    assert seen["min_cluster_size"] == 9


def test_min_samples_never_exceeds_min_cluster_size(settings: Settings, monkeypatch):
    """A min_samples above min_cluster_size makes HDBSCAN reject everything."""
    seen = capture_clusterer(monkeypatch)
    cluster_labels(
        blobs(), replace(settings, hdbscan_min_cluster_size=3, hdbscan_min_samples=50)
    )
    assert seen["min_samples"] == 3


def test_a_typical_landscape_is_mostly_clustered(settings: Settings):
    """The product requirement: few grey dots.

    Three well-separated groups of twenty. The default configuration must
    assign nearly all of them to a real cluster.
    """
    labels = cluster_labels(blobs(sizes=(20, 20, 20)), settings)
    assert count_clusters(labels) == 3
    assert int(np.sum(labels == -1)) <= 3


def test_centroids_land_near_their_blobs(settings: Settings):
    coords = blobs(sizes=(12, 12, 12))
    labels = cluster_labels(coords, settings)
    centroids = cluster_centroids(coords, labels)
    for label, (x, _y) in centroids.items():
        if label == -1:
            continue
        members = coords[[i for i, lab in enumerate(labels) if lab == label]]
        assert abs(x - members[:, 0].mean()) < 0.5


def test_layout_returns_aligned_coords_and_labels(settings: Settings):
    matrix = np.random.default_rng(11).normal(size=(40, 12)).astype(np.float32)
    result = layout(matrix, settings)
    assert result.coords.shape == (40, 2)
    assert result.labels.shape == (40,)
    assert result.cluster_count + (1 if result.unclustered_count else 0) >= 1


# --------------------------------------------------------------------------- #
# Coordinate normalization and colors
# --------------------------------------------------------------------------- #


def test_normalization_scales_into_unit_range():
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, -30.0]], dtype=np.float32)
    normalized = normalize_coords(coords)
    assert np.abs(normalized).max() == pytest.approx(1.0)


def test_normalization_centers_the_projection():
    coords = np.array([[100.0, 100.0], [102.0, 100.0]], dtype=np.float32)
    assert np.allclose(normalize_coords(coords).mean(axis=0), [0.0, 0.0], atol=1e-6)


def test_normalization_of_a_degenerate_spread_is_the_origin():
    coords = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    assert np.array_equal(normalize_coords(coords), np.zeros_like(coords))


def test_normalization_of_nothing_is_empty():
    assert normalize_coords(np.zeros((0, 2), dtype=np.float32)).shape == (0, 2)


def test_unclustered_has_its_own_muted_color():
    assert color_for_label(-1) == UNCLUSTERED_COLOR
    assert color_for_label(0) != UNCLUSTERED_COLOR


def test_cluster_colors_are_stable_and_wrap():
    assert color_for_label(0) == color_for_label(0)
    assert color_for_label(0) != color_for_label(1)
    assert color_for_label(999) == color_for_label(999 % 12)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def test_embedding_text_includes_title_and_abstract():
    paper = make_paper(title="T", abstract="A")
    assert embedding_text(paper) == "T\nA"


def test_embeddings_are_cached_across_calls(settings: Settings, conn):
    papers = [make_paper(paper_id="p1"), make_paper(paper_id="p2")]
    store.upsert_papers(conn, papers)
    embedder = FakeEmbedder()

    first = embed_papers(papers, settings, conn, embedder=embedder)
    assert embedder.encode_calls == 1

    second = embed_papers(papers, settings, conn, embedder=embedder)
    assert embedder.encode_calls == 1  # served from the cache
    assert np.allclose(first["p1"], second["p1"])


def test_only_new_papers_are_encoded(settings: Settings, conn):
    """Growing a landscape must not re-embed what it already knows."""
    embedder = FakeEmbedder()
    existing = [make_paper(paper_id=f"p{i}") for i in range(3)]
    store.upsert_papers(conn, existing)
    embed_papers(existing, settings, conn, embedder=embedder)
    assert embedder.encode_calls == 1

    newcomer = make_paper(paper_id="new")
    store.upsert_papers(conn, [newcomer])
    vectors = embed_papers([*existing, newcomer], settings, conn, embedder=embedder)

    assert embedder.encode_calls == 2
    assert set(vectors) == {"p0", "p1", "p2", "new"}


def test_embeddings_roundtrip_through_the_store(settings: Settings, conn):
    papers = [make_paper(paper_id="p1")]
    store.upsert_papers(conn, papers)
    embedder = FakeEmbedder(dim=4)
    expected = embed_papers(papers, settings, conn, embedder=embedder)["p1"]

    fetched = store.fetch_embeddings(conn, embedder.model_name, ["p1"])["p1"]
    restored = np.frombuffer(fetched, dtype=np.float32)
    assert np.allclose(restored, expected)


def test_embeddings_are_scoped_by_model(settings: Settings, conn):
    papers = [make_paper(paper_id="p1")]
    store.upsert_papers(conn, papers)
    embed_papers(papers, settings, conn, embedder=FakeEmbedder(model_name="model-a"))
    other = FakeEmbedder(model_name="model-b")
    embed_papers(papers, settings, conn, embedder=other)
    # A different model has no cache, so it must actually encode.
    assert other.encode_calls == 1


def test_embedding_of_nothing_is_empty(settings: Settings):
    assert embed_papers([], settings, None, embedder=FakeEmbedder()) == {}


def test_a_mismatched_vector_count_is_an_error(settings: Settings):
    class ShortEmbedder(FakeEmbedder):
        def encode(self, texts):
            self.encode_calls += 1
            return np.zeros((1, 4), dtype=np.float32)  # always one row

    with pytest.raises(ValueError, match="vectors for"):
        embed_papers([make_paper(paper_id="a"), make_paper(paper_id="b")], settings, None, embedder=ShortEmbedder())


def test_embedding_matrix_preserves_the_requested_order():
    vectors = {
        "b": np.array([2.0, 0.0], dtype=np.float32),
        "a": np.array([1.0, 0.0], dtype=np.float32),
    }
    matrix = embedding_matrix(["a", "b"], vectors)
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][0] == pytest.approx(2.0)


def test_embedding_matrix_refuses_to_silently_drop_a_missing_paper():
    """Dropping a row would shift every later coordinate onto the wrong paper."""
    with pytest.raises(ValueError, match="Missing embeddings"):
        embedding_matrix(["a", "missing"], {"a": np.zeros(3, dtype=np.float32)})


def test_embedding_matrix_of_nothing_is_empty():
    assert embedding_matrix([], {}).shape == (0, 0)
