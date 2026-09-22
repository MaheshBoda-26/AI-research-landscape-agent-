"""Tests for synthesis: reference validation and graceful degradation.

``validate_synthesis`` is the guard that stops a hallucinated arXiv id from
becoming a graph edge pointing at nothing. It is tested adversarially — every
mutation a model might plausibly produce is fed in and must be dropped, not
trusted.
"""

from __future__ import annotations

from config import Settings
from conftest import make_paper
from models import (
    ClusterLabel,
    ClusterNaming,
    EdgeClaim,
    LandscapeSynthesis,
    OpenProblem,
    PaperExtraction,
    ReadingStep,
    Tension,
)
from pipeline.synthesize import (
    MAX_READING_PATH,
    fallback_synthesis,
    group_by_cluster,
    label_clusters,
    placeholder_label,
    synthesize_landscape,
    validate_synthesis,
)

IDS = {"p1", "p2", "p3"}


def edge(src="p1", dst="p2", kind="extends", weight=0.6, rationale="r") -> EdgeClaim:
    return EdgeClaim(src_paper_id=src, dst_paper_id=dst, kind=kind, weight=weight, rationale=rationale)


def synthesis(**overrides) -> LandscapeSynthesis:
    data = {"title": "T", "summary": "S"}
    data.update(overrides)
    return LandscapeSynthesis(**data)


class ScriptedCompleter:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    def complete_json(self, *, system, user, schema, max_repairs=None, temperature=0.0):
        self.prompts.append(user)
        if not self._responses:
            return None
        outcome = self._responses.pop(0)
        return outcome


# --------------------------------------------------------------------------- #
# Edge validation
# --------------------------------------------------------------------------- #


def test_a_valid_edge_survives():
    result = validate_synthesis(synthesis(edges=[edge()]), IDS)
    assert len(result.edges) == 1


def test_an_edge_to_an_unknown_paper_is_dropped():
    result = validate_synthesis(synthesis(edges=[edge(dst="2406.99999")]), IDS)
    assert result.edges == []


def test_an_edge_from_an_unknown_paper_is_dropped():
    result = validate_synthesis(synthesis(edges=[edge(src="invented/0000")]), IDS)
    assert result.edges == []


def test_self_loops_are_dropped():
    result = validate_synthesis(synthesis(edges=[edge(src="p1", dst="p1")]), IDS)
    assert result.edges == []


def test_duplicate_edges_are_deduplicated():
    result = validate_synthesis(synthesis(edges=[edge(), edge(), edge()]), IDS)
    assert len(result.edges) == 1


def test_the_same_pair_with_different_kinds_is_kept():
    result = validate_synthesis(
        synthesis(edges=[edge(kind="extends"), edge(kind="contradicts")]), IDS
    )
    assert len(result.edges) == 2


def test_valid_edges_among_invalid_ones_are_kept():
    result = validate_synthesis(
        synthesis(edges=[edge(dst="nope"), edge(src="p1", dst="p3"), edge(src="x", dst="y")]), IDS
    )
    assert len(result.edges) == 1
    assert result.edges[0].dst_paper_id == "p3"


def test_edge_count_is_capped():
    many = [edge(src="p1", dst=f"p{i}", kind="extends") for i in range(3)]
    many += [edge(src=p, dst=q) for p in ("p1", "p2", "p3") for q in ("p1", "p2", "p3")]
    assert len(validate_synthesis(synthesis(edges=many), IDS).edges) <= 60


# --------------------------------------------------------------------------- #
# Tensions
# --------------------------------------------------------------------------- #


def test_a_tension_naming_two_real_papers_survives():
    result = validate_synthesis(
        synthesis(tensions=[Tension(statement="they disagree", paper_a_id="p1", paper_b_id="p2")]),
        IDS,
    )
    assert len(result.tensions) == 1


def test_a_tension_naming_an_unknown_paper_is_dropped():
    result = validate_synthesis(
        synthesis(tensions=[Tension(statement="x", paper_a_id="p1", paper_b_id="ghost")]), IDS
    )
    assert result.tensions == []


def test_a_tension_between_a_paper_and_itself_is_dropped():
    result = validate_synthesis(
        synthesis(tensions=[Tension(statement="x", paper_a_id="p1", paper_b_id="p1")]), IDS
    )
    assert result.tensions == []


def test_an_empty_tension_list_stays_empty():
    """No disagreement is a legitimate finding, not a gap to fill."""
    assert validate_synthesis(synthesis(tensions=[]), IDS).tensions == []


# --------------------------------------------------------------------------- #
# Open problems
# --------------------------------------------------------------------------- #


def test_unknown_supporting_ids_are_stripped_but_the_problem_is_kept():
    result = validate_synthesis(
        synthesis(
            open_problems=[
                OpenProblem(statement="s", why_open="w", supporting_paper_ids=["p1", "ghost"])
            ]
        ),
        IDS,
    )
    assert len(result.open_problems) == 1
    assert result.open_problems[0].supporting_paper_ids == ["p1"]


# --------------------------------------------------------------------------- #
# Reading path
# --------------------------------------------------------------------------- #


def test_reading_path_is_renumbered_densely_after_drops():
    result = validate_synthesis(
        synthesis(
            reading_path=[
                ReadingStep(paper_id="ghost", position=1, why="x"),
                ReadingStep(paper_id="p2", position=2, why="y"),
                ReadingStep(paper_id="p3", position=5, why="z"),
            ]
        ),
        IDS,
    )
    assert [step.paper_id for step in result.reading_path] == ["p2", "p3"]
    assert [step.position for step in result.reading_path] == [1, 2]


def test_reading_path_duplicates_are_removed():
    result = validate_synthesis(
        synthesis(
            reading_path=[
                ReadingStep(paper_id="p1", position=1, why="a"),
                ReadingStep(paper_id="p1", position=2, why="b"),
            ]
        ),
        IDS,
    )
    assert len(result.reading_path) == 1


def test_reading_path_order_follows_the_model_intent_not_the_input_order():
    result = validate_synthesis(
        synthesis(
            reading_path=[
                ReadingStep(paper_id="p3", position=3, why="last"),
                ReadingStep(paper_id="p1", position=1, why="first"),
            ]
        ),
        IDS,
    )
    assert [step.paper_id for step in result.reading_path] == ["p1", "p3"]


def test_reading_path_is_capped():
    steps = [ReadingStep(paper_id=f"p{i}", position=i) for i in range(50)]
    many_ids = {f"p{i}" for i in range(50)}
    assert len(validate_synthesis(synthesis(reading_path=steps), many_ids).reading_path) == MAX_READING_PATH


def test_an_entirely_invalid_synthesis_degrades_to_empty_sections():
    result = validate_synthesis(
        synthesis(
            edges=[edge(src="a", dst="b")],
            tensions=[Tension(statement="x", paper_a_id="a", paper_b_id="b")],
            reading_path=[ReadingStep(paper_id="a", position=1)],
        ),
        IDS,
    )
    assert result.edges == [] and result.tensions == [] and result.reading_path == []
    # The prose survives: it is the only part that cannot be verified by id.
    assert result.title == "T"


# --------------------------------------------------------------------------- #
# Cluster labeling
# --------------------------------------------------------------------------- #


def test_placeholder_label_is_visibly_a_placeholder():
    assert placeholder_label(3).label == "Cluster 3"
    assert "could not be named" in placeholder_label(3).description


def test_labeling_without_a_completer_keeps_placeholders(settings: Settings):
    members = {0: [make_paper(paper_id="p1")], 1: [make_paper(paper_id="p2")]}
    clusters = label_clusters("topic", members, {}, None, settings)
    assert [c.local_label for c in clusters] == [0, 1]
    assert all(c.label.startswith("Cluster") for c in clusters)


def test_noise_is_never_named_as_a_cluster(settings: Settings):
    members = {-1: [make_paper(paper_id="p1")], 0: [make_paper(paper_id="p2")]}
    clusters = label_clusters("topic", members, {}, None, settings)
    assert [c.local_label for c in clusters] == [0]


def test_a_successful_label_is_applied(settings: Settings):
    members = {0: [make_paper(paper_id="p1", title="GraphRAG")]}
    completer = ScriptedCompleter(
        [ClusterNaming(label="Graph-Structured Retrieval", description="graph indices", representative_paper_ids=["p1"])]
    )
    clusters = label_clusters("topic", members, {}, completer, settings)
    assert clusters[0].label == "Graph-Structured Retrieval"
    assert clusters[0].representative_paper_ids == ["p1"]


def test_representative_ids_from_other_clusters_are_dropped(settings: Settings):
    members = {0: [make_paper(paper_id="p1")]}
    completer = ScriptedCompleter(
        [ClusterNaming(label="L", representative_paper_ids=["p1", "p-from-nowhere"])]
    )
    clusters = label_clusters("topic", members, {}, completer, settings)
    assert clusters[0].representative_paper_ids == ["p1"]


def test_a_failed_label_call_keeps_the_placeholder(settings: Settings):
    members = {0: [make_paper(paper_id="p1")], 1: [make_paper(paper_id="p2")]}
    completer = ScriptedCompleter([None, ClusterNaming(label="Named")])
    clusters = label_clusters("topic", members, {}, completer, settings)
    labels = {c.local_label: c.label for c in clusters}
    assert labels[1] == "Named"
    assert labels[0] == "Cluster 0"


def test_an_empty_label_falls_back_to_the_placeholder(settings: Settings):
    members = {0: [make_paper(paper_id="p1")]}
    completer = ScriptedCompleter([ClusterNaming(label="   ")])
    clusters = label_clusters("topic", members, {}, completer, settings)
    assert clusters[0].label == "Cluster 0"


def test_labeling_no_clusters_does_nothing(settings: Settings):
    assert label_clusters("topic", {-1: [make_paper()]}, {}, None, settings) == []


# --------------------------------------------------------------------------- #
# Synthesis orchestration
# --------------------------------------------------------------------------- #


def test_without_a_completer_the_landscape_says_synthesis_did_not_run(settings: Settings):
    papers = [make_paper(paper_id="p1"), make_paper(paper_id="p2")]
    result = synthesize_landscape(
        "topic", [ClusterLabel(local_label=0, label="L")], {0: papers}, papers, {}, None, settings
    )
    assert "not produced" in result.summary
    assert result.edges == []


def test_a_successful_synthesis_is_validated(settings: Settings):
    papers = [make_paper(paper_id="p1"), make_paper(paper_id="p2")]
    completer = ScriptedCompleter(
        [
            synthesis(
                title="Graph Retrieval",
                edges=[edge(src="p1", dst="p2"), edge(src="p1", dst="ghost")],
                reading_path=[ReadingStep(paper_id="p1", position=1)],
            )
        ]
    )
    result = synthesize_landscape(
        "topic", [ClusterLabel(local_label=0, label="L")], {0: papers}, papers, {}, completer, settings
    )
    assert result.title == "Graph Retrieval"
    assert len(result.edges) == 1  # the ghost edge was dropped


def test_synthesis_failure_falls_back_instead_of_crashing(settings: Settings):
    papers = [make_paper(paper_id="p1")]
    completer = ScriptedCompleter([None])
    result = synthesize_landscape(
        "topic", [ClusterLabel(local_label=0, label="L")], {0: papers}, papers, {}, completer, settings
    )
    assert "not produced" in result.summary


def test_computed_cluster_labels_override_whatever_the_model_returned(settings: Settings):
    """The clustering is computed; the model only supplies names for it."""
    papers = [make_paper(paper_id="p1")]
    completer = ScriptedCompleter(
        [synthesis(clusters=[ClusterLabel(local_label=99, label="Bogus")])]
    )
    result = synthesize_landscape(
        "topic", [ClusterLabel(local_label=0, label="Real")], {0: papers}, papers, {}, completer, settings
    )
    assert [c.local_label for c in result.clusters] == [0]
    assert result.clusters[0].label == "Real"


def test_the_prompt_only_offers_supplied_ids(settings: Settings):
    papers = [make_paper(paper_id="p1"), make_paper(paper_id="p2")]
    completer = ScriptedCompleter([synthesis()])
    synthesize_landscape(
        "topic", [ClusterLabel(local_label=0, label="L")], {0: papers}, papers, {}, completer, settings
    )
    assert "p1" in completer.prompts[0] and "p2" in completer.prompts[0]
    assert "the only ids you may reference" in completer.prompts[0]


def test_fallback_synthesis_reports_the_real_counts():
    papers = [make_paper(paper_id="p1"), make_paper(paper_id="p2")]
    result = fallback_synthesis("my topic", [ClusterLabel(local_label=0, label="L")], papers)
    assert result.title == "my topic"
    assert "2 papers" in result.summary


def test_group_by_cluster_preserves_input_order():
    groups = group_by_cluster(["p3", "p1", "p2"], {"p1": 0, "p2": 0, "p3": 1})
    assert groups[0] == ["p1", "p2"]
    assert groups[1] == ["p3"]


def test_group_by_cluster_defaults_missing_labels_to_unclustered():
    groups = group_by_cluster(["p1"], {})
    assert set(groups) == {-1}


def test_extractions_reach_the_cluster_prompt(settings: Settings):
    members = {0: [make_paper(paper_id="p1")]}
    extractions = {"p1": PaperExtraction(problem="the gap", contribution="the claim")}
    completer = ScriptedCompleter([ClusterNaming(label="L")])
    label_clusters("topic", members, extractions, completer, settings)
    assert "the gap" in completer.prompts[0]
    assert "the claim" in completer.prompts[0]
