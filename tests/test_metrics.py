"""Tests for the question-quality metrics (pure functions + a stub model)."""

from typing import Any

import pytest

from mind2web_vqg.metrics import (
    answer_scores,
    chrf,
    compute_metrics,
    containment,
    distinct_n,
    exact_match,
    is_well_formed,
    normalize,
    rouge_l,
    round_trip,
    self_bleu,
    token_f1,
    unique_ratio,
)


class StubModel:
    """Duck-typed stand-in for a QuestionGenerator (records what it was asked)."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, Any]] = []

    def generate(self, prompt: str, image: Any = None) -> str:
        self.calls.append((prompt, image))
        return self.answer


# --------------------------------------------------------------------------- #
# Answer comparison
# --------------------------------------------------------------------------- #
def test_normalize() -> None:
    assert normalize("  Search,  now! ") == "search now"
    assert normalize(None) == ""


def test_exact_match_and_containment() -> None:
    assert exact_match("Search", "search") == 1.0
    assert exact_match("search field", "search") == 0.0
    assert exact_match("", "") == 0.0
    assert containment("the search field", "search") == 1.0
    assert containment("nothing", "search") == 0.0


def test_token_f1() -> None:
    assert token_f1("search field", "search field") == 1.0
    assert token_f1("search", "search field") == pytest.approx(2 / 3)
    assert token_f1("totally different", "search field") == 0.0
    assert token_f1("", "search") == 0.0


# --------------------------------------------------------------------------- #
# Question overlap
# --------------------------------------------------------------------------- #
def test_rouge_l() -> None:
    assert rouge_l("What text is on the button?", "What text is on the button?") == 1.0
    assert rouge_l("completely unrelated words", "another phrase entirely") < 0.2


def test_chrf() -> None:
    assert chrf("what is the label", "what is the label") == 1.0
    assert chrf("what is the label", "where is the element") < 0.5
    assert chrf("", "where is the element") == 0.0


# --------------------------------------------------------------------------- #
# Diversity and format
# --------------------------------------------------------------------------- #
def test_distinct_and_unique() -> None:
    same = ["a b c", "a b c"]
    assert unique_ratio(same) == 0.5
    assert distinct_n(same, 1) == 3 / 6
    assert unique_ratio(["a b", "c d"]) == 1.0


def test_self_bleu_detects_collapse() -> None:
    assert self_bleu(["same question here?"] * 4) == 1.0
    diverse = ["what is the label?", "where is it located?", "what kind of field?", "what value?"]
    assert self_bleu(diverse) < 0.5
    assert self_bleu(["only one?"]) == 0.0


def test_is_well_formed() -> None:
    assert is_well_formed("What text is on the button?")
    assert not is_well_formed("no question mark here")
    assert not is_well_formed("hi?")
    assert not is_well_formed("")
    assert not is_well_formed("Here is a question about the page?")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def records() -> list[dict[str, Any]]:
    return [
        {
            "id": "a:0::target_label",
            "target": "What text is shown on the button?",
            "answer": "Search",
            "image": "images/a.jpg",
            "meta": {"category": "target_label"},
            "prediction": "What does the button say?",
        },
        {
            "id": "a:0::target_region",
            "target": 'In which part of the page is "Search" located?',
            "answer": "top-right",
            "image": "images/a.jpg",
            "meta": {"category": "target_region"},
            "prediction": "Where is the search button?",
        },
    ]


def test_compute_metrics_overall_and_per_category() -> None:
    metrics = compute_metrics(records())
    assert metrics["n"] == 2
    assert metrics["well_formed_rate"] == 1.0
    assert metrics["empty_rate"] == 0.0
    assert metrics["unique_ratio"] == 1.0
    assert 0.0 <= metrics["rouge_l"] <= 1.0
    assert set(metrics["by_category"]) == {"target_label", "target_region"}
    assert metrics["by_category"]["target_label"]["n"] == 1


def test_compute_metrics_handles_empty_predictions() -> None:
    rows = records()
    rows[0]["prediction"] = ""
    metrics = compute_metrics(rows)
    assert metrics["empty_rate"] == 0.5
    assert metrics["well_formed_rate"] == 0.5
    assert metrics["n"] == 2


def test_compute_metrics_on_an_empty_run() -> None:
    metrics = compute_metrics([])
    assert metrics["n"] == 0
    assert metrics["well_formed_rate"] == 0.0


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
def test_round_trip_asks_and_scores() -> None:
    model = StubModel("Search")
    rounds = round_trip(records(), model)
    assert [row["question"] for row in rounds] == [row["prediction"] for row in records()]
    assert all(row["image"] == "images/a.jpg" for row in rounds)
    assert rounds[0]["em"] == 1.0  # "Search" == gold "Search"
    assert rounds[1]["em"] == 0.0  # "Search" != gold "top-right"
    assert all("Question:" in prompt for prompt, _ in model.calls)

    scores = answer_scores(rounds)
    assert scores["n"] == 2
    assert scores["em"] == 0.5
    assert scores["by_category"]["target_label"]["em"] == 1.0


def test_round_trip_can_ask_the_reference_question() -> None:
    model = StubModel("top-right")
    rounds = round_trip(records(), model, use_reference=True)
    assert rounds[0]["question"] == records()[0]["target"]
    assert answer_scores(rounds)["em"] == 0.5
