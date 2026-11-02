"""Experience summaries share decisions without posing as source or public measurements."""
from copy import deepcopy

import pytest

import build
import experience


def reported_card(catalog):
    return next(c for c in catalog.cards.values() if c["evidence_level"] == "reported_experience")


def test_reported_experience_is_distinct_from_source_evidence(catalog):
    card = reported_card(catalog)
    assert not card["source_evidence"]
    assert card["experience"]
    assert card["status"] == "candidate_only"
    broken = deepcopy(catalog)
    broken.cards[card["id"]]["evidence_level"] = "source_observed"
    with pytest.raises(ValueError, match="needs source_evidence"):
        broken.validate()


def test_reported_card_cannot_lose_its_evidence(catalog):
    broken = deepcopy(catalog)
    reported_card(broken)["experience"] = []
    with pytest.raises(ValueError, match="needs experience"):
        broken.validate()


@pytest.mark.parametrize("mutation", ["raw_trace", "latency", "empty_conditions", "empty_summary",
                                     "unqualified_source", "validation_promotion", "unknown_outcome"])
def test_invalid_public_summary_is_rejected(catalog, mutation):
    entry = deepcopy(reported_card(catalog)["experience"][0])
    if mutation == "raw_trace":
        entry["trace_path"] = "/private/run/trace.json"
    elif mutation == "latency":
        entry["latency_us"] = 1.0
    elif mutation == "empty_conditions":
        entry["conditions"] = []
    elif mutation == "empty_summary":
        entry["summary"] = ""
    elif mutation == "unqualified_source":
        entry["provenance"]["references"] = ["https://github.com/AMD-AGI/GEAK/blob/main/summary.md"]
    elif mutation == "validation_promotion":
        entry["provenance"]["verification"] = "internally_validated"
    elif mutation == "unknown_outcome":
        entry["outcome"] = "fastest_everywhere"
    assert experience.validate("test", [entry])


def test_private_validation_requires_context_without_a_public_run(catalog):
    broken = deepcopy(catalog)
    card = reported_card(broken)
    entry = card["experience"][0]
    entry["provenance"] = {"kind": "private_summary", "references": [],
                           "artifacts": "not_published", "verification": "internally_validated"}
    with pytest.raises(ValueError, match="complete public comparison context"):
        broken.validate()
    entry["context"] = {"gfx": "gfx950", "software": "test fixture revision",
                        "dtype": "bf16", "shape_layout": "M=16, N=128, K=256; contiguous",
                        "baseline": "test fixture baseline", "protocol": "paired eager timing",
                        "correctness": "independent reference and unchanged tolerance"}
    broken.validate()
    _, text = build.render_decisions(broken, broken.card_owners[card["id"]])
    assert "internally_validated" in text
    assert "test fixture baseline" in text
    entry["context"]["baseline"] = ""
    assert experience.validate("test", [entry])


def test_general_cards_use_the_same_validation_and_rendering(catalog):
    path, text = build.render_decisions(catalog, "general")
    assert path.read_text() == text
    for card in catalog.general["decisions"]["cards"]:
        assert catalog.card_owners[card["id"]] == "general"
        assert f'<a id="{card["id"]}"></a>' in text
        assert "not_reproduced" in text
    broken = deepcopy(catalog)
    broken.general["decisions"]["cards"][0]["axes"].append("unknown_axis")
    with pytest.raises(ValueError, match="unknown performance axis"):
        broken.validate()


def test_existing_source_card_renders_both_evidence_channels(catalog):
    card = next(c for c in catalog.cards.values() if c["source_evidence"] and c.get("experience"))
    _, text = build.render_decisions(catalog, catalog.card_owners[card["id"]])
    assert "source_observed" in text
    assert "Reported experience" in text
    assert "Historical summary" in text
    assert card["source_evidence"][0] in text
