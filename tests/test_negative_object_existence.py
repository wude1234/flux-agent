"""Test constraint question generation for negative object existence."""
import pytest
from src.constraint_questions import _negative_object_existence_constraints


def test_negative_object_existence_no_x_pattern():
    """Test "no bowl and no spoon" pattern."""
    constraints = [
        "A yellow cereal box stands on a counter with no bowl and no spoon nearby."
    ]
    entities = ["cereal box", "counter"]

    checks = _negative_object_existence_constraints(constraints, entities)

    assert len(checks) == 2
    # Should extract "bowl" and "spoon"
    objects = {c["object"] for c in checks}
    assert "bowl" in objects
    assert "spoon" in objects

    for c in checks:
        assert c["category"] == "negative_object_existence"
        assert c["expected"] == "yes"  # yes, it's absent
        assert c["negative"] is True
        assert "absent" in c["question"].lower()


def test_negative_object_existence_no_window_sign():
    """Test "no window and no sign" pattern."""
    constraints = [
        "A blue door has a round handle but no window and no sign."
    ]
    entities = ["door", "handle"]

    checks = _negative_object_existence_constraints(constraints, entities)

    assert len(checks) == 2
    objects = {c["object"] for c in checks}
    assert "window" in objects
    assert "sign" in objects


def test_negative_object_existence_with_no_visible():
    """Test "with no visible zipper pull" pattern."""
    constraints = [
        "A black backpack is closed, with no visible zipper pull and no side pocket."
    ]
    entities = ["backpack"]

    checks = _negative_object_existence_constraints(constraints, entities)

    assert len(checks) == 2
    objects = {c["object"] for c in checks}
    assert "zipper pull" in objects
    assert "side pocket" in objects


def test_negative_object_existence_skip_verbs():
    """Test that verb phrases like 'sitting' are filtered out."""
    constraints = [
        "A green park bench is empty, with no people and no animals sitting on it."
    ]
    entities = ["park bench"]

    checks = _negative_object_existence_constraints(constraints, entities)

    # Should extract "people" and "animals", NOT "sitting"
    objects = [c["object"] for c in checks]
    assert "people" in objects or "animals" in objects
    assert "sitting" not in objects


def test_negative_object_existence_skip_symbol_text():
    """Test that symbol/text constraints are skipped (handled by other function)."""
    constraints = [
        "A red sign with no text or logo visible."
    ]
    entities = ["sign"]

    checks = _negative_object_existence_constraints(constraints, entities)

    # Should skip because "text" keyword triggers symbol/text handler
    assert len(checks) == 0


def test_negative_object_existence_skip_relation():
    """Test that relation negations are skipped (handled by other function)."""
    constraints = [
        "The cup is not touching the plate."
    ]
    entities = ["cup", "plate"]

    checks = _negative_object_existence_constraints(constraints, entities)

    # Should skip because "not touching" triggers relation handler
    assert len(checks) == 0


def test_negative_object_existence_without_pattern():
    """Test 'without X' pattern (only nouns, not verb phrases)."""
    constraints = [
        "A blue bird perches on the rim of a brown basket without sitting inside it."
    ]
    entities = ["bird", "basket", "rim"]

    checks = _negative_object_existence_constraints(constraints, entities)

    # "without sitting" should be filtered (verb phrase)
    # This case is actually about action/relation, not object absence
    # So we expect 0 checks or filtered results
    # (The real fix for interaction_005 may need action-level negation handling)
    assert len(checks) == 0 or all("sitting" not in c["object"] for c in checks)
