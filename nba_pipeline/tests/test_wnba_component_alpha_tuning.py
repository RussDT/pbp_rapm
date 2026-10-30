from wnba_test.tune_wnba_component_alphas import (
    classify_trust,
    coordinate_descent,
)


def test_coordinate_descent_recovers_joint_component_side_minimum():
    target = {"rim": (2, 3), "tov": (1, 2)}

    def score(alphas):
        return sum(
            (alphas[name][0] - expected[0]) ** 2
            + (alphas[name][1] - expected[1]) ** 2
            for name, expected in target.items()
        )

    selected, best, _ = coordinate_descent(
        seed={"rim": (1, 1), "tov": (1, 1)},
        components=["rim", "tov"],
        grid=[1, 2, 3],
        score=score,
        max_passes=3,
    )

    assert selected == target
    assert best == 0


def test_trust_requires_development_and_confirmation_gain():
    assert classify_trust(20_000, 320_000, 2.0, 1.5) == "supported"
    assert classify_trust(20_000, 320_000, 2.0, -0.1) == "mixed"
    assert classify_trust(20_000, 320_000, 0.5, 5.0) == "prior_only"
    assert classify_trust(320_000, 320_000, 0.0, 0.2) == "prior_only"
