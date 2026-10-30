import pandas as pd

from nba_pipeline.scripts.fit_joint_possession_age_curve import fit_joint_age_curve


def test_joint_curve_is_reference_anchored_and_flips_defense_positive_good():
    ages = list(range(20, 31))
    age_effects = pd.DataFrame(
        {
            "age": ages,
            "off_raw_effect_per_100": [-(age - 27) ** 2 / 10 for age in ages],
            # Raw solver defense is points allowed, so positive values are bad.
            "def_raw_effect_per_100": [(age - 27) ** 2 / 20 for age in ages],
            "off_slots": [1000] * len(ages),
            "def_slots": [1000] * len(ages),
        }
    )

    curve, receipt = fit_joint_age_curve(
        age_effects,
        degree=2,
        age_min=20,
        age_max=30,
        reference_age=27.0,
    )

    reference = curve.loc[curve["age"] == 27].iloc[0]
    age_30 = curve.loc[curve["age"] == 30].iloc[0]
    assert abs(reference["orapm_age_effect"]) < 1e-10
    assert abs(reference["drapm_age_effect"]) < 1e-10
    assert round(age_30["orapm_age_effect"], 8) == -0.9
    assert round(age_30["drapm_age_effect"], 8) == -0.45
    assert receipt["defense_output_sign"] == "positive_good"
    assert receipt["fit_age_min"] == 20
    assert receipt["fit_age_max"] == 30
