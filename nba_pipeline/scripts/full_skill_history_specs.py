"""Curated independent measurements for the 1996-97..2025-26 skill library.

Counts, conditional proportions and exposure have explicit units. Provider
percentages, ratings, metadata and duplicated tracking vintages are not extra
independent skills. Makes and combined shooting summaries are derived later.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    numerator: str
    denominator: str
    family: str
    kind: str = "rate"
    initial: float = 0.0
    first_season: int = 1997
    last_season: int = 2026

    def record(self):
        return dict(asdict(self), units="fraction" if self.kind == "pct" else "per_100_" + self.denominator)


def registry():
    result = []
    def rate(name, numerator, side="off_poss", family="core", first=1997, last=2026):
        result.append(Skill(name, numerator, side, family, "rate", 0., first, last))
    def pct(name, makes, attempts, family="shooting", initial=.5, first=1997, last=2026):
        result.append(Skill(name, makes, attempts, family, "pct", initial, first, last))

    zones = ["at_rim", "short_midrange", "long_midrange", "corner3", "arc3"]
    for zone in zones:
        rate(zone + "_fga100", "core." + zone + "_fga", family="shooting")
        pct(zone + "_pct", "core." + zone + "_fgm", "core." + zone + "_fga",
            initial=.35 if zone in {"corner3", "arc3"} else .5)
        pct(zone + "_assisted_make_share", "core." + zone + "_fgm_assisted",
            "core." + zone + "_fgm", family="assistedness")
        rate("assists_" + zone + "100", "core.ast_" + zone, family="passing")
    rate("fta100", "core.fta", family="shooting")
    pct("ft_pct", "core.ftm", "core.fta", initial=.75)
    rate("tov100", "core.tov", family="turnovers")
    for name in ["bad_pass_any", "lost_ball_any", "traveling", "remainder"]:
        pct("tov_" + name + "_share", "core.tov_" + name, "core.tov", family="turnovers")
    pct("tov_bad_pass_oob_share", "core.tov_bad_pass_oob", "core.tov_bad_pass_any", family="turnovers", first=2016)
    pct("tov_lost_ball_oob_share", "core.tov_lost_ball_oob", "core.tov_lost_ball_any", family="turnovers", first=2004)
    for name, first in [("step_oob", 2004), ("offensive_foul", 2018), ("other", 2018)]:
        rate("tov_" + name + "100", "core.tov_" + name, family="turnovers", first=first)
    for name in ["shooting", "offensive", "technical", "flagrant", "loose_ball", "double", "personal_group"]:
        rate("foul_" + name + "100", "core.foul_" + name, "total_poss", "fouls")
    rate("foul_away_from_play100", "core.foul_away_from_play", "total_poss", "fouls", last=2020)
    rate("foul_inbound100", "core.foul_inbound", "total_poss", "fouls", last=2018)
    for name, side in [("oreb", "off_poss"), ("dreb", "def_poss"), ("stl", "def_poss"), ("blk", "def_poss")]:
        rate(name + "100", "core." + name, side)
    for reb in ["oreb", "dreb"]:
        for shot in ["2pt", "3pt", "ft", "other"]:
            pct(reb + "_after_" + shot + "_share", "core." + reb + "_after_" + shot,
                "core." + reb, family="rebounding")
    for name in ["and_one", "transition_points", "second_chance_points", "halfcourt_points", "penalty_points", "paint_points", "fast_break_points", "from_turnover_points"]:
        rate(name + "100", "core." + name, family="scoring_context")

    off = ["potential_ast", "secondary_ast", "passes_made", "passes_received", "touches", "front_ct_touches",
           "drives", "drive_ast", "drive_passes", "drive_tov", "drive_pf", "time_of_poss",
           "elbow_touches", "paint_touches", "post_touches", "pfd"]
    for name in off:
        rate(name + "100", "report." + name, family="creation")
    for name in ["dreb_chances", "dreb_chance_defer", "dreb_contest", "dreb_uncontest"]:
        rate(name + "100", "report." + name, "def_poss", "rebounding_tracking")
    for name in ["oreb_chances", "oreb_chance_defer", "oreb_contest", "oreb_uncontest"]:
        rate(name + "100", "report." + name, "off_poss", "rebounding_tracking")
    for name, side in [
        ("screen_assists", "off_poss"), ("screen_ast_pts", "off_poss"),
        ("box_outs", "total_poss"), ("def_boxouts", "def_poss"), ("off_boxouts", "off_poss"),
        ("box_out_player_rebs", "total_poss"), ("box_out_player_team_rebs", "total_poss"),
        ("deflections", "def_poss"), ("contested_shots", "def_poss"),
        ("contested_shots_2pt", "def_poss"), ("contested_shots_3pt", "def_poss"),
        ("charges_drawn", "def_poss"), ("def_loose_balls_recovered", "def_poss"),
        ("off_loose_balls_recovered", "off_poss"),
    ]:
        rate(name + "100", "report.hustle_" + name, side, "hustle")
    for stem, attempts, makes in [
        ("overall_def", "overall_def_d_fga", "overall_def_d_fgm"),
        ("two_pt_def", "two_pt_def_fg2a", "two_pt_def_fg2m"),
        ("three_pt_def", "three_pt_def_fg3a", "three_pt_def_fg3m"),
        ("less_6ft_def", "less_6ft_def_fga_lt_06", "less_6ft_def_fgm_lt_06"),
        ("less_10ft_def", "less_10ft_def_fga_lt_10", "less_10ft_def_fgm_lt_10"),
        ("more_15ft_def", "more_15ft_def_fga_gt_15", "more_15ft_def_fgm_gt_15"),
    ]:
        rate(stem + "_fga100", "report." + attempts, "def_poss", "defended_shooting")
        pct(stem + "_fg_pct", "report." + makes, "report." + attempts, "defended_shooting")
    for stem in ["catch_shoot", "pull_up", "drive"]:
        for three in ([False, True] if stem != "drive" else [False]):
            tag = "fg3" if three else ("fg2" if stem != "drive" else "fg")
            name = stem + ("_three" if three else ("_two" if stem != "drive" else ""))
            rate(name + "_fga100", "report." + stem + "_" + tag + "a", family="shooting_context")
            pct(name + "_pct", "report." + stem + "_" + tag + "m", "report." + stem + "_" + tag + "a", "shooting_context")
    for bucket in ["very_tight", "tight", "open", "wide_open"]:
        rate(bucket + "_fga100", "report." + bucket + "_fga", family="defender_distance")
        pct(bucket + "_pct", "report." + bucket + "_fgm", "report." + bucket + "_fga", "defender_distance")
    assert len({x.name for x in result}) == len(result)
    return result


GENERIC_SKILLS = registry()
SOURCE_RULES = json.loads((Path(__file__).resolve().parents[1] / "config/full_skill_history_sources.json").read_text())


def versioned_registry():
    result = []
    fields = SOURCE_RULES["versioned_fields"]
    for spec in GENERIC_SKILLS:
        versioned = [x for x in [spec.numerator, spec.denominator] if x.startswith("report.") and x[7:] in fields]
        if not versioned:
            result.append(spec)
            continue
        for version in ["v1", "v2"]:
            if not all(version in fields[x[7:]] for x in versioned):
                continue
            result.append(replace(spec, name=spec.name + "__" + version,
                                  numerator=spec.numerator + "_" + version if spec.numerator in versioned else spec.numerator,
                                  denominator=spec.denominator + "_" + version if spec.denominator in versioned else spec.denominator,
                                  first_season=max(spec.first_season, SOURCE_RULES["v2_first_season"] if version == "v2" else 1997)))
    return result


SKILLS = versioned_registry()


def canonical_aliases(frame):
    result = frame.copy()
    for spec in GENERIC_SKILLS:
        if spec.name in result:
            continue
        old, new = spec.name + "__v1", spec.name + "__v2"
        if old in result or new in result:
            result[spec.name] = result.get(old, float("nan"))
            if new in result:
                mask = result.season.ge(SOURCE_RULES["v2_first_season"])
                result.loc[mask, spec.name] = result.loc[mask, new]
    return result
