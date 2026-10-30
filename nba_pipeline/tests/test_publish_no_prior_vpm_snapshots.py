import json

import pandas as pd

from nba_pipeline.scripts.publish_no_prior_vpm_snapshots import build_publish_frame


def test_build_publish_frame_exposes_calendar_keys_and_model_contract(tmp_path):
    components = {"DECOMP_NET_RIM_FREQ": {"off_alpha": 750, "def_alpha": 750}}
    base = tmp_path / "2020"
    for label, day in (("oct01", "2019-10-01"), ("jan01", "2020-01-01"), ("may01", "2020-05-01"), ("aug30", "2020-08-30")):
        target = base / f"{label}_{day}"
        target.mkdir(parents=True)
        pd.DataFrame([{
            "player_id": 1, "player_name": "Player", "season_end_year": 2020,
            "as_of_date": day, "pricing": "net_continuation", "prior_scale": 0.0,
            "possessions": 1.0, "off_poss": 1.0, "def_poss": 1.0,
            "o_DECOMP_NET_RIM_FREQ": 0.1, "d_DECOMP_NET_RIM_FREQ": 0.2,
        }]).to_csv(target / "player_estimates.csv", index=False)
        (target / "manifest.json").write_text(json.dumps({"components": components}))

    # Repeat the four-file fixture across all seven expected season directories.
    for year in range(2021, 2027):
        for source in base.iterdir():
            target = tmp_path / str(year) / source.name.replace("2020", str(year)).replace("2019", str(year - 1))
            target.mkdir(parents=True)
            frame = pd.read_csv(source / "player_estimates.csv")
            frame["player_id"] = year
            frame["season_end_year"] = year
            frame["as_of_date"] = target.name.split("_", 1)[1]
            frame.to_csv(target / "player_estimates.csv", index=False)
            (target / "manifest.json").write_text(json.dumps({"components": components}))

    frame = build_publish_frame(tmp_path)
    assert len(frame) == 28
    assert frame["snapshot_date"].nunique() == 28
    assert set(frame["snapshot_month"]) == {1, 5, 8, 10}
    assert set(frame["time_decay_half_life_days"]) == {550}
    assert set(frame["rubberband_model"]) == {"smooth_clock_aware"}
    assert set(frame["prior_scale"]) == {0.0}
