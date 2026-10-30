# Historical Mechanism Backfills

## Decision

The mechanisms no longer have to be discarded merely because the original
current-board inputs were short. The repository now has three explicitly
separated historical lanes:

| Mechanism | Historical coverage | What is exact | What is reconstructed |
| --- | --- | --- | --- |
| STOP | 1997-2026 RS+PS files | Turnover stops throughout; the full turnover-or-defense-recovered-block definition from 2014 onward | The 1997-2013 raw feed does not carry opposing block descriptions, so those files are turnover-only and must not be called full STOP |
| Transition frequency | 1997-2026 RS+PS | Provider labels in the 2025-2026 labeled era | Earlier rows use the frozen PBP classifier and retrospective pooled calibration |
| Transition-rim frequency | 1997-2026 RS+PS | Provider labels in the 2025-2026 labeled era | Earlier rows use the frozen PBP classifier and retrospective pooled calibration |
| Playtype proxy points | 1997-2026 RS+PS | Actual FGA outcomes, zones, and ordinary PBP descriptions | A seven-family PBP taxonomy plus the frozen transition backcast replaces unavailable provider descriptors |

The reconstructed series use `_BACKCAST` prefixes and separate registry IDs.
They do not overwrite or impersonate the exact current provider targets.

## STOP

`process_stop.py` can be rebuilt directly from ordinary raw play-by-play. The
historical run produced all 60 regular-season/postseason files. A temporary
2021 rebuild matched the existing canonical 2021 files exactly in row keys and
all STOP flags.

The important seam is 2014. Beginning in 2014 the raw feed includes the
opposing block description on missed-shot rows, so STOP retains its intended
identity:

```text
STOP = offensive turnover OR defense-recovered blocked FGA
```

In 1997-2013, those block descriptions are absent. The files remain useful as
an auditable turnover-stop precursor, but the forward-alpha screen deliberately
uses only the four modern blocks beginning with a 2014-2018 training window.

## Transition backcasts

The source archive is the completed historical transition-classifier run under
`results/sq_transition_backfill/`: 60 files and 6,328,268 FGA from 1997-2026.
`build_historical_transition_mechanisms.py` joins each classified FGA one-to-one
to the canonical FGA lineup surface and writes:

```text
TRANSITION_FREQ_BACKCASTxx[_PS].parquet
TRANSITION_RIM_BACKCASTxx[_PS].parquet
```

Every row retains the classifier probability, calibration scope, feature
contract, and target provenance. Pre-2025 values are retrospective compatible
measurements, not as-of provider observations. The labeled-era discrimination
and calibration evidence remains documented in
[Historical Transition Backfill](./HISTORICAL_TRANSITION_BACKFILL.md).

Against the exact 2026 processed provider labels, the regular-season binary
backcast has `0.778` correlation / `0.812` F1 for transition and `0.840`
correlation / `0.852` F1 for transition-rim. Postseason samples are smaller and
weaker (`0.640` / `0.674` and `0.695` / `0.711`, respectively). This supports a
compatible historical research lane, not relabeling it as observed history.

## Playtype proxy backcast

Exact historical provider descriptors do not exist in the local 2014-2025 raw
files. `build_historical_playtype_proxy_backcast.py` therefore creates a
separate compatible series from seven mutually exclusive ordinary-PBP families:

1. transition;
2. cuts at the rim;
3. self-created threes;
4. other threes;
5. self-created twos;
6. other rim twos;
7. other twos.

Each regular season and postseason prices its category by actual points per FGA
relative to that phase's overall FGA mean, with the same 100-attempt shrinkage
used by the provider-descriptor proxy. The output is
`PLAYTYPE_PROXY_PTS_BACKCASTxx[_PS].parquet`.

On 2026 rows, where both definitions can be observed, the compatible backcast
correlates `0.719` with exact provider `PLAYTYPE_PROXY_PTS` in the regular
season and `0.605` in the postseason. That is useful historical signal, but it
is not close enough to merge the two names or claim exact historical recovery.

## Rebuild

```bash
.venv/bin/python nba_pipeline/scripts/reprocess_metric.py STOP 1997 2020 --workers 8

.venv/bin/python nba_pipeline/scripts/build_historical_transition_mechanisms.py \
  --start-year 1997 --end-year 2026 --overwrite

.venv/bin/python nba_pipeline/scripts/build_historical_playtype_proxy_backcast.py \
  --start-year 1997 --end-year 2026 --overwrite
```

The processed parquets and generated manifests remain untracked artifacts.
The checked-in registry records the semantic seam and keeps exact provider and
compatible-backcast series distinct.

Completed artifact audit:

| Series | Files | Rows | Missing targets | Canonical FGA row mismatches |
| --- | ---: | ---: | ---: | ---: |
| STOP | 60 | 7,157,906 possessions | 0 nonbinary targets | n/a |
| Transition frequency backcast | 60 | 6,319,861 FGA | 0 | 0 |
| Transition-rim backcast | 60 | 6,319,861 FGA | 0 | 0 |
| Playtype proxy points backcast | 60 | 6,319,861 FGA | 0 | 0 |

The playtype pricing receipt contains all 30 seasons, both phases, and all
seven categories (`420` rows). The STOP seam audit finds zero recovered-block
flags before 2014 and 93,723 from 2014 onward, matching the raw-description
coverage break described above.

## Alpha use

The historical files make real forward validation possible. The primary screen
fits a five-season zero-prior TD550 model and predicts the next non-overlapping
two seasons. It uses four modern cuts from 2014-2018 -> 2019-2020 through
2020-2024 -> 2025-2026 and searches offense and defense penalties separately.

These tests can recommend alphas for the explicitly labeled backcast boards.
They are supporting evidence—not automatic permission to assign the same alpha
to the exact 2026-only provider board or to promote any mechanism as a VPM prior.

The organized 2021-2026 research board is reproducible with:

```bash
.venv/bin/python nba_pipeline/scripts/run_mechanism_rapm_registry.py \
  --future-selected --workers 4 --cores-per-rapm 2
```

Its manifest contains 15/15 completed zero-prior TD550 mechanisms and no
skipped inputs. Exact current-provider and compatible historical series remain
separate mechanism IDs.
