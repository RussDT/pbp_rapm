# VPM Cloudflare reproduction plan

Prepared November 3, 2026. Plan only; no cloud resources deployed or schedule enabled.

## Goal and scope

Run the working full-feature VPM refresh, including current tracking and ShotQuality, in Cloudflare Containers and reproduce the saved November 2 board before considering daily automation. Keep the frozen models, source definitions, date cutoffs, missingness, penalties, centering and numerical checks. This is a runtime migration, not new model research or another full-history rebuild.

Use one container, R2 for durable files, and a small Worker to start/check a job. First run manually. Add scheduling only after reproduction succeeds. Keep the selected model pointer and existing readers unchanged throughout the experiment.

## Fixed comparison target

Local artifact root: `nba_pipeline/results/estimated_skills/` in RAPM.

- Output: `current_refresh_20261102_sq_board_v1/`; manifest SHA256 `3a983c8eec2cc1e588eeed3215cf987654d72090bd597cb0d0f859c9dcf29550`.
- Runner/result: `current_refresh_20261102_sq_v1/emit_current_vpm_sq_refresh.py`, `result.json`, `closeout.json`.
- Estimate date November 2, 2026; games strictly earlier than that date, through November 1. Explicit season ending 2027.
- 92 current-season games / 18,389 appended native rows; 37,937 native games total; 2,908 players, 168 columns, 20 arrays. Recent-player ranking contains 641 players under the existing positive-exposure rule.
- Frozen VPM mapper SHA256 `0d60bfb7b5fee5904ca2e429874caf5ab1ac945cca2079aaaa47b1083ecf3d48`.
- Frozen SQ inference: `current_sq_20261102_v1/`; 11,068 scored mid/three shots and 9,664 eligible native first-chance replacements. Preserve all original eligibility exclusions and actual-point targets.
- Existing SQ models are the 2026 models trained through June 22, 2025, transported without fitting. Cached predictive EPM skills remain dated as in the local run; latest cached request September 20, 2026. Do not fetch newer covariates into this reproduction.
- Resume `vpm_full_feature_history_v1/compact_history_fast_v1/checkpoints/2026-10-01` in the sibling VPM harness. Six files total about 57 MiB by file sizes; this is not the size of the complete dependency bundle.
- Observed local board-emitter time: 38.47 seconds. SQ inference: 2.43 seconds. These exclude other preparation, historical input loading and cloud downloads and are not an end-to-end cloud runtime estimate.

## 1. Package the existing run

Create one explicit input bundle covering the Oct1 geometry checkpoint, frozen VPM mapper/artifacts, two SQ models and their metadata, frozen skill parameters/source state, source code, historical SQ talent prefix/seed, cached EPM skill snapshots, and the saved November2 raw cloud input snapshot. Include files reached through actual runtime reads and existing manifest path checks; model files alone are insufficient.

Record file hashes and total extracted bytes in one bundle manifest. Resolve required symlink targets; do not upload the entire research workspace. Existing artifact hashes and historical records remain unchanged.

The runner currently embeds Mac absolute paths, dated paths and some source transformations. For the first reproduction, retain its November2 scope and provide a filesystem compatibility layout inside the container so serialized absolute-path checks still resolve. Keep the existing source files unchanged; use a thin entrypoint for setup and stage ordering. Generalize dates only after the fixed-date reproduction passes.

Pin the working versions: Python3.9 runtime, NumPy2.0.2, pandas2.3.3, SciPy1.13.1, scikit-learn1.6.1, PyArrow21.0.0, joblib1.5.3, fastavro1.12.2 and required remaining dependencies discovered from the executed path. Build native kernels for Linux from the same C++ sources; do not copy macOS binaries. Keep BLAS/OpenMP thread counts at two.

## 2. Reproduce in a Linux container locally

First use the saved complete source reference and SQ-bearing native atoms to reproduce the board emitter. Then run the same container from the saved raw92-game snapshot through core/tracking state refresh, SQ inference, native residual construction and board emission. This second run must recompute intermediates rather than copy the expected board.

Record stage timings, peak RSS and peak temporary-disk use. Include model/input download and cold-start costs separately when testing in Cloudflare. The existing runtime requires 15 GiB free disk: retain it for this test and include it when sizing the container; do not silently weaken that check.

Select a Cloudflare instance only after measuring the extracted bundle, scratch needs, image and free-space margin. The published predefined maximum is standard-4: 4vCPU, 12GiB RAM, 20GB disk; custom configurations/limit increases may be available. If required memory/disk does not fit the available account configuration, report that concrete result rather than redesigning the math.

## 3. Run one manual Cloudflare reproduction

Upload the input bundle to an isolated R2 experiment prefix. Use source-read credentials and a separately scoped output prefix; do not depend on the Mac's interactive Wrangler login or put credentials in the image.

Start one container manually for `2026-11-02`. Download the bundle to local container disk before numerical work; avoid repeated remote reads in the inner loop. Run the identical entrypoint, upload outputs/logs/measurements to a new run-specific R2 prefix, then stop the container. R2 is the durable store because container disks are ephemeral. Do not enable a cron job yet.

## 4. Compare and decide

Produce one comparison report containing:

- Exact game/player IDs, date/season, row/column/array shapes, ordering and missing-value masks; identical frozen model and input identities.
- SQ event probabilities, native residuals and eligible identity counts; all37 unchanged actual/non-SQ atom fields and actual game-point totals preserved.
- Every player rating and lane/prior value, all20 arrays (including recomputed source bases), centering and intercepts. Compare underlying values, not compressed file bytes or rounded rankings.
- Try bit-for-bit parity first. For platform floating-point differences, report maxima by field/array. Predeclare at most1e-9 points/100 for displayed ratings/lanes and1e-11 for native per-possession impact arrays; retain existing stricter basis/prior checks and all original normal-equation/Net closure guards. Investigate any failure without widening tolerances.
- Existing numerical guards: normal-equation error at most1e-10; Net/centering closure at most1e-9. The local target observed4.24e-15 and1.43e-15 respectively.
- Runtime, memory, disk, transferred bytes and estimated recurring cost from measured usage. Report whether it fits and whether it is practical for daily refreshes; do not promise a speedup from moving hosts.

If it fails, fix only the identified packaging/runtime difference and repeat this comparison. No extra scientific fits, expanded model features, new verification framework or dependency-version cascade.

## 5. Only after reproduction passes: test a fresh daily cut

Parameterize estimate date, explicit NBA season and manifest-derived completed-game count; remove the November2/92-game assumptions without changing the estimator. Keep an explicit America/Los_Angeles estimate-date contract and strict prior-day observation cutoff. Fetch current cloud inputs, preserve report missingness, apply frozen parameters/SQ models, and compare cloud versus local on the same frozen fresh snapshot.

For this first migration, preserve the existing Oct1-checkpoint-plus-current-season method; do not introduce an unproven incremental source-state algorithm. Record its growing season-to-date cost. Any later optimization is separate work.

## 6. Only after the fresh-cut comparison passes: automate

Enable one scheduled Worker invocation after upstream data readiness is established. Use a durable date/run lock to prevent overlapping or duplicate publication. Save immutable dated outputs and a small job-status record in R2. Update the cloud latest-successful reference only after the complete upload and checks succeed; failures retain the previous successful result. Test restart/retry once and notify on failures. Decide reader/API cutover separately from the reproduction experiment.

## Deliverables

One container/entrypoint, one input bundle manifest, one cloud reproduction output and comparison report, and measured runtime/resource/cost figures. These answer whether the current model is reproducible on Cloudflare. Deployment and automation are later execution steps of this plan, not actions already performed.

## Platform references checked November 3, 2026

- [Cloudflare Containers limits and instance types](https://developers.cloudflare.com/containers/platform/limits/)
- [Container disk persistence and lifecycle](https://developers.cloudflare.com/containers/faq/)
- [Container pricing](https://developers.cloudflare.com/containers/platform/pricing/)
