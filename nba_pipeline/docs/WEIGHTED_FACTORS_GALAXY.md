# Weighted Factors Galaxy

Build a reusable 3D player-similarity galaxy from any `weighted_factors*.csv` file.

The generator lives at:
- [build_weighted_factors_galaxy.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/build_weighted_factors_galaxy.py)

## Design

The galaxy uses two or three different spaces for different jobs:

1. **Similarity space**
   - nearest-neighbor edges come from the selected factor space
   - default: scaled raw factors plus `0.5 * standardized net_rapm`, not 3D screen distance

2. **Cluster space**
   - default: scaled raw factors
   - optional: PCA -> UMAP 5D latent space
   - default clustering is `kmeans` with `10` clusters
   - optional methods: `agglomerative`, `hdbscan`

3. **Visible layout**
   - separate UMAP 3D projection for rendering

This keeps comps tied to the full factor set while still giving the app a 3D scene.
The default is the more interpretable version: raw-factor neighbors and raw-factor clusters, with UMAP 3D used only for visible `x/y/z`. A half-weighted `net_rapm` term is added by default so extreme quality gaps are penalized without letting `net_rapm` dominate the whole geometry. Clusters default to `kmeans` with `10` buckets so the app always gets archetype groups.

## Presets

### 6-factor preset

Uses the canonical six-factor core:

```text
oTS, oTOV, oREB, dTS, dTOV, dREB
```

### 8-factor preset

Splits TS into non-FT scoring plus FT pressure:

```text
oEFG, oFT, oTOV, oREB, dEFG, dFT, dTOV, dREB
```

Derived columns:

```text
oEFG = oTS - oFT
dEFG = dTS - dFT
```

`oFT` / `dFT` are passthrough informational columns in `weighted_factors`; they are not part of the six-factor regression fit itself. The 8-factor galaxy is intentionally a visualization/comps split, not a claim that the public weighted-factors regression changed.

## Examples

### Standard lifetime 6-factor galaxy

```bash
cd /Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set six_factor \
  --min-poss 2000
```

### Standard lifetime 8-factor galaxy

```bash
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set eight_factor \
  --min-poss 2000
```

### Time-decay galaxy from another weighted file

```bash
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_21_26_all_rb_td700.csv \
  --feature-set eight_factor \
  --neighbor-space raw \
  --top-k 15
```

### Disable the net penalty

```bash
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set eight_factor \
  --net-weight 0
```

### Force 10 agglomerative clusters instead

```bash
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set eight_factor \
  --cluster-method agglomerative \
  --n-clusters 10
```

### Apply the current eight-factor display names

```bash
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set eight_factor \
  --cluster-name-profile eight_factor_kmeans_10_v1
```

### Optional latent-space clustering experiment

```bash
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set eight_factor \
  --cluster-method hdbscan \
  --cluster-space latent
```

### Custom factor list

```bash
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set custom \
  --features oTS,oTOV,oREB,dTS,dTOV,dREB,oFT,dFT
```

## Outputs

Default outputs land in `nba_pipeline/results/galaxy/`:

- `{input_stem}_{preset}_galaxy.json`
- `{input_stem}_{preset}_embeddings.csv`

The JSON includes:

- `nodes`
- `edges`
- `clusters`
- `metadata`

The CSV includes:

- source weighted-factors columns
- selected factor columns
- `latent_1..latent_5`
- `x`, `y`, `z`
- `cluster`
- `cluster_name`

## Notes

- The script expects a weighted-factors style file with `player_name`, and usually `player_id`, `Latest_Year`, and `possessions`.
- If `oEFG` / `dEFG` are requested and not present, they are derived automatically from `oTS/oFT` and `dTS/dFT`.
- `net_rapm` is included by default as an extra weighted similarity dimension with weight `0.5`. Set `--net-weight 0` to remove it or choose a different weight for experiments.
- By default, both comps and clusters come from the scaled raw factor space, and clusters are forced with `kmeans --n-clusters 10`.
- `--cluster-name-profile eight_factor_kmeans_10_v1` applies the current hand-tuned display names for the `eight_factor + kmeans + 10` build.
- Use `--cluster-space latent` only when you explicitly want the draft-style latent clustering behavior.
- `hdbscan` still uses `-1` for noise points; fixed-cluster methods do not.
- The script is generic over the input weighted file path, so the same pipeline works for standard, rubberband, age-curve, season-effects, or time-decay weighted-factors exports as long as the requested feature columns exist.
