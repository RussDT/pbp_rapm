# Legacy Alt3 Compatibility

Load this reference only when a task explicitly names
`weighted_factors_alt3_efg_value_*`, `weighted_factors_alt3_*`,
`ALT_EFG_BASELINE`, or the June 2026 Alt3 research snapshot.

## Status

- Treat `weighted_factors_decomp_*` as the active public family.
- Treat `weighted_factors_alt3_efg_value_*` as a legacy EFG-value display and
  audit surface.
- Treat `weighted_factors_alt3_*` as the older legacy/audit surface.
- Do not start current VPM work from root Alt3 scripts or published Alt3
  snapshots.

## Legacy EFG-Value Identities

The old display used six atomic shot-value aliases:

```text
ALT_EFG_RIM_FREQ
+ ALT_EFG_RIM_FG
+ ALT_EFG_MID_FREQ
+ ALT_EFG_MID_FG
+ ALT_EFG_THREE_FREQ
+ ALT_EFG_THREE_FG
= ALT_EFG_ATOMS_VALUE
```

Its public shot total was:

```text
ALT_EFG_VALUE = ALT_EFG_BASELINE + ALT_EFG_ATOMS_VALUE
```

`ALT_EFG_BASELINE` was a balancing/replacement-shot bucket. It was not second
chance and must not be reintroduced into active DECOMP.

Other legacy identities:

```text
ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT
ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE = ALT_TOV_VALUE
FIRST_CHANCE + SECOND_CHANCE_CLEAN = possession scoring layer
```

In the EFG-value bundle, `oSC` and `dSC` were direct
`SECOND_CHANCE_CLEAN` coefficients, not residual buckets. The still older
Alt3 path reconstructed second chance as a residual; do not mix those schemas.

## Sign and Display Rules

- Offensive components use their solver sign.
- Defensive display columns are sign-flipped so positive means good defense.
- Derived rim/mid/three totals are sums of their atomic children.
- `ASSIST_POINTS`, `RIM_ASSIST`, and similar mechanism drilldowns are not
  additive display children and must not enter the total.

## Migration Rule

When migrating a consumer:

1. Identify its exact expected schema.
2. Replace Alt3 inputs with `weighted_factors_decomp_*` only through an audited
   mapper.
3. Prove player-universe, sign, and additive identities.
4. Label any decade/support fallback that still uses Alt3 explicitly.
5. Never describe a legacy fallback as the active current decomposition.
