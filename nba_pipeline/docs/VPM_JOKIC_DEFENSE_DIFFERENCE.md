# Why Jokić's defensive VPM differs between the two readers

The historical October 1, 2026 board gives Jokić **+2.935 defensive VPM**, compared with **+2.595** in the selected modern September 1 board. Both use defense mapper penalty100 and Parent4000/8000/TD550. The historical mapper gives him more credit from contested defensive rebounds, steals and playing-time context, and it lacks several modern tracking fields whose combined contribution is negative for him. These are different fitted models and input definitions, rather than a missing defensive mapper.

## The prior accounts for most of the starting difference

Raw saved defensive priors are3.338 versus2.056. That1.281 difference includes different reference baselines. Centering each prior on its own complete native population with defensive TD exposure gives2.638 versus2.097: **a0.540 relative-prior gap**.

The following table includes each numeric input and its missingness flag. It is exact linear contribution accounting within each saved model, not the causal effect of removing or changing one feature.

| Centered defensive prior contribution | Selected modern | Historical | Difference |
|---|---:|---:|---:|
| Contested defensive rebounds |1.497|1.692|+0.195|
| Steals |0.400|0.491|+0.091|
| Minutes per game and missingness context |0.285|0.382|+0.097|
| Blocks |0.075|0.072|−0.004|
| Modern-only tracking fields |−0.160|0.000|+0.160|
| Other foul/fallback terms |0.001|0.002|+0.001|
| **Total** |**2.097**|**2.638**|**+0.540**|

The minutes/context increase does **not** mean a larger minutes forecast. The actual inputs are nearly identical:34.844 historical versus34.858 selected. The numeric minutes contribution changes by−0.019; the missingness/reference contribution changes by+0.116, giving the net+0.097. Both of his minutes inputs are observed. The missing-flag term arises from population centering; it does not mean his own minutes are missing or that a different availability forecast was used.

The historical model's effective contested-rebound slope is0.598, compared with0.488 in the selected model, about22% steeper. Its source prediction is4.820 versus4.734, but the historical mapper's fitted physical support clips its input to4.524. The higher rebound contribution therefore persists despite a lower effective input; the learned slope and reference population matter more than the small source-forecast increase. These slopes are conditional on each model's other inputs and transformations.

The modern-only tracking block contains rim points saved, rim contests, defended attempts, deflections, offensive fouls drawn and contested shots. Its net−0.160 contribution is the sum of positive and negative terms, not six uniformly negative signals. The historical compact bank omits this block and refits the remaining slopes. Its absence is not a causal ablation proving that these fields lower his final rating by0.160.

## Where the final0.341 difference appears

The final saved standard-Net lanes explain the difference exactly:

| Defensive lane | Historical minus selected |
|---|---:|
| Three-point conversion |+0.163|
| Rim conversion |+0.148|
| Midrange/reference conversion |+0.117|
| Rim frequency |+0.047|
| Free throws |−0.113|
| Turnovers |−0.025|
| Second-chance prevention |+0.009|
| Three-point frequency |−0.006|
| **Total** |**+0.341**|

Second-chance prevention is already high in both models:2.596 versus2.587. It explains much of the positive level of his defense, but almost none of the increase between readers. The increase appears primarily in team shot-value lanes, not an increase in saved second-chance value. These lanes describe lineup-adjusted team outcomes, not Jokić's personal defended shooting percentages.

The final rating difference is smaller than the independently centered prior difference. The gap between each final rating and its centered prior is0.298 historically versus0.497 in the selected model. This is accounting for the remaining lineup/SQ estimation stages, including coupled player adjustments; it is not an independently identified causal lineup effect.

## A controlled check of the newer mapper vintage

The historical2026 mapper can use the forward window ending October1,2026, with its last training game June13,2026. The selected modern mapper's last training game is June22,2025. A later report date therefore unlocks another forward-training cohort, not just another month's player observations.

To test that explanation, the diagnostic keeps the **same current historical source packet, all-player lineup geometry, prices, SQ adjustments and centering weights**, then changes only which saved historical2025/2026 mapper supplies the O/D priors. No filter or mapper is trained, no scientific output is rewritten, and no alternative is selected for deployment.

| Saved historical mapper used for priors | Jokić defensive VPM |
|---|---:|
|2025 offense and2025 defense |2.908890|
|2026 offense and2025 defense |2.904556|
|2025 offense and2026 defense |2.939793|
|2026 offense and2026 defense |2.935459|

The annual mapper update contributes only**+0.026570** under this fixed historical setup: defense-prior changes add0.030903, partly offset by−0.004333 through the coupled offense update. Thus most of the0.341 cross-reader difference already exists with the previous historical mapper. This isolates vintage within the historical family; it does not isolate every source, preprocessing, feature, cohort or universe difference from the modern model.

## Evidence and scope

- [Small source-backed accounting](evidence/vpm_jokic_defense_difference.json).
- Ignored exact lane/centering/fixed-geometry diagnostic: `nba_pipeline/results/estimated_skills/jokic_defense_difference_v1/receipt.json`, SHA `5230728f456f223031119845a6a1bb0a801b660ac427b58e755c4286e8afbd9c`. The same-current historical solve reproduces all saved O/D/Total values with maximum gap0; four fixed-geometry arms satisfy relative normal errors below4e−15.
- Independent feature/predictor attribution: `nba_pipeline/results/estimated_skills/jokic_defense_prior_attribution_v1/receipt.json`, SHA `0c85bf1ef07e22ee3ba623aacbc7aeea55b83d3d38a7319c29dfa79305b2ef18`. Both raw priors reproduce exactly within numerical precision; feature and flag contributions sum to the saved values.

Background weight is0 for Jokić in both models. This diagnostic establishes why their estimates differ, not which defensive estimate is more accurate. Historical source/publication and finite-optimizer qualifications from the historical build remain. Selected VPM, all completed products and2027 outcomes remain unchanged.
