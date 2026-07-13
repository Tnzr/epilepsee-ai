# W&B Epoch Visualization Review (Updated Panels)

Date: 2026-07-13
Scope: Updated W&B-exported inference panels in [epoch_visualizations](./epoch_visualizations)

Reviewed panels:
- [Ep5 panel](./epoch_visualizations/Epilepsee-AI_Ep5_TrainingVisualization.png)
- [Ep9 panel](./epoch_visualizations/Epilepsee-AI_Ep9_TrainingVisualization.png)
- [Ep23 panel](./epoch_visualizations/Epilepsee-AI_Ep23_TrainingVisualization.png)

## 1) High-level interpretation

Training behavior is not uniformly collapsed. The model shows mixed modes:
- Some windows have informative temporal response (Ep9 is the clearest example).
- Some windows show delayed or post-event response behavior (seen in Ep5-like patterning).
- Some windows show flatter probability heads and lower token dynamism (observed in part of Ep23 and other epochs).

This is consistent with partial context learning plus case-specific under-response, not a total training failure.

## 2) Epoch-by-epoch qualitative findings

### Ep5
- Alarm trajectory decreases through parts of pre-ictal region and rises more strongly after onset.
- This can indicate post-event confirmation bias (good detection of transition), but weaker anticipation lead.
- Heatmap channels are active but not strongly discriminative before onset.

### Ep9
- Most coherent anticipation structure among the selected panels.
- Probability heads move in a more interpretable way around pre-ictal/ictal landmarks.
- Token summary and derived waterfall show non-trivial structure rather than pure flat occupancy.

### Ep23
- Mixed quality: segments with meaningful variation, but also broad stretches of flatter head dynamics.
- Token channels are active, though less sharply contrasted than the best mid-run windows.
- Supports the view that behavior is case-dependent rather than globally collapsed.

## 3) Why only ~213 batches/epoch looked "small"

The 213 count is expected under distributed training with 2 GPUs:
- Train set size from run artifacts is about 6840 windows.
- Batch size is 16.
- Global steps if single-process would be about 6840 / 16 = 427.5.
- With 2 DDP ranks, each rank sees about half, giving about 213 steps per rank (matching logs).

So 213 is per-rank step count, not total-data under-utilization by itself.

## 4) Flat-head / flat-token concern: likely causes (ranked)

1. Window-level class prior + calibration pressure
- Heads staying roughly in a middle band (about 0.35 to 0.7) can happen when objective pressures favor conservative probabilities over sharp separation.

2. Case heterogeneity and sparse informative segments
- Some recordings appear strongly informative, others less so; this produces alternating "interesting" and "dull" panels.

3. Augmentation strategy mismatch
- Current config uses heavy online pre-ictal augmentation and high augmentation factor. If augmented samples are repetitive, this can encourage smoother, less contrastive latent structure.

4. Signal quality episodes
- The run repeatedly flagged modality issues (for example flat/missing channels in some epochs), which can flatten both token transitions and head excursions for affected windows.

## 5) Evidence from latest successful short full-scale run

Reference run artifacts in [models/real_sanity_stability_20260713_fullscale_4h](../real_sanity_stability_20260713_fullscale_4h):
- Best validation MAE around 2.4946 min.
- AUROC around 0.606 on test.
- Clinical lead-time profile remained asymmetric (strong at very short horizon, weak by 5-10 min), consistent with "late/near-event" confidence behavior.

This lines up with your observation: visual quality is good for presentation, but anticipation sharpness is inconsistent across cases.

## 6) Recommended next ablations (targeted)

1. Augmentation calibration ablation
- Reduce augmentation_factor (for example 15 -> 4-6).
- Lower online_preictal_augmentation_prob (for example 0.7 -> 0.35-0.5).
- Goal: test whether flatter latent dynamics are partially augmentation-induced.

2. Context density increase for diagnostics
- Remove or raise sampling caps for analysis runs (for example allow denser windows per recording).
- Goal: distinguish true flat modeling from sparse-panel aliasing artifacts.

3. Case-stratified evaluation pass
- Split visual/metric review by recording quality and flagged modality issues.
- Goal: quantify how much flat behavior is data-quality driven versus model-behavior driven.

4. Head sharpness regularization experiment
- Add/adjust objective term to increase separation only in validated pre-ictal intervals.
- Goal: improve anticipation contrast without increasing post-event overreaction.

## 7) Bottom line

- Training is successful operationally and scientifically useful.
- Ep9-style behavior shows the architecture can express good anticipation trajectories.
- Main remaining gap is consistency across recordings, especially reducing flat or delayed-response cases.
- Your selected visuals are strong for reporting, and they also clearly motivate the next optimization cycle.
