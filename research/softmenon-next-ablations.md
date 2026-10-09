# Next SoftMenon ablations

Review date: 2026-10-09. Baseline: current refined SoftMenon at
`04836b961d808c72cf3e703cd9617831721778a6`.

The strongest next opportunities are **more reliable direction scores and
selective chroma refinement**. Start with changes that reuse existing samples
and preserve the current pipeline. These are research-backed hypotheses, not
new benchmark results or a claim that any proposal will improve quality or
speed. This review changes documentation only.

## What the existing results tell us

The [complete quality results](../benchmark/results/softmenon-quality-summary.json)
and [exploratory screen](../benchmark/results/softmenon-exploratory-screen.json)
give us more useful targets than another general algorithm survey:

| Observation | Implication for the next experiments |
|---|---|
| Current SoftMenon averages 37.4928 dB versus paper Menon's 36.9285 and initial SoftMenon's 36.6116 on 442 images. | Keep current SoftMenon as the parent of every new ablation; keep full paper Menon as an unchanged external control. |
| McMaster drops from 34.8191 to 34.4812 dB versus initial SoftMenon, although it remains above paper Menon. Fifteen of its 18 images regress. | The full median-derived green update is not universally beneficial. Test locally adaptive correction and support, rather than weakening it everywhere. |
| Urban100 image 011 is 7.6304 dB behind paper Menon, but improves over initial SoftMenon. Its paper-relative deficit remains 2.1966 dB after Inset16. | This is partly an inherited reconstruction problem. Diagnose direction scoring and interior errors as well as borders; do not attribute the whole gap to the new cleanup. |
| Urban100 image 055 loses about 1.543 dB versus initial SoftMenon, including an interior regression. | Keep a case where the new refinement actually hurts, alongside inherited failures. |
| Half/quarter green correction, an 8-DN correction cap, and a 16-DN correction-magnitude gate all lose to full correction in the 442-image mean. | Those are already-tested controls, not new research ideas. A local confidence rule would test a different mechanism. |

Earlier classifier alternatives were evaluated on **31 images with the initial
cleanup**, not with today's green refinement:

| Earlier variant | Mean PSNR change versus its initial-SoftMenon parent |
|---|---:|
| Co-located color-difference scores, linear weights | +0.2170 dB |
| Co-located scores, squared weights | +0.3209 dB |
| Wider posterior scores, linear weights | +0.2466 dB |
| Wider posterior scores, squared weights | +0.4771 dB |
| Native scores, squared weights | −0.0263 dB |

These small-screen results make classifier revisits attractive, but they are
**not** full-corpus gains and cannot be added to the gain from green refinement.
Test a new score with the current linear blend first. Then test a 2×2 comparison
of native/new score and linear/squared weighting if warranted. Changing both
at once would obscure which mechanism helped.

## Ten experiments to prioritize

This is a suggested experiment order, balancing evidence, cost, and coverage.
The cost column is an implementation assessment, not a timing measurement.
Each row is a separate child of the current baseline; do not stack the rows.

| Order / ID | Single changed mechanism | Research basis | Expected extra work |
|---|---|---|---|
| 1. `late_final_clip` | Keep signed refined green until the other missing color has been reconstructed. | Unclipped intermediate reconstruction in the Menon reference [S1]; our numerical ablation. | Almost none: same samples and medians. |
| 2. `colocated_score` | Compare color differences formed at matching spatial locations. | Posterior color-difference reasoning [S1]; positive prior screen on the old parent. | Wider RAW support and neighboring candidate calculations; vectorization needed. |
| 3. `center_weight3` | Give the center chroma value weight three in each median. | Center-weighted medians and weighted-median demosaicing [S2, S3]. | Same loads, more order-statistic work. |
| 4. `correlation_gate` | Reduce only the median-derived green correction when local chroma variation is large relative to green variation. | Adaptive inter-channel correlation [S4] and edge-strength filtering [S5]. | Local differences, sums, and a division; existing 3×3 samples suffice. |
| 5. `guide_gate16` | Replace cross-edge neighbors with center chroma before the median. | Edge-adaptive artifact suppression [S6]. | Eight guide comparisons and masked selections; same median network. |
| 6. `posterior_score` | Aggregate directional color-difference evidence over the wider posterior stencil. | Menon DDFAPD [S1]; positive prior screen on the old parent. | More candidate/gradient calculations or tiled scratch; greater speed risk than row 2. |
| 7. `ppg_score` | Replace only direction scores with PPG's wider gradient statistic. | PPG source [S14]. | Radius-three cross, extra differences; no new pass. |
| 8. `ratio_green` | Replace the initial green predictor with a low-pass ratio predictor. | RCD author derivation [S8]. | Wider RAW reads or a low-pass buffer and several divisions. |
| 9. `refresh_green_site_chroma` | Recompute R/B only at measured-green sites using neighboring refined green. | Sequential Bayer-aware correction [S9]. | Small arithmetic, but a new dependency stage/pass unless tiled. |
| 10. `one_step_ri_rb` | Replace initial R/B interpolation with one bounded guided prediction plus residual interpolation. | RI/MLRI analysis and reference code [S10]. | Local statistics and residual storage; a higher-cost quality reference first. |

### Exact first probes

Use the current rounding convention: nearest integer, half ties toward positive
infinity. `clip` means `[0,255]`. Unless explicitly changed, preserve reflect101,
all measured CFA samples, initial reconstruction arithmetic, and immutable
cleanup reads. Use wide intermediates where needed. `I` is the complete initial
image, `D_R=I.R-I.G`, `D_B=I.B-I.G`, and `mR,mB` are their 3×3 medians.
The parameters below are proposed starting points, not fitted values or a
claim to reproduce the cited full algorithms. Freeze the executable definitions,
source hashes, and any deviations before scoring.

**1. Delayed final clipping.** At a measured color `C`, calculate
`g*=C-mC`. Store `G=clip(g*)`, but reconstruct the opposite color as
`clip(g*+mOther)` instead of `clip(clip(g*)+mOther)`. Green sites stay unchanged
from the baseline. Only cases with out-of-range `g*` can differ: report their
count and contribution to error. This might reduce a clipping-induced hue bias
or make saturated edges worse. It does not defer clipping throughout the entire
pipeline or require a wider scratch image.

**2. Co-located scores.** Let `Gh(q)` be the existing horizontal candidate
evaluated at chromatic site `q`, and `dh(q)=C(q)-Gh(q)`. Replace the horizontal
score by `abs(dh(p)-dh(p-2x))+abs(dh(p)-dh(p+2x))`, with the analogous vertical
score. Keep the current linear inverse-score blend. Unlike the native score,
every difference subtracts green and color at the same location. This is the
recorded `colocated_linear_weights` design. Its initial-green RAW radius grows
from two to four; downstream interpolation/cleanup expands total support
further. Reuse candidate calculations in tiles if scalar screening justifies
an optimized implementation.

**3. Center weight three.** Independently for each chroma plane, take the
median of eight neighbors and **three copies of the center**: eleven values.
Keep the complete current green/R/B reconstruction. Equivalently, clamp the
center chroma between the fourth and sixth order statistics of the original
nine samples. The current median-only comparator network does not necessarily
produce those ranks, so this needs its own exact network. It may preserve
colored detail, but it can also retain a bad center estimate.

**4. Local correlation gate.** A concrete inexpensive proxy to try uses the
four cardinal neighbors `N` of the unchanged image:

```text
eG = 2 * sum(q in N, abs(I.G(q) - I.G(p)))
eC = sum(q in N, abs(D_R(q) - D_R(p)) + abs(D_B(q) - D_B(p)))
w = (eG + 1) / (eG + eC + 1)
Gnew = clip(round(I.G(p) + w * (Cmeasured - mC - I.G(p))))
```

Apply this only at measured R/B sites, then reconstruct the opposite color
from `Gnew` and its unchanged median as usual. Measured-green sites use the
baseline. This is our local proxy, **not AICC's published correlation estimator
or ESF's filter equation**. Chroma artifacts themselves can inflate `eC` and
incorrectly block a useful correction. That is a reason to test this gate,
not to assume its confidence interpretation is correct.

**5. Guide-based support masking.** For each noncenter neighbor `q` in the
3×3 window, replace both `D_R(q)` and `D_B(q)` by their respective center values
when `abs(I.G(q)-I.G(p))>16`. Run the existing median9 on those values and keep
the current reconstruction. The same mask serves both chroma planes. This
16-DN threshold applies to **neighbor green differences**, unlike the already
tested threshold on green-correction magnitude. It can fail on an incorrect
green guide or a chromatic boundary with little green contrast.

**6. Posterior scores.** Reuse the previous eight-tap classifier, whose source
hash is recorded under `independent_family.frozen.builds` in the
[screen artifact](../benchmark/results/softmenon-exploratory-screen.json). Its stencil is:
apply weights `[1,1,1,3,3,1,1,1]` at offsets
`[(0,2),(-2,2),(-1,1),(0,0),(-2,0),(-1,-1),(0,-2),(-2,-2)]`
to `abs(dh(q)-dh(q+2x))`. Transpose offsets and use vertical candidates for the
vertical score. Keep current candidate rounding and linear weights. At the
opposite chromatic phase, `C(q)` means the color measured there. This is the
previous adapted classifier with reflected RAW support; importing it does not
turn SoftMenon into full paper Menon. Time tiled reuse as well as arithmetic:
naively recomputing every neighboring candidate is an unfavorable deployment.

**7. PPG direction scores.** Keep native green candidates and linear blending.
For horizontal offsets use the statistic from the PPG source:

```text
Sh = 3*(abs(C(-2)-C0) + abs(C(+2)-C0) + abs(G(-1)-G(+1)))
   + 2*(abs(G(+3)-G(+1)) + abs(G(-3)-G(-1)))
```

Transpose for vertical. Unlike the previously tested Hamilton–Adams score,
the two chromatic deviations cannot cancel, and outer green samples contribute
directional evidence. This uses a radius-three RAW cross and no extra pass.
Attaching it to our predictor and soft blend is an adaptation. Larger scores
reduce the relative effect of the existing `+1` stabilizer, and outer samples
may obscure thin detail.

**8. Ratio-based green candidates.** Compute the normalized Bayer low-pass
`L` with the 3×3 separable `[1,2,1]` kernel. For each cardinal direction `d`,
form `E_d=G(p+d)*2*L(p)/(L(p)+L(p+2d))`, falling back to `G(p+d)` if the
denominator is zero. Average east/west for `Gh` and north/south for `Gv`,
rounding each completed candidate once. Retain the native scores computed from
the **original Hamilton–Adams candidates**, and the current blend/RB/cleanup
stages. This isolates prediction from classification; retaining the old
candidate calculations is part of its cost. Do not import RCD's direction
estimator too. The source derivation motivates
this predictor; its behavior near low signal and its division/memory costs
need explicit checks. Keep low-pass and ratio arithmetic unquantized until
the specified candidate rounding in the reference prototype.

**9. Refresh chroma at measured-green sites.** After current cleanup, use
the two measured target-color neighbors `q1,q2` on that color's Bayer axis:
`Cnew=clip(Gmeasured+round(((C(q1)-Gnew(q1))+(C(q2)-Gnew(q2)))/2))`.
Apply separately to R and B only at green sites. Read an immutable refined
green plane; do not let scan order affect outputs. This isolates stale green
guidance in the last reconstruction step, but sacrifices the robust 3×3 median
at these sites and introduces a real dependency barrier.

**10. One-step guided residual R/B reconstruction.** Keep initial green
fixed and immutable. At each pixel, for each target color, fit `C≈aG+b` over
only its measured sites in the surrounding 5×5 window:
`a=cov(G,C)/(var(G)+1)`, `b=mean(C)-a*mean(G)`, using population
statistics and an epsilon of one DN squared. Predict a tentative target-color
plane. Keep fits, predictions, and residuals unrounded and unclipped. At
measured sites compute `residual=observed-tentative` **before**
restoring observed values. Bilinearly interpolate these sparse residuals
(two axial samples at green sites, four diagonal samples at opposite-color
sites). At missing sites write `clip(round(prediction+interpolated_residual))`
and restore measured color values exactly. Then run today's cleanup unchanged.
Residuals computed after measured
samples are restored would all be zero and would not test RI. This bounded
stage replacement is our adaptation, not complete RI or ARI.

## Further ideas worth retaining

- **Logistic green weights.** LED [S7] suggests replacing the current rational
  blend by `wH=1/(1+exp(0.05*(Sh-Sv)))`, holding native candidates and scores
  fixed. Start with a frozen Q12 lookup over integer score differences −256
  through +256, clamping the index at the ends (1,026 bytes). LED's coefficient
  operates on its own score units; its transfer to ours is exploratory. This
  differs from already-tested squared inverse weights and does not reproduce
  full LED. A lookup might save arithmetic or cost more in SIMD gathers.
- **Coupled chroma filtering.** Replace separate scalar medians by the L1
  medoid of nine `(R-G,B-G)` pairs, with center-first then raster tie-breaking
  [S11]. It uses the same samples but 36 distinct pair distances and score
  reductions. A smaller alternative is to test a direct median of `R-B` for
  opposite-color reconstruction; that would add a third median and is a
  different experiment. Neither has demonstrated a gain here.
- **Sparse wider support.** Freeman describes a nine-point chroma support
  spanning 7×5 [S12]. Test support alone if local masks help but leave broader
  fringes. Same sample count does not mean same row traffic, and the asymmetric
  footprint deserves rotation tests. Plain cross5/diagonal5 filters were
  already screened on the initial version.
- **Tiny learned correction or gate.** BLADE explicitly postprocesses a Menon
  reconstruction with selected learned filters [S13]. First try an offline-fit
  small gain table on existing confidence features, or a small phase-specific
  residual filter bank. Those are separate adaptations with explicit cache
  and compute budgets, not the complete BLADE model. Preserve measured samples
  and constant-color responses. Training and untouched evaluation require
  scene-disjoint manifests; generic BLADE filtering timings do not establish
  full SoftMenon-plus-correction throughput.
- **Near-Nyquist detection and multiscale direction consensus.**
  [AMaZE](https://github.com/RawTherapee/RawTherapee/blob/dev/rtengine/amaze_demosaic_RT.cc)
  and [fully directional estimation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5053961/)
  suggest treating repeated textures separately. Read and freeze a specific
  detector before implementation; do not hide a detector plus a new predictor
  plus a new cleanup inside one ablation.
- **Noise-aware and temporal extensions.** Keep these as distinct studies:
  noise-aware confidence inspired by [green-guided Bayer denoising](https://pmc.ncbi.nlm.nih.gov/articles/PMC3967728/),
  and confidence-gated history inspired by [spatial/temporal CFA filtering](https://pmc.ncbi.nlm.nih.gov/articles/PMC5492375/).
  A true denoiser may intentionally alter measured samples; a temporal method
  adds state, motion handling, and latency constraints. Neither is a direct
  substitution under the current stateless, sample-preserving contract.

## How to evaluate the next round

1. **Freeze and isolate.** Create one variant per row against the pinned current
   parent. Validate the generated baseline byte for byte. Preserve all scored
   variants and failures. Combine winners only after isolated comparisons;
   include parent, A, B, and A+B to measure interactions. Keep the paper baseline
   unchanged.
2. **Score the complete existing corpus.** Reuse the 442-image decoder and
   remosaicing rules from [benchmarks.md](../benchmarks.md). Report original and
   Inset16 separately, both phases paired by source, full/interior/border RGB
   errors, per-channel errors, mean-image and pooled PSNR, dataset means,
   wins/losses, lower-tail quantiles, and worst-scene crops. Inset16 is a second
   boundary condition on the same images, not extra independent data.
3. **Make the failure slices visible.** Include all McMaster images, Urban011
   and Urban055, fine colored edges, and saturated regions. Record how many
   pixels each gate changes; inspect green error separately from chroma error.
   Never choose a winner solely from a crop or a single dataset mean.
4. **Use fresh scenes for confirmation.** The current 442 are already a
   development set. The acquired Waterloo, DIV2K training, and BSDS train/val
   pools offer additional candidates; see [dataset reconstruction](../benchmarks.md).
   Audit prior scoring/training use and exact/near duplicates before reserving
   a deterministic scene-disjoint confirmation manifest. Acquisition alone
   does not prove a set is untouched. Freeze choices before opening its scores.
5. **Measure actual deployment cost.** After a correctness prototype, time
   optimized CPU1, CPU8, forced AVX2, and CUDA with the existing interleaved
   control runner. Use the matched current baseline in the same run, warm
   1080p host-to-host latency, per-phase samples, and device-only CUDA time to
   expose changes hidden by transfers. Report extra scratch and launches.
   For the existing no-slowdown goal, require a quality gain and no repeatable
   latency regression beyond control drift on the deployment backend. Keep
   slower winners as separate quality references. Jetson/ARM still needs real
   target-device measurements.
6. **Check transfer to other domains separately.** REDS/DAVIS can expose flicker
   under synthetic mosaicing. MSR linear references are uint16 and need a
   declared precision conversion or a separate wider-precision implementation;
   do not silently score them as native uint8 RAW. Real RAW photographs without
   independent RGB truth support artifact inspection, not equivalent PSNR.

## Sources and evidence level

- **S1:** [Menon reference implementation](https://colour-demosaicing.readthedocs.io/en/latest/generated/colour_demosaicing.demosaicing_CFA_Bayer_Menon2007.html), [paper](https://doi.org/10.1109/TIP.2006.884928), and our archived screen definitions. Reference equations/implementation; earlier local results explicitly use a different parent.
- **S2:** Ko and Lee, [Center Weighted Median Filters and Their Applications to Image Enhancement](https://pure.korea.ac.kr/en/publications/center-weighted-median-filters-and-their-applications-to-image-en/). Published operator; our center weight and attachment point are separate choices.
- **S3:** Li and Randhawa, [Color Filter Array Demosaicking Using High-Order Interpolation Techniques With a Weighted Median Filter for Sharp Color Edge Preservation](https://pubmed.ncbi.nlm.nih.gov/19556197/). Abstract describes an edge-classified weighted median of directional interpolants, not our spatial chroma median.
- **S4:** Duran and Buades, [A Demosaicking Algorithm with Adaptive Inter-Channel Correlation](https://www.ipol.im/pub/art/2015/145/), [full text](https://www.ipol.im/pub/art/2015/145/article.pdf). Full text and source available. Its correlation adaptation and nonlocal stage are broader than our proposed gate.
- **S5:** Pekkucuksen and Altunbasak, [Edge Strength Filter Based Color Filter Array Interpolation](https://pubmed.ncbi.nlm.nih.gov/21606035/). Abstract checked: edge strength adjusts application of the color-difference rule; exact ESF equations were not reproduced here.
- **S6:** Chang and Tan, [Hybrid color filter array demosaicking for effective artifact suppression](https://doi.org/10.1117/1.2183325), [paper](http://elynxsdk.free.fr/ext-docs/Demosaicing/more/JEI013003.pdf). Edge-adaptive median precedent; the green-only mask in this review is our simplification.
- **S7:** Niu et al., [Low Cost Edge Sensing for High Quality Demosaicking](https://arxiv.org/abs/1806.00771), [author code](https://github.com/shmilyo/Low-Cost-Edge-Sensing-for-High-Quality-Demosaicking). Logistic directional decisions; no paper speedup is transferred to our hardware.
- **S8:** Luis Sanz Rodríguez, [RCD author derivation and code](https://github.com/LuisSR/RCD-Demosaicing), [author discussion of interchangeable algorithm stages](https://discuss.pixls.us/t/about-rcd-and-other-demosaicing-methods/22126). Inspectable ratio formulas; discussion is design guidance, not a controlled comparison.
- **S9:** Lukac, Martin, and Plataniotis, [Demosaicked image postprocessing using local color ratios](https://www.comm.utoronto.ca/~kostas/Publications2008/pub/53.pdf). Sequential refinement precedent; our two-neighbor color-difference refresh is a different formula.
- **S10:** Jin et al., [A Mathematical Analysis and Implementation of Residual Interpolation Demosaicking Algorithms](https://www.ipol.im/pub/art/2021/358/). Full analysis and reference source for RI/MLRI/ARI; our fixed 5×5 single-stage probe omits their complete constructions.
- **S11:** Astola, Haavisto, and Neuvo, [Vector Median Filters](https://researchportal.tuni.fi/en/publications/vector-median-filters/). Joint vector selection precedent; proposed application is to reconstructed chroma pairs.
- **S12:** Freeman, [US4774565A](https://patents.google.com/patent/US4774565A/en). Explicit sparse chroma-filter support; support-only testing omits the complete described pipeline.
- **S13:** Getreuer et al., [BLADE author page](https://getreuer.info/papers/getreuer2018blade/index.html), [paper](https://arxiv.org/abs/1711.10700). A learned-filter postprocessing precedent including a Menon-based demosaicing example; our small gain table is a further approximation.
- **S14:** Dave Coffin, [dcraw PPG source](https://github.com/ncruces/dcraw/blob/master/dcraw.c), `ppg_interpolate`. Inspectable direction statistic; only that component is proposed here.
