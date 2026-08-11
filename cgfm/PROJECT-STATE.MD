# Project state

Last updated: 2026-08-06, after the second implementation pass, which added the Stage 1 sweep and the Stage 5 analysis.

## Where this stands

Every stage of the plan is implemented, tested and smoke-tested end to end on CPU. Nothing has been trained. The next
action is the Stage 1 go/no-go sweep, which decides whether to spend the MPTS-52 budget at all.

## Research question

Does a learned partition of a crystal into coordination-scale atom groups define a better flow-matching probability path
than standard atomwise periodic interpolation? Four arms on composition-conditioned MPTS-52 crystal structure
prediction, three seeds each, identical in everything except the path for fractional coordinates.

## Done

- Two additive edits to OMatG (`omg/si/abstracts.py`, `omg/si/stochastic_interpolants.py`) that let an interpolant ask
  for the full data sample. No shipped interpolant changes behaviour.
- The path: group-mean operator, centred `Delta`, the interpolant, its velocity and its loss.
- The four arms behind one class, so they cannot differ in anything but the grouping and `eta`.
- Both fixed partitions, their file format with material identifiers, and the offline precompute pass.
- The learned grouping network, registered so Lightning trains, moves and checkpoints it.
- Collapse diagnostics and temperature annealing as Lightning callbacks, with the adjusted Rand index reported against
  both fixed partitions.
- Configuration for MPTS-52, one overlay per arm, an MP-20 reduced-budget overlay for the Stage 1 sweep, and one smoke
  overlay.
- Scripts to precompute, sweep, launch, evaluate, score, collect, analyse and smoke-test.
- 60 tests, all passing.

## Verified

- All 60 tests in `cgfm/tests` pass.
- OMatG's own suite (`omg/tests`, nine and a half minutes): 65 passed, 17 failed. None are attributable to the edits:
  - Fourteen are score-based-diffusion parametrisations, six in `test_integrators_ODE.py` and eight in
    `test_integrators_SDE.py`. Each fails inside `SingleStochasticInterpolant.__init__` with "the interpolant should not
    be a ScoreBasedDiffusionModelInterpolantVP ... because it requires antithetic sampling on the level of alpha(t)".
    The test matrix asks for a combination the constructor refuses outright, so they cannot reach any edited code.
  - One is a `RuntimeError` in `test_coupled_integrator.py`.
  - Two are sandbox file-permission errors in `test_analysis.py`.
  - The ODE failures were separately re-confirmed to fail with that same constructor error, which is the check that
    matters, since the ODE path is the only one the coarse-to-fine interpolant uses.
- `cgfm/scripts/smoke.sh` runs clean end to end on a 64-structure split: partitions precomputed, all four arms trained
  for two epochs under the real `match_rate` validation, the four collapse diagnostics asserted present in the CSV log,
  two draws generated and scored, and both result tables collected.
- The four arm configurations were diffed against each other after merging. They differ only in the grouping kind,
  `eta`, the group file, the grouper hyperparameters and the annealing callback. Denoiser, optimiser, batch size,
  integrator, `use_min_perm_dist`, both velocity annealing factors and the centre-of-mass correction are identical,
  which is the Stage 4 invariant the comparison rests on.
- The learned arm was checked through the real `StochasticInterpolants.losses` path: every parameter of the grouping
  network receives a non-zero gradient, and every diagnostic is populated with a plausible value. At initialisation the
  adjusted Rand index sits near 0.18 against both fixed partitions, group extents near 4.5 Angstrom.

## Decisions that departed from the original specification

Each of these is a judgement call worth revisiting.

**Group count comes from chemistry, not from `round(N / 5)`.** CrystalNN coordination shells decide how many groups a
structure has, and the other coarse-grained arms are given that same number. On MPTS-52 this works out at about
2.8 atoms per group rather than 5. The reason is that groups are sets of atoms *in the cell*, so neighbours that are
periodic images of one site count once: mean coordination is about 3.4 distinct sites, and the polyhedron centres form
an independent set of roughly a third of the atoms. Forcing 5 would mean the shells arm was no longer testing
coordination shells.

**Overlapping shells are resolved by a dominating set, not by peeling.** Peeling whole shells from the highest
coordination downwards was tried first and gave one large polyhedron surrounded by more than half singleton groups,
which is barely a coarse-graining. Centres are now chosen greedily as a dominating set of the bonding graph and every
other atom joins the nearest centre that actually bonds it. Singletons fell from 55 to 30 per cent, matching what
k-medoids produces at the same group count, which makes the two fixed arms comparable.

**Soft assignments, annealed, instead of straight-through Gumbel.** Endpoint exactness needs the bump to vanish at
`t = 0` and `t = 1`, not the group-mean operator to be a projector, so nothing requires a hard partition during
training. A tempered softmax is differentiable everywhere with no estimator. Temperature is annealed to 0.1 over the
first 1000 epochs so that the coarse and fine schedules become exact and the final partition can be named. Anchor
selection stays discrete, with gradient reaching the scorer through the gating trick from top-k graph pooling.

**Explicit mean squared error instead of OMatG's loss.** OMatG drops the constant `E||u_t||^2` from the velocity loss,
which is correct when the target is fixed. It is not fixed once the grouping is learned: the dropped term's absence
leaves the grouping network minimising `-2 E[b . u_t]`, which is unbounded below and is optimised by inflating the
target rather than by finding better groups. The arms therefore use `E||b - u_t||^2`, which has identical gradients with
respect to the denoiser.

**Multi-sample metrics are computed here, not by `omg csp_metrics`.** The plan assumed `csp_metrics --metre=True` would
give METRe and cRMSE at five samples per composition. It cannot: `match_rmsds` and `metre_rmsds` both raise when the
generated list is longer than the reference list, and OMatG's METRe rate counts the fraction of *generated* structures
matching some reference, so it measures precision and would fall as sampling grew more diverse. `scripts/score.py`
computes a genuine best-of-n rate on the same `ValidAtoms` validation and the same `StructureMatcher` call, and the
one-shot numbers still come from OMatG so that they stay comparable with published results. This was confirmed with the
user before being implemented.

**Both fixed partitions travel with every batch.** The plan asks for the adjusted Rand index against k-medoids *and*
against the coordination shells, which needs both on the same batch. `CGOMGDataset` derives the sibling group file from
the name of the one that defines the arm's path, so the arm overlays still name a single file.

**Hyperparameters are OMatG's MP-20 values.** OMatG ships no tuned MPTS-52 configuration, so the denoiser, integrator,
annealing factors, optimiser and batch size come from `omg/conf_examples/csp_linear_ode_mp_20.yaml`. All arms are biased
identically, so the comparison holds, but the absolute numbers should not be reported as tuned MPTS-52 results.

## Bugs found and fixed during the second pass

- **`distance_bias` had an unidentifiable parameter.** The final layer of the grouping network's distance bias carried a
  bias term whose output is added to logits that are immediately softmaxed over anchors, where a constant cancels.
  Shifting it by 7.0 changed the assignment by 3e-16. It is now `bias=False`. A test asserting that gradient reaches
  every parameter caught this, but only intermittently, which is what prompted the investigation.
- **Test flakiness from an unseeded global generator.** Module constructors draw weights from the global torch
  generator, so adding an unrelated test file changed the initialisation of another test's network and turned it red.
  The autouse fixture now seeds the global state.
- **The sweep pointed at the wrong partitions.** The arm overlay names the MPTS-52 group files, so applying it after the
  dataset overlay silently paired perov-5 structures with MPTS-52 partitions. The overlay order is now reversed, and the
  material identifiers now stored in each group file would have caught it at runtime regardless.
- **Group files could not be checked against their split.** They were positional, validated only by length and atom
  count. They now store the material identifier of every structure, checked on load, which catches a reordered or
  filtered split rather than only a differently sized one.

## Open questions to settle before or during the full run

1. **Validation cadence.** `check_val_every_n_epoch: 50` with `validation_mode: "match_rate"` generates and matches the
   whole 5,000-structure validation split forty times over 2,000 epochs. The matching is CPU-bound and may dominate
   wall-clock time. If it does, either raise the interval or add a validation subset.
2. **`eta = 0.5` is unswept for the full runs.** The Stage 1 sweep tunes it on MP-20 at reduced budget, but the value
   carried into MPTS-52 is currently the specified 0.5 regardless of what the sweep finds. Decide whether to carry the
   sweep's best value across.
3. **`max_epochs: 2000`** is OMatG's MP-20 value. MPTS-52 has 27,380 training structures, so at batch 1024 this is
   roughly 54,000 steps. Worth confirming against the loss curve of the baseline arm.
4. **Whether the learned grouping deserves its own learning rate.** It currently shares the denoiser's. The
   specification asked for no auxiliary objective and no separate schedule, and that is what is implemented, but if the
   grouping is still near its initialisation late in training this is the first thing to change.
5. **Whether 2.8 atoms per group is too fine to matter.** If every coarse-grained arm ties with atomwise, merging
   shells to a coarser target is cheaper to test than any other explanation.

## Kill criteria

Decided in advance, so that a null result stays reportable.

- If no `eta` beats atomwise on MP-20 in the Stage 1 sweep, do not scale to MPTS-52.
- If `cg_ari_vs_kmedoids` climbs above roughly 0.8, the learned arm has rediscovered geometric clustering. Report it as
  such; do not tune it into looking different.
- If `cg_fine_energy_fraction` falls below the k-medoids arm's, the grouping is shrinking the residual rather than
  learning structure. Switch to the target-normalised loss variant.
- If `cg_singleton_fraction` approaches one, the grouping has collapsed to atomwise and the arm carries no information.
- If the learned arm does not beat `atomwise` on test match rate, the interesting comparison becomes `shells` against
  `atomwise`: whether coordination motifs help when supplied, even if they cannot be discovered.

## Next actions

1. Run `cgfm/scripts/sweep_eta.sh`. This is the go/no-go and it precomputes the MP-20 partitions itself.
2. If MP-20 clears, precompute the MPTS-52 partitions and launch the twelve jobs. Check after the first few validation
   epochs that `cg_fine_energy_fraction` is not already collapsing in the learned arm.
3. Evaluate every finished run with `evaluate.sh`, then `nfe_curve.sh` for the results-versus-cost curve.
4. Collect with `collect_results.py` and read the atom-count bins, not just the headline numbers.
5. Run `analyse_groups.py` on the learned arm and compare its partitions against both fixed ones.
6. Optionally close Stage 0 by reproducing the published MPTS-52 match rate from the released checkpoint, which is the
   only external reference number.
