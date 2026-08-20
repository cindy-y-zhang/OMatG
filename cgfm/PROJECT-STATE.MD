# Project state

Last updated: 2026-08-20, after the joint-geometry descriptor, implementation, and local oracle stop gates.

## Joint geometry state: local stop-gate result

The compact descriptor gate did not select CN–RDF4.
Against the full 17-channel radial state, CN–RDF4 retained 62.85 per cent of the coordination information and 69.84 per cent of the geometry information.
CN–RDF8 retained 87.20 and 92.62 per cent respectively, so its coordination result still missed the pre-registered 90 per cent threshold.
The nested fallback therefore selected the standardized full `radial17` state.
Its invariance, shuffled-target, noisy-endpoint, retrieval, and independent Gaussian-prior controls passed.

The backbone gate selected stock `fc`.
Its clean-endpoint chemistry-relative R2 was 0.2769 with a 95 per cent interval of 0.2659 to 0.2879, statistically tied with `periodic_distance` at 0.2786 and therefore preferred by the pre-registered cheapest-overlapping rule.
The deployed joint arms consequently use `radial17`, geometry width 17, and the `fc` graph.

The fourth stochastic-interpolant field, independent Gaussian prior, shared encoder input, descriptor head, and frozen calibrated loss are implemented.
The final seed-zero coefficient is **0.03648035**, which gives an estimated mean shared-trunk geometry-gradient share of exactly 0.0707 over the 20 calibration batches.
The promoted configuration passed the five-step smoke run and the ten-structure, 100-step memorisation check, whose objective fell by 81.99 per cent relative to its initial magnitude.

**The local oracle gate failed, so the sequential protocol stopped before the real-versus-dummy screens.**
After 200 warm-started optimiser steps on each arm and paired 210-step inference over 1,024 validation compositions, `D` matched **26.66 per cent** and perfect-trajectory `O` matched **24.61 per cent**.
The paired difference `O - D` was **-2.05 match points**, with a 95 per cent bootstrap interval from -4.20 to 0.10 points and a two-sided McNemar p-value of 0.0755.
Position target MSE improved slightly from 0.03151 to 0.03140, but cell target MSE worsened from 4.3293 to 4.4751.
This misses both parts of the pre-registered oracle gate: at least a two-point match gain and no worsening of position or cell error.

This is a local 200-step warm-start conclusion, not a full MPTS-52 training conclusion.
The `D/H/P/J` real-versus-dummy screens and the optional `R` comparison were not run because the oracle ceiling was already negative.
No checkpoint was promoted, no five-draw jitter study was triggered, no A100 bundle was created, and there is no full-run joint-geometry claim.

## Where this stands

**Read this first, because it changes how every accuracy number below should be read.**
In crystal-structure prediction the species are given rather than generated — the sampler mirrors them and the species loss is identically zero — so every atom's element and its structure's whole composition survive intact at *every* noise level.
A classifier on any per-atom label can therefore score well while reading nothing but chemistry, and **absolute accuracy on such a label is uninterpretable.**
The only readable quantity is the gain over an identically trained control whose coordinates and cell are replaced by a prior draw.
Two experiments in this project were nearly decided on absolute accuracies before that control was run, and both reversed when it was.

**The motif-token programme is stopped at its instrument check, twice over.**
The learned vocabulary turned out to be three per cent shape and seventy-three per cent chemistry, so Experiment B was cancelled before its six arms ran.
The replacement label is geometric by construction and audits clean — 28 named coordination polyhedra, 4.36 bits, 0.4 per cent masked, and a deterministic reader recovers 96 per cent of it from a structure displaced by 0.05 Angstrom.
But the free instrument check that was supposed to clear the way for it failed: on plain coordination number, the easiest geometric label there is, a classifier shown an essentially clean crystal beat the geometry-free control by **1.57 accuracy points against a pre-registered gate of ten**.
The target is readable and the apparatus is not reading it, so the pivot is held and the question moves to the encoder.
The leading suspect is measured rather than argued: `CSPNet` never computes an interatomic distance and aggregates with a mean, which makes composition free and coordination number hard.
See the motif-token section.

**The 1200-epoch comparison is in, and the block parameterisation lost by 9.8 points.**
Atomwise reached **26.82 per cent** (best, epoch 1099; 26.48 last) against centres-only blocks at **17.04** (16.72 last), on the same split, the same budget and the same effective batch.
The pre-registered response therefore triggers: the block stack stops being a generative parameterisation of its own and becomes, if anything, a conditioning interface on an atomwise backbone.
Every "17.9 per cent baseline" in the older sections below was a 400-epoch reading of a still-climbing curve; **26.8 is the number to beat from here on**, and the tolerance and oracle-mix tables should be read against it rather than against 17.9.

The two curves never cross and the gap opens monotonically after epoch 100:

| epoch | 99 | 199 | 399 | 599 | 799 | 999 | 1099 | 1199 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atomwise | 3.94 | 11.20 | 18.58 | 21.54 | 23.20 | 24.94 | **26.82** | 26.48 |
| centres-only blocks | 4.56 | 8.60 | 12.64 | 15.88 | 15.40 | 16.50 | **17.04** | 16.72 |

Blocks are ahead at the first validation and behind from the second onwards, which is the signature of a representation that is easier to fit and has a lower ceiling rather than one that is slower to start.
Blocks are flat from epoch 600 (15.9, 15.6, 15.4, 16.3, 16.5, 17.0) while atomwise is still gaining 2 points per 200 epochs at the end, so the gap at convergence is wider than 9.8, not narrower.
`centre_error` fell from 1.674 to 1.441 Angstrom over the run, against the 0.3 Angstrom the readout needs to deliver its 73 per cent, so the centres were never close to the accuracy the representation's own ceiling requires.

**What survives, and is now the premise of the next experiment.**
The readout result is untouched: given *true* poses, the block vocabulary rebuilds 90 per cent of unseen MPTS-52 structures, and correct centres alone give 51 points less than that.
Local geometry really does live on a small discrete set.
What failed twice is making that set carry the *generative path* — first as a coarse-to-fine schedule, which the spectral study killed on first principles, then as a rigid-pose parameterisation, which lost 9.8 points here.
So the vocabulary claim is separated from the parameterisation claim and retested with the parameterisation removed entirely: keep atomwise interpolation exactly as it is, and attach to each atom a **rotationally invariant discrete label describing its environment**.
No poses, no orientations, no partition, no per-object degrees of freedom — the three things that failed.
See the motif-token section for the design and the gates.

Dropping the rotation objective moved the overfit gate from 7 to 66 per cent, a ninefold change.
Two of the metrics this document previously reasoned from were measuring the wrong thing, and correcting them changes what the earlier results mean.

**The translation sweep worked.**
`sweep_block_gate.sh translation` was listed here as the next action and as the experiment that decides everything.
It ran.
At 105000 optimiser steps on the 100-structure gate: the joint control peaks at 7 per cent, removing the rotation *input* alone changes nothing at 7, removing the rotation *objective* reaches 62, and removing both reaches 66.
The atomwise control on the same task at the same budget is 79.
So the recovery came from the loss weight rather than from the trunk input, and the block model is in the same league as atomwise for the first time.

**The metric that said otherwise was broken.**
`translation_error` compared block `i` to labelled target `i`.
Centres of one element are interchangeable, so a structure with eight lithium centres has `8!` equally correct labellings that the matcher accepts, and the conditional velocity target is centred per structure, so the generated centroid stays wherever the uniform prior put it and the model cannot move it.
A perfect generation therefore reads about `0.45 * V ** (1/3)`, which on MPTS-52 is the 3.4 Angstrom that "no intervention moved".
It stayed flat across all four translation arms while their match rates spanned 7 to 66 per cent.
It is now `centre_error`, which solves the per-element assignment and quotients out one global translation; the old reading survives as `translation_error_labelled`.
On a one-epoch run the two read 2.01 and 3.80 Angstrom.

**An earlier comparison was unfair.**
The production run and the first version of `run_centres_mpts52.sh` loaded `block_oracle_mpts52.yaml`, which copies each centre's true coordination number into the base state.
Coordination number is a property of the target structure, not of the composition, so it leaked a summary of the answer that the atomwise baseline never received and that no real inference could obtain.
The headline mode now predicts it.
`blocks-oracle` keeps the old behaviour for the diagnostic question only, and its number must not be quoted against atomwise.

**The coarse-to-fine premise has a first-principles problem, now measured rather than argued.**
Two readings, and the second is the one that matters.
The usable spatial frequency range of these datasets, from the fundamental `1/L` to the bond scale near `1/2.2` inverse Angstrom, is **1.51 octaves on MP-20 and 2.00 on MPTS-52**, against 8 for a 256-pixel image; a cascade needs more bands than that to cascade through.
And **the spectrum runs the other way from an image's**: after dividing out atomic form factors, the normalised intensity that would sit at 1.0 for independently placed atoms instead sits at 0.002 in the longest-wavelength shells and only approaches 1 near the bond scale.
The coarse scales of a crystal are not where its information is, so a coarse-to-fine schedule spends its early capacity on the emptiest bands.
See the spectral study section for the mechanism and for what this implies about featurisation.

## Research questions, and what has been answered

*Does a learned partition of a crystal into coordination-scale atom groups define a better flow-matching probability path than standard atomwise periodic interpolation?*

**No, settled.**
See Stage 1 below, and the spectral study for why.

*Does a coordination-motif representation with rigid poses carry a crystal well enough to generate one?*

**The representation yes, the full pose no.**
The readout rebuilds 90 per cent of unseen MPTS-52 structures from true poses, and orientation is nearly redundant within that.

*Can a model place motif centres and the cell?*

**Yes, but not well enough to compete.**
**Settled.**
66 per cent on the memorisation gate against an atomwise control of 79, and on full MPTS-52 at 1200 epochs, 17.04 per cent against atomwise's 26.82.
Centre error ends at 1.44 Angstrom where the readout needs 0.3.
Placing the objects accurately is the binding constraint, and it does not come from a better loss weight or more depth: the curve is flat for the last 600 epochs.

*Do the inductive biases about bonding help, or should the structure just be learned from data?*

**Three distinct claims were conflated.**
**Two are settled and the third is what is being tested now.**
A *multiscale* claim — resolve coarse then fine — is not supported, twice tested, with under two octaves of usable frequency range to work in and a spectrum that runs the wrong way.
A *parameterisation* claim — generate objects with poses instead of atoms — is not supported either, and cost 9.8 match-rate points.
A *vocabulary* claim — local geometry lives on a small discrete set — is strongly supported at 51 points on the readout and has never been tested on its own, because both previous tests bundled it with a parameterisation change.

*Does an invariant discrete label per atom, describing its environment, help an otherwise unchanged atomwise generator?*

**Open.**
**This is the current experiment.**
The premise is a subtraction rather than an addition: the interpolant, the prior, the network and the loss stay exactly as the 26.82 per cent baseline has them, and the only change is that each atom also carries a token summarising the local geometry it sits in.
That isolates the one claim that has survived from the two that have not.
The token is rotation-, translation- and permutation-invariant by construction and excludes the centre atom from its own descriptor, so it names an environmental *role* — `Si` in an oxygen tetrahedron and `Ge` in the same tetrahedron get the same token — while the element continues to reach the model through the existing embedding.
Because the previous two failures were both diagnosed only after a full parameterisation had been built and trained, the ceiling is measured *first* this time: Experiment B hands the model perfect tokens computed from the answer, which is not a deployable model and bounds every scheme for predicting them.
If perfect tokens do not buy 3 to 5 points, nothing downstream can, and the reportable result becomes Experiment A's characterisation of the vocabulary plus the ceiling itself.

## Stage 1: the partition sweep

Four arms on composition-conditioned MP-20 structure prediction, identical in everything but the path for fractional coordinates, 400 epochs, seed 0.
Validation match rate in per cent:


| arm                 | eta 0     | eta 0.25 | eta 0.5 | eta 0.75 |
| ------------------- | --------- | -------- | ------- | -------- |
| atomwise            | **56.15** |          |         |          |
| k-medoids           |           | 56.00    | 53.44   | 48.23    |
| coordination shells |           | 51.81    | 50.99   | 49.59    |
| learned             |           | 52.69    | 50.61   | 5.23     |


Every coarse-grained arm ties with or loses to atomwise, and the loss grows with `eta`, which is the strength of the coarse-to-fine bump.
The learned arm at `eta = 0.75` collapsed outright and never recovered.
The k-medoids `eta = 0.75` run ended at epoch 299 rather than 399; it was already 8 points behind.

One pattern is worth keeping.
Six of the nine coarse-grained runs *led* at epoch 49, the learned arms by 2.8 and 2.5 points, and every lead was gone by convergence.
That is the signature of a path that is easier to optimise early rather than one that models the data better, which is what the flow-matching literature reports for conditional path reparameterizations generally: they buy sampling efficiency, not converged quality.

### Why it failed

Four reasons, in order of how much they explain.

**A partition cannot hold a coordination environment.**
Polyhedra share atoms; an anion belongs to several cation polyhedra at once, and that sharing is the structural content.
`cgfm/shells.py` had to destroy it to produce a partition, and its own docstring records the cost.
The MP-20 group files hold 2.75 atoms per group with 33 per cent singletons for k-medoids and 36 per cent for coordination shells, against the five-atom shells the specification assumed.
The shells arm was therefore not testing coordination environments; it was testing small geometric neighbourhoods, which is what k-medoids already tests.
That explains why the two fixed arms behave alike.

**The group never becomes a unit with a pose.**
Per atom the path reads `x_t = x_0 + (t + eta s(t)) d_g + (t - eta s(t)) delta_i`, where `d_g` is the group's mean displacement and `delta_i` the within-group residual.
The centroid component does run ahead of the residual, so the schedule story is real, but the motif's shape is interpolated out of noise the whole way.
There is no time at which a correct polyhedron exists and is carried to a site, no orientation variable, and no template.
The intention, place the group and then its atoms, is not what the algebra does.

**Nothing in the model knows about symmetry.**
The motivation is stated in terms of Wyckoff sites, but no space group, orbit or multiplicity appears anywhere.
A geometric cluster's centroid has no reason to sit at a special position.

**The bias never reaches generation.**
Because the bump vanishes at both endpoints, the modelled distribution is identical for every `eta`.
The grouping network reads the clean target, which does not exist at sampling time, so it contributes nothing to generation.
The inductive bias entered only as a modified regression target that the denoiser had to learn to infer.

### Ruled out: the sampler

Before accepting the result, the obvious confound was tested.
Generation multiplies predicted velocities by `(1 + factor * t)`, and that factor was tuned for the atomwise path, so a bumped path might simply have been sampled wrongly.
The factor was exposed per arm and swept from 2 to 50 on the existing checkpoints, no retraining.

Retuning lifted every arm and changed no ordering.
On the full validation split the best figures are atomwise 58.31 at factor 40, k-medoids `eta = 0.25` 57.44 at factor 32, and k-medoids `eta = 0.5` 54.78 at factor 25.
The gains also cost geometric precision: match rate keeps climbing past the point where root-mean-square displacement starts to worsen, so the high factors are overshooting into the tolerance rather than placing atoms better.
The comparison was not confounded.

## The replacement: rigid blocks on SE(3)

Both of the first two failures come from the same source, a hard partition of Euclidean coordinates.
Dropping it gives a representation in which the motif hypothesis is actually expressible.

- Blocks **overlap**.
  A structure is a set of cation-centred coordination polyhedra, and an atom may be a vertex of several.
  Sharing is kept rather than resolved away.
- A block has a **pose**.
  It carries a rigid template for its type, a position and an orientation, so placing a motif is one move.
  This is the parameterization behind MOFFlow and MolCrystalFlow, which cut `3N` coordinates to `6M` pose parameters and beat atom-level baselines on metal-organic and molecular crystals.
- Atom positions are the **consensus** of the blocks claiming them.

The difference from those two precedents is that their building blocks are disjoint and given, while inorganic polyhedra share atoms.
Which blocks share which atoms therefore has to be generated, and that is the part with no published precedent for inorganic crystals.

### Measured ceiling on MP-20

`cgfm/scripts/block_ceiling.py` gives away everything a model would have to predict: every polyhedron is replaced by the canonical template of its type, placed at its true centre with the best-fitting rotation, and shared atoms are averaged.
1500 validation structures, templates fitted on the training split.


| pose error (translation / rotation) | match rate | per-atom displacement |
| ----------------------------------- | ---------- | --------------------- |
| oracle                              | **98.73%** | 0.116 A               |
| 0.05 A / 2 deg                      | 98.47%     | 0.151 A               |
| 0.10 A / 5 deg                      | 98.13%     | 0.198 A               |
| 0.20 A / 10 deg                     | 97.73%     | 0.306 A               |
| 0.40 A / 20 deg                     | 90.73%     | 0.527 A               |


**The representation is not the limit.**
98.7 per cent against 58.3 per cent achieved leaves about 40 points unclaimed, so the binding constraint is pose prediction, not expressiveness.

**Pose precision is affordable.**
The curve is flat out to 0.2 Angstrom and 10 degrees.

**Sharing is what makes it work, and this was not expected.**
Polyhedra are floppy: an instance sits 0.476 Angstrom from its type template on average, far too much to rebuild a structure from any one of them.
But each atom is a vertex of 4.5 blocks, 66 per cent of atoms are shared, and averaging their disagreeing predictions cancels most of the individual distortion, leaving 0.116 Angstrom.
The overlap that made a partition impossible is precisely what makes rigid templates accurate, because the system is over-determined.

### The dimensionality question, now answered

MOFFlow's case rests on `6M` being far below `3N`, which holds for cells of hundreds of atoms.
It does not hold here, and moving to MPTS-52 did not rescue it:


| split                      | blocks | atoms | pose dof `6M + 9` | atomwise dof `3N + 9` | ratio | centres only `3M + 9` | ratio    |
| -------------------------- | ------ | ----- | ----------------- | --------------------- | ----- | --------------------- | -------- |
| gate split, 100 structures | 8.68   | 19.25 | 61.1              | 66.8                  | 0.92  | 35.0                  | **0.52** |
| MPTS-52 validation, 5000   | 11.52  | 24.44 | 78.1              | 82.3                  | 0.95  | 43.5                  | **0.53** |


So the pose parameterization buys no compression at either scale, and whatever it offered had to come from the inductive bias and the over-determined consensus.
Dropping orientations is the first version of this that is genuinely smaller than atomwise, by a factor of two.
Centres are 45 to 47 per cent of atoms.

## What was built

`cgfm/blocks.py` (decomposition, templates, alignment), `cgfm/so3.py` (exponential and logarithmic maps, geodesics, stabiliser canonicalisation, vertex distance, 6D rotation parameterization), `cgfm/block_si.py` (coupled CN, SE(3) and lattice interpolants), `cgfm/block_sampler.py`, `cgfm/block_encoder.py` (CSPNet over blocks), `cgfm/block_lightning.py`, `cgfm/blockdata.py` (leakage-safe preprocessing), `cgfm/readout.py` (assignment-free consensus readout by clustering), `cgfm/scripts/precompute_blocks.py` and the launch, gate and sweep scripts.
Full block tables for MPTS-52 exist at `cgfm/blocks/mpts_52`, 905 coarse types and 9142 fine types over 27380 training structures.

## What the gates said

**Phase-0 G1, readout fidelity on real validation data.**
**Missed, and the threshold was unreachable as written.**
The coarse free readout reaches 88.73 per cent against a threshold of 90, but the noiseless ceiling of the same readout is 89.93 per cent, so no reduction in pose error could ever have passed it.
The fine vocabulary scores identically, so G2's question is settled and the vocabulary is not the constraint.
Pose tolerance is not either: 88.2 per cent survives 0.20 Angstrom and 10 degrees, and 83.6 per cent survives 0.40 and 20.
The lost 11 per cent is in the clustering step that groups vertex votes into atoms, and varying the linkage rule moves it very little.

**G3, memorising 100 structures.**
**Failed against a control that passes — but only for the full-pose parameterization, which the translation sweep has since superseded at 66 per cent.**
The threshold was 80 per cent best training-set match.
The block model reached 5 per cent at the gate's optimiser budget and 11 to 13 per cent at ten times it.
The atomwise control on the same 100 structures, against the same references with the same matcher, reached 27 per cent at the gate budget and **79 per cent** at ten times it.
So the gate is achievable and the budget matters enormously — for the control.
Ten times the budget moved the block model by one point.

On this split the readout costs nothing, since oracle poses rebuild all 100 structures, so the G3 comparison against the atomwise control is clean.
The readout does cap the block model near 90 per cent on real validation data, which matters for the production comparison below but not here.

## What was eliminated

Every intervention below was run on the gate and is reported at its best arm.
The elimination is the useful part: it is what makes the decomposition that follows trustworthy rather than one more guess.
Read it as the record of what did *not* help before the orientation came out; the arm that did help is in the translation sweep section.


| intervention                                                              | best match on 100 structures |
| ------------------------------------------------------------------------- | ---------------------------- |
| baseline, 10500 steps                                                     | 5%                           |
| ten times the budget, 105000 steps                                        | 6%                           |
| full batch                                                                | 3%                           |
| twice the width                                                           | 7%                           |
| twice the depth, 12 layers                                                | **11–13%**                   |
| learning rate from 1e-4 to 2e-3, and no gradient clipping                 | 3–8%                         |
| six-arm loss-weight grid: consensus weight and translation/rotation split | 3–8%                         |
| **atomwise control, same budget**                                         | **79%**                      |


Depth is the only thing that helped, and doubling it doubled the score.
Templates and readout were already clear, since true poses rebuild every structure of this split, and an oracle naming exact endpoints integrates to 100 per cent, so the integrator and the SO(3) path are clear too.

### The dataset-size ladder

Ten thousand five hundred steps each, on subsets of increasing size.
The cell volume ratio is the median at the last validation, relative to the target.


| structures | best match | last match | cell volume |
| ---------- | ---------- | ---------- | ----------- |
| 1          | 0%         | 0%         | 0.014       |
| 3          | 33%        | 0%         | 0.188       |
| 10         | 20%        | 0%         | 0.381       |
| 30         | 10%        | 10%        | 0.664       |
| 100        | 5%         | 1%         | 0.784       |


The model cannot memorise three structures, and the smaller the subset the worse the cell collapses, which is the opposite of the expected direction.
Several arms peak and then decay.

## The production run

Full MPTS-52, oracle coordination numbers, 400 epochs, 10800 optimiser steps at an effective batch of 1024, about 80 minutes.
Match rate 0.80 per cent at the first validation, peaking at 1.90 and ending at 1.18.
The atomwise baseline on the same split and budget read 17.92, 17.42 and 16.16 per cent over three runs, **all still climbing when they stopped** — and they were right to be: the 1200-epoch run reached 26.82.

Every per-block error is flat across all 400 epochs: rotation 2.197 radians against a random-orientation baseline of 2.213, translation 3.79 to 3.81 Angstrom, vertex 9.8 square Angstrom.
The cell volume ratio sits between 0.90 and 1.21.

Two caveats, both discovered afterwards.
The flat translation reading is the broken metric and says nothing.
And this run was handed the true coordination numbers, so its comparison against the atomwise baseline was not fair in the first place; it should be read as an upper bound on what that parameterization achieved, not as a like-for-like result.

## The translation sweep: what removing the orientation recovered

`sweep_block_gate.sh translation`, four arms on the 100-structure gate, 15000 epochs and 105000 optimiser steps each.
Files under `cgfm/overfit_runs/translation_grid/`.


| arm                                        | what it removes                      | first | best    | last | labelled trans |
| ------------------------------------------ | ------------------------------------ | ----- | ------- | ---- | -------------- |
| joint                                      | nothing, the control                 | 3%    | 7%      | 1%   | 3.48 A         |
| no rotation input                          | nine numbers of prior noise per node | 2%    | 7%      | 5%   | 3.55 A         |
| no rotation objective                      | half the loss weight                 | 8%    | **62%** | 51%  | 3.46 A         |
| both                                       | the orientation apparatus entirely   | 10%   | **66%** | 61%  | 3.45 A         |
| **atomwise control**, same task and budget |                                      | 10%   | **79%** | 73%  | n/a            |


Three readings.

**The loss weight was the cost, not the input.**
Blinding the trunk to the orientation on its own does nothing at all; removing its share of the gradient budget is worth 55 points.
Half the objective was being spent on a field the oracle substitution had already shown to be worth 0.7 points.

**The gap to atomwise is now 13 points on a memorisation task, not a factor of six.**
Both arms are still well below the control, and both peak and then decay, so this is not yet a win.
It is the first version of the parameterization for which the real-data comparison is worth running.

**The per-block translation error stayed flat at 3.45 to 3.55 Angstrom across every arm** while the match rate moved by a factor of nine.
Nothing about the model explains that; the metric does.

## The decomposition: which degree of freedom is binding

`diagnose_rotations.py --checks oracle_mix` replaces one head's output with the exact conditional velocity towards the target and integrates, so the rate with a head replaced is the rate the run would earn if that head were solved.
Nothing is trained; it reads off an existing checkpoint in minutes.

**Production checkpoint, 150 unseen MPTS-52 validation structures:**


| given to the oracle                               | match rate |
| ------------------------------------------------- | ---------- |
| nothing (the run as it stands)                    | 0.0%       |
| rotation                                          | 0.7%       |
| cell                                              | 3.3%       |
| translation                                       | 6.0%       |
| rotation and cell                                 | 6.0%       |
| translation and rotation                          | 8.7%       |
| **translation and cell**                          | **72.0%**  |
| translation and cell, orientations from the prior | **73.3%**  |
| everything                                        | 90.0%      |


**Best block checkpoint (twelve layers), the 100-structure gate split:**


| given to the oracle                               | match rate |
| ------------------------------------------------- | ---------- |
| nothing (the run as it stands)                    | 13.0%      |
| rotation                                          | 14.0%      |
| cell                                              | 25.0%      |
| rotation and cell                                 | 27.0%      |
| translation                                       | 58.0%      |
| translation and rotation                          | 78.0%      |
| **translation and cell**                          | **90.0%**  |
| translation and cell, orientations from the prior | 77.0%      |
| everything                                        | 100.0%     |


Three readings.

**Translation is binding, by a wide margin.**
Perfect orientations are worth 0.7 points on the real task and one point on the gate.
Perfect translations and cell are worth 72 and 77.
The earlier reading, that orientation was the dead head because its error sat at chance, was wrong: that error is a mean over blocks whose orientations are partly degenerate, a third of which have fewer than three vertices and no orientation at all, and it does not predict the match rate.
Oracle substitution does.

**Orientation is not merely redundant, it is expensive.**
On the real task the learned orientations score marginally *worse* than orientations drawn afresh from the prior at every step.
Meanwhile they cost nine numbers of prior noise per node entering the shared trunk, and half the loss weight.

**The readout ceiling is consistent across three independent measurements.**
90.0 per cent here, 89.93 per cent from the noiseless G1 study, 88.73 per cent from the G1 gate itself.

### The control that stopped a wrong conclusion

The gate arms' per-block errors are flat while the match rate climbs from 1 to 13 per cent, which suggested the matches might be structures whose poses cannot matter: the matcher is invariant to a global translation, so a single-block structure needs no translation, and a block with fewer than three template vertices has no orientation to get right.
The gate split has 5 of the former and 13 of the latter, which brackets the 5 to 13 per cent the arms score.

It is not what is happening.
`--checks attribution` splits the matches by group: 12 of the 82 structures where poses *must* be right matched, which is 92 per cent of all matches, while the 13 orientation-free structures matched none.
The pose learning is small but genuine, and the flat means simply cannot see 12 structures out of 100.

## What the readout is worth, and what it requires

`read_out` uses the composition only as a multiset, its order is discarded, and atoms it cannot reach are piled onto centres, which costs the match rather than borrowing the answer.
But the standard matcher tolerances are permissive enough that the rates need a floor to be read against.
On the same 150 validation structures, with the exact cell throughout:


| generated structure                                            | match rate |
| -------------------------------------------------------------- | ---------- |
| every atom exactly right                                       | 100.0%     |
| centres right, every other atom at a uniformly random position | 22.0%      |
| centres right, every other atom piled on the centres           | 8.0%       |
| centres right, every other atom at the cell origin             | 8.0%       |


So the motif vocabulary is worth **51 points**: 22 per cent from correct centres alone, 73 per cent once the templates place the remaining atoms at random orientations.
On the gate split the same floor is 34 per cent, which is why that split's numbers read higher throughout and should not be quoted as the headline.

**Read the oracle-mix tables above with the metric fault in mind.**
The rows are match rates from oracle substitution, which is a sound procedure and is unaffected.
The `translation_error` column those runs printed is not, and the earlier reading of "the model sits at 3.87 Angstrom, which is chance" does not follow from it.
`diagnose_rotations --checks oracle_mix` now prints both columns.

### Required centre accuracy

Exact cell, orientations from the prior, centres displaced by an isotropic Gaussian, 150 unseen validation structures:


| centre error | match rate |
| ------------ | ---------- |
| exact        | 71.3%      |
| 0.05 A       | 72.7%      |
| 0.10 A       | 73.3%      |
| 0.20 A       | 76.0%      |
| 0.30 A       | 73.3%      |
| 0.50 A       | 48.7%      |
| 0.75 A       | 23.3%      |
| 1.00 A       | 10.7%      |


The tolerance is mild in the sense that it does not demand crystallographic precision, but read it against the finished baseline rather than the 17.9 per cent this section originally quoted: **0.3 Angstrom is needed to clear atomwise's 26.82, and 0.5 Angstrom is already below it.**
The flat region out to 0.3, and the fact that 0.20 scores slightly above exact, say the clustering is not sensitive at this scale and a little jitter does it no harm.

**These are ceilings given perfect coarse placement, not model results.**
The live run has since answered the model question and answered it badly: `centre_error` ended at **1.441 Angstrom**, which on this curve is worth close to nothing, and the arm scored 17.04 per cent.
The earlier figure of 3.87 Angstrom came from the broken metric, so the true reading is better than that number and still far outside the tolerance the representation needs.

## Bugs found and fixed

Recorded because several of them had been silently shaping conclusions.

1. `translation_error` **measured gauge, not structure, and it was load-bearing.**
   Index-aligned against a block labelling the task does not fix, and against a global offset the per-structure centred velocity target leaves free.
   It cannot fall below about `0.45 * V ** (1/3)` even for a perfect generation, which is the 3.4 Angstrom that was read as a translation head learning nothing.
   Replaced by `centre_error`, which solves the per-element assignment with `linear_sum_assignment` and quotients out one global translation, with tests asserting that a globally translated and within-element permuted structure scores zero while the old reading exceeds one Angstrom.
   The old value is kept as `translation_error_labelled`.
2. **The consensus term was degenerate.**
   Residuals were measured in the *predicted* lattice, so shrinking the cell reduced the loss without improving anything.
   It was inflating or collapsing generated cells depending on its weight.
   Now measured in the target lattice, with a test asserting a shrunk cell cannot improve the objective.
3. `cell_volume_ratio` **was logged as a mean.**
   A volume ratio is floored at zero and unbounded above, so a run whose median cell was two thirds of the target reported 0.99 and was read as having learned the lattice.
   Now a median.
4. `blockdata` **did not propagate** `floating_point_precision`, so graphs arrived double against a single-precision model and training crashed at the first linear layer.
5. **The overfit gate ruled on undertrained runs.**
   It now refuses to return a verdict below a stated optimiser-step budget, which matters because the control needs ten times the original budget to pass.
6. **Checkpoint loading built the encoder under the double-precision default.**
   `BlockCSPNet` holds tensors that are neither parameters nor buffers, which `.to()` does not reach, so every diagnostic loading a checkpoint would have run with mixed precision.
7. **The readout re-derived every composition's centres, which was most of what a validation pass cost.**
   A sampling profile put 16 of 20 stacks inside pymatgen's oxidation-state solver, reached once per structure through `centre_elements` at about 0.2 seconds a composition, with the GPU idle throughout.
   There was a cache, but it was keyed on the atom ordering rather than on the composition the answer depends on, and it died with the process, so every fresh evaluation paid in full and a paired evaluation paid once per arm per seed.
   Decisions are now keyed on the composition and persisted to `{block_dir}/cache/centres.pkl`, which `BlockDataModule` primes on load and the validation hook writes back to; `precompute_blocks` ships new tables warm.
   A pass over 1024 structures fell from 246 to 56 seconds and the full 5000-structure split now takes 156 seconds.
   The reordering is behaviour-preserving: 400 real MPTS-52 compositions give identical cations under both orderings.

The rotation head was also changed from predicting a body velocity to predicting an **endpoint orientation** through a 6D Gram-Schmidt parameterization, with the velocity recovered analytically at inference.
That is a design change rather than a bug fix, and it is now moot if orientations are dropped.

## Decisions that departed from the original specification

Each is a judgement call worth revisiting.

**Polyhedron centres are chosen from the composition alone.**
The first implementation picked centres by bond-valence analysis of the structure, which is circular: the number of centres is the number of poses a model must emit, and at inference there is only the composition.
Centres now come from a charge-balanced assignment of oxidation states where one exists, from electronegativity ordering where it does not, and otherwise every atom becomes a centre.
The rule refuses to split a composition whose electronegativity gap is under 0.4.
Making the rule honest cost nothing: the MP-20 ceiling moved from 98.80 to 98.73 per cent.
CrystalNN is still used to find which atoms are a polyhedron's *vertices*, which is not circular, because that only ever runs on a known training structure to build a target.

**The generative vocabulary is central element and coordination number.**
The fine key including vertex composition gave 2561 types for 1500 MP-20 structures with 40 per cent needing a fallback.
On MPTS-52 the coarse key gives 905 types, and the G1 study shows the fine key scores identically, so nothing is lost.

**The orientation loss is minimised over the template's point group.**
An octahedron has 24 rotational symmetries, so the regression target is a set, not a point.
Training against an arbitrary representative would inject noise the size of a whole rotation.
This came out of a unit test rather than the literature.

**Group count came from chemistry, not from** `round(N / 5)`**.**
This is why Stage 1 groups held 2.75 atoms rather than five, and in hindsight it was the moment that arm lost its hypothesis.
The fix was not a different group count but abandoning the partition.

**Explicit mean squared error instead of OMatG's loss.**
OMatG drops the constant `E||u_t||^2`, which is correct when the target is fixed.
It is not fixed once the grouping is learned: without it the grouping network minimises `-2 E[b . u_t]`, unbounded below and optimised by inflating the target rather than by finding better groups.

**Both sides of any production comparison run the same number of epochs.**
The atomwise baseline was still climbing at epoch 399 and the atomwise control tripled its score with ten times the budget, so a block result declared better at a budget the baseline had not converged at would say nothing.
`run_centres_mpts52.sh` runs 1200 epochs a side and a test asserts the two match.

**Hyperparameters are OMatG's MP-20 values.**
No tuned MPTS-52 configuration ships with OMatG.
All arms are biased identically so comparisons hold, but the absolute numbers are not tuned MPTS-52 results.

## Open items

1. ~~**Why the translation head learns nothing is unexplained.**~~
   **Retracted; it was a measurement artefact.**
   The evidence was `translation_error` sitting at chance after 105000 steps.
   That reading is index-aligned against a labelling the task does not fix and a global offset the centred velocity target leaves free, so it is pinned near chance for any model at all, including a perfect one.
   It stayed flat while the same runs moved from 7 to 66 per cent match.
   `centre_error` replaces it.
   The remaining, much narrower question is how far centre placement gets on real data, which is what the live run measures.
2. **Neither model uses the minimum permutation distance coupling.**
   It exists in `omg`, both configs set `use_min_perm_dist: False`, and it targets exactly the position regression.
   Untried, cheap, and the natural next lever if the live run falls short.
   Note the connection to item 1: the coupling addresses the same exchangeability that the old metric ignored, so it is better motivated now than when it was first noted.
3. **The G1 clustering loses 11 per cent and it is not understood.**
   It is not the vocabulary and not pose tolerance.
   Worth attention only if the centre model works, since it caps everything downstream at about 90 per cent.
4. **Depth was tested once.**
   Twelve layers doubled the gate score over six and produced the only block arm whose cell volume reached the target.
   Nothing beyond twelve has been tried, and depth has not been retested since the rotation objective came out, so the two may compound.
5. **The neighbour rule produces blocks at both extremes.**
   CrystalNN returns up to 24 vertices for some sites, which is not a coordination polyhedron; the cap is 12.
   Separately, 28 per cent of blocks on the gate split and 33 per cent on MPTS-52 validation have fewer than three vertices, so they have no orientation to predict and no vertex votes to contribute.
   This matters much less now that orientation is not predicted at all.
6. **The gate split is small and reads high.**
   Its matcher floor is 34 per cent against 22 on real validation data, so the 66 per cent above should not be quoted as a headline and the live MPTS-52 run is the number that counts.
7. **The 52-atom cap blocks the application.**
   Li10GeP2S12 has 50 atoms in its conventional cell, so it is just inside MPTS-52.
   Li21Ge8P3S34 has 66 and is outside it.
   The new-ratio generation programme needs a training set without that cap, which means Alex-MP-20 or MP-full, before its results mean anything.
   See the lithium programme section for what is affected and what is not.
8. **The framework rule has a table but no trained model.**
   `centre_rule="framework"` is implemented end to end through `decompose`, `orphan_free`, `record_structure` and the settings digest, and is tested, but no thio-conductor split has been assembled and nothing has been trained on it.
   The sampler also has no entry point that starts from a composition alone: every block set today comes from a precomputed table, so generating for a *novel* composition needs `block_specification` wired into a base-state constructor.
   That is the next piece of the conditioning path.
9. **A seeded validation pass is not reproducible, so a sub-point effect cannot be read from one run.**
   Three runs of the same `validate` command, same checkpoint, same code and `--seed_everything 0`, returned match rates of 14.65, 15.33 and 15.23 per cent over the same 1024 structures.
   The spread is what nondeterministic GPU scatter operations in the integrator would produce, perturbing the trajectory enough to flip borderline structures, and it is the same size as the binomial standard error of the split itself.
   Anything below about a point therefore needs repeated passes.
   This is what the paired EMA design assumed a fixed seed would remove, and it does not.



## Kill criteria and their outcome

Decided in advance so a null result stays reportable.

- *Stage 1: if no* `eta` *beats atomwise on MP-20, do not scale to MPTS-52.*
  **Triggered.**
  MPTS-52 was not launched for that design.
- *Phase-0 G1: coarse free readout at or above 90 per cent.*
  **Missed at 88.73, and unreachable as written**, since the noiseless ceiling of the same readout is 89.93.
  The threshold, not the readout, was wrong.
- *G3: 80 per cent best training-set match on 100 structures.*
  **Failed at 5 to 13 per cent**, against an atomwise control that reaches 79 on the same task, so the gate was fair and the failure is real.
- *If the learned arm does not beat atomwise, the interesting comparison becomes whether coordination motifs help when supplied.*
  They do, and this is now quantified: 51 points over correct centres alone, with no orientation prediction.



## The lithium conductor programme: blocks as a conditioning interface

This does not depend on the benchmark outcome, and it is the one thing the representation can do that an atomwise model cannot do at all.

**The composition enumerator.**
`scripts/enumerate_thio.py`.
With `a` tetravalent `XS4` tetrahedra, `b` pentavalent ones and `c` bridging sulfurs, `S = 4 (a + b) - c` and `Li = 4 a + 3 b - 2 c`.
It reproduces all five known compounds exactly, including Li7P3S11 whose `P2S7` dimer is a single bridge and is therefore the case that pins the bridging term rather than merely being consistent with it.
Li10GeP2S12 falls out at `(1, 2, 0)` and Li21Ge8P3S34 at `(8, 3, 10)`.
The 5-to-2 ratio the project asked about is Li26Ge5P2S28 isolated, and each bridge added removes two lithium — condensation and mobile lithium are in direct competition, which is the trade-off at the centre of this chemistry.

**The targeted block rule.**
`blocks.py`, `centre_rule="framework"`, opt-in through `precompute_blocks.py --centre-rule framework`.
The generic composition rule calls every cation a centre, which on a lithium conductor means every lithium: 32 blocks over 66 atoms for Li21Ge8P3S34, spending six numbers per block to explain barely two atoms each, so it lands at 201 continuous parameters against 207 atomwise.
It also lets lithium compete for the sulfurs — on a synthetic Li7GePS8 cell the generic rule builds `(Li, 2)` and `(Li, 1)` polyhedra out of tetrahedron corners.
The framework rule gives every atom one of three roles from the composition alone: `Ge`, `Si`, `Sn`, `P`, `As`, `Sb` are poses; anything more electronegative than every former is a vertex placed by votes; everything else, lithium in particular, is a free atom carrying a translation and no orientation.
For Li21Ge8P3S34 that is 11 poses, 21 free atoms and the cell: **138 parameters against 207 atomwise**, a third fewer.

Leaving lithium free is the substantive choice, not a bookkeeping one.
A superionic conductor is by definition a material whose lithium does *not* sit in a well-defined coordination environment, so fitting it a rigid template is fitting a distribution as though it were a shape.
Three numbers with no orientation is the honest parameterisation of an atom in a diffuse site.

**Why this is conditioning and not leakage.**
Under the framework rule the coordination number of a tetrahedron is four by *stipulation of the hypothesis being tested*, not by measurement of a target structure, so `block_specification` may supply it.
That is the opposite of the `cn_mode: oracle` fault recorded above, where the coordination number was read off the answer.
"Eight `GeS4` to three `PS4`" is then a direct input: eight blocks with a germanium centre and coordination four, three with a phosphorus centre and coordination four, and twenty-one free lithium.
An atomwise model conditioned on the same material is told sixty-six atomic numbers and has no variable that holds a tetrahedron.

**The blocker, recorded explicitly.**
Li10GeP2S12 has 50 atoms in its conventional cell, so recovery is in distribution for MPTS-52 but only just.
Li21Ge8P3S34 has 66 and is outside it, and no model in this repository can emit a cell that large: `MAX_STRUCTURE_ATOMS` is 52 and the padded graph-level composition tensors are built against it.
So the new-ratio half of the programme needs a training set without the cap — Alex-MP-20 or MP-full — before its results mean anything.
Recovery of the known compound does not, and is the right first test.

## Motif tokens: the vocabulary claim without the parameterisation

The third and last form of the bonding-prior question, and the one the two failures above did not test.
Code in `cgfm/descriptors.py`, `cgfm/vocabulary.py`, `cgfm/motifdata.py`, `cgfm/motif_encoder.py`, `cgfm/motif_lightning.py`, `cgfm/motif_si.py`, `cgfm/motif_main.py`, and the scripts `precompute_motifs.py`, `precompute_coordination.py`, `vocabulary_report.py`, `run_motif_expB.sh`.

**The premise.**
Blocks changed three things at once: what the model generates (objects, not atoms), how many degrees of freedom it generates (poses, not coordinates), and what it knows (a vocabulary of local geometries).
The first two lost 9.8 points.
The third has never been isolated.
So: change nothing about the generative problem — same atomwise interpolant, same prior, same trunk, same losses, same 26.82 per cent configuration — and give each atom one extra piece of information, a discrete label naming the local environment it sits in.

**What a token is.**
For atom `i`, the neighbours within 6.0 Angstrom, with the centre atom excluded from its own density.
A species-resolved power spectrum of that neighbourhood (`n_max` 6 radial Gaussians under a cosine cutoff, real spherical harmonics to `l_max` 4, four alchemical species channels from a PCA of a standardised pymatgen element-property table, L2 normalised, about 1500 dimensions), PCA-whitened to 32 dimensions on the training split alone, then a Gaussian mixture whose responsibilities are the soft per-atom target.
Exactly invariant to rotation, translation, atom reordering and choice of cell vectors, and tested to be so.
The 6.0 Angstrom cutoff is not a guess: it is the band this project's own spectral study identified as carrying the signal, and the descriptor is deliberately blind to everything the spectral study found to be empty.

Excluding the centre is the design decision that makes one shared vocabulary coherent across the periodic table.
`Si` in a tetrahedron of oxygens and `Ge` in the same tetrahedron receive the *same* token, because the token names an environmental role rather than a species; the element continues to reach the model through the existing `node_embedding`, so nothing is lost.
Token-versus-element mutual information should therefore be near zero, and the report measures it: a non-trivial value means the descriptor is leaking the centre's identity and the vocabulary is not actually shared.

**Why the ceiling is measured first.**
Both previous failures were diagnosed after a full parameterisation had been built and trained to 1200 epochs — the readout ceiling in one case, the orientation objective's 0.7-point contribution in the other.
Both were answerable in one arm by supplying the model with perfect values of the thing in question.
So Experiment B hands the model the *true* token of every atom, computed from the target structure.
That is not a deployable model and its match rate must never be quoted as one: a token is a property of the answer.
It is a bound.
If perfect tokens do not help, no scheme for predicting them can, and Experiment C's eight arms are not run.

**Two changes in `omg` core**, both opt-in and behaviour-preserving, with tests asserting bit-identical behaviour.
First, `StochasticInterpolants.losses` was calling the model twice per step with byte-identical inputs, once for `pos` and once for `cell`; the `# TODO: Cache return of model function` at that spot is now implemented, halving the forward cost of every run in the repository.
Second, `integrate` gained no-op `_on_integration_start` and `_on_step_end` hooks so that self-conditioning state can be threaded through sampling without duplicating the integration loop.
The caching pays for the extra self-conditioning pass with change left over: 1.5 forwards per step against the 2 the baseline was spending.

**The loss weight is calibrated, not chosen.**
The rigid-pose failure is directly attributable to a hand-set loss weight — the orientation objective took half the weight and returned 0.7 points.
Here the motif weight is set from the ratio of the two objectives' gradient norms *on the shared trunk*, averaged over 20 batches at step zero, to put the motif objective at 7 per cent of the trunk gradient, and re-measured every 500 steps with a warning logged if it drifts outside 5 to 10 per cent.
Measured on the smoke split the calibration lands on a visibly different weight per arm — 0.099 for coordination numbers, 0.138 for learned tokens, 0.212 for shuffled ones — which is the point: the same gradient share requires a different weight for each target, and a single hand-set number would have silently given three different experiments.

**Pre-registered gates.**
Stop and write up Experiment A alone unless: tokens carry substantial information beyond coordination number and beat CrystalNN's 0.56 Angstrom stability half-life; oracle learned tokens give at least 3 to 5 match-rate points over the atomwise arm, or a substantial convergence or low-NFE improvement; oracle learned tokens beat oracle `(element, CN)`; shuffled-within-element oracle tokens give no gain; and tokens stay predictable above the element and composition prior in the 0.3 to 0.8 Angstrom regime.
Because a seeded validation pass is not reproducible to better than about 0.7 points (open item 9), every gate reading is the mean and standard error of five independent passes.

### Experiment A: the vocabulary, with no generator involved

`cgfm/reports/motifs/vocabulary.txt` and `.json`.
Fitted on the 27380 training structures (501334 environments; 32 principal components retain 97.6 per cent of the 1501-dimensional descriptor variance), and every number below is measured on the **val** split, which the vocabulary never saw: 98444 environments for the information estimates and 48323 for the stability curve.

**The gate cleared, and not marginally.**
Coordination number accounts for between 0.098 and 0.151 of the token's entropy, depending on `K`.
The handcrafted `(element, CN)` vocabulary is therefore not a substitute for the learned one, and the project does not stop here.
The full picture at `K = 32`, in bits and as a fraction of the token's 4.50 bits of entropy:

| feature | bits | fraction of token entropy |
| --- | --- | --- |
| neighbour composition | 1.798 | 0.400 |
| element | 1.476 | 0.328 |
| bridging versus terminal anion | 0.607 | 0.135 |
| coordination number | 0.559 | 0.124 |
| octahedrality | 0.144 | 0.032 |
| tetrahedrality | 0.133 | 0.030 |

Two of these deserve comment.
**Element is 0.33 rather than the near-zero the design predicted**, and the explanation is not descriptor leakage: the centre is provably excluded, and `Si-O4` equals `Ge-O4` exactly in the tests.
It is that in real crystals the neighbourhood predicts the centre — a cation sits among anions — and indeed element information is *below* neighbour-composition information at every `K`, consistent with being mediated by it rather than read off the centre.
The token is a genuine statement about environment; it is just that environment and identity are correlated in nature.
Since the element reaches the model directly anyway, what matters is the token's marginal contribution, which is what Experiment B measures.
And **tetrahedrality and octahedrality are almost absent, at 0.03**, which says the vocabulary is not organised around the two polyhedra a chemist would have drawn by hand.
That is the same lesson as the 22-token effective size: the partition it finds is finer and less nameable than the one the rigid-block work assumed.

**Utilisation is high and the mixture is not collapsing.**
Effective size `exp(H)` runs 6.6, 12.6, 22.6, 40.8 and 77.5 against nominal 8, 16, 32, 64 and 128, with the starved fraction rising from 0.000 to 0.180.
So `K = 128` is the first size where a fifth of the vocabulary is decoration, and nothing below 64 wastes any of it.

**The two selection criteria point in opposite directions, monotonically, with no interior optimum.**
This is the finding that shaped the experiment:

| K | surviving bits, `H(Z\|element) - E[-log q]` | stability half-life | clears CrystalNN's 0.56 A? |
| --- | --- | --- | --- |
| 8 | 1.519 | > 0.8 A | yes |
| 16 | 1.779 | 0.77 A | yes |
| 32 | **2.226** | **0.63 A** | **yes** |
| 64 | 2.501 | 0.52 A | no |
| 128 | 2.917 | 0.38 A | no |

The bound rises all the way to 128 and never turns over, so the noise-persistence criterion on its own has no answer: a larger vocabulary always carries more recoverable information, because the entropy it adds is recoverable faster than it is lost.
But the pre-registered gate also requires beating CrystalNN's 0.56 Angstrom half-life, and above `K = 32` the tokens are *less* stable under displacement than the handcrafted alternative they are supposed to beat.
**`K = 32` is the largest vocabulary that clears the gate**, and it retains 2.226 of the 2.917 bits the peak reaches.
Experiment B therefore runs at `K = 32` with flanking arms at 8 and 128, which is a better sweep than an arbitrary factor of four in each direction: 8 sits well inside the stability bar with less information and 128 sits well outside it with more, so the size arms measure whether the generator is limited by how much the tokens say or by how reliably they say it.

**The soft targets are soft in name only, and the reason is interesting.**
Responsibilities come out effectively one-hot at every `K`, and the noise-averaged softening the plan specified as the fallback — the mean over eight draws of a 0.1 Angstrom displacement — barely moves them: target entropy stays at 0.06 to 0.13 nats.
The stability curve explains why.
At 0.8 Angstrom of displacement the argmax has changed for two thirds of environments at `K = 32`, yet the posterior entropy has risen only from 0.08 to 0.15 bits against a marginal floor of 4.84.
**The vocabulary is sharp but not stable**: displacement does not blur an environment across a boundary, it moves it confidently into a different cell.
So `KL(pi || q)` is in practice cross-entropy against hard labels, and the soft-target machinery is decorative rather than wrong.
The interesting consequence is for Experiment C: a model asked to predict the token of a noised structure faces a target that is confidently wrong rather than honestly uncertain, which is the harder of the two problems.

### The oracle pilot: the sign of the effect, at an eighth of the budget

`cgfm/motif_runs/pilot/oracle/`.
One arm, 200 epochs on real MPTS-52 at `K = 32`, oracle tokens fed in, read against the existing atomwise run at the same epochs.
Both arms are the same trunk at an effective batch of 1024 with a constant learning rate, so the epoch-for-epoch reading is legitimate.

| epoch | oracle tokens | atomwise | difference |
| --- | --- | --- | --- |
| 49 | 1.24 | not measured | |
| 99 | 4.16 | 3.94 | +0.22 |
| 149 | 8.78 | not measured | |
| 199 | **12.46** | **11.20** | **+1.26** |

**The sign is positive and the size is small.**
Against the roughly 0.7-point spread of a single seeded validation pass (open item 9), a difference of two single passes carries about a point of noise, so +1.26 is about 1.3 standard errors: real if the trend is taken with it, indistinguishable from nothing if the endpoint is taken alone.
What makes it worth reporting is that it grows with budget, from +0.22 at epoch 99 to +1.26 at 199, which is the pattern of an effect rather than of noise.
It is also well short of the 3 to 5 points the pre-registered gate asks for, at an eighth of the budget the gate is defined at.

**The token is being used, and modestly.**
This matters because a null would otherwise have two explanations.
In the trained weights the 32 token embeddings have mean norm 4.19 against the element embedding's 21.9, so the token contributes about a fifth of what the element does to the same node features, and the spread across tokens is 4.17, nearly the norm itself, so they are 32 distinct vectors and not 32 copies of one.
The path is alive; it is simply not worth much yet.
`real_prediction` stayed at 0.074, which is expected rather than broken: in oracle mode every input is the informative one, so a constant offset marking it as such has nothing to discriminate and no gradient pressure to grow.

**What this does and does not license.**
It licenses spending the six A100 arms, which is what the pilot was for: the effect is not negative, and the ceiling arm is not measuring a disconnected input.
It does not license any expectation about the gate.
If the 1600-epoch oracle arm lands near +1 point, the honest conclusion is that perfect knowledge of the environment token is worth about a point to this generator, and Experiment C should not be run, because no scheme for *predicting* the token can beat being handed it.

**State.**
Phase 0 and all of Experiment A are done, with 503 tests passing in `cgfm/` and 95 in `omg/`.
Descriptors and codebooks are precomputed at `K` in 8, 16, 32, 64, 128, and the handcrafted coordination-number vocabulary is built alongside for the control arm.
Every Experiment B arm has been smoke-tested end to end on the 100-structure overfit split, including a full sampling and match-rate validation pass, and again from inside the unpacked handoff bundle, so a shape or configuration error cannot surface after the arms are queued on another machine.

**Experiment B was then cancelled, before it ran.**
Not because of the pilot, but because a diagnostic run after the pilot found that the classifier which was supposed to predict the tokens learned nothing from geometry: two 60-epoch arms at `K = 32`, identical but for one seeing the interpolated structure and the other seeing base-distribution coordinates at every time, finished at validation cross-entropy 0.6017 against 0.5997 nats and accuracy 0.7916 against 0.7903.
**The chemistry-only null won.**
A fixed-time sweep confirmed it, 0.698 top-1 at 0.23 Angstrom of corruption against 0.695 at 4.39.
Decomposing the token's 4.50 bits explained why: neighbour composition 1.80 bits, the atom's own element 1.48, bridging anion 0.61, coordination number 0.56, and then octahedrality 0.14 and tetrahedrality 0.13.
**The vocabulary was three per cent shape.**
The Experiment A gate it passed — "coordination number is at most 0.15 of token entropy, so the tokens are not coordination numbers with extra steps" — passed for the wrong reason: the token was not coordination number because it was mostly composition.
Spending six 1600-epoch arms on an oracle ceiling for a target with no geometric content would have measured the value of being handed a composition summary.

### The deterministic replacement, and the instrument check that stopped it

The correction was to replace the learned codebook with a label that is geometric by construction, and it is built in `cgfm/coordination_geometry.py`: the neighbour set and coordination number from `CrystalNN().get_nn_info`, then `LocalStructOrderParams` over the polyhedron templates belonging to *that coordination number only*, argmax within it.
Never across coordination numbers, because a tetrahedrality score and a bond-orientational parameter are not commensurable and matminer's fingerprint reports coordination weight times order parameter rather than the parameter.
Coordination numbers with fewer than two templates carry no shape to resolve and are labelled by the number alone, which covers CN 1, CN 9 through 11 where matminer has only q2, q4 and q6, and CN 12 and above.
**28 hard classes with human-readable names**, such as `tetrahedral CN_4` and `octahedral CN_6`, plus a per-atom validity mask; failures are masked, never filled with a uniform target or an unknown class.

**The label is sound, and was audited before anything was spent on it.**
On 4000 training structures and 72,982 atoms (`reports/geometry/audit_train.txt`): all 28 classes populated, 24 of them above one per cent, **4.362 bits** of label entropy against the learned vocabulary's 4.50, so the two are roughly information-matched.
Only 0.37 per cent of atoms are masked and every one of those is a coordination number of zero — the neighbour finder never raised and the order parameters never returned an undefined value.
The coordination numbers agree with the separately precomputed CrystalNN tables at **exactly 100.0000 per cent, zero atoms differing**, which is the right answer for the same finder on the same structures and is the check that would have caught either pipeline drifting.
The full precompute over all three splits is in `cgfm/motifs/mpts_52_geom/`: 501,334, 122,203 and 202,522 sites at 0.38, 0.34 and 0.28 per cent masked.
ChemEnv agrees with CrystalNN on the coordination number of 64.2 per cent of sites, which is why it is a cross-check and not the label.

**E1, identifiability with no learning.**
Displace every atom by a known root-mean-square amount, relabel with the same pipeline, compare (`reports/geometry/identifiability_val.txt`, 2000 validation structures, 48,323 atoms):

| RMS displacement / Angstrom | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 |
| --- | --- | --- | --- | --- | --- |
| 28-class label agreement | 0.924 | 0.857 | 0.702 | 0.388 | 0.142 |
| coordination number agreement | 0.961 | 0.922 | 0.822 | 0.547 | 0.233 |
| `H(clean given displaced)` / bits | 0.586 | 0.983 | 1.747 | 3.092 | 4.072 |

The label half-life is **0.329 Angstrom** and the coordination-number half-life **0.461**, both under the 0.5 the gate asked for and under CrystalNN's own stated 0.56 precision.
ChemEnv put through the identical displacements gives a coordination-number half-life of 0.409, so **the brittleness is a property of the geometry and not an artefact of one neighbour finder.**
The more important reading is the other one: relabelling is one particular predictor of the clean label, and at 0.05 Angstrom it recovers 96.1 per cent of coordination numbers.
That is the floor a learned classifier has to beat.

**E3, the instrument check, failed.**
Three classifiers on the already-precomputed CrystalNN coordination number, 13 classes, 60 epochs each, identical but for what geometry they see (`reports/geometry/instrument_check.txt`):

| arm | accuracy at best validation cross-entropy | best accuracy | last accuracy |
| --- | --- | --- | --- |
| geometry, interpolated state | 0.6066 | 0.6274 | 0.6208 |
| chemistry-only, prior coordinates at every time | 0.6129 | 0.6232 | 0.6159 |
| clean, time pinned at 0.999 | 0.6081 | **0.6389** | 0.6314 |

`delta A_geom` for the clean arm is **+1.57 accuracy points** at its best accuracy and **+1.55** at a matched epoch.
**The gate asked for ten.**
Counting neighbours is the most learnable geometric label there is, the structures are essentially clean — the positions use a periodic linear interpolant, so at `t = 0.999` the corruption is about 0.01 Angstrom — and the head reads the full six-layer trunk rather than the mid-network layer.
The apparatus cannot read coordination number off a clean crystal.

**Four things it is not.**
It is not a mislabelled target: on real MPTS-52 data the tensors the model receives match freshly recomputed labels for 96 of 96 atoms across seven structures, and `MotifTable.check_against` verifies identifiers and the per-structure atomic-number sequence on every construction.
It is not underfitting: the clean arm drove its *training* cross-entropy to **0.083 nats**, near-perfect memorisation, while validation climbed to 2.06 — and geometry demonstrably helped it memorise, since at a matched epoch it sat at 0.658 nats against the chemistry-only arm's 0.909.
It is not a too-easy floor: an explicit lookup table conditioned on the atom's element gets only **34.3 per cent** top-1 on this label, so the chemistry-only arm's 62.3 comes from generalising across chemistry rather than from tabulating it.
And it is not the pivot being wrong about the target — E1 shows the geometry determines the label, and a deterministic reader recovers 96 per cent of it at a displacement a hundred times larger than what the clean arm was shown.

**What it looks like instead is what the encoder is given.**
In `CSPNet` the graph is fully connected, each edge carries a 768-dimensional sinusoidal embedding of the *fractional* offset together with the nine numbers of the lattice Gram matrix, and `node_model` aggregates with `reduce='mean'`.
So the Cartesian distance is never computed: recovering it means learning the quadratic form `|dr|^2 = df^T (L L^T) df` out of a Fourier basis of `df` against nine lattice numbers, uniformly over every cell shape in the dataset, and the raw `df` is not even an input.
Meanwhile mean aggregation suppresses precisely what a coordination number is, since counting neighbours inside a distance is a *sum* of a soft indicator and a mean divides that sum by the total atom count, which is not an input feature at all.
Composition, by contrast, is what mean aggregation over species embeddings hands the network for free.
That is a coherent account of a network that reads chemistry easily and geometry not at all.

**Tested, and it is most of the answer.**
`cgfm/distance_encoder.py` appends a radial-basis expansion of the true minimum-image Cartesian distance to every edge, zero-initialised so the model starts out identical to the baseline, and changes nothing else.
The same two arms, same label, same budget (`reports/geometry/distance_fc.txt`):

| encoder | clean arm, best accuracy | floor, best accuracy | `delta A_geom` at best | at matched epoch |
| --- | --- | --- | --- | --- |
| baseline, fractional offsets only | 0.6389 | 0.6232 | +1.57 | +1.55 |
| Cartesian distance exposed | **0.6932** | 0.6204 | **+7.29** | **+6.97** |

**One input feature took the geometry signal from 1.6 points to 7.3, a factor of 4.6, while the floor did not move.**
The floor not moving is the control that makes it readable: an arm whose coordinates are a prior draw gains nothing from being told the distances between them, which is what should happen and confirms the gain is geometric rather than a capacity effect.
The diagnosis was right.
The label was never the problem, and neither was the budget; the encoder was not being given the quantity the label is a function of.

**A second structural defect, now under test.**
`gen_edges` with `edge_style='fc'` puts exactly one edge between each pair of atoms, carrying the offset wrapped into the unit cell.
But in a crystal most of an atom's neighbours are periodic *images*, and a single edge per pair cannot represent how many.
In a caesium-chloride cell each atom has fourteen neighbours inside 7 Angstrom — eight at `a * sqrt(3) / 2` and six at `a` — and the fully connected graph on a two-atom cell offers it two edges.
**The coordination number is not a countable property of the graph the model is given.**
This is the same trap the labeller itself had to avoid, where passing bare site indices to the order parameters presented rock salt's octahedron as three points.
`CSPNet` already supports the alternative: `edge_style='knn'` builds a periodic radius graph through `radius_graph_pbc` at a 7 Angstrom cutoff with up to twenty neighbours, one edge per image, and it is verified against the caesium-chloride numbers in the tests.
It is off by default and every run in this repository has used `fc`.
The same two arms on the radius graph are running now.

One upstream constraint found while wiring it: `radius_graph_pbc` builds its image offsets with a hardcoded single-precision dtype and multiplies them by the caller's lattice, so the radius graph raises under float64.
The training configuration is `32-true`, so this restricts the tests rather than the runs, but it is why one test sets the dtype locally.

## Next actions

1. ~~**Run**~~ `sweep_block_gate.sh translation`~~**.**~~
   **Done, and it worked**: 7 to 66 per cent.
   See the sweep section.
2. ~~`run_centres_mpts52.sh blocks` **and** `atomwise`, 1200 epochs each~~ **Done, and blocks lost: 17.04 against 26.82.**
   Both arms are in `cgfm/centre_runs/`; read with `run_centres_mpts52.sh report`.
   See "Where this stands" for the curves.
   The pre-registered response applies and has been taken: the block stack stops being a generative parameterisation.
   One byproduct worth recording: `centre_error` falls monotonically, 1.674 to 1.441 Angstrom, while `translation_error_labelled` sits at 3.80 to 3.82 for all 1200 epochs.
   The metric fault diagnosed above reproduces itself live, on real data, at exactly the predicted `0.45 * V ** (1/3)` value.
3. **The untried levers on the block stack are not being spent.**
   The minimum permutation distance coupling, depth beyond twelve without the rotation objective, and a centres-before-cell warmup would each be worth a point or two against a 9.8-point deficit on a curve that has been flat for 600 epochs.
   That is the wrong place to spend the remaining compute.
4. **Motif tokens, handed off.**
   Experiment A is done and cleared its gate; the vocabulary carries information well beyond coordination number and `K = 32` is the largest size beating CrystalNN's stability half-life.
   The 200-epoch oracle pilot is done and positive but small, at +1.26 points.
   Experiment B, the six-arm oracle ceiling at 1600 epochs, is packaged for the A100 box: `cgfm/scripts/package_motif_bundle.sh` builds it, `cgfm/scripts/run_motif_expB.sh launch` runs it.
   Experiment C stays gated on B.
   The ceiling is measured before the mechanism this time, which is the procedural lesson from items 2 and 3: both failures were diagnosable months earlier by an oracle run costing one arm.
5. **Independent of the outcome**, the conditioning programme in `scripts/enumerate_thio.py` and the `XS4_TETRAHEDRA` block rule do not depend on beating the benchmark.
   A block-level composition such as eight `GeS4` to three `PS4` is a native input to this representation and is not expressible for an atomwise model at all, and the target family has 50 to 70 atom cells where the degree-of-freedom argument starts to hold.
6. **If none of it works**, the presentation rests on four complete results: the representation is worth 51 points over correct centres alone on unseen data with a mild tolerance curve; the multiscale premise fails and the spectral study says why, in under two octaves of usable frequency range against eight for images; the rigid-pose parameterisation fails too, by 9.8 points at 1200 epochs, with the binding constraint identified as centre placement rather than the vocabulary; and two widely-read diagnostics in this project were measuring gauge rather than structure, which is worth reporting on its own.



## The spectral study: is a crystal dataset like an image dataset?

`scripts/spectra.py`, `spectra.py`, and `reports/spectra/`.
The motivating analogy is that natural images have power spectra falling as `f ** -alpha` with `alpha` near 2, Gaussian noise is flat in frequency, so noising destroys high frequencies first and a diffusion model is roughly predicting the next frequency band given the preceding ones.

The analogy has an exact form here rather than a loose one.
Adding iid Gaussian noise of variance `sigma ** 2` to atomic positions multiplies the structure factor by `exp(-2 pi ** 2 sigma ** 2 |k| ** 2)`, which is the Debye-Waller factor of crystallography.
Noising coordinates *is* a low-pass filter in reciprocal space, with a closed form and a resolution cutoff near `1 / (2 pi sigma)`.
The closed forms were checked against Monte Carlo on real cells and agree in variance to within 4.6 per cent on MP-20 and 3.9 per cent on MPTS-52, so the readings below are the interpolant's, not an approximation of it.

Four findings.
The second is the one that reframes the project, and the fourth is the one that is separately fixable.

**(1) There is no cascade to exploit.**
Usable frequency range is 1.51 octaves on MP-20 and 2.00 on MPTS-52, against 8 for a 256-pixel image.
A power law fitted over two octaves is not a power law, and the fitted exponents behave accordingly: `-2.8` with `r ** 2` 0.87 on MP-20 and `-30` with `r ** 2` 0.53 on MPTS-52.

**(2) A crystal's spectrum is blue, and an image's is red.**
This is the finding.
Normalising `|F| ** 2` by the atomic form factors gives `|E| ** 2`, which equals 1.0 in every shell when atoms are placed independently, so it reads directly as how much structure there is relative to random.
Median `|E| ** 2` on MPTS-52 runs 0.000, 0.000, 0.009, 0.074, 0.113, 0.340, 0.336 from the longest wavelength shell to the bond scale.
Long-wavelength content is two orders of magnitude *below* random.
Two mechanisms, both structural: a crystal is a dense nearly incompressible packing, so long-wavelength density fluctuations are the one thing it cannot have, and symmetry forbids a large share of the low-order reflections outright — 66 per cent of them are exactly zero in the coarsest MPTS-52 shell.
Given the composition and the cell, the coarse scales are close to determined and essentially all of the uncertainty sits at the bond scale.

**(3) The signal-to-noise ordering is not coarse-to-fine, and changing the prior cannot make it so.**
The coarsest populated shell never leads in per-shell SNR, at any time, under either OMatG's prior or a Gaussian one.
The lead sits at `|G|` of 0.17 to 0.30 inverse Angstrom, which is 3 to 6 Angstrom, the coordination-shell scale, and it barely moves along the trajectory.
The filter does favour low frequencies, monotonically and provably so for a Gaussian prior, but by finding (2) the data's content there is two orders of magnitude below random, so the signal deficit beats the filter advantage.
The SNR profile is a hump in the middle of the accessible range, not a ramp.

**(4) OMatG's prior has resonant dead zones, which is a real bug and is fixable.**
The position prior is uniform on the torus, so the factor applied to reflection `(h, k, l)` is exactly a product of three sincs, and a sinc has zeros: the reflection is annihilated whenever `(1 - t) h`, `(1 - t) k` or `(1 - t) l` is a non-zero integer.
At `t = 0.5` every even index dies outright while `(3, 0, 0)` passes through with 21 per cent of its amplitude, above lower-frequency neighbours that are gone.
Measured depth of the `t = 0.5` dead zone, as the factor by which the worst shell falls below the interpolation of its neighbours: **5.7e30 on MP-20 and 7.9e30 on MPTS-52, against 1.1 and 1.0 for a Gaussian reference.**
Whole families of reflections are switched off and back on mid-trajectory at times the sampler walks straight through.
A Gaussian prior gives the Debye-Waller factor, which is strictly positive everywhere, and removes this outright.
Worth doing on its own terms, but by (3) it will not produce a coarse-to-fine ordering, because there is not one to get.

**What the noise destroys, in direct space.**
Half-life is the RMS displacement at which half the recoverable signal is gone, measured against each featurisation's own fully-randomised floor.
On MPTS-52: `g(r)` 0.24 Angstrom, CrystalNN 0.56, powder pattern 0.93, and the compositional control never.
Everything geometric dies inside a 3.9x band of displacement that sits *below* the 2.2 Angstrom bond scale rather than straddling it, which is the direct-space statement of (1).

**What this says about featurisation, which is the constructive half.**
Four consequences follow from (2) and (3) rather than from taste:

- A featurisation should carry `|E| ** 2` rather than `|F| ** 2`, because dividing out the form factors is what turns a spectrum dominated by which elements are present into one about where they are.
  Every reading above changed character under that normalisation.
- Systematic absences are labels, not zeros.
  Two thirds of the coarsest MPTS-52 reflections are exactly zero by symmetry, so a model that spends capacity predicting them is fitting a deterministic consequence of the space group.
  The absence *pattern* identifies the space group, which is worth predicting; the amplitudes are not.
- Resolution shells are the wrong bins for the informative region.
  All of the SNR sits in one or two shells around 3 to 6 Angstrom, so a featurisation with equal weight per octave puts almost all of its resolution where `|E| ** 2` is 0.002.
  Bin where the variance is: finely across the coordination-shell band, coarsely below it.
- A useful feature has to survive tenths of an Angstrom, because that is the band where all the geometric information lives.
  `g(r)` at a 0.1 Angstrom bin dies at 0.24 Angstrom, which is bin width rather than physics; the powder pattern survives to 0.93 because it integrates over orientation.
  Angular quantities that are stable to bond-length noise — which polyhedron, how the polyhedra connect — are the ones that hold their signal across that band, and they are what the block vocabulary already encodes.



## Diagnostics available

All of these read an existing checkpoint and train nothing.
`cgfm/scripts/diagnose_rotations.py --checks`:

`templates`, `floor`, `target`, `consensus`, `cell` (does the lattice reach its target along the trajectory), `matcher` (what the tolerances accept when only centres are right), `tolerance` (required centre accuracy), `attribution` (which structures a match rate comes from), `oracle_mix` (which degree of freedom is binding), `gradients`, `solo` (each field trained alone), `integrate`, `heads`.

`--limit` takes the first N structures of a split, which is what makes the validation-set versions affordable.

## Code

- `cgfm/blocks.py`, `cgfm/so3.py`, `cgfm/readout.py`: representation, group theory, consensus readout.
  `blocks.py` holds both centre rules, `free_elements`, and `block_specification`, which is the composition-only conditioning interface.
- `cgfm/spectra.py`: structure factors, Wilson normalisation, both interpolant filters, and the SNR decomposition.
- `cgfm/block_si.py`, `cgfm/block_sampler.py`, `cgfm/block_encoder.py`, `cgfm/block_lightning.py`, `cgfm/blockdata.py`: the training stack.
  `BlockCSPNet` takes `use_rotation_input`, which blinds the trunk to the orientation.
- `cgfm/scripts/`: `precompute_blocks.py`, `block_ceiling.py`, `diagnose_rotations.py`, `check_overfit_gate.py`, `run_gates_mpts52.sh`, `sweep_block_gate.sh` (`objective`, `optimiser`, `translation`), `run_unattended.sh`, `run_centres_mpts52.sh` (`blocks`, `blocks-oracle`, `atomwise`, `report`), `spectra.py`, `enumerate_thio.py`.
- `cgfm/EXPLAINER.md`: the pipeline and architecture written for a machine learning audience with no materials background, including the honest scorecard and both corrected metrics.
- `cgfm/tests/`: 362 tests, all passing.
- `cgfm/reports/`: measurement output, including `readout_g1_coarse.json`, `away/summary.txt` and `oracle_mix_production.log`.
