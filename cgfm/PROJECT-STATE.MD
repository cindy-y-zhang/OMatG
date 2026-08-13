# Project state

Last updated: 2026-08-12, after the Stage 1 go/no-go sweep and the rigid-block ceiling measurement.

## Where this stands

The Stage 1 sweep ran and the answer is no. No coarse-to-fine arm beat the atomwise baseline on MP-20 at any value of
`eta`, so the kill criterion decided in advance forbids spending the MPTS-52 budget. The reason is now understood, and it
is not a tuning failure: the path reparameterization cannot express the hypothesis it was built to test.

A replacement was proposed, measured and cleared its first test. Representing each coordination polyhedron as a rigid
body with a position and an orientation reconstructs 98.7 per cent of MP-20 validation structures, against the 58.3 per
cent that is the best any arm has achieved. The representation is therefore not the bottleneck, and the direction is
worth building. Nothing of it is built yet.

## Research question, and what the sweep answered

*Does a learned partition of a crystal into coordination-scale atom groups define a better flow-matching probability path
than standard atomwise periodic interpolation?*

No. Four arms on composition-conditioned MP-20 structure prediction, identical in everything but the path for fractional
coordinates, 400 epochs, seed 0. Validation match rate in per cent:

| arm | eta 0 | eta 0.25 | eta 0.5 | eta 0.75 |
| --- | --- | --- | --- | --- |
| atomwise | **56.15** | | | |
| k-medoids | | 56.00 | 53.44 | 48.23 |
| coordination shells | | 51.81 | 50.99 | 49.59 |
| learned | | 52.69 | 50.61 | 5.23 |

Every coarse-grained arm ties with or loses to atomwise, and the loss grows with `eta`, which is the strength of the
coarse-to-fine bump. The learned arm at `eta = 0.75` collapsed outright and never recovered. The k-medoids `eta = 0.75`
run ended at epoch 299 rather than 399; it was already 8 points behind.

One pattern is worth keeping. Six of the nine coarse-grained runs *led* at epoch 49, the learned arms by 2.8 and 2.5
points, and every lead was gone by convergence. That is the signature of a path that is easier to optimise early rather
than one that models the data better, which is what the flow-matching literature reports for conditional path
reparameterizations generally: they buy sampling efficiency, not converged quality.

## Why it failed

Four reasons, in order of how much they explain.

**A partition cannot hold a coordination environment.** Polyhedra share atoms; an anion belongs to several cation
polyhedra at once, and that sharing is the structural content. `cgfm/shells.py` had to destroy it to produce a partition,
and its own docstring records the cost. The MP-20 group files hold 2.75 atoms per group with 33 per cent singletons for
k-medoids and 36 per cent for coordination shells, against the five-atom shells the specification assumed. The shells arm
was therefore not testing coordination environments; it was testing small geometric neighbourhoods, which is what
k-medoids already tests. That explains why the two fixed arms behave alike.

**The group never becomes a unit with a pose.** Per atom the path reads
`x_t = x_0 + (t + eta s(t)) d_g + (t - eta s(t)) delta_i`, where `d_g` is the group's mean displacement and `delta_i` the
within-group residual. The centroid component does run ahead of the residual, so the schedule story is real, but the
motif's shape is interpolated out of noise the whole way. There is no time at which a correct polyhedron exists and is
carried to a site, no orientation variable, and no template. The intention, place the group and then its atoms, is not
what the algebra does.

**Nothing in the model knows about symmetry.** The motivation is stated in terms of Wyckoff sites, but no space group,
orbit or multiplicity appears anywhere. A geometric cluster's centroid has no reason to sit at a special position.

**The bias never reaches generation.** Because the bump vanishes at both endpoints, the modelled distribution is
identical for every `eta`. The grouping network reads the clean target, which does not exist at sampling time, so it
contributes nothing to generation. The inductive bias entered only as a modified regression target that the denoiser had
to learn to infer.

## Ruled out: the sampler

Before accepting the result, the obvious confound was tested. Generation multiplies predicted velocities by
`(1 + factor * t)`, and that factor was tuned for the atomwise path, so a bumped path might simply have been sampled
wrongly. The factor was exposed per arm and swept from 2 to 50 on the existing checkpoints, no retraining.

Retuning lifted every arm and changed no ordering. On the full validation split the best figures are atomwise 58.31 at
factor 40, k-medoids `eta = 0.25` 57.44 at factor 32, and k-medoids `eta = 0.5` 54.78 at factor 25. The gains also cost
geometric precision: match rate keeps climbing past the point where root-mean-square displacement starts to worsen, so
the high factors are overshooting into the tolerance rather than placing atoms better. The comparison was not confounded.

## The replacement: rigid blocks on SE(3)

Both of the first two failures come from the same source, a hard partition of Euclidean coordinates. Dropping it gives a
representation in which the motif hypothesis is actually expressible.

- Blocks **overlap**. A structure is a set of cation-centred coordination polyhedra, and an atom may be a vertex of
  several. Sharing is kept rather than resolved away.
- A block has a **pose**. It carries a rigid template for its type, a position and an orientation, so placing a motif is
  one move. This is the parameterization behind MOFFlow and MolCrystalFlow, which cut `3N` coordinates to `6M` pose
  parameters and beat atom-level baselines on metal-organic and molecular crystals.
- Atom positions are the **consensus** of the blocks claiming them.

The difference from those two precedents is that their building blocks are disjoint and given, while inorganic polyhedra
share atoms. Which blocks share which atoms therefore has to be generated, and that is the part with no published
precedent for inorganic crystals.

### Measured ceiling

`cgfm/scripts/block_ceiling.py` gives away everything a model would have to predict: every polyhedron is replaced by the
canonical template of its type, placed at its true centre with the best-fitting rotation, and shared atoms are averaged.
What remains is only what the parameterization cannot hold. 1500 validation structures, templates fitted on the training
split so nothing is scored against a template fitted to it.

| pose error (translation / rotation) | match rate | per-atom displacement |
| --- | --- | --- |
| oracle | **98.73%** | 0.116 A |
| 0.05 A / 2 deg | 98.47% | 0.151 A |
| 0.10 A / 5 deg | 98.13% | 0.198 A |
| 0.20 A / 10 deg | 97.73% | 0.306 A |
| 0.40 A / 20 deg | 90.73% | 0.527 A |

Three readings.

**The representation is not the limit.** 98.7 per cent against 58.3 per cent achieved leaves about 40 points unclaimed,
so the binding constraint is pose prediction, not expressiveness.

**Pose precision is affordable.** The curve is flat out to 0.2 Angstrom and 10 degrees. A model does not need extreme
accuracy to reach 97 per cent.

**Sharing is what makes it work, and this was not expected.** Polyhedra are floppy: an instance sits 0.476 Angstrom from
its type template on average, far too much to rebuild a structure from any one of them. But each atom is a vertex of 4.5
blocks, 66 per cent of atoms are shared, and averaging their disagreeing predictions cancels most of the individual
distortion, leaving 0.116 Angstrom. The overlap that made a partition impossible is precisely what makes rigid templates
accurate, because the system is over-determined.

### What the measurement also showed, less comfortably

**There is no dimensionality saving on MP-20.** MOFFlow's case rests on `6M` being far below `3N`, which holds for cells
of hundreds of atoms. MP-20 averages 10.2 atoms in 5.7 heavily shared polyhedra, so the block parameterization carries
*more* degrees of freedom than the atomwise one, a factor of 0.92. Whatever rigid blocks buy here is the inductive bias
and the over-determined consensus, not a smaller search space. If the dimensionality is the point, the experiment belongs
on MPTS-52 or larger cells.

**The type vocabulary does not generalise as keyed.** Central element, coordination number and vertex composition gives
2561 types for 1500 structures, and 40 per cent of blocks need the composition-blind fallback because their exact type
never appears in training. The generative vocabulary should be central element and coordination number, with vertex
composition predicted separately.

**A symmetric polyhedron has no unique pose.** An octahedron has 24 rotational symmetries, so the regression target for
its orientation is a set, not a point. The SO(3) loss has to be minimised over the template's point group, or the target
canonicalised. Training against an arbitrary representative would inject noise the size of a whole rotation. This came
out of a unit test rather than the literature.

## Decisions that departed from the original specification

Each is a judgement call worth revisiting.

**Polyhedron centres are chosen from the composition alone.** The first implementation picked centres by bond-valence
analysis of the structure, which is circular: the number of centres is the number of poses a model must emit, and at
inference there is only the composition. Centres now come from a charge-balanced assignment of oxidation states where one
exists (55 per cent of MP-20), from electronegativity ordering where it does not (19 per cent), and otherwise every atom
becomes a centre (27 per cent, mostly alloys and elemental structures). The rule refuses to split a composition whose
electronegativity gap is under 0.4, because naming the marginally less electronegative element of an alloy its cation
puts the centres somewhere arbitrary. Making the rule honest cost nothing: the ceiling moved from 98.80 to 98.73 per
cent. CrystalNN is still used to find which atoms are a polyhedron's *vertices*, which is not circular, because that only
ever runs on a known training structure to build a target.

**Group count came from chemistry, not from `round(N / 5)`.** Coordination shells decided how many groups a structure
has and the other arms were given that number. This is why groups hold 2.75 atoms rather than five: groups are sets of
atoms in the cell, so neighbours that are periodic images of one site count once. Forcing five would have stopped the
shells arm testing coordination shells. In hindsight this was the moment the arm lost its hypothesis, and the fix was not
a different group count but abandoning the partition.

**Overlapping shells were resolved by a dominating set, not by peeling.** Peeling whole shells gave one large polyhedron
and more than half singletons. Greedy dominating-set centres with every other atom joining the nearest centre that bonds
it brought singletons from 55 to 33 per cent.

**Soft assignments, annealed, instead of straight-through Gumbel.** Endpoint exactness needs only the bump to vanish at
`t = 0` and `t = 1`, not the group-mean operator to be a projector, so no hard partition is required during training. A
tempered softmax is differentiable everywhere. Anchor selection stays discrete, with gradient reaching the scorer through
the gating trick from top-k graph pooling.

**Explicit mean squared error instead of OMatG's loss.** OMatG drops the constant `E||u_t||^2`, which is correct when the
target is fixed. It is not fixed once the grouping is learned: without it the grouping network minimises `-2 E[b . u_t]`,
unbounded below and optimised by inflating the target rather than by finding better groups. The arms use
`E||b - u_t||^2`, which has identical gradients with respect to the denoiser.

**Multi-sample metrics are computed here, not by `omg csp_metrics`.** `match_rmsds` and `metre_rmsds` both raise when the
generated list is longer than the reference list, and OMatG's METRe rate measures precision, so it would fall as sampling
grew more diverse. `scripts/score.py` computes a genuine best-of-n rate on the same validation and the same
`StructureMatcher` call.

**Hyperparameters are OMatG's MP-20 values.** No tuned MPTS-52 configuration ships with OMatG. All arms are biased
identically so the comparison holds, but the absolute numbers are not tuned MPTS-52 results.

## Open items

1. **The shells arm's runtime diagnostic contradicts its own group file.** The MP-20 shells partition holds 36 per cent
   singletons, and the k-medoids arm's logged `cg_singleton_fraction` of 0.305 matches its file's 33 per cent. The shells
   arm logged 0.001. Either the diagnostic is computed differently for that arm or the runs were fed a different file, so
   the shells results should not be quoted until this is resolved.
2. **Temperature annealing was scheduled over 1000 epochs and Stage 1 ran for 400.** The learned arms never reached a
   hard partition, so their adjusted Rand index of 0.17 against both fixed partitions describes a soft assignment rather
   than a partition that could be named.
3. **The low-step-count claim is untested.** Since the coarse arms led early and reparameterizations are known to buy
   sampling efficiency, `nfe_curve.sh` on the existing checkpoints could still support a defensible result about cheap
   sampling. It needs no training.
4. **CrystalNN returns up to 24 vertices for some sites**, which is not a coordination polyhedron. Worth a cap or a
   different neighbour rule before Stage A is built.

## Kill criteria and their outcome

Decided in advance so a null result stays reportable.

- *If no `eta` beats atomwise on MP-20, do not scale to MPTS-52.* **Triggered.** Best coarse-grained arm is k-medoids
  `eta = 0.25` at 56.00 against 56.15, a tie, and everything else is worse. MPTS-52 was not launched.
- *If `cg_ari_vs_kmedoids` exceeds roughly 0.8, the learned arm has rediscovered geometric clustering.* Not triggered; it
  sat at 0.17, but see open item 2 before reading that as a finding.
- *If `cg_singleton_fraction` approaches one, the grouping has collapsed to atomwise.* Not triggered, though the learned
  `eta = 0.75` run reached 0.59 as it collapsed.
- *If the learned arm does not beat atomwise, the interesting comparison becomes shells against atomwise: whether
  coordination motifs help when supplied, even if they cannot be discovered.* Shells lost too, and the ceiling
  measurement explains why: what was supplied was not coordination motifs.

## Next actions

1. Build Stage A, the discrete model: coordination number and environment type per centre, and the sharing graph saying
   which anions are vertices of which polyhedra. The block count follows from the composition, so only the graph is
   generated. This is the contribution and the risk.
2. Build Stage B, the continuous model: flow matching for block translations on the torus and orientations on SO(3), plus
   the lattice, with atom positions read out by consensus so the sharing constraints hold by construction. Minimise the
   orientation loss over each template's point group.
3. Repeat the ceiling measurement on MPTS-52, where the dimensionality argument may hold, before committing to MP-20 as
   the testbed.
4. Optionally, run `nfe_curve.sh` on the existing checkpoints to settle open item 3, which is the only claim the
   completed sweep can still support.

## Code added for the replacement

- `cgfm/blocks.py`: overlapping polyhedral decomposition, composition-only centre rule, species-restricted Kabsch
  alignment with exact enumeration or assignment fallback, per-type template fitting, consensus reconstruction.
- `cgfm/scripts/block_ceiling.py`: the ceiling and pose-sensitivity measurement.
- `cgfm/tests/test_blocks.py`: 21 tests. Full `cgfm` suite is 84 tests, all passing.
- `cgfm/reports/`: the measurement output.
