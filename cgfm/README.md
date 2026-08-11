# Coarse-to-fine flow matching for crystal structure prediction

Does a partition of a crystal into coordination-scale atom groups define a better flow-matching probability path than
standard atomwise periodic interpolation, and can a useful partition be learned end to end?

This package adds one probability path to [OMatG](https://github.com/FERMat-ML/OMatG) and nothing else. The denoiser,
sampler, optimiser, schedule and data are OMatG's, unchanged, so any difference between arms is attributable to the path.

## The path

Let `d = Log(x_0 -> x_1)` be the periodic displacement from the base sample to the data sample, that is, the
minimum-image separation OMatG's periodic corrector already computes. Given a partition of the structure's atoms, let
`S` be the operator that replaces each atom's value by its group's mean, and

```
Delta = (2 S - I) d,   centred so that its mean over each structure is zero.
```

The path and its conditional velocity are

```
x_t = wrap( (1 - t) x_0 + t x_1 + eta s(t) Delta ),
u_t = (x_1 - x_0) + eta s'(t) Delta,        s(t) = t^p (1 - t)^q.
```

Since `s(0) = s(1) = 0`, the marginals at both endpoints are exactly the baseline's. This is a reparameterisation of the
conditional path, not a change of the modelled distribution. At `eta = 0` the added term is exactly the float zero, so
the baseline is reproduced bit for bit.

For any hard partition `S` is a projector, and the path is then algebraically identical to running the group centroids
on the schedule `a(t) = t + eta s(t)` and the within-group geometry on `b(t) = t - eta s(t)`. Coordination-scale
arrangement resolves before internal geometry. Both schedules are checked for monotonicity at construction, which for
the default bump means `eta < 1`.

## The four arms


| arm        | grouping                      | eta | what it isolates           |
| ---------- | ----------------------------- | --- | -------------------------- |
| `atomwise` | none                          | 0   | the OMatG baseline         |
| `kmedoids` | periodic geometric clusters   | 0.5 | coarse-graining as such    |
| `shells`   | CrystalNN coordination shells | 0.5 | the hypothesis, supplied   |
| `learned`  | anchor-and-membership network | 0.5 | the hypothesis, discovered |


All four instantiate `cgfm.arms.ArmStochasticInterpolants`, which builds every interpolant itself and exposes only the
grouping and `eta`. Selecting an arm therefore cannot change anything else. The coordination shells set the number of
groups per structure, and the other coarse-grained arms are given that same number, so the arms differ in *which* atoms
are grouped and not in *how much* the structure is compressed.

## Running it

Everything is run from the OMatG repository root. Start with the smoke test: it builds a 64-structure split, precomputes
partitions on it, trains all four arms for two epochs, checks the collapse diagnostics were logged, then generates two
draws and scores them. A couple of minutes on a laptop CPU.

```bash
cgfm/scripts/smoke.sh
```



### Stage 1, the go/no-go

Before spending the MPTS-52 budget, check that any coarse-to-fine path beats atomwise at all. This sweeps
`eta` over `{0, 0.25, 0.5, 0.75}` with the k-medoids arm on MP-20 at a fifth of the full schedule, precomputing its
partitions if needed. `eta = 0` runs the atomwise arm, since a grouping has no effect there. Metrics are logged locally
and to the `cindyz/cgfm-s1` Weights & Biases project.

```bash
cgfm/scripts/sweep_eta.sh kmedoids  # eta = 0, 0.25, 0.5, 0.75
cgfm/scripts/sweep_eta.sh learned   # eta = 0.25, 0.5, 0.75; reuses the atomwise result above
```

If no `eta` beats atomwise, stop.

### Stages 2 to 5, the full experiment

```bash
# Precompute the fixed partitions once per split. Both fixed arms come out of one pass, because they must agree on
# the number of groups. About four minutes at 16 workers for the 27,380 training structures.
for split in train val; do
    python -m cgfm.scripts.precompute_groups \
        --data omg/data/mpts_52/${split}.lmdb --out-dir cgfm/groups/mpts_52 --workers 16
done

# Train. Twelve jobs: four arms at three seeds.
cgfm/scripts/launch_all.sh plan     # see the jobs
cgfm/scripts/launch_all.sh local    # run them, one per visible GPU
cgfm/scripts/launch_all.sh slurm    # or submit an sbatch array

# Generate and score. "auto" finds the run's best_match_rate checkpoint. Five draws by default.
cgfm/scripts/evaluate.sh shells 0 auto

# The results-versus-cost curve, one draw at each of several step counts.
cgfm/scripts/nfe_curve.sh shells 0

# Collect.
python -m cgfm.scripts.collect_results --runs runs --nfe 210
python -m cgfm.scripts.analyse_groups \
    --checkpoint $(find runs/learned/seed0 -name 'best_match_rate*.ckpt' | head -1) \
    --data omg/data/mpts_52/val.lmdb --groups cgfm/groups/mpts_52 --split val
```



## Reading the output

Training losses are **not** comparable across arms. Different paths have different conditional velocity distributions,
so a lower loss says nothing about a better model. Arms are compared on test match rate and corrected RMSD only, and
checkpoints are selected on validation match rate.

`score.json` reports three things per run, plus each of them split into the atom-count bins `1-10`, `11-20`, `21-36`
and `37-52`:


| metric           | meaning                                                                               |
| ---------------- | ------------------------------------------------------------------------------------- |
| `one_shot`       | one structure per composition, matched to its own reference. The headline comparison. |
| `one_shot_metre` | the same draw, but each structure may match any reference. Polymorph-aware.           |
| `best_of_n`      | a composition counts as solved if any of the five draws matches it.                   |


The size-binned result is the one that separates "useful intermediate resolution" from "just a different schedule". At
small atom counts the group count falls to one, where the coarse component is a pure global translation that the
centre-of-mass correction already removes, so a gain that grows with atom count is the mechanistically expected
signature.

The learned arm needs one further check. The flow-matching loss can be lowered by shrinking the within-group residual
rather than by finding better groups, and at a fixed group count the partition that minimises that residual is plain
geometric clustering. The diagnostics logged to `metrics.csv` are what distinguishes success from that collapse:


| metric                    | what it says                                                             |
| ------------------------- | ------------------------------------------------------------------------ |
| `cg_ari_vs_kmedoids`      | agreement with geometric clustering; climbing towards one means collapse |
| `cg_ari_vs_shells`        | agreement with coordination shells; this is the research question        |
| `cg_fine_energy_fraction` | `                                                                        |
| `cg_singleton_fraction`   | a partition of mostly singletons is not a coarse-graining                |
| `cg_group_extent_mean`    | group diameter in Angstrom; coordination scale is a few Angstrom         |
| `cg_assignment_entropy`   | how far the soft assignment is from a real partition                     |
| `cg_temperature`          | where the annealing schedule currently is                                |


Both adjusted Rand indices are reported for every arm, and `analyse_groups.py` additionally reports the index between
the two fixed partitions, which calibrates the scale: it says how far apart two reasonable partitions of the same
structures already are.

## Layout


| file                       | contents                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `blur.py`                  | the group-mean operator and the centred `Delta`                                          |
| `interpolant.py`           | the path, its velocity and its loss                                                      |
| `grouping.py`              | the grouping interface and the precomputed grouping                                      |
| `grouper.py`               | the learned anchor-and-membership network                                                |
| `graph.py`                 | dense periodic neighbourhoods for the network                                            |
| `kmedoids.py`, `shells.py` | the two fixed partitions                                                                 |
| `groupfile.py`, `data.py`  | the partition file format and its route into a batch                                     |
| `arms.py`                  | the four arms, built so they can only differ in the path                                 |
| `lightning.py`, `main.py`  | registering the grouping network, and the command line                                   |
| `diagnostics.py`           | collapse diagnostics and temperature annealing                                           |
| `configs/`                 | the shared configuration, one overlay per arm, two for the sweep, one for the smoke test |
| `scripts/`                 | precompute, sweep, launch, evaluate, score, collect, analyse, smoke                      |
| `tests/`                   | 60 tests, including that `eta = 0` is bit-for-bit the baseline                           |




## Changes to OMatG itself

Two, both in `omg/si/`, together about fifteen lines. OMatG's interpolants act elementwise on one data field, but a path
that couples atoms of the same structure needs the rest of the structure. `StochasticInterpolant` gains a
`requires_aux` class attribute, default `False`, and `StochasticInterpolants` passes `aux=x_1` to `interpolate` and
`loss` only for interpolants that set it. Every interpolant shipped with OMatG keeps its current signature and
behaviour.

## Why the metrics are computed here rather than by `omg csp_metrics`

The one-shot numbers do come from OMatG, through the same `match_rmsds` and `metre_rmsds` it uses. The multi-sample ones
cannot: both functions raise if the generated list is longer than the reference list, so five samples per composition
cannot be passed at all. Its METRe rate is also the fraction of *generated* structures matching some reference, which is
a precision measure, so feeding it more samples would make it fall as the model became more diverse. `scripts/score.py`
therefore builds the best-of-n rate on the same `ValidAtoms` validation and the same pymatgen `StructureMatcher` call,
so the two sit on comparable footings.

## Known limitations

- Hyperparameters come from OMatG's tuned MP-20 crystal-structure-prediction configuration, not from an MPTS-52 sweep.
All arms are biased identically, so the comparison holds, but the absolute numbers are not tuned MPTS-52 results.
- Only ODE training without a latent variable is supported. The antithetic and score-based branches would each need
their own coarse-to-fine derivation.
- The CrystalNN shells give about 2.8 atoms per group on MPTS-52, not the 5 the original specification assumed. Since
that group count is shared across arms, every coarse-grained arm inherits it. If the coarse-graining turns out too
fine to matter, this is the first thing to revisit.
- Stage 0 of the plan, reproducing the published MPTS-52 match rate from the released checkpoint, has not been done.
Nothing here depends on it, but it is the only external reference number.

