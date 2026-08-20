# Joint geometry state

This package adds a compact per-site radial environment as a fourth OMatG
stochastic-interpolant field. The deployable sampler starts that field from an
independent standard Gaussian and predicts/integrates it together with
positions and cells; it never reads a clean descriptor at inference.

## Fixed arms

- `S`: stock atomwise benchmark.
- `D`: selected distance-readable backbone, without geometry state.
- `R`: radial descriptor recomputed from the current noisy structure.
- `H`: geometry head and loss, with state-to-structure input disabled.
- `O`: leaked perfect geometry path, used only as an oracle ceiling.
- `P`: carried state whose clean endpoints are shuffled within element.
- `J`: carried state with real CN-RDF endpoints.

`J-D` is the deployable architectural contrast, `J-H` isolates trajectory
feedback, and `J-P` isolates descriptor content.

## Reproducible stages

```bash
# Build all nested descriptors.
for representation in cn-rdf4 cn-rdf8 radial17; do
  .venv/bin/python -m joint_geometry.scripts.build_descriptor \
    --representation "$representation" \
    --out-dir "joint_geometry/artifacts/mpts_52/$representation"
done

# Run descriptor compression/information and backbone readability gates.
.venv/bin/python -m joint_geometry.scripts.probe_descriptor
.venv/bin/python -m joint_geometry.scripts.probe_backbone

# Execute sequential local stop gates.
bash joint_geometry/scripts/run_local.sh all
```

The local launcher uses paired sampler priors, writes one outcome per
composition, calibrates the descriptor-loss coefficient once, and refuses to
continue when an earlier gate fails. Generated artifacts and runs are not
scientific conclusions; the JSON reports in `joint_geometry/reports/` are the
evidence trail.
