# Weights & Biases logging, shared by the local and A100 launchers so the two cannot drift apart.
#
# Sourced, not executed. Sets WANDB_ARGS to the extra `omg fit` arguments that add a wandb logger, and WANDB_MODE to
# online or offline depending on whether this machine actually has credentials.
#
# WHY THE CSV LOGGER STAYS
#
# wandb is added *beside* the CSVLogger the baseline config pins, never instead of it. Two reasons, both load-bearing:
#
#   - every report in this package reads `metrics.csv`, and so does the DG3 gate. A run whose numbers live only in a
#     remote service is a run whose numbers cannot be re-read offline, hashed, or shipped in the bundle;
#   - Lightning resolves both `trainer.log_dir` and ModelCheckpoint's directory from `loggers[0]` alone. Keeping the
#     CSV logger first is what keeps checkpoints at `<run>/checkpoints/` and `metrics.csv` at `<run>/metrics.csv`.
#     Appending wandb in front of it would silently relocate every checkpoint the evaluation scripts look for.
#
# WHY THE ARGUMENTS MUST COME LAST
#
# jsonargparse applies `--trainer.logger.init_args.*` to the most recently named logger. The launchers set the CSV
# logger's save_dir, name and version first; everything WANDB_ARGS adds is applied after the append and therefore lands
# on wandb. Emitting these earlier would send the CSV logger's directory to wandb and leave metrics.csv in the repository
# root.
#
# WHY OFFLINE IS DETECTED RATHER THAN ASSUMED
#
# Without credentials `wandb.init` can block on an interactive prompt. One prompt would hang an unattended twelve-run
# sweep for as long as the job is allowed to live, so a machine without credentials is put into offline mode instead and
# told how to sync afterwards. This matters on the A100 node, which is a different machine from the one that packed the
# bundle.

DG_WANDB_PROJECT="${DG_WANDB_PROJECT:-geo-adapter}"

# Resolve online or offline once, so that every run in a sweep agrees and the decision is visible in the log.
wandb_resolve_mode() {
    if [[ "${DG_WANDB:-1}" != "1" ]]; then
        WANDB_RESOLVED_MODE="disabled"
    elif [[ -n "${WANDB_API_KEY:-}" ]] || grep -qs "api.wandb.ai" "${NETRC:-${HOME}/.netrc}"; then
        WANDB_RESOLVED_MODE="online"
    else
        WANDB_RESOLVED_MODE="offline"
    fi
    export WANDB_RESOLVED_MODE
}

# Announce the decision. Called once per launcher invocation rather than per run.
wandb_announce() {
    case "${WANDB_RESOLVED_MODE}" in
        disabled) echo "$(date -u +%H:%M) wandb logging off (DG_WANDB=0); metrics.csv is still written per run" ;;
        online)   echo "$(date -u +%H:%M) wandb project '${DG_WANDB_PROJECT}', runs auto-named, group in the UI by the" \
                       "logged config keys" ;;
        offline)  echo "$(date -u +%H:%M) no wandb credentials on this machine, so logging offline to <run>/wandb." \
                       "Sync later with: wandb sync <run>/wandb/offline-run-*" ;;
    esac
}

# Build the arguments for one run, into the WANDB_ARGS array.
#
# $1 run directory, $2 run mode, $3 arm, $4 message graph, $5 feature mode, $6 seed, $7 overlay config.
#
# Only the identity the launcher itself owns is logged. Epoch counts, batch sizes and accumulation live in the overlay
# configs, so restating them here would create a second copy to keep in step; the overlay's name is logged instead and
# pins all of them. Nothing in this project should have to trust a hand-copied number.
#
# Runs are deliberately left unnamed: wandb generates its own, and the identity that matters is in the config keys, which
# is what the wandb UI groups on. The tags carry the same identity again for filtering.
wandb_arguments() {
    WANDB_ARGS=()
    [[ "${WANDB_RESOLVED_MODE}" == "disabled" ]] && return 0

    local run_dir="$1" mode="$2" arm="$3" graph="$4" features="$5" seed="$6" overlay="$7"
    local config
    # Flat keys, because the wandb UI groups on a key and not on a path.
    printf -v config \
        '{"run_mode":"%s","arm":"%s","message_graph":"%s","feature_mode":"%s","seed":%s,"overlay":"%s","commit":"%s"}' \
        "${mode}" "${arm}" "${graph}" "${features}" "${seed}" "$(basename "${overlay}")" \
        "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

    WANDB_ARGS=(--trainer.logger+=lightning.pytorch.loggers.WandbLogger
                "--trainer.logger.init_args.project=${DG_WANDB_PROJECT}"
                "--trainer.logger.init_args.save_dir=${run_dir}"
                "--trainer.logger.init_args.config=${config}"
                "--trainer.logger.init_args.tags=[${mode},arm-${arm},${graph},features-${features},seed-${seed}]")
}
