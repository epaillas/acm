#!/bin/bash
#SBATCH --account=desi
#SBATCH -q preempt
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

if shopt -oq posix; then
    exec /bin/bash "$0" "$@"
fi

set -euo pipefail

DEFAULT_CONFIG=/global/u1/e/epaillas/code/acm/scripts/desi/cubic/abacushod_config_base.yaml
DEFAULT_COSMOS=0
DEFAULT_PHASES=0
DEFAULT_REDSHIFTS=0.500
DEFAULT_SIM_TYPE=base
DEFAULT_SEED=600
DEFAULT_OVERWRITE=0

CONFIG=$DEFAULT_CONFIG
OUTPUT_DIR=
COSMO_SPEC=$DEFAULT_COSMOS
PHASE_SPEC=$DEFAULT_PHASES
REDSHIFT_SPEC=$DEFAULT_REDSHIFTS
SIM_TYPE=$DEFAULT_SIM_TYPE
SEED=$DEFAULT_SEED
OVERWRITE=$DEFAULT_OVERWRITE
DRY_RUN=0

usage ()
{
    cat <<'EOF'
Usage:
  generate_abacus_subsamples.sh [options]

Generate AbacusHOD subsample files for selected cosmologies, phases, and
redshifts by calling:
  python -m abacusnbody.hod.prepare_sim

Options:
  --cosmos LIST       Cosmology IDs, comma-separated with optional ranges.
                      Example: 0,1,13,100-126
                      Default: 0
  --phases LIST       Phase IDs, comma-separated with optional ranges.
                      Example: 0-4,3000-3020
                      Default: 0
  --redshifts LIST    Redshifts, comma-separated.
                      Example: 0.5,0.8
                      Default: 0.500
  --config PATH       Path to abacusHOD config yaml.
  --output-dir PATH   Override sim_params.subsample_dir from the config yaml.
                      A trailing slash is added if missing.
  --sim-type NAME     AbacusSummit simulation type.
                      Default: base
  --seed INT          Random number seed passed as --newseed.
                      Default: 600
  --overwrite VALUE   Value passed to prepare_sim --overwrite.
                      Default: 0
  --dry-run           Print commands without sourcing the environment or running.
  -h, --help          Show this help.

Examples:
  sbatch generate_abacus_subsamples.sh --cosmos 0,1,13 --phases 0-4 --redshifts 0.5,0.8
  sbatch generate_abacus_subsamples.sh --output-dir /pscratch/sd/e/epaillas/summit_subsamples/boxes/base/
  sbatch generate_abacus_subsamples.sh --cosmos 100-126 --phases 0 --redshifts 0.800 --overwrite 0
  ./generate_abacus_subsamples.sh --dry-run --cosmos 0,13 --phases 0-2 --redshifts 0.5,0.8
EOF
}

die ()
{
    echo "ERROR: $*" >&2
    exit 1
}

require_value ()
{
    local option=$1
    local value=${2:-}

    if [[ -z $value || $value == --* ]]; then
        die "$option requires a value"
    fi
}

expand_integer_spec ()
{
    local spec=$1
    local token start end value
    local -a tokens

    [[ -n $spec ]] || die "integer list cannot be empty"

    IFS=',' read -ra tokens <<< "$spec"
    for token in "${tokens[@]}"; do
        [[ -n $token ]] || die "empty item in integer list '$spec'"

        if [[ $token =~ ^[0-9]+-[0-9]+$ ]]; then
            start=${token%-*}
            end=${token#*-}
            start=$((10#$start))
            end=$((10#$end))

            [[ $start -le $end ]] || die "range '$token' must be increasing"
            for ((value = start; value <= end; value++)); do
                printf '%s\n' "$value"
            done
        elif [[ $token =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$((10#$token))"
        else
            die "invalid integer list item '$token'"
        fi
    done
}

expand_csv_spec ()
{
    local spec=$1
    local token
    local -a tokens

    [[ -n $spec ]] || die "comma-separated list cannot be empty"

    IFS=',' read -ra tokens <<< "$spec"
    for token in "${tokens[@]}"; do
        [[ -n $token ]] || die "empty item in comma-separated list '$spec'"
        printf '%s\n' "$token"
    done
}

leading_zero_fill ()
{
    printf "%0$1d\n" "$2"
}

ensure_trailing_slash ()
{
    local path=$1

    if [[ $path == */ ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/\n' "$path"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cosmos)
            require_value "$1" "${2:-}"
            COSMO_SPEC=$2
            shift 2
            ;;
        --cosmos=*)
            COSMO_SPEC=${1#*=}
            shift
            ;;
        --phases)
            require_value "$1" "${2:-}"
            PHASE_SPEC=$2
            shift 2
            ;;
        --phases=*)
            PHASE_SPEC=${1#*=}
            shift
            ;;
        --redshifts)
            require_value "$1" "${2:-}"
            REDSHIFT_SPEC=$2
            shift 2
            ;;
        --redshifts=*)
            REDSHIFT_SPEC=${1#*=}
            shift
            ;;
        --config)
            require_value "$1" "${2:-}"
            CONFIG=$2
            shift 2
            ;;
        --config=*)
            CONFIG=${1#*=}
            shift
            ;;
        --output-dir)
            require_value "$1" "${2:-}"
            OUTPUT_DIR=$2
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR=${1#*=}
            [[ -n $OUTPUT_DIR ]] || die "--output-dir requires a value"
            shift
            ;;
        --sim-type)
            require_value "$1" "${2:-}"
            SIM_TYPE=$2
            shift 2
            ;;
        --sim-type=*)
            SIM_TYPE=${1#*=}
            shift
            ;;
        --seed)
            require_value "$1" "${2:-}"
            SEED=$2
            shift 2
            ;;
        --seed=*)
            SEED=${1#*=}
            shift
            ;;
        --overwrite)
            require_value "$1" "${2:-}"
            OVERWRITE=$2
            shift 2
            ;;
        --overwrite=*)
            OVERWRITE=${1#*=}
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option '$1'. Run with --help for usage."
            ;;
    esac
done

[[ -n $CONFIG ]] || die "--config cannot be empty"
[[ -n $SIM_TYPE ]] || die "--sim-type cannot be empty"
[[ $SEED =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
[[ $OVERWRITE =~ ^[0-9]+$ ]] || die "--overwrite must be a non-negative integer"

mapfile -t COSMOS < <(expand_integer_spec "$COSMO_SPEC")
mapfile -t PHASES < <(expand_integer_spec "$PHASE_SPEC")
mapfile -t REDSHIFTS < <(expand_csv_spec "$REDSHIFT_SPEC")

RUN_CONFIG=$CONFIG
TEMP_CONFIG=

if [[ $DRY_RUN -eq 0 ]]; then
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    export PYTHONPATH="/global/cfs/cdirs/desicollab/users/epaillas/code/abacusutils:${PYTHONPATH:-}"
fi

if [[ -n $OUTPUT_DIR ]]; then
    OUTPUT_DIR=$(ensure_trailing_slash "$OUTPUT_DIR")
    TEMP_CONFIG=$(mktemp "${TMPDIR:-/tmp}/abacushod_config.XXXXXX.yaml")
    RUN_CONFIG=$TEMP_CONFIG
    trap '[[ -z ${TEMP_CONFIG:-} ]] || rm -f "$TEMP_CONFIG"' EXIT

    python - "$CONFIG" "$TEMP_CONFIG" "$OUTPUT_DIR" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
output_dir = sys.argv[3]

lines = source.read_text().splitlines(keepends=True)
in_sim_params = False
updated = False
out = []

for line in lines:
    stripped = line.strip()
    if re.match(r'^sim_params\s*:', line):
        in_sim_params = True
        out.append(line)
        continue
    if in_sim_params and line and not line.startswith((' ', '\t')) and stripped:
        in_sim_params = False
    if in_sim_params and re.match(r'^\s*subsample_dir\s*:', line):
        indent = line[:len(line) - len(line.lstrip())]
        quote_safe_output_dir = output_dir.replace("'", "''")
        newline = '\n' if line.endswith('\n') else ''
        out.append(f"{indent}subsample_dir: '{quote_safe_output_dir}'{newline}")
        updated = True
    else:
        out.append(line)

if not updated:
    raise SystemExit(f"Could not find sim_params.subsample_dir in {source}")

target.write_text(''.join(out))
PY

    printf 'Using output directory override: %s\n' "$OUTPUT_DIR"
    printf 'Using temporary config: %s\n' "$RUN_CONFIG"
fi

for cosmo in "${COSMOS[@]}"; do
    cosmo_padded=$(leading_zero_fill 3 "$cosmo")

    for phase in "${PHASES[@]}"; do
        phase_padded=$(leading_zero_fill 3 "$phase")
        alt_simname="AbacusSummit_${SIM_TYPE}_c${cosmo_padded}_ph${phase_padded}"

        for alt_z in "${REDSHIFTS[@]}"; do
            cmd=(
                python -m abacusnbody.hod.prepare_sim
                --path2config "$RUN_CONFIG"
                --alt_simname "$alt_simname"
                --alt_z "$alt_z"
                --newseed "$SEED"
                --overwrite "$OVERWRITE"
            )

            printf 'Preparing %s z%s\n' "$alt_simname" "$alt_z"
            printf 'Command:'
            printf ' %q' "${cmd[@]}"
            printf '\n'

            if [[ $DRY_RUN -eq 0 ]]; then
                "${cmd[@]}"
            fi
        done
    done
done
