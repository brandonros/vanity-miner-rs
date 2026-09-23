#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'HELP'
Usage:
  vast-run.sh rsync [LOCAL_BINARY [LOCAL_PTX_DIR]] -- COMMAND [ARGS...]
  vast-run.sh github VERSION -- COMMAND [ARGS...]
  vast-run.sh actions RUN_ID_OR_URL -- COMMAND [ARGS...]

Examples:
  ./scripts/vast-run.sh rsync -- solana-vanity --prefix aaaa
  ./scripts/vast-run.sh rsync /path/to/linux/vanity-miner /path/to/ptx -- self-test
  LLVM_VARIANT=llvm7 ./scripts/vast-run.sh github v1.24.0 -- bitcoin-vanity --prefix bc1q
  ./scripts/vast-run.sh actions 34923712713 -- ethereum-vanity --prefix 5555
  ./scripts/vast-run.sh actions https://github.com/brandonros/vanity-miner-rs/actions/runs/34923712713/job/104237136178 -- self-test

Everything after -- is passed literally to vanity-miner; no interactive menu.

Environment: HOST (ssh2.vast.ai), PORT (37827), REMOTE_USER (root),
LLVM_VARIANT (llvm21), LOCAL_BINARY, LOCAL_PTX_DIR, GITHUB_ASSET,
BATCH_SIZE (optional CUDA candidate count per launch), THREADS_PER_BLOCK (256),
STACK_SIZE (defaults to 16384 bytes for Bitcoin/Ethereum; other commands use
the application's per-thread stack limit). Explicit STACK_SIZE overrides win.
Default local binary: <repo>/target/runner/release/vanity-miner.
Default local PTX: <repo>/target/<LLVM_VARIANT>/release/ptx.
Explicit relative paths are resolved from your current directory.
GITHUB_ASSET overrides vanity-miner-<remote architecture>.
Rsync uploads both the Linux binary and PTX; they must have matching interfaces.
Actions mode requires local gh authentication (gh auth login) and rsync.
It downloads the selected run's runner for the remote CPU and PTX bundle for
LLVM_VARIANT, then uploads both. Both artifacts must still be available.
LLVM 21 PTX targets sm_100 and requires a compatible GPU and driver.
HELP
}

# Separate deployment arguments from the miner's own CLI arguments.
DEPLOY_ARGS=()
MINER_ARGS=()
while [ "$#" -gt 0 ]; do
    if [ "$1" = -- ]; then
        shift
        MINER_ARGS=("$@")
        break
    fi
    DEPLOY_ARGS+=("$1")
    shift
done
set -- "${DEPLOY_ARGS[@]}"
MODE=${1:---help}
case "$MODE" in
    -h|--help) usage; exit 0 ;;
    rsync) [ "$#" -le 3 ] || { usage >&2; exit 1; } ;;
    github) [ "$#" -eq 2 ] || { usage >&2; exit 1; } ;;
    actions) [ "$#" -eq 2 ] || { usage >&2; exit 1; } ;;
    *) usage >&2; exit 1 ;;
esac

if [ "${#MINER_ARGS[@]}" -eq 0 ]; then
    echo "ERROR: specify a miner command after -- (e.g. -- bitcoin-vanity --prefix bc1q)" >&2
    exit 1
fi
case "${MINER_ARGS[0]}" in
    solana-vanity) PTX_MODULE=solana ;;
    bitcoin-vanity) PTX_MODULE=bitcoin ;;
    ethereum-vanity) PTX_MODULE=ethereum ;;
    shallenge) PTX_MODULE=shallenge ;;
    rsa-pss-signature-vanity) PTX_MODULE=rsa_pss ;;
    p256-public-key-vanity) PTX_MODULE=p256_public_key ;;
    p256-signature-vanity) PTX_MODULE=p256_signature ;;
    self-test) PTX_MODULE=self_test ;;
    -h|--help|-V|--version) PTX_MODULE="" ;;
    *) echo "ERROR: unknown miner command: ${MINER_ARGS[0]} (use -- --help)" >&2; exit 1 ;;
esac

PORT=${PORT:-37827}
HOST=${HOST:-ssh2.vast.ai}
REMOTE_USER=${REMOTE_USER:-root}
LLVM_VARIANT=${LLVM_VARIANT:-llvm21}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LOCAL_BINARY=${2:-${LOCAL_BINARY:-$REPO_ROOT/target/runner/release/vanity-miner}}
LOCAL_PTX_DIR=${3:-${LOCAL_PTX_DIR:-$REPO_ROOT/target/$LLVM_VARIANT/release/ptx}}
SSH=(ssh -p "$PORT" "$REMOTE_USER@$HOST")
VERSION=""

if [ "$MODE" = actions ]; then
    RUN_INPUT=$2
    if [[ "$RUN_INPUT" =~ ^[0-9]+$ ]]; then
        RUN_ID=$RUN_INPUT
    elif [[ "$RUN_INPUT" =~ ^https://github\.com/brandonros/vanity-miner-rs/actions/runs/([0-9]+)(/|\?|\#|$) ]]; then
        RUN_ID=${BASH_REMATCH[1]}
    else
        echo "ERROR: expected a run ID or a vanity-miner-rs Actions run/job URL" >&2
        exit 1
    fi
    command -v gh >/dev/null || { echo "ERROR: install gh and run gh auth login" >&2; exit 1; }
    command -v rsync >/dev/null
    REMOTE_ARCH=$("${SSH[@]}" uname -m)
    case "$REMOTE_ARCH" in
        x86_64|aarch64) ;;
        *) echo "ERROR: unsupported remote architecture: $REMOTE_ARCH" >&2; exit 1 ;;
    esac
    ACTIONS_ARTIFACT="vanity-miner-$REMOTE_ARCH"
    DOWNLOAD_DIR=$(mktemp -d)
    trap 'rm -rf -- "$DOWNLOAD_DIR"' EXIT
    echo "DOWNLOAD :: Actions run $RUN_ID / $ACTIONS_ARTIFACT"
    gh run download "$RUN_ID" --repo brandonros/vanity-miner-rs \
        --name "$ACTIONS_ARTIFACT" --dir "$DOWNLOAD_DIR"
    LOCAL_BINARY="$DOWNLOAD_DIR/$ACTIONS_ARTIFACT"
    LOCAL_PTX_DIR="$DOWNLOAD_DIR/ptx"
    gh run download "$RUN_ID" --repo brandonros/vanity-miner-rs \
        --name "ptx-$LLVM_VARIANT" --dir "$LOCAL_PTX_DIR"
    tar -C "$LOCAL_PTX_DIR" -xzf "$LOCAL_PTX_DIR/ptx-bundle-$LLVM_VARIANT.tar.gz"
    rm -- "$LOCAL_PTX_DIR/ptx-bundle-$LLVM_VARIANT.tar.gz"
fi

if [ "$MODE" != github ]; then
    command -v rsync >/dev/null
    [ -f "$LOCAL_BINARY" ] || { echo "ERROR: binary not found: $LOCAL_BINARY" >&2; exit 1; }
    if [ "$MODE" != github ]; then
        if [ "$PTX_MODULE" = self_test ]; then
            shopt -s nullglob
            PTX_FILES=("$LOCAL_PTX_DIR"/self_test_*.ptx)
            [ "${#PTX_FILES[@]}" -gt 0 ] || { echo "ERROR: no self-test PTX in $LOCAL_PTX_DIR" >&2; exit 1; }
        elif [ -n "$PTX_MODULE" ]; then
            [ -f "$LOCAL_PTX_DIR/$PTX_MODULE.ptx" ] || { echo "ERROR: missing $LOCAL_PTX_DIR/$PTX_MODULE.ptx" >&2; exit 1; }
        fi
        [ -d "$LOCAL_PTX_DIR" ] || { echo "ERROR: PTX directory not found: $LOCAL_PTX_DIR" >&2; exit 1; }
    fi
    BINARY_INFO=$(file -b "$LOCAL_BINARY")
    REMOTE_ARCH=$("${SSH[@]}" uname -m)
    case "$REMOTE_ARCH:$BINARY_INFO" in
        x86_64:*ELF*x86-64*|aarch64:*ELF*ARM\ aarch64*) ;;
        *) echo "ERROR: local binary does not match remote Linux $REMOTE_ARCH: $BINARY_INFO" >&2; exit 1 ;;
    esac
else
    VERSION=$2
fi

# Quote arguments for the remote shell before invoking bash.
printf -v REMOTE_SETUP 'bash -s -- %q' "$MODE"
"${SSH[@]}" "$REMOTE_SETUP" <<'EOF'
set -euo pipefail
cd "$HOME"
MODE=$1
banner() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

banner "ENV CHECK :: deployment tools"
PKGS=""
command -v killall &> /dev/null || PKGS="$PKGS psmisc"
command -v patchelf &> /dev/null || PKGS="$PKGS patchelf"
if [ "$MODE" != github ]; then
    command -v rsync &> /dev/null || PKGS="$PKGS rsync"
else
    command -v curl &> /dev/null || PKGS="$PKGS curl"
fi
if [ -n "$PKGS" ]; then
    apt update
    apt install -y $PKGS
else
    echo "deployment tools already installed"
fi

mkdir -p ptx
EOF

banner_local() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

if [ "$MODE" != github ]; then
    banner_local "UPLOAD :: binary via rsync"
    rsync -avz -e "ssh -p $PORT" -- "$LOCAL_BINARY" "$REMOTE_USER@$HOST:vanity-miner.next"
fi
if [ "$MODE" != github ]; then
    banner_local "UPLOAD :: PTX via rsync"
    rsync -avz -e "ssh -p $PORT" -- "$LOCAL_PTX_DIR/" "$REMOTE_USER@$HOST:ptx/"
fi

printf -v REMOTE_RUN 'bash -s -- %q %q %q %q %q %q %q %q' "$MODE" "$VERSION" "$LLVM_VARIANT" "${GITHUB_ASSET:-}" "${STACK_SIZE:-}" "$PTX_MODULE" "${BATCH_SIZE:-}" "${THREADS_PER_BLOCK:-256}"
printf -v QUOTED_MINER_ARGS ' %q' "${MINER_ARGS[@]}"
REMOTE_RUN+=$QUOTED_MINER_ARGS
"${SSH[@]}" "$REMOTE_RUN" <<'EOF'
set -euo pipefail
cd "$HOME"
MODE=$1
VERSION=$2
LLVM_VARIANT=$3
ASSET=$4
REQUESTED_STACK_SIZE=$5
PTX_MODULE=$6
REQUESTED_BATCH_SIZE=$7
REQUESTED_THREADS_PER_BLOCK=$8
shift 8
MINER_ARGS=("$@")
banner() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

if [ "$MODE" = github ]; then
    ARCH=$(uname -m)
    ASSET=${ASSET:-vanity-miner-$ARCH}
    banner "DOWNLOAD :: $VERSION / $ASSET"
    curl -fL -o vanity-miner.next "https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/$ASSET"
    curl -fL -o ptx-bundle.tar.gz "https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/ptx-bundle-$LLVM_VARIANT.tar.gz"
    mkdir -p ptx
    tar -C ptx -xzf ptx-bundle.tar.gz
    rm -- ptx-bundle.tar.gz
fi

banner "PREPARE :: binary"
chmod +x vanity-miner.next
# Patch only Nix-linked binaries; support both Linux host architectures.
INTERPRETER=$(patchelf --print-interpreter vanity-miner.next)
if [[ "$INTERPRETER" == /nix/store/* ]]; then
    case "$(uname -m)" in
        x86_64) SYSTEM_LD=/lib64/ld-linux-x86-64.so.2 ;;
        aarch64) SYSTEM_LD=/lib/ld-linux-aarch64.so.1 ;;
        *) echo "Unsupported host architecture" >&2; exit 1 ;;
    esac
    [ -f "$SYSTEM_LD" ] || { echo "Missing loader: $SYSTEM_LD" >&2; exit 1; }
    patchelf --set-interpreter "$SYSTEM_LD" vanity-miner.next
    patchelf --remove-rpath vanity-miner.next
fi
killall vanity-miner || true
mv -f vanity-miner.next vanity-miner
ls -lh vanity-miner

# All deployment modes supply standalone modules matching the runner revision.
unset CUBIN_PATH
export PTX_PATH="$HOME/ptx"
echo "PTX_PATH=$PTX_PATH"

banner "GPU INFO :: nvidia-smi"
echo "The CUDA Version in nvidia-smi is driver capability, not the installed toolkit."
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

banner "REMOTE CUDA TOOLKIT :: nvcc"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || echo "Could not query nvcc; continuing with the CUDA driver."
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    /usr/local/cuda/bin/nvcc --version || echo "Could not query nvcc; continuing with the CUDA driver."
else
    echo "nvcc not found in PATH or /usr/local/cuda/bin (toolkit is optional for running this miner)."
fi

banner "BUILD CUDA TOOLKIT + PTX REQUIREMENTS"
echo "Deployment mode: $MODE; selected variant: $LLVM_VARIANT"
if [ "$PTX_MODULE" = self_test ] || [ -z "$PTX_MODULE" ]; then
    shopt -s nullglob
    PTX_INPUTS=("$PTX_PATH"/"${PTX_MODULE}"*.ptx)
else
    PTX_INPUTS=("$PTX_PATH/$PTX_MODULE.ptx")
fi
echo "Inspecting standalone PTX for ${MINER_ARGS[0]}"
# Read the selected modules' compiler headers.
if [ "${#PTX_INPUTS[@]}" -eq 0 ] || ! LC_ALL=C grep -ahoE 'Cuda compilation tools, release [0-9.]+, V[0-9.]+|\.version[[:blank:]]+[0-9.]+|\.target[[:blank:]]+sm_[0-9]+[a-z]?' "${PTX_INPUTS[@]}" | LC_ALL=C sort -u; then
    echo "No readable PTX metadata found; build toolkit/PTX requirements could not be determined."
fi
echo "PTX .version is the PTX ISA version; .target is the GPU architecture target."
echo "The driver must support the PTX version; the remote nvcc version does not determine JIT support."

banner "RUNTIME ENV"
if [ -n "$REQUESTED_BATCH_SIZE" ]; then
    export BATCH_SIZE="$REQUESTED_BATCH_SIZE"
else
    unset BATCH_SIZE
fi
echo "BATCH_SIZE=${BATCH_SIZE:-automatic}"
export CUDA_LOG_FILE="stdout"
export THREADS_PER_BLOCK="$REQUESTED_THREADS_PER_BLOCK"
if [ -n "$REQUESTED_STACK_SIZE" ]; then
    export STACK_SIZE="$REQUESTED_STACK_SIZE"
else
    case "${MINER_ARGS[0]}" in
        bitcoin-vanity|ethereum-vanity) export STACK_SIZE=16384 ;;
        *) unset STACK_SIZE ;;
    esac
fi
echo "CUDA_LOG_FILE=$CUDA_LOG_FILE"
echo "THREADS_PER_BLOCK=$THREADS_PER_BLOCK"
echo "STACK_SIZE=${STACK_SIZE:-application default}"

banner "RUN :: vanity-miner"
printf 'Command: ./vanity-miner'
printf ' %q' "${MINER_ARGS[@]}"
printf '\n'
exec ./vanity-miner "${MINER_ARGS[@]}"
EOF
