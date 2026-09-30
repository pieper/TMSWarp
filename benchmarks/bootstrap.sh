#!/usr/bin/env bash
# Run the TMSWarp benchmarks on a fresh Linux machine, start to finish.
#
#   curl -sSL https://raw.githubusercontent.com/pieper/TMSWarp/main/benchmarks/bootstrap.sh \
#     | bash -s -- --label vast-3070 --install-simnibs
#
# Everything goes under ~/tmswarp-bench (override with TMSWARP_BENCH_DIR):
# the TMSWarp checkout, a Python virtual environment, the datasets, SimNIBS
# if requested, and the results.  Re-running reuses what is already there.
#
# Options
#   --suite quick|standard|full   which benchmarks to run (default: standard)
#   --label NAME                  short name for this machine in the results
#   --provider NAME               e.g. vast.ai, jetstream2 (vast.ai is detected)
#   --install-simnibs             download and install SimNIBS (1.3 GB) so that
#                                 its solvers are timed on the same machine
#   --branch NAME                 TMSWarp branch to benchmark (default: main)
#   --no-optimization             skip the coil optimization benchmark
#   anything else is passed to "python -m tmswarp.bench run"

set -euo pipefail

SUITE="standard"
LABEL=""
PROVIDER=""
INSTALL_SIMNIBS=0
OPTIMIZATION=1
BRANCH="${TMSWARP_BRANCH:-main}"
WORK="${TMSWARP_BENCH_DIR:-$HOME/tmswarp-bench}"
SIMNIBS_VERSION="4.6.0"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --suite) SUITE="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --install-simnibs) INSTALL_SIMNIBS=1; shift ;;
    --no-optimization) OPTIMIZATION=0; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# Applications such as 3D Slicer set library paths that break system tools
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME || true

mkdir -p "$WORK"
cd "$WORK"
exec > >(tee -a "$WORK/bootstrap.log") 2>&1
rm -f "$WORK/DONE" "$WORK/FAILED"
trap 'echo "bootstrap failed at line $LINENO"; touch "$WORK/FAILED"' ERR

echo "=== TMSWarp benchmark bootstrap: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    directory $WORK, branch $BRANCH, suite $SUITE"

for tool in git python3; do
  command -v "$tool" >/dev/null || { echo "$tool is required"; exit 1; }
done

# --- TMSWarp ---------------------------------------------------------------
if [ -d TMSWarp/.git ]; then
  git -C TMSWarp fetch --quiet origin "$BRANCH"
  git -C TMSWarp checkout --quiet "$BRANCH"
  git -C TMSWarp reset --quiet --hard "origin/$BRANCH"
else
  git clone --quiet --branch "$BRANCH" https://github.com/pieper/TMSWarp.git
fi
echo "    TMSWarp at $(git -C TMSWarp rev-parse --short HEAD)"

# --- Python environment ----------------------------------------------------
if [ ! -x venv/bin/python ]; then
  if ! python3 -m venv venv 2>/dev/null; then
    rm -rf venv
    if command -v uv >/dev/null; then
      uv venv venv
      uv pip install --python venv/bin/python pip
    elif sudo -n true 2>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv
      python3 -m venv venv
    else
      echo "Could not create a virtual environment: install python3-venv"
      exit 1
    fi
  fi
fi
venv/bin/python -m pip install --quiet --upgrade pip
venv/bin/python -m pip install --quiet -e "TMSWarp[bench]" rpyc
venv/bin/python -c "import warp, tmswarp; print('    warp', warp.__version__)"

# --- The optimizer, which lives in SlicerTMS --------------------------------
RUN_ARGS=()
if [ "$OPTIMIZATION" = 1 ]; then
  mkdir -p SlicerTMS/Experiments
  if venv/bin/python - << 'PY'
import urllib.request
url = ("https://raw.githubusercontent.com/SlicerTMS/SlicerTMS/"
       "tmsservice/Experiments/TMSService.py")
urllib.request.urlretrieve(url, "SlicerTMS/Experiments/TMSService.py")
PY
  then
    RUN_ARGS+=(--tmsservice "$WORK/SlicerTMS/Experiments/TMSService.py")
  else
    echo "    could not download TMSService.py; skipping the optimization"
    RUN_ARGS+=(--no-optimization)
  fi
else
  RUN_ARGS+=(--no-optimization)
fi

# --- SimNIBS ---------------------------------------------------------------
SIMNIBS_PYTHON="$WORK/SimNIBS/simnibs_env/bin/python"
if [ "$INSTALL_SIMNIBS" = 1 ] && [ ! -x "$SIMNIBS_PYTHON" ]; then
  if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ]; then
    echo "    installing SimNIBS $SIMNIBS_VERSION"
    venv/bin/python - << PY
import urllib.request
url = ("https://github.com/simnibs/simnibs/releases/download/"
       "v$SIMNIBS_VERSION/simnibs_installer_linux.tar.gz")
urllib.request.urlretrieve(url, "simnibs_installer_linux.tar.gz")
PY
    tar xzf simnibs_installer_linux.tar.gz
    ./simnibs_installer/install -s -t "$WORK/SimNIBS" > simnibs_install.log 2>&1
    rm -rf simnibs_installer simnibs_installer_linux.tar.gz
  else
    echo "    the SimNIBS installer used here is for Linux x86_64; skipping"
  fi
fi
if [ -x "$SIMNIBS_PYTHON" ]; then
  RUN_ARGS+=(--simnibs-python "$SIMNIBS_PYTHON")
fi

# --- Run -------------------------------------------------------------------
[ -n "$LABEL" ] && RUN_ARGS+=(--label "$LABEL")
[ -n "$PROVIDER" ] && RUN_ARGS+=(--provider "$PROVIDER")

venv/bin/python -m tmswarp.bench run --suite "$SUITE" \
  --output "$WORK/results" "${RUN_ARGS[@]}" "${EXTRA[@]+"${EXTRA[@]}"}"

touch "$WORK/DONE"
echo "=== finished: $(date -u +%Y-%m-%dT%H:%M:%SZ); results in $WORK/results ==="
