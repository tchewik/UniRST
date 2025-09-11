#!/usr/bin/env sh
# Portable runner for multiple_runs.py (no bashisms).

set -eu
# DEBUG=1 ./src/commands.sh ...  # to see commands
[ "${DEBUG:-0}" = "1" ] && set -x

PY="${PY:-python}"
SCRIPT="${SCRIPT:-src/universal_parser/multiple_runs.py}"

# Common toggles (env)
RUNS="${RUNS:-3}"
CUDA="${CUDA:-}"      # e.g. 0
AUG="${AUG:-}"       # true|false (monolingual only)
MASKED="${MASKED:-}" # true|false (MU setups)
SSEG="${SSEG:-}"     # true|false (single segmenter for MU)
ONLY="${ONLY:-}"     # regex filter for mono all

# Unified corpora (single source of truth)
UNIRST_MUH_LIST='
eng.rst.rstdt
eng.erst.gum
eng.rst.sts
eng.rst.oll
eng.rst.umuc
fra.sdrt.annodis
rus.rst.rrt
rus.rst.rrg
spa.rst.rststb
ces.rst.crdt
deu.rst.pcc
eus.rst.ert
fas.rst.prstc
por.rst.cstn
spa.rst.sctb
zho.rst.sctb
zho.rst.gcdt
nld.rst.nldt
'

json_single() { printf '["%s"]' "$1"; }

json_from_list() {
  # Input: newline-separated items on stdin
  # Output: JSON array
  printf '['
  first=1
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    [ $first -eq 0 ] && printf ','
    printf '"%s"' "$line"
    first=0
  done
  printf ']'
}

opt_flags() {
  # Emit optional CLI flags based on env
  [ -n "$CUDA" ]  && printf ' --cuda_device %s' "$CUDA"
  [ -n "$AUG" ]   && printf ' --data_aug %s' "$AUG"
  [ -n "$MASKED" ]&& printf ' --masked_union %s' "$MASKED"
  if [ -n "$SSEG" ]; then
    if [ "$SSEG" = "true" ]; then
      printf ' --segmenter_separated false'
    else
      printf ' --segmenter_separated true'
    fi
  fi
}

run_cmd() {
  corps_json=$1
  mode=$2   # train|evaluate
  shift 2
  echo ">> ${PY} ${SCRIPT} --corpora ${corps_json} ${mode} --n_runs ${RUNS}$(opt_flags) $*"
  # shellcheck disable=SC2086
  ${PY} ${SCRIPT} --corpora "${corps_json}" "${mode}" --n_runs "${RUNS}" $* $(opt_flags)
}

list_corpora() {
  printf "%s\n" "$UNIRST_MUH_LIST" | sed '/^\s*$/d'
}

mono() {
  mode="${1:-train}"            # train|evaluate
  which="${2:-eng.rst.rstdt}"   # corpus or "all"
  shift 2 || true

  if [ "$which" = "all" ]; then
    list="$(list_corpora)"
    if [ -n "$ONLY" ]; then
      list="$(printf "%s\n" "$list" | grep -E "$ONLY" || true)"
    fi
    if [ -z "$list" ]; then
      echo "No corpora matched ONLY='$ONLY'." >&2; exit 1
    fi
    echo "==> Monolingual over:"
    echo "$list" | sed 's/^/   - /'
    echo
    printf "%s\n" "$list" | while IFS= read -r c; do
      [ -z "$c" ] && continue
      run_cmd "$(json_single "$c")" "$mode" "$@"
    done
  else
    run_cmd "$(json_single "$which")" "$mode" "$@"
  fi
}

unir_muh() {
  corps="$(list_corpora | json_from_list)"
  run_cmd "$corps" "${1:-train}" "${2+"$2"}"
}

unir_muh_mseg() { SSEG="${SSEG:-false}"; unir_muh "$@"; }
unir_muh_sseg() { SSEG="${SSEG:-true}";  unir_muh "$@"; }

unir_uu() {
  run_cmd '["CONCAT"]' "${1:-train}" "${2+"$2"}"
}

unir_mu_mseg() { MASKED="${MASKED:-true}"; SSEG="${SSEG:-false}"; unir_muh "$@"; }
unir_mu_sseg() { MASKED="${MASKED:-true}"; SSEG="${SSEG:-true}";  unir_muh "$@"; }

usage() {
  cat <<'EOF'
Usage:
  src/commands.sh mono <train|evaluate> [corpus|all]
  src/commands.sh unir-muh-mseg <train|evaluate>
  src/commands.sh unir-muh-sseg <train|evaluate>
  src/commands.sh unir-uu  <train|evaluate>
  src/commands.sh unir-mu-mseg <train|evaluate>
  src/commands.sh unir-mu-sseg <train|evaluate>
  src/commands.sh list

Env toggles:
  RUNS=3 CUDA=0 AUG=true MASKED=true SSEG=true ONLY='eng|rus' DEBUG=1

Examples:
  bash src/commands.sh list
  bash src/commands.sh mono train eng.erst.gum
  ONLY='eng|rus' bash src/commands.sh mono evaluate all
  CUDA=0 bash src/commands.sh unir-mu-mseg train
EOF
}

# --- CLI parsing: accept either order: "<preset> <mode>" or "<mode> <preset>"
arg1="${1:-}"; arg2="${2:-}"; arg3="${3:-}"

case "$arg1" in
  train|evaluate)
    # user gave mode first
    mode="$arg1"; preset="${arg2:-mono}"; shift 2 || true
    case "$preset" in
      mono)           mono "$mode" "${1:-eng.rst.rstdt}";;
      unir-muh-mseg)  unir_muh_mseg "$mode";;
      unir-muh-sseg)  unir_muh_sseg "$mode";;
      unir-uu)        unir_uu  "$mode";;
      unir-mu-mseg)   unir_mu_mseg "$mode";;
      unir-mu-sseg)   unir_mu_sseg "$mode";;
      ""|-h|--help)   usage;;
      *) echo "Unknown preset: $preset"; usage; exit 1;;
    esac
    ;;
  mono|unir-muh|unir-muh-mseg|unir-muh-sseg|unir-uu|unir-mu-mseg|unir-mu-sseg)
    preset="$arg1"; shift
    case "$preset" in
      mono)           mono       "${1:-train}" "${2:-eng.rst.rstdt}";;
      unir-muh-mseg)  unir_muh_mseg "${1:-train}";;
      unir-muh-sseg)  unir_muh_sseg "${1:-train}";;
      unir-uu)        unir_uu    "${1:-train}";;
      unir-mu-mseg)   unir_mu_mseg "${1:-train}";;
      unir-mu-sseg)   unir_mu_sseg "${1:-train}";;
    esac
    ;;
  list) list_corpora;;
  ""|-h|--help) usage;;
  *)
    echo "Unrecognized arguments: $*"; usage; exit 1;;
esac
