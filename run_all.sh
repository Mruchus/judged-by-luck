#!/usr/bin/env bash
# Run every model on every domain and variant, then score the results.
# Finished result files are skipped, so the script can be re-run after a crash.

set -u
cd "$(dirname "$0")"

SAMPLES="${SAMPLES:-5}"
OLLAMA="http://localhost:11434/v1"

# One model per line: "<model id>|<server URL, or '-' for Claude>|<extra run.py args>".
MODELS=(
  "claude-code:claude-haiku-5-5|-|--workers 4"
  "claude-code:claude-opus-5-5|-|--workers 4"
  "qwen3:4b-instruct-2507-q4_K_M|$OLLAMA|"
)

DOMAINS=(medical agents hiring)
VARIANTS=(main warn no_odds)

run_model() {
  local id url extra
  IFS='|' read -r id url extra <<< "$1"; shift
  if [ "$url" = "-" ]; then
    python3 run.py --model "$id" $extra "$@"
  else
    OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}" python3 run.py --model "$id" --base-url "$url" $extra "$@"
    local rc=$?
    # Unload the local model so the next one does not have to share memory with it.
    command -v ollama > /dev/null && ollama stop "$id" > /dev/null 2>&1
    return $rc
  fi
}

# Run all variants for a list of models, main variant first, and refresh the figures after each variant.
run_lane() {
  [ $# -eq 0 ] && return
  for v in "${VARIANTS[@]}"; do
    for d in "${DOMAINS[@]}"; do
      for spec in "$@"; do
        echo "== $(date +%H:%M) $d | $v | ${spec%%|*}"
        run_model "$spec" --domain "$d" --variant "$v" --samples "$SAMPLES" \
          || echo "!! failed: $d $v ${spec%%|*} (re-run the script to retry)"
      done
    done
    python3 score.py > /dev/null 2>&1
  done
}

echo "== $(date) starting"
CLAUDE_OK=(); LOCAL_OK=()
for spec in "${MODELS[@]}"; do
  if run_model "$spec" --check; then
    case "$spec" in *"|-|"*) CLAUDE_OK+=("$spec");; *) LOCAL_OK+=("$spec");; esac
  else echo "!! skipping ${spec%%|*} (check failed)"; fi
done

if [ $(( ${#CLAUDE_OK[@]} + ${#LOCAL_OK[@]} )) -eq 0 ]; then echo "!! no working models"; exit 1; fi

# Claude models run remotely, so they run in parallel with the local models.
run_lane ${CLAUDE_OK[@]+"${CLAUDE_OK[@]}"} &
run_lane ${LOCAL_OK[@]+"${LOCAL_OK[@]}"} &
wait

python3 score.py
echo "== $(date) done"
