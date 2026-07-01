#!/usr/bin/env bash
set -u

if [ "$#" -lt 2 ]; then
  echo "Usage: bash runlog.sh <run_name> <command> [args...]"
  exit 2
fi

mkdir -p logs

name="$1"
shift

ts=$(date +"%Y%m%d_%H%M%S")
log="logs/${ts}_${name}.log"

script_file=""
for arg in "$@"; do
  if [ -f "$arg" ] && [ "${arg##*.}" = "sh" ]; then
    script_file="$arg"
    break
  fi
done

{
  echo "time_start: $(date -Is)"
  echo "cwd: $(pwd)"
  echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "command: $*"
  if [ -n "$script_file" ]; then
    echo "script_file: $script_file"
    echo "script_snapshot_start"
    sed -n '1,240p' "$script_file"
    echo "script_snapshot_end"
  fi
  echo "----------------------------------------"

  "$@"
  code=$?

  echo "----------------------------------------"
  echo "time_end: $(date -Is)"
  echo "exit_code: $code"

  exit $code
} 2>&1 | tee "$log"

exit ${PIPESTATUS[0]}
