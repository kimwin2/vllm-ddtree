#!/usr/bin/env bash
#
# ddtree_debug.sh — server-side log dissection helper for the ddtree
# integration. Given a vllm server log (default /tmp/vllm-on-s3c1.log),
# this prints:
#
#   1. Detected stage marker
#   2. Backend probe verdict
#   3. KV layout audit (S3c-4 one-time INFO)
#   4. Compaction stats over time (average accepted path length,
#      copies, dry_run flag) with the most recent N entries
#   5. spec_decode metric history (if the /metrics endpoint is reachable
#      on $PORT — defaults to 8000)
#
# Each section is independent — feel free to grep, awk, or pipe to
# `tee` for finer slicing.

set -u

LOG="${1:-/tmp/vllm-on-s3c1.log}"
PORT="${2:-8000}"
RECENT_N="${3:-10}"

if [ ! -f "$LOG" ]; then
  echo "ERROR: log file not found: $LOG"
  echo "Usage: $0 [log_path] [port] [recent_N]"
  exit 1
fi

# ────────────────────────────────────────────────────────────────────
# 1. Stage marker (S1, S1+S2, ..., S1+S2+S3b+S3c-1+S3c-2+S3c-3+S3c-4)
# ────────────────────────────────────────────────────────────────────
echo "===== 1. Active stage marker ====="
KNOWN_STAGES=(
  "S1+S2+S3b+S3c-1+S3c-2+S3c-3+S3c-4"
  "S1+S2+S3b+S3c-1+S3c-2+S3c-3"
  "S1+S2+S3b+S3c-1+S3c-2"
  "S1+S2+S3b+S3c-1"
  "S1+S2+S3b"
  "S1+S2"
  "S1"
)
STAGE="UNKNOWN"
for tag in "${KNOWN_STAGES[@]}"; do
  if grep -q "$tag path active" "$LOG"; then
    STAGE="$tag"
    break
  fi
done
echo "  active stage  : $STAGE"
echo "  verify_tree   : $(grep -oE 'verify=tree:(True|False)' "$LOG" | head -1 | cut -d: -f2)"
echo "  num_spec      : $(grep -oE 'num_speculative_tokens=[0-9]+' "$LOG" | head -1 | cut -d= -f2)"
echo "  budget        : $(grep -oE 'budget=[0-9]+' "$LOG" | head -1 | cut -d= -f2)"

# ────────────────────────────────────────────────────────────────────
# 2. Backend probe verdict
# ────────────────────────────────────────────────────────────────────
echo
echo "===== 2. Backend probe verdict ====="
grep -nA4 "ddtree backend probe:" "$LOG" | tail -5

# ────────────────────────────────────────────────────────────────────
# 3. KV layout audit (S3c-4 one-time INFO)
# ────────────────────────────────────────────────────────────────────
echo
echo "===== 3. KV layout audit (S3c-4) ====="
LAYOUT_BLOCK=$(awk '/ddtree KV layout audit:/{flag=1; print; next} flag && /^\(.*INFO.*\)/{flag=0} flag {print}' "$LOG")
if [ -n "$LAYOUT_BLOCK" ]; then
  echo "$LAYOUT_BLOCK"
else
  echo "  (no KV layout audit found — server may be pre-S3c-4 or audit not yet logged)"
fi

# ────────────────────────────────────────────────────────────────────
# 4. Compaction stats over time
# ────────────────────────────────────────────────────────────────────
echo
echo "===== 4. Compaction history (most recent $RECENT_N) ====="
COMPACT_LINES=$(grep "ddtree compact:" "$LOG")
if [ -z "$COMPACT_LINES" ]; then
  echo "  (no 'ddtree compact:' log lines yet — server may be in OFF mode"
  echo "   or no traffic ran with verify_tree=true)"
else
  TOTAL=$(echo "$COMPACT_LINES" | wc -l)
  echo "  total compact calls logged : $TOTAL"
  # Extract avg_K values for histogram-lite summary
  AVG_KS=$(echo "$COMPACT_LINES" | grep -oE 'avg_K=[0-9.]+' | cut -d= -f2)
  if [ -n "$AVG_KS" ]; then
    MIN=$(echo "$AVG_KS" | sort -g | head -1)
    MAX=$(echo "$AVG_KS" | sort -g | tail -1)
    SUM=$(echo "$AVG_KS" | awk '{s+=$1} END {print s}')
    COUNT=$(echo "$AVG_KS" | wc -l)
    MEAN=$(awk -v s="$SUM" -v c="$COUNT" 'BEGIN {printf "%.3f", s/c}')
    echo "  avg_K stats                : min=$MIN max=$MAX mean=$MEAN n=$COUNT"
    echo "  (baseline dflash cumprod: avg_K = 3.228; S3c-4 target: > 3.228)"
  fi
  echo
  echo "  --- most recent ${RECENT_N} compact lines ---"
  echo "$COMPACT_LINES" | tail -n "$RECENT_N" | sed 's/^/  /'
fi

# ────────────────────────────────────────────────────────────────────
# 5. spec_decode metrics (if endpoint reachable)
# ────────────────────────────────────────────────────────────────────
echo
echo "===== 5. spec_decode metrics ====="
if curl -fs "http://127.0.0.1:$PORT/metrics" >/dev/null 2>&1; then
  M=$(curl -s "http://127.0.0.1:$PORT/metrics")
  DRAFTS=$(echo "$M" | grep '^vllm:spec_decode_num_drafts_total' | head -1 | awk '{print $NF}')
  DRAFT_TOKENS=$(echo "$M" | grep '^vllm:spec_decode_num_draft_tokens_total' | head -1 | awk '{print $NF}')
  ACCEPTED=$(echo "$M" | grep '^vllm:spec_decode_num_accepted_tokens_total' | head -1 | awk '{print $NF}')
  if [ -n "${DRAFTS:-}" ] && [ -n "${ACCEPTED:-}" ]; then
    ACCEPT_PER_DRAFT=$(awk -v a="$ACCEPTED" -v d="$DRAFTS" 'BEGIN{if(d+0>0) printf "%.3f", a/d; else print "n/a"}')
    echo "  drafts                     : $DRAFTS"
    echo "  draft_tokens               : $DRAFT_TOKENS"
    echo "  accepted                   : $ACCEPTED"
    echo "  accept_len per draft round : $ACCEPT_PER_DRAFT  (baseline 3.228)"
  fi
  echo "  per-position accepted counts (first 10):"
  echo "$M" | grep '^vllm:spec_decode_num_accepted_tokens_per_pos_total' \
        | head -10 | sed 's/^/    /'
else
  echo "  (port $PORT not reachable; pass a different port as the 2nd arg)"
fi

echo
echo "Done. log=$LOG  port=$PORT  recent=$RECENT_N"
