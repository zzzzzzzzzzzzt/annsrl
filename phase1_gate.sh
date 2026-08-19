#!/usr/bin/env bash
#
# Phase 1 gate: does the graph-edit policy beat random edge addition?
#
# Run A samples both halves of the swap from the policy. Run B sets the temperature
# to 1e-6, which flattens the candidate softmax to uniform, and the gradient with
# it -- so B is a frozen, uniform-chance control.
#
# B randomizes BOTH halves, not just additions. logit_scale is applied inside
# get_pair_logits, which is what score_candidates calls for the neighbour rows too,
# so at 1e-6 the drop softmax is uniform as well. (This header used to claim the drop
# side was untouched, on the grounds that a positive rescale cannot change a ranking.
# That held while drops were a deterministic argmin; drop_mode='policy' SAMPLES from
# softmax(-score), where a rescale changes the distribution.) The gate therefore asks
# "does the trained policy beat chance", and does not isolate which half is
# responsible -- worth knowing given that an untrained sampled drop measured 1.5 sigma
# WORSE than uniform.
#
# The gate: A's val/recall@k must clearly exceed B's. If it does not, the gradient is
# flowing but points somewhere useless, meaning the defect is in the reward or the
# credit assignment -- not in the hyperparameters. Do not proceed to phase 2/3 until
# this passes.
#
# The start state is a RANDOM graph, not HNSW. Measured on SIFT100K from an HNSW
# start, 0 of 20 random swap batches improved recall (best -2.4e-04), while from
# a random start 20% improved recall and 60% improved the shaped reward: HNSW is
# already a local optimum of this bounded swap, and the learner is greedy
# (gamma = 0, no critic), so on HNSW it is guaranteed to FAIL regardless of how
# good the policy is. Set GRAPH_TYPE=nsw to re-run the HNSW version once a
# critic and gamma > 0 exist.
#
# Usage:  bash phase1_gate.sh                 # 300 steps; measured ~3 min per arm, of
#                                             # which training is ~90 s and the rest is
#                                             # the final ef sweep
#         MAX_STEPS=30 bash phase1_gate.sh    # quick check
#         GRAPH_TYPE=nsw bash phase1_gate.sh  # old HNSW start
#         K=1 bash phase1_gate.sh             # judge on recall@1 (weaker; see K below)
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-/mnt/HDD0/home/zjw25/anaconda3/envs/annsrl/bin/python}
PRETRAINED=${PRETRAINED:-models/mlplink_SIFT100K_dot_best.pth}
MAX_STEPS=${MAX_STEPS:-300}
GRAPH_TYPE=${GRAPH_TYPE:-random}
# 512 instead of 4096: the graph-level delta is split over every node the probe
# visited, so with 4096 edited nodes the per-node credit was ~1e-01 while the
# true per-node effect measured ~5e-07. Fewer edits per step keeps one step's
# reward closer to something a single node can be held responsible for.
NODES_PER_STEP=${NODES_PER_STEP:-512}
# Fixed-cost regime: a hard DCS budget in the search kernel makes cost a constant of
# the environment, so recall is comparable across graphs and beta can be 0 -- no
# recall/DCS exchange rate to pick. The budget only works if it BINDS: measured on
# SIFT100K at budget=300, the fraction of queries that saturate it is 0.49 at ef=10,
# 0.85 at ef=16, and 1.000 at ef>=32. Below ef=32 the walk still stops on its own and
# DCS is free to move, which is the leak this is meant to close.
DCS_BUDGET=${DCS_BUDGET:-300}
EVAL_EF=${EVAL_EF:-32}
# Fixed so A and B are a PAIRED comparison; override to repeat the gate on a
# different draw of s_0 (one seed says nothing about how the result generalizes).
SEED=${SEED:-1234}
# 'policy' trains both halves of the swap. Set argmin to reproduce pre-2026-08 runs.
DROP_MODE=${DROP_MODE:-policy}
# 10, not the trainer's default of 1. Measured on SIFT100K at identical cost, recall/SE
# is 5 at k=1 and 18 at k=10, so recall@1 from a weak s_0 is mostly quantization noise.
# This also decides which metric the verdict uses: val/recall@10 is only logged when
# k > 1, and dump_diag.py falls back to recall@1 when it is absent. Must stay <= EVAL_EF
# (the kernel's result heap holds only ef entries, and train_sift100k_ppo.py asserts it).
K=${K:-10}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=log/phase1_gate_$STAMP
mkdir -p "$LOG_DIR"

RUN_A=p1_policy
RUN_B=p1_random

# commit_stride 1 so MAX_STEPS steps produce MAX_STEPS real commits: the graph
# has to move enough for val/recall@1 to leave its starting point at all.
common_args=(
  --graph_type "$GRAPH_TYPE"
  --pretrained_path "$PRETRAINED"
  --max_steps "$MAX_STEPS"
  --commit_stride 1
  --max_grad_norm 1.0
  --nodes_per_step "$NODES_PER_STEP"
  --dcs_budget "$DCS_BUDGET"
  --ef "$EVAL_EF"
  --k "$K"
  # Both arms share the seed, so they share s_0, the per-step node sample and the
  # per-step query batch: the only difference left is the thing under test. The
  # previous verdict was invalid without this -- the arms started 0.0418 apart in
  # mean reward, larger than the +0.0334 advantage the gate reported for A.
  --seed "$SEED"
  --init_seed "$SEED"
  # beta 0 because the budget already fixes the cost: with DCS pinned at
  # DCS_BUDGET for every query there is nothing left for a cost penalty to buy,
  # and a nonzero beta would just reintroduce the exchange rate the budget
  # removed. Measured previously: 91% of the reward gain came from r_cost.
  --beta 0
  --drop_mode "$DROP_MODE"
  --no_plot
)

# A second run writing into an existing runs/<name>/ leaves TWO event files
# there, and dump_diag.py reads the whole directory: overlapping step ranges
# then interleave silently and the table is quietly wrong. Move any previous
# attempt aside rather than merging into it.
for name in "$RUN_A" "$RUN_B"; do
  if [ -d "runs/$name" ]; then
    mv "runs/$name" "runs/${name}.superseded_$STAMP"
    echo "[warn] runs/$name existed -> runs/${name}.superseded_$STAMP"
  fi
done

run_one() {   # name, logit_scale, description
  local name=$1 scale=$2 desc=$3
  echo "[$(date +%H:%M:%S)] $name ($desc): logit_scale=$scale, $MAX_STEPS steps"
  # -u so per-step diagnostics reach the log while the run is still going.
  "$PY" -u train_sift100k_ppo.py "${common_args[@]}" \
    --run_name "$name" --logit_scale "$scale" \
    > "$LOG_DIR/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] $name done"
}

echo "[cfg] graph_type=$GRAPH_TYPE nodes_per_step=$NODES_PER_STEP noop=on drop_mode=$DROP_MODE"
run_one "$RUN_A" 20   'policy-chosen swaps'
run_one "$RUN_B" 1e-6 'uniform swaps, frozen policy'

# Sample the run evenly rather than hardcoding 0/50/100/200, so a shorter
# MAX_STEPS still yields a meaningful table.
steps=()
for frac in 0 6 12 25 50 75 100; do
  s=$(( (MAX_STEPS - 1) * frac / 100 ))
  if [ "${#steps[@]}" -eq 0 ] || [ "$s" != "${steps[-1]}" ]; then
    steps+=("$s")
  fi
done

for name in "$RUN_A" "$RUN_B"; do
  echo
  "$PY" dump_diag.py "$name" "${steps[@]}" | tee "$LOG_DIR/$name.diag.txt"
done

# One comparison is the whole point of the phase, so state the verdict instead
# of leaving two tables to be cross-read by hand.
echo
"$PY" dump_diag.py --compare "$RUN_A" "$RUN_B" | tee "$LOG_DIR/verdict.txt"

echo
echo "logs + tables: $LOG_DIR"
