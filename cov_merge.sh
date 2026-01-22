#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export COVERAGE_FILE="$ROOT/.coverage"

rm -f "$ROOT/.coverage" "$ROOT/.coverage".* "$ROOT/coverage.xml"
rm -rf "$ROOT/htmlcov"

# 1) dataset creator run (still cd, but writes coverage into ROOT because COVERAGE_FILE is absolute)
(
  cd "$ROOT/DatasetCreator"
  coverage run --parallel-mode --source="$ROOT" prepare_datasets.py --dataset HCP_dummy --problem HCP
)

# 2) main run from repo root
cd "$ROOT"
coverage run --parallel-mode --source="$ROOT" argparse_ray_main.py \
  --GPUs 2 --IsingMode HCP_dummy --EnergyFunction HCP --N_anneal 300 \
  --n_diffusion_steps 4 --minib_diff_steps 4 --batch_size 100 --n_basis_states 1000 \
  --minib_basis_states 25 --noise_potential annealed_obj --project_name HCP_main --seed 420 \
  --debug --jit --train_mode PPO --use-sample 0 --AnnealSchedule linear --temps 0.001 \
  --T_target 0.005 --embedding_dim 32 --node_transformer_layers 4 --lrs 0.0001

coverage combine
coverage report -m
coverage html

echo "Open: $ROOT/htmlcov/index.html"