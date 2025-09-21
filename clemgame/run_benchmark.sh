#!/bin/bash
# Usage: scripts/run_benchmark.sh

# activate  the virtual environment, assuming that the virtual environment files are located under "venv" folder (adjust if it is set up differently)
source /Users/karlosswald/repositories/rllm/toh-venv/bin/activate
export PYTHONPATH=.:$PYTHONPATH

cd /Users/karlosswald/repositories/rllm/tower_of_hanoi/clemgame || exit

mkdir -p logs

games=(
  "toh_single_turn", "toh_multi_turn", "toh_single_asp", "toh_multi_asp"
)

# choose models to run, from global list (clem list models) or local list (the ones in model_registry.json)

models=(
  "mock"
)

echo
echo "==================================================="
echo "RUNNING: Benchmark Run"
echo "==================================================="
echo


# Runs each model and game separately
for game in "${games}"; do
  for model in "${models}"; do
    echo "Testing ${model} on ${game}"
    # currently, instances.json is the default file for the current run
    # to input a specific instances file, so we could also use the version number here like this: -i instances_file (no need to .json extension)
    { time clem run -g "${game}" -m "${model}" -r "results"; }
    { time clem transcribe -g "${game}" -r "results"; }
    { time clem score -g "${game}" -r "results"; }
  done
done
echo "Evaluating results"
{ time clem eval -r "results"; }

echo "==================================================="
echo "FINISHED: Benchmark Run Version"
echo "==================================================="