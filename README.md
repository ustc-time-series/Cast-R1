# Cast-R1

Implementation of **Learning Agentic Time Series Forecasting via Trajectory Distillation and Memory-Based Reinforcement Learning**.

The paper workflow has three stages: diagnostic evidence acquisition, forecasting-model selection, and forecast refinement. A single Qwen3-8B policy is trained on ten datasets through teacher trajectory distillation and multi-turn GRPO. Each policy state is freshly constructed from forecast-origin inputs and episode-local Analysis History / Prediction Results; previous assistant and raw tool messages are excluded.

## Paper training settings

| Setting | SFT | GRPO |
| --- | --- | --- |
| Backbone | Qwen3-8B | SFT-initialized Qwen3-8B |
| Optimizer / schedule | AdamW / cosine | AdamW / cosine |
| Learning rate | 1e-5 | 1e-6 |
| Epochs | 1 | 1 |
| Global batch size | 16 | 64 inputs |
| Group size | — | 8 trajectories per input |
| Prompt / response budget | 4096 / 2048 | 8192 / 4096 |
| Gradient accumulation | 1 | 4 for a 64-row policy minibatch on 4 GPUs |
| KL coefficient | — | 0.04 |
| Temperature | — | 1.0 |
| Hardware | 4 × A800 80GB | 4 × A800 80GB |

GRPO selects the terminal reward using explicit step indices, normalizes across trajectories for the same input, and applies the resulting trajectory advantage to every generated response token. It uses no separate critic. Tool observations are prompt context, never supervised response tokens. Intermediate tool calls have no auxiliary reward.

The active terminal reward is `p * (0.10 + 0.70*q(prediction MSE) + 0.05*q(seasonal MSE) + 0.15*q(trend MSE) + 0.10*maximum hit rate + 0.10*minimum hit rate)`, where `q(e) = 1 / (1 + log(1 + e))` and `p = min(1, prediction length / target length)`. Prediction and target use the same target mean and standard deviation for reward normalization. Missing answer tags give -1. Reported MSE/MAE remain on the raw scale.

## Environment

Use Python 3.11 on the CUDA training host:

```bash
pip install -r requirements.txt
pip install 'verl[vllm]==0.8.0'
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The trainer integration targets the published verl 0.8.0 interfaces. An installed package or a local `verl/` checkout can provide its Hydra configuration. The CPU verification environment uses PyTorch 2.8, Transformers 4.57.6, NumPy 1.26 and PyArrow 20. GPU training and inference require a compatible CUDA/vLLM installation.

## Forecast-origin data

Prepare these paths below `datasets/RAW`:

```text
ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
Wind/wind.csv
NP/EPF_NP_full.csv
PJM/EPF_PJM_full.csv
BE/EPF_BE_full.csv
DE/EPF_DE_full.csv
FR/EPF_FR_full.csv
```

CSV files need `date`, the target (`OT` for ETT/EPF, `target` for Wind), and the benchmark covariates. Column names are stripped of surrounding whitespace. The serialized task keeps historical covariates. For EPF, numeric covariates named as forecast fields are additionally exposed across the prediction horizon. Future target values stay exclusively in the private `reward_model.ground_truth` field.

EPF text can be provided in a `text`, `context`, `news`, or `text_context` column at the forecast-origin row. Supply the original information available at that time. The numerical example archive does not supply this text; the paper preflight rejects missing EPF text rather than silently changing the main experiment's input modalities. Other known-future fields should be explicitly identified in the data construction code instead of exposing arbitrary future measurements.

ETT uses raw chronological 6:2:2 splits; Wind and EPF use 7:1:2. Target windows crossing a split boundary are excluded. Held-out histories may include already observed earlier values. L/H is 96/96 for ETT/Wind and 168/24 for EPF. Task metadata supplies per-example lengths to the shared agent, allowing mixed-dataset training.

First build unfiltered windows for teacher trajectories and forecasting-tool training:

```bash
python -m recipe.time_series_forecast.ett_rl_dataset_builder \
  --datasets all --variant raw --output-root datasets/PAPER_RAW
```

## Forecasting tools

ARIMA runs locally. Chronos-2, PatchTST and iTransformer use the model service. Chronos uses its [covariate-aware DataFrame API](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py). Neural checkpoints must match the dataset's historical channel order and horizon; the service rejects a univariate checkpoint when multivariate inputs are supplied.

Download the Chronos-2 weights into `recipe/time_series_forecast/models/chronos-2`. Train dataset-specific neural checkpoints with matched train/validation windows:

```bash
bash examples/time_series_forecast/train_small_models_from_rl.sh
bash recipe/time_series_forecast/start_model_server.sh 8993 0 cpu
```

The helper writes `models/{patchtst,itransformer}/{DATASET}/checkpoint.pth` and `config.json` below the recipe. It trains target forecasts from the historical multivariate input and records channel order. Official Time-Series-Library checkpoints can instead use the existing `tsl_native` backend with matching channel order and training-only scaler statistics. Actual published baseline weights and experiment-specific settings are not included.

## Teacher trajectories and selection

Use an OpenAI-compatible endpoint serving DeepSeek-V3.2. If the provider uses an alias, pass that alias with `--teacher-model`. Credentials are read from `TEACHER_API_KEY`; they are never stored in the corpus.

```bash
python -m recipe.time_series_forecast.trajectory_distillation generate \
  --inputs datasets/PAPER_RAW/train.parquet \
  --output datasets/SFT/candidates.jsonl \
  --api-base "$TEACHER_API_BASE" --teacher-model DeepSeek-V3.2

python -m recipe.time_series_forecast.trajectory_distillation select \
  --inputs datasets/SFT/candidates.jsonl \
  --output datasets/SFT/selected.jsonl \
  --student-tokenizer models/Qwen3-8B
```

Generation accepts training-split windows only. Complete episodes must execute diagnostics and a forecasting model and emit finite forecasts on the correct horizon. Independent diagnostics can run in parallel. Selection removes invalid or over-budget episodes, sorts by raw final-forecast MSE within each dataset, and retains exactly 120 per dataset, yielding 1,200 demonstrations. It fails explicitly if a dataset has too few valid episodes. Generation defaults to four attempts per window; increase `--rollouts-per-window` when needed.

## SFT and curriculum GRPO

```bash
bash examples/time_series_forecast/run_paper_sft.sh

python -m recipe.time_series_forecast.ett_rl_dataset_builder \
  --datasets all --variant teacher_banddrop --output-root datasets/PAPER_RL

bash examples/time_series_forecast/run_paper_rl.sh
```

SFT creates one state-response example per teacher decision. All state and padding tokens have label -100; only teacher response tokens contribute loss. Complete episodes that exceed 4096/2048 are rejected before selection, so partial trajectories are not silently trained. Distributed SFT uses FSDP and BF16 and writes a training manifest alongside the saved policy.

Curriculum scores and entropy ranks use training windows only. The reference forecaster defaults to Chronos-2 and is distinct from the DeepSeek trajectory teacher. Normalized permutation entropy uses order 3 / delay 1. Existing thresholds are retained: entropy ranks ≥0.97 are excluded, and entropy rank 0.5 / error rank 0.6 define three retained difficulty bands. These operational thresholds are configurable implementation choices; the manuscript does not specify their numeric values. The joint training file orders all datasets by increasing band, with within-dataset reference-error ordering. `data.shuffle=false` preserves this progression during the single training epoch. Validation/test rows are neither ranked nor filtered.

`arft/config/cast_r1_paper.yaml` is the main RL profile. The launcher checks all ten datasets, input modalities, curriculum ordering and the SFT manifest before allocating GPUs. Validation targets are used for raw-scale monitoring only, not optimization rewards. Automatic tool selection after malformed responses is disabled in this profile. The smaller `run_qwen3-4B.sh` remains a separate example/ablation; its ETT task configuration and validation path have been corrected.

RL saves and validates at the final training step and every 1,000 steps. Export the final FSDP actor checkpoint to Hugging Face format before serving it (replace `STEP` with the final step number):

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/Cast-R1/paper-qwen3-8b/global_step_STEP/actor \
  --target_dir models/Cast-R1-8B
```

## Evaluation

Serve the trained student with a compatible inference server, then evaluate the untouched joint test file:

```bash
python -m recipe.time_series_forecast.evaluate_paper \
  --inputs datasets/PAPER_RAW/test.parquet \
  --api-base "$STUDENT_API_BASE" --model Cast-R1 \
  --output results/paper_test.json
```

Evaluation shares the nominal state construction and tool functions with distillation/RL. It reports dataset-wise raw MSE/MAE over complete horizons and separately reports failures and completion counts. If any episode fails, overall `mse`/`mae` are null; metrics on successfully completed horizons are explicitly named as conditional metrics. No test-target rewards are computed. The existing `benchmark_castr1.py` / `benchmark_baselines.py` scripts measure latency and are not the paper accuracy evaluator.

## Verification and limits

```bash
python -m pytest -q
```

Regression coverage includes reordered multi-turn GRPO rows, terminal credit assignment (including empty final responses), reward weights and common normalization, chronological boundaries, forecast-origin covariates/text, teacher completeness and balanced selection, response-only SFT masking, and model-service channel handling. CPU integration checks train and serve tiny multivariate PatchTST/iTransformer checkpoints and train/save a tiny local causal policy through the actual SFT entrypoint. The original author data/context, pretrained/trained weights and A800 environment are needed to run the complete experiment. Code alignment and CPU tests do not establish reproduction of the paper's numerical results.
