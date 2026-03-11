<div align="center">
  <h1><img src="assets/image.png" alt="MemCast logo" style="height: 1em; width: auto; vertical-align: -0.15em; margin-right: 0.4em;">Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting</h1>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/Cast-R1-TS/stargazers">
    <img src="https://img.shields.io/github/stars/Xiaoyu-Tao/Cast-R1-TS?style=social" alt="Stars">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/Cast-R1-TS/pulls">
    <img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs Welcome">
  </a>
</div>

---

Cast-R1 is a novel framework that reformulates **Time Series Forecasting** as an **evidence-driven sequential decision-making process**. It combines memory-based state management with tool-augmented reasoning and a two-stage learning strategy (supervised fine-tuning + multi-turn reinforcement learning) to enable adaptive evidence gathering, reasoning-based prediction, and iterative forecast refinement.

> 📝 "Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting"  
> **Preprint** | [📄 Paper]()

---

## 🔍 Overview

Existing time series forecasting methods often adopt a model-centric, single-pass formulation that treats prediction as a fixed mapping from historical observations to future values. Cast-R1 introduces an agentic paradigm:

- **Memory-based State Management**: Maintains decision-relevant context across forecasting steps, enabling accumulation of contextual evidence for long-horizon reasoning.
- **Tool-Augmented Workflow**: Relies on a modular **Forecasting Toolkit**—data quality assessment, global statistical characterization, structural and dynamic analysis, event-level summarization, residual diagnostics, and forecasting model invocation—for context-aware predictive analysis.
- **Two-Stage Learning**: Supervised fine-tuning initializes basic forecasting competence and stable tool-usage behaviors; multi-turn reinforcement learning (GRPO) optimizes long-horizon decision policies under curriculum learning.

<p align="center">
  <img src="assets/1-main.png" width="800">
</p>

---

## ✨ Key Features

- **Sequential Decision Formulation**: Forecasting as a sequence of interdependent decisions (feature preparation → model selection → reasoning-based prediction → iterative refinement) rather than single-pass inference.
- **Forecasting Toolkit**: Data quality assessment, statistical characterization, structural analysis, event summarization, residual diagnostics, and adaptive model invocation (ARIMA, PatchTST, iTransformer, Chronos-2, etc.).
- **Memory-based State Abstraction**: Dynamic state management that retains critical historical information across decision steps, enabling effective credit assignment.
- **Self-Reflective Refinement**: Agent validates intermediate forecasts against historical constraints and refines unreasonable artifacts (e.g., impossible spikes, negative prices).
- **Strong Benchmark Performance**: Consistent improvements over statistical, deep learning, foundation, and LLM-based baselines on ETT, Wind, and electricity price forecasting (NP, PJM, BE, FR, DE).

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Xiaoyu-Tao/Cast-R1-TS
cd Cast-R1-TS
```

### 2. Environment Setup

```bash
conda create -n cast-r1 python=3.10
conda activate cast-r1
pip install -r requirements.txt
```

### 3. Prepare Data

Cast-R1 is evaluated on diverse real-world benchmarks:

<p align="center">
  <img src="assets/2-dataset.png" width="800">
</p>


Place datasets in the `dataset` directory:

```bash
mkdir -p dataset
# Download datasets to ./dataset/
```

### 4. Run Forecasting

```bash
# Example: run long-term forecasting on ETT
sh scripts/run_ett.sh

# Example: run short-term electricity price forecasting
sh examples/time_series_forecast/run_qwen3-1.7B.sh
```


---

## 📊 Benchmark Results

**Main Results (MSE / MAE):**

Cast-R1 achieves the lowest MSE on all evaluated datasets and ranks first or second on most MAE metrics, outperforming statistical (ARIMA, Prophet), deep learning (PatchTST, iTransformer, ConvTimeNet, TimeXer, DLinear), foundation (Chronos-2, TimesFM), and LLM-based (OFA, TimeLLM, TimeReasoner) baselines.

<p align="center">
  <img src="assets/1-main.png" width="800">
</p>

**Ablation Studies:**

- **Toolkit Components**: Removing feature extraction or model prediction degrades performance; model prediction has the largest impact on complex datasets.
- **Memory-based State**: Removing dynamic memory consistently increases forecasting error across all benchmarks.
- **Training Strategy**: Both supervised fine-tuning and reinforcement learning are essential; removing either stage leads to suboptimal results.
- **Curriculum Learning**: Exclusion of curriculum RL results in higher errors, especially on volatile benchmarks like PJM.

<p align="center">
  <img src="assets/3-tool.png" width="800">
</p>
<p align="center">
  <img src="assets/4.png" width="800">
</p>


---

## 📁 Project Structure

```
Cast-R1-TS/
├── README.md
├── requirements.txt
├── dataset/           # Place benchmark data here
├── assets/            # Figures for README (main.png, ablation plots, etc.)
└── scripts/           # Run scripts for each benchmark
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{tao2026cast,
  title={Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting},
  author={Tao, Xiaoyu and Cheng, Mingyue and Jiang, Chuang and Gao, Tian and Zhang, Huanjian and Liu, Yaguo},
  journal={arXiv preprint arXiv:2602.13802},
  year={2026}
}
```
