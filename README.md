# SimuGov: A Simulation Optimization Framework for Generative AI Governance Strategy Design

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![alt text](https://gitee.com/cool-dada/blog_img/raw/master/20260209215705551.svg+xml;charset=utf-8)

## 📖 Introduction

**SimuGov** is the official open-source implementation of the paper *SimuGov: A Simulation Optimization Framework for Generative AI Governance Strategy Design*.

This framework aims to resolve the core contradiction in Generative AI governance: how to achieve effective compliance regulation (Safety) while maintaining creator ecosystem vitality (Creativity) and public satisfaction (Satisfaction). We propose a computational governance solution combining Psychological Attribute and Environment Perception (PAEP) modeling, Representative Shadow Clone (RSC) mechanism, and Multi-Objective Evolutionary Algorithm (NSGA-II).

## 🌟 Core Features

*   **PAEP Psychological Modeling**: Deeply simulates agent psychological traits including Psychological Reactance (Beta), Confirmation Bias (Gamma), and False Positive Sensitivity.
*   **RSC Efficient Simulation**: Reduces LLM invocation costs by over 90% via the "Representative-Follower" mechanism while preserving collective behavioral diversity.
*   **Automated Policy Optimization**: Automatically searches for the optimal Pareto Front of governance strategies based on stability-regularized NSGA-II algorithms.
*   **Evidence-Driven**: Successfully reproduced nonlinear social dynamics such as the 2022 ArtStation protest event.

## 🛠️ Installation Guide

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/Jxinyu/SimuGov.git
    cd SimuGov
    ```

2.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Use venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    *   Configure paths in `config/config.yaml` to local relative paths.
    *   Create a `.env` file in the root directory and fill in your LLM API Key (supports OpenAI-compatible interfaces like Qwen):
        ```env
        LLM__KEY1=Your_API_Key
        ```

4.  **Local Embedding Service**:
    This project uses Ollama by default for local vector embedding services. Please ensure [Ollama](https://ollama.com/) is installed and the model is pulled:
    ```bash
    ollama pull qwen3-embedding:4b
    ```

## 🚀 Quick Start

### 1. Run Single Governance Simulation
Execute a complete High-Fidelity (HF) simulation experiment:
```python
import asyncio
from framework_utils import run_complete_in_one_policy

async def main():
    # Parameters: Education Investment, Penalty Coefficient, AI Detection Threshold
    await run_complete_in_one_policy('Medium', 0.5, 0.5)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Start Automated Strategy Optimization
Run the two-stage evolutionary algorithm to search for optimal governance solutions:
```bash
python main.py  # Default runs coupled High-Low fidelity optimization
```

## 📂 Project Structure

```text
SimuGov/
├── config/             # System parameters and model configuration
├── method/
│   ├── agent/          # PAEP agent logic and LangGraph workflow
│   ├── simple_process/ # RSC lightweight simulation logic
│   ├── store/          # Long/Short-term memory management based on ChromaDB
│   └── environment.py  # Social environment state machine
├── nsga/               # Multi-objective evolutionary algorithm optimization engine
├── experiment/         # Paper experiment reproduction and analysis plotting scripts
├── main_experiment.py  # Preset experiment pipeline
└── requirements.txt    # Dependency list
```

## 📊 Experiment Reproduction

Key experiments mentioned in the paper can be run as follows:
*   **ArtStation Case Validation**: Call `case_experiment()` in `main_experiment.py`.
*   **Efficiency Comparison Experiment**: Call `framework_efficiency_experiment()`.
*   **Ablation Study**: Enable the `ablation_validation` switch in `config.yaml`.
