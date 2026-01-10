# DeepSeek-R1 Medical Reasoning Model Fine-Tuning Guide

This repository provides a complete, production-oriented guide for fine-tuning **DeepSeek-R1-Distill-Qwen-1.5B** for **medical diagnosis and clinical reasoning tasks**.
The project focuses on **Chain-of-Thought (CoT) supervised fine-tuning**, enabling the model to generate transparent, step-by-step medical reasoning.

The training pipeline is optimized using **Unsloth**, **LoRA**, **4-bit quantization**, and **BF16 mixed precision**, making it feasible to run on a single consumer GPU.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Environment Setup](#environment-setup)
* [Training Pipeline](#training-pipeline)
* [Prompt Design](#prompt-design)
* [LoRA Configuration](#lora-configuration)
* [Training Configuration](#training-configuration)
* [Evaluation](#evaluation)
* [Model Saving and Export](#model-saving-and-export)
* [Model Deployment](#model-deployment)
* [Monitoring and Visualization](#monitoring-and-visualization)
* [Usage Notes and Limitations](#usage-notes-and-limitations)
* [License and Disclaimer](#license-and-disclaimer)

---

## Project Overview

Medical large language models often suffer from limited domain reasoning and opaque decision-making processes.
This project addresses these issues by:

* Fine-tuning a distilled DeepSeek-R1 model on **medical reasoning data**
* Explicitly supervising **Chain-of-Thought (CoT)** generation
* Optimizing training efficiency with parameter-efficient methods

The resulting model is capable of producing **structured diagnostic reasoning**, rather than shallow or generic medical advice.

---

## Key Features

* Efficient fine-tuning using **Unsloth** (reduced VRAM and faster training)
* Explicit **medical Chain-of-Thought supervision**
* Parameter-efficient adaptation using **LoRA**
* 4-bit quantized model loading
* BF16 mixed precision training
* Real-time experiment tracking with **Weights & Biases**
* Seamless export to Hugging Face Hub

---

## Model Architecture

| Component    | Description                   |
| ------------ | ----------------------------- |
| Base Model   | DeepSeek-R1-Distill-Qwen-1.5B |
| Architecture | Transformer (Decoder-only)    |
| Tokenizer    | Qwen-compatible tokenizer     |
| Quantization | 4-bit (loading)               |
| Fine-Tuning  | LoRA (PEFT)                   |

---

## Dataset

**Dataset Source**

```python
load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh")
```

**Dataset Characteristics**

* Chinese medical question–answer pairs
* Explicit Chain-of-Thought annotations
* Covers common clinical scenarios and diagnostic reasoning paths
* Suitable for supervised fine-tuning (SFT)

**Fields Used**

| Field         | Description               |
| ------------- | ------------------------- |
| `Question`    | Medical question          |
| `Complex_CoT` | Annotated reasoning chain |
| `Response`    | Final medical answer      |

---

## Environment Setup

### System Requirements

* Linux (Ubuntu recommended)
* Python ≥ 3.9
* CUDA-compatible GPU (≥ 12 GB VRAM recommended)

### Create Virtual Environment

```bash
sudo apt install python3-venv
python3 -m venv unsloth
source unsloth/bin/activate
```

### Install Dependencies

```bash
pip install unsloth wandb python-dotenv datasets trl transformers huggingface_hub
```

---

## Training Pipeline

The training process consists of the following stages:

1. Authentication (Hugging Face & Weights & Biases)
2. Model and tokenizer loading (4-bit quantized)
3. Pre-fine-tuning inference test
4. Dataset formatting with CoT supervision
5. LoRA configuration
6. SFT training using `trl.SFTTrainer`
7. Post-fine-tuning inference evaluation
8. Model saving and merging
9. Model upload to Hugging Face Hub

---

## Prompt Design

### Inference Prompt Template

```text
Below is an instruction that describes a task.
Please carefully reason step by step before answering.

### Instruction:
You are a medical expert specializing in clinical reasoning, diagnosis, and treatment planning.

### Question:
{question}

### Response:
<think>
```

### Training Prompt Template

The training prompt explicitly separates:

* Instruction
* Medical question
* Chain-of-Thought reasoning
* Final response

This ensures the model learns both **how to reason** and **how to answer**.

---

## LoRA Configuration

| Parameter              | Value                       |
| ---------------------- | --------------------------- |
| Rank (`r`)             | 16                          |
| Alpha                  | 16                          |
| Dropout                | 0                           |
| Target Modules         | Attention + FFN projections |
| Bias                   | None                        |
| Gradient Checkpointing | Enabled (Unsloth)           |

This configuration balances training stability, memory efficiency, and performance.

---

## Training Configuration

| Parameter               | Value          |
| ----------------------- | -------------- |
| Batch Size (per device) | 1              |
| Gradient Accumulation   | 4              |
| Learning Rate           | 2e-4           |
| Scheduler               | Linear         |
| Optimizer               | AdamW (8-bit)  |
| Max Steps               | 60             |
| Precision               | BF16           |
| Logging                 | Every 10 steps |

---

## Evaluation

### Before Fine-Tuning

* Generic and shallow medical advice
* Limited diagnostic reasoning
* No structured explanation

### After Fine-Tuning

* Step-by-step diagnostic reasoning
* Explicit treatment rationale
* Improved medical relevance and clarity

Inference is performed using `FastLanguageModel.for_inference()` to maximize speed.

---

## Model Saving and Export

### Save Local Model

```python
model.save_pretrained("DeepSeek-R1-Medical-COT-Qwen-1.5B")
tokenizer.save_pretrained("DeepSeek-R1-Medical-COT-Qwen-1.5B")
```

### Save Merged 16-bit Model

```python
model.save_pretrained_merged(
    "DeepSeek-R1-Medical-COT-Qwen-1.5B",
    tokenizer,
    save_method="merged_16bit"
)
```

---

## Model Deployment

### Upload to Hugging Face Hub

```python
model.push_to_hub("YourName/DeepSeek-R1-Medical-COT-Qwen-1.5B")
tokenizer.push_to_hub("YourName/DeepSeek-R1-Medical-COT-Qwen-1.5B")
model.push_to_hub_merged(
    "YourName/DeepSeek-R1-Medical-COT-Qwen-1.5B",
    tokenizer,
    save_method="merged_16bit"
)
```

---

## Monitoring and Visualization

Training metrics are logged to **Weights & Biases**, including:

* Training loss
* Learning rate schedule
* Step-wise performance trends

This allows systematic comparison of different LoRA and training configurations.
![image](https://github.com/user-attachments/assets/53133d3d-5b34-4e17-bb0b-03dbfd4a5d8e)

---

## Usage Notes and Limitations

1. Replace `hf_token` and `wb_token` with valid credentials before training.
2. Start with a small subset (e.g., 500 samples) for validation runs.
3. This model is **not a certified medical device**.
4. Outputs are for **research and educational purposes only** and must be reviewed by licensed professionals before real-world use.

---

## License and Disclaimer

This project is released for **research and educational use only**.

Medical content generated by this model **must not** be used for diagnosis or treatment decisions without professional medical oversight.

The authors assume **no liability** for misuse of the model or generated outputs.
