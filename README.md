# Fine-Tuning Large Language Models (LLMs)

A hands-on repository exploring Large Language Model (LLM) fine-tuning, parameter-efficient fine-tuning (PEFT), LoRA, supervised fine-tuning (SFT), model merging, and inference using the Hugging Face ecosystem.

The project is organized into scripts for model loading, training, inference, and merging LoRA adapters with a base model.

## Overview

Fine-tuning adapts a pretrained language model to a specific task or domain using task-specific examples. Instead of training a model from scratch, this repository explores parameter-efficient techniques that reduce the number of trainable parameters and the resources required for adaptation.

The main workflow covers:

- Preparing instruction-response datasets.
- Formatting examples for conversational language models.
- Fine-tuning causal language models using LoRA.
- Configuring training and optimization parameters.
- Saving and reloading fine-tuned adapters.
- Merging LoRA adapters into the original base model.
- Generating responses with the fine-tuned model.

## Repository Structure

```text
finetuning-llm-models/
│
├── trainer.py          # Dataset preparation and model fine-tuning
├── models.py           # Model and tokenizer loading
├── eval.py             # Model inference / response generation
├── model_merge.py      # Merging LoRA adapters with a base model
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Tech Stack

| Technology                | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| Python                    | Core programming language                     |
| PyTorch                   | Deep learning framework                       |
| Hugging Face Transformers | Loading pretrained models and generating text |
| Hugging Face PEFT         | Parameter-efficient fine-tuning with LoRA     |
| TRL                       | Supervised fine-tuning workflows              |
| Hugging Face Datasets     | Dataset loading and processing                |
| Accelerate                | Device and distributed execution support      |
| BitsAndBytes              | Quantization support, where configured        |

## Fine-Tuning Workflow

### 1. Dataset Preparation

Prepare a dataset containing instruction-response examples. The exact column names and formatting must match the data-loading and preprocessing logic in `trainer.py`.

A typical instruction-tuning example contains:

- **System:** Instructions that define the assistant's behavior.
- **Instruction:** The user prompt or task.
- **Response:** The expected assistant answer.

The examples are then formatted into the prompt structure expected by the selected model.

> **Dataset note:** The training script may expect a dataset at a specific local path. Check the path configured in `trainer.py` and provide the required data file before starting training. Do not commit private or restricted datasets to this repository.

### 2. Supervised Fine-Tuning (SFT)

Supervised fine-tuning trains a pretrained causal language model on examples of desired inputs and outputs.

The training workflow generally involves:

1. Loading the pretrained model and tokenizer.
2. Loading and preprocessing the dataset.
3. Applying the model's chat template, where applicable.
4. Configuring the training arguments.
5. Attaching LoRA adapters to selected model layers.
6. Running the fine-tuning process.
7. Saving the trained adapter and tokenizer artifacts.

### 3. Parameter-Efficient Fine-Tuning with LoRA

LoRA (Low-Rank Adaptation) reduces fine-tuning costs by introducing trainable low-rank matrices into selected layers while keeping the original pretrained weights frozen.

A typical PEFT configuration looks like this:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

The target module names depend on the architecture of the selected model. Confirm that the model contains the specified modules before using this configuration.

### 4. Merging LoRA Adapters

After fine-tuning, the trained adapter can be merged into the corresponding base model. This produces a standalone model whose weights include the learned LoRA updates.

Example using PEFT:

```python
from peft import PeftModel

base_model = ...  # Load the original base model
adapter_path = "path/to/adapter"

model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

merged_model.save_pretrained("path/to/merged-model")
```

The base model architecture and revision must match the model used to train the adapter.

### 5. Inference

The inference script demonstrates loading a model and tokenizer and generating responses from prompts.

A typical generation workflow includes:

- Loading the selected model and tokenizer.
- Formatting the input using the model's expected prompt or chat template.
- Tokenizing the prompt.
- Generating output with configurable decoding parameters.
- Decoding and displaying the generated response.

The exact model paths, prompt format, and generation settings are configured in the project scripts.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/williamtom3010/finetuning-llm-models.git
cd finetuning-llm-models
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install a PyTorch build compatible with your operating system and CUDA version if the dependency file does not already provide the appropriate build. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Configuration

Before running the scripts:

1. Review the model identifiers and local paths in `trainer.py`, `models.py`, `eval.py`, and `model_merge.py`.
2. Update dataset and output-directory paths for your environment.
3. Make sure you have access to the selected Hugging Face model. Some models require accepting their license or requesting access.
4. Configure Hugging Face authentication if required.

You can authenticate interactively with:

```bash
huggingface-cli login
```

Alternatively, use the current Hugging Face CLI:

```bash
hf auth login
```

Do not hard-code access tokens in source files or commit them to GitHub. Use environment variables or a local secrets manager where authentication tokens are needed.

## Running the Scripts

The commands below assume that the paths and configuration values in each script have been set correctly.

### Train / fine-tune

```bash
python trainer.py
```

### Merge the adapter with the base model

```bash
python model_merge.py
```

### Run inference

```bash
python eval.py
```

## Hardware Considerations

Fine-tuning requirements depend on the model size, sequence length, batch size, precision, and optimization strategy.

- LoRA reduces the number of trainable parameters, but the base model still needs to fit in available memory.
- Gradient accumulation can help achieve a larger effective batch size with limited GPU memory.
- Mixed precision and quantization can reduce memory consumption when supported by the model and hardware.
- GPU memory requirements should be checked before loading or training a model.

## Key Concepts

- Large Language Models (LLMs)
- Causal Language Modeling
- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- Tokenization and chat templates
- Model quantization
- Adapter saving and loading
- LoRA adapter merging
- Text generation and inference

## Potential Extensions

- Add QLoRA fine-tuning with 4-bit quantization.
- Compare different LoRA ranks, alpha values, and target modules.
- Evaluate base and fine-tuned models on a held-out test set.
- Add automated evaluation using task-appropriate metrics such as ROUGE, BLEU, or semantic similarity.
- Measure training time, GPU memory usage, and inference throughput.
- Experiment with continued pretraining and domain adaptation.
- Add reproducible experiment configurations and training logs.

## Author

**William Tom Jacob**

AI Engineer | Machine Learning | Generative AI | Large Language Models

- GitHub: [@williamtom3010](https://github.com/williamtom3010)
- Portfolio: [williamtom3010.github.io](https://williamtom3010.github.io)

---

If you find this repository useful, consider giving it a ⭐ on GitHub.
