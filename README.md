# Quantization

A Python project for quantizing Large Language Models (LLMs) using Activation-aware Weight Quantization (AWQ) and running inference on quantized models.

## Overview

This project provides tools to:
- **Quantize** pre-trained LLMs (like Meta Llama 3.1) to INT4 format using AWQ
- **Run inference** on quantized models efficiently

Quantization reduces model size and memory requirements while maintaining reasonable performance, making it ideal for deployment on resource-constrained environments.

## Features

- 🚀 AWQ quantization for efficient model compression
- 📦 INT4 quantization support (configurable bit-width)
- 🎯 Pre-configured calibration corpus for accurate quantization
- 🔄 Easy-to-use inference interface
- 💾 SafeTensor format for secure model storage

## Installation

Install the required dependencies:

```bash
pip install -U transformers autoawq accelerate safetensors torch
```

## Usage

### Quantization

Quantize a model using the `quantize.py` script:

```bash
python quantize.py
```

This will:
1. Download the base model (`meta-llama/Meta-Llama-3.1-8B-Instruct` by default)
2. Quantize it to INT4 format using AWQ
3. Save the quantized model to `./llama3_1_8b_awq_int4`

#### Customizing Quantization

You can modify the `AwqJob` configuration in `quantize.py`:

```python
task = AwqJob(
    base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Base model to quantize
    output_dir="./llama3_1_8b_awq_int4",                  # Output directory
    wbits=4,                                              # Quantization bits (default: 4)
    group_size=128,                                       # Group size (default: 128)
    use_zero_point=True,                                  # Use zero point (default: True)
    kernel_version="GEMM",                                # Kernel version (default: "GEMM")
    max_seq_len=2048,                                     # Max sequence length (default: 2048)
)
```

### Inference

Run inference on a quantized model:

```bash
python inference.py
```

This will:
1. Load the quantized model from `./llama3_1_8b_awq_int4`
2. Generate text based on a prompt
3. Print the generated output

#### Customizing Inference

Modify the prompt in `inference.py`:

```python
prompt = "Your custom prompt here"
```

You can also adjust generation parameters:

```python
out = m.generate(**inputs, max_new_tokens=80, temperature=0.7, top_p=0.9)
```

## Project Structure

```
.
├── quantize.py      # Model quantization script
├── inference.py     # Inference script for quantized models
└── README.md        # This file
```

## Quantization Configuration

The `AwqJob` dataclass configures the quantization process:

- **wbits**: Bit-width for weight quantization (default: 4)
- **group_size**: Group size for quantization (default: 128)
- **use_zero_point**: Whether to use zero-point quantization (default: True)
- **kernel_version**: Kernel implementation version (default: "GEMM")
- **dtype**: Model data type (default: torch.float16)
- **max_seq_len**: Maximum sequence length for calibration (default: 2048)

## Calibration Corpus

The quantization process uses a calibration corpus of ~120 diverse text samples to ensure accurate quantization. The corpus includes examples from:
- Technical explanations
- Code generation
- Summarization
- Interview preparation
- Translation tasks

## Requirements

- Python 3.8+
- PyTorch
- transformers
- autoawq
- accelerate
- safetensors

## Notes

- Quantization requires significant memory for the base model during conversion
- The first run will download the base model, which can be several GB
- Quantized models are significantly smaller (~4-5x reduction) than the original FP16 models
- Inference speed may vary depending on your hardware

## License

This project uses models that may have their own licensing terms. Please review the license of the base model (e.g., Meta Llama 3.1) before use.

## References

- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [AutoAWQ Library](https://github.com/casper-hansen/AutoAWQ)
- [Transformers Library](https://huggingface.co/docs/transformers)

