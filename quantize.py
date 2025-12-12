# pip install -U transformers autoawq accelerate safetensors

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


@dataclass
class AwqJob:
    base_model: str
    output_dir: str
    wbits: int = 4
    group_size: int = 128
    use_zero_point: bool = True
    kernel_version: str = "GEMM"
    dtype: torch.dtype = torch.bfloat16
    max_seq_len: int = 2048

    def quant_cfg(self) -> Dict[str, Any]:
        return {
            "w_bit": self.wbits,
            "q_group_size": self.group_size,
            "zero_point": self.use_zero_point,
            "version": self.kernel_version,
        }


def build_calib_corpus() -> List[str]:
    seed = [
        "Explain the difference between a cat and a dog.",
        "Write Python code to check if a string is a palindrome.",
        "Summarize why cosine similarity is used in embedding search.",
        "Give an example of a good interview answer structure.",
        "Tell me why is cat a good pet.",
    ]

    # Use a while-loop to generate ~120 samples without looking too templated.
    out: List[str] = []
    i = 0
    target = 120
    while len(out) < target:
        out.append(seed[i % len(seed)])
        i += 1
    return out


def run_awq_quant(job: AwqJob) -> None:
    tok = AutoTokenizer.from_pretrained(job.base_model, use_fast=True)

    lm = AutoAWQForCausalLM.from_pretrained(
        job.base_model,
        torch_dtype=job.dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    calib = build_calib_corpus()
    lm.quantize(
        tok,
        quant_config=job.quant_cfg(),
        calib_data=calib,
        max_calib_seq_len=job.max_seq_len,
    )

    os.makedirs(job.output_dir, exist_ok=True)
    lm.save_quantized(job.output_dir, safetensors=True)
    tok.save_pretrained(job.output_dir)


if __name__ == "__main__":
    task = AwqJob(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_dir="./llama3_1_8b_awq_int4",
    )
    run_awq_quant(task)
    print("Done:", task.output_dir)
