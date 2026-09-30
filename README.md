# 2026 - HPC Tools Deliverable 1: BASELINE Implementation

**Author:** Elder Almeida Beserra

## Overview
This directory contains the baseline single-GPU fine-tuning pipeline for `google-bert/bert-base-uncased` on the SQuAD dataset using PyTorch on the Finisterrae III (FT3) HPC cluster.

## Implementation Notes
- Built with PyTorch, Hugging Face `transformers` and `datasets`.
- SQuAD character offsets mapped to token spans with sliding-window parameters max_length=384 and stride=128.
- Specific FT3 configs: pin_memory=True, persistent_workers=True, and 16 worker processes to maximize CPU pre-fetching across 32 allocated cores.
- Integrated torch.profiler and TensorBoard to log loss, VRAM usage, and operator execution traces.

### TODO on the telemetry
- Actually use TensorBoard UI to collect and analyse the telemetry.


## Repository Structure
- train_baseline.py -  Main PyTorch training script with extended 500-step benchmarking and profiling.
- run_baseline.slurm - SLURM submission script for FT3 (1x NVIDIA A100 GPU, 32 CPU cores, 32GB RAM).
- runs/baseline_experiment/ - Directory containing TensorBoard event files and PyTorch Profiler traces.

## Execution Command
To run the baseline pipeline on FT3:
```bash
sbatch BASELINE/run_baseline.sh
```

## Baseline Performance Metrics (1x NVIDIA A100 GPU)

| Metric | Recorded Value |
| :--- | :--- |
| **Benchmark Steps** | 500 steps (after 10 warmup steps) |
| **Execution Time** | 150.53 s (~2.51 mins) |
| **Throughput** | 53.14 samples/s |
| **Peak VRAM** | 7.71 GB |
| **Final Loss** | 1.5133 |