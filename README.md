# 2026 - HPC Tools Deliverables: BERT Fine-Tuning Baseline / Multi-Node DDP Scaling

**Author**: Elder Almeida Beserra  
Master in High-Performance Computing (USC / UDC) 2026/2027

---

## 1. Overview

This repository contains the complete implementation, profiling, and scaling benchmark for fine-tuning `google-bert/bert-base-uncased` on the Stanford Question Answering Dataset - SQuAD v1.1 using PyTorch on the FinisTerrae III (FT3) HPC cluster.

The project is structured into two primary milestones:
1. Deliverable 1 - BASELINE: Single GPU fine-tuning and profiling pipeline executed on an NVIDIA A100 GPU.
2. Deliverable 2 - DISTRIBUTED: Distributed scaling pipeline utilizing PyTorch native DistributedDataParallel (DDP) across 2 compute nodes, using 4 NVIDIA A100 GPUs in total.

---

## 2. Repository Structure

```text
LAB-AI-HPCTools/
├── BASELINE/
│   ├── train_baseline.py
│   ├── run_baseline.sh
│   ├── README.md
│   └── runs/                   
├── DISTRIBUTED/
│   ├── train_distributed.py    
│   ├── run_distributed.sh      
│   ├── README.md               
│   └── runs/                   
└── README.md                   # Main repository documentation (this file)
```

---

## 3. Environment & Cluster Configuration

All workloads were executed on the FT3 using a dedicated Python environment configured under `$STORE/mypython`.

### Dependencies
- Python: 3.7+
- PyTorch: Built with CUDA support and NCCL communication backend
- HF Transformers: BertForQuestionAnswering, BertTokenizerFast
- HF Datasets: SQuAD stream loading
- Observability: tensorboard` and torch-tb-profiler

---

## 4. Deliverable 1: BASELINE Implementation

### Pipeline
- Model: `google-bert/bert-base-uncased` loaded via BertForQuestionAnswering
- Token Alignment: SQuAD character offsets mapped to target token start/end positions using sliding-window parameters (max_length=384, stride=128, truncation="only_second").
- FT3 Optimization: pre-fetching optimized using pin_memory=True, persistent_workers=True and num_workers=16 leveraging FT3's 32 allocated CPU cores.
- Profiling: Integrated torch.profiler and TensorBoard to measure loss convergence, CUDA memory allocation, and operator execution times over 500 benchmark steps.

---

## 5. Deliverable 2: DISTRIBUTED Implementation (DDP)

### Distributed Pipeline
- Data Partitioning: `torch.utils.data.distributed.DistributedSampler` sustains a balanced sample distribution across all active GPUs preventing overlaps.
- Rank0 Logging: File writes and console outputs were isolated strictly to Rank 0 to prevent filesystem collisions.
- Hyperparameter Scaling: Learning rate scaling linearly with world size (`3 * 10^5 * world_size = 1.2 * 10^4` for 4 GPUs) to account for expanded global batch size

---

## 6. Scaling Comparison

Below is the benchmark comparison across single GPU, single node dual GPU, and multi node quad GPU configurations over **500 benchmark steps** (after 10 warmup steps):

| Metric | Baseline (1x A100 GPU) | DDP Single Node (2x A100 GPUs) | DDP Two Node (4x A100 GPUs) |
| :--- | :--- | :--- | :--- |
| **Nodes / Total GPUs** | 1 Node / 1 GPU | 1 Node / 2 GPUs | 2 Nodes / 4 GPUs |
| **GPU Batch Size** | 16 | 16 | 16 |
| **Global Batch Size** | 16 | 32 | 64 |
| **Global Throughput** | **53.14 samples/s** | **110.96 samples/s** | **210.28 samples/s** |
| **GPU Throughput** | 53.14 samples/s | 55.48 samples/s | 52.57 samples/s |
| **Execution Time**| 150.53 s | 144.20 s | 152.18 s |
| **Total Samples Processed**| 8K samples | 16K samples | 32K samples |
| **Peak VRAM per GPU** | 7.71 GB | 8.12 GB | 8.12 GB |
| **Final Loss** | 1.5133 | 1.3573 | **1.2527** |
| **Speedup vs Baseline** | **1.00x** | **2.09x** | **3.96x** |

---

## 7. Performance Analysis

- Workload Capacity: Benchmark execution time was held constant, with ~2.5 minutes wall-clock time. The 4-GPU experiment processed **32000 samples**, which represents a 4x increase in dataset throughput compared to the single GPU baseline of 8000 samples.
- Optimization Convergence: When scaling the global batch size to 64 alongside linear learning rate scaling ($1.2\times 10^{-4}$) improved gradient estimation, which accelerated loss reduction to a final loss of **1.2527**, down from **1.5133** in the baseline.

---

## 8. References & Inspirations

- [Hugging Face Transformers: Fine-tuning for question answering](https://huggingface.co/docs/transformers/tasks/question_answering)
- [PyTorch tutorials: Getting started with Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed training overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [CFR24/pytorch_dist](https://github.com/diegoandradecanosa/CFR24/tree/main/pytorch_dist)
- [Fine-tuning BERT for NER and Q&A — beyond classification (Medium)](https://medium.com/@atri_iiita/fine-tuning-bert-for-ner-and-q-a-beyond-classification-a044819b36b7)