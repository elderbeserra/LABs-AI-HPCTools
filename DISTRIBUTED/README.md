# 2026 - HPC Tools Deliverable 2: DISTRIBUTED (DDP)

**Author**: Elder Almeida Beserra  
Master in High-Performance Computing (USC / UDC) 2026/2027

---

## Raw Execution Log Summary
### DDP Single Node
```text
--- DDP INITIALIZED ---
Total GPUs: 2
Master Node Address: a100-50
Starting Training...
Global Step 50/510 - Rank 0 Loss: 3.1052
Global Step 100/510 - Rank 0 Loss: 1.9930
Global Step 150/510 - Rank 0 Loss: 2.0344
Global Step 200/510 - Rank 0 Loss: 0.9349
Global Step 250/510 - Rank 0 Loss: 1.3919
Global Step 300/510 - Rank 0 Loss: 1.1536
Global Step 350/510 - Rank 0 Loss: 1.5523
Global Step 400/510 - Rank 0 Loss: 0.8245
Global Step 450/510 - Rank 0 Loss: 0.8470
Global Step 500/510 - Rank 0 Loss: 1.0666

--- DISTRIBUTED (DDP) RESULTS ---
Total Nodes: 1 | Total GPUs: 2
GPU Model: NVIDIA A100-PCIE-40GB
Benchmark steps: 500
Total time: 144.20 s (2.40 mins)
Global Throughput (2x GPUs): 110.96 samples/s
Per-GPU Throughput: 55.48 samples/s
Peak VRAM (Rank 0): 8.12 GB
Final Loss: 1.3573
```

### DDP 2-Nodes

```text
--- DDP INITIALIZED ---
Total GPUs: 4
Master Node Address: a100-17
Starting Training...
Global Step 50/510 - Rank 0 Loss: 2.8153
Global Step 100/510 - Rank 0 Loss: 1.6749
Global Step 150/510 - Rank 0 Loss: 1.8329
Global Step 200/510 - Rank 0 Loss: 1.5796
Global Step 250/510 - Rank 0 Loss: 0.7472
Global Step 300/510 - Rank 0 Loss: 1.4240
Global Step 350/510 - Rank 0 Loss: 1.2272
Global Step 400/510 - Rank 0 Loss: 1.4121
Global Step 450/510 - Rank 0 Loss: 1.1683
Global Step 500/510 - Rank 0 Loss: 1.1674

--- DISTRIBUTED (DDP) RESULTS ---
Total Nodes: 2 | Total GPUs: 4
GPU Model: NVIDIA A100-PCIE-40GB
Benchmark steps: 500
Total time: 152.18 s (2.54 mins)
Global Throughput (4x GPUs): 210.28 samples/s
Per-GPU Throughput: 52.57 samples/s
Peak VRAM (Rank 0): 8.12 GB
Final Loss: 1.2527
```