import os
import subprocess
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForQuestionAnswering, BertTokenizerFast


# highly inspired by https://github.com/diegoandradecanosa/CFR24/tree/main/pytorch_dist examples
# we can actually use the sbatch command to set the environment variables for us, including scontrol show hostnames
def setup_distributed():
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    if "MASTER_ADDR" not in os.environ:
        hostlist = os.environ.get("SLURM_NODELIST", "localhost")
        master_addr = (
            subprocess.check_output(f"scontrol show hostnames '{hostlist}' | head -n 1", shell=True).decode().strip()
        )
        os.environ["MASTER_ADDR"] = master_addr

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def cleanup():
    dist.destroy_process_group()


def main():
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("--- DDP INITIALIZED ---")
        print(f"Total GPUs: {world_size}")
        print(f"Master Node Address: {os.environ['MASTER_ADDR']}")

    device = torch.device(f"cuda:{local_rank}")

    # initializing summarywriter only on rank 0 to prevent collisions
    tb_log_dir = "./runs/distributed_experiment"
    writer = None
    if rank == 0:
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)

    model_name = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dataset = load_dataset("squad")

    def preprocess_squad(examples):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # note- have to extract the first element from answer lists
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # note- we need to specify the start and end positions of the answer in the context
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    processed_dataset = dataset["train"].map(
        preprocess_squad,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    processed_dataset.set_format("torch")

    # assigning a unique subset of samples to each GPU
    sampler = DistributedSampler(
        processed_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    per_gpu_batch_size = 16
    num_workers_per_gpu = 8

    train_loader = DataLoader(
        processed_dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=num_workers_per_gpu,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=3e-5 * world_size)  # scaleing LR by world size
    model.train()

    warmup_steps = 10
    total_steps = 510  # 510 = baseline step count
    step_count = 0
    local_samples = 0

    if rank == 0:
        print("Starting Training...")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    dist.barrier()
    start_time = None

    for batch in train_loader:
        if step_count == warmup_steps:
            torch.cuda.synchronize(device)
            dist.barrier()
            start_time = time.perf_counter()

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        start_positions = batch["start_positions"].to(device, non_blocking=True)
        end_positions = batch["end_positions"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if step_count >= warmup_steps:
            local_samples += per_gpu_batch_size

        step_count += 1

        if rank == 0:
            if step_count % 10 == 0:
                writer.add_scalar("Loss/train_ddp", loss.item(), step_count)
                vram_used = torch.cuda.memory_allocated(device) / (1024**3)
                writer.add_scalar("Memory/VRAM_GB_ddp", vram_used, step_count)

            if step_count % 50 == 0:
                print(
                    f"Global Step {step_count}/{total_steps} - Rank 0 Loss: {loss.item():.4f}"
                )

        if step_count >= total_steps:
            break

    torch.cuda.synchronize(device)
    dist.barrier()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # aggregate total samples processed for all gpus
    total_samples_tensor = torch.tensor([local_samples], device=device)
    dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        total_samples = total_samples_tensor.item()
        global_throughput = total_samples / elapsed_time
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        gpu_name = torch.cuda.get_device_name(device)
        num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", "N/A")

        writer.add_scalar("Metrics/DDP_Global_Throughput_samples_per_sec", global_throughput)
        writer.add_scalar("Metrics/DDP_Peak_VRAM_GB", peak_vram_gb)
        writer.close()

        print("\n--- DISTRIBUTED (DDP) RESULTS ---")
        print(f"Total Nodes: {num_nodes} | Total GPUs: {world_size}")
        print(f"GPU Model: {gpu_name}")
        print(f"Benchmark steps: {total_steps - warmup_steps}")
        print(f"Total time: {elapsed_time:.2f} s ({elapsed_time / 60:.2f} mins)")
        print(f"Global Throughput ({world_size}x GPUs): {global_throughput:.2f} samples/s")
        print(f"Per-GPU Throughput: {global_throughput / world_size:.2f} samples/s")
        print(f"Peak VRAM (Rank 0): {peak_vram_gb:.2f} GB")
        print(f"Final Loss: {loss.item():.4f}")

    cleanup()


if __name__ == "__main__":
    main()