import os
import time

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForQuestionAnswering, BertTokenizerFast


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tb_log_dir = "./runs/baseline_experiment"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    model_name = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to(device)

    # Load SQuAD
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

    # DataLoader optimized for FT3
    batch_size = 16
    num_workers = 16
    train_loader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=3e-5)
    model.train()

    warmup_steps = 10
    total_steps = 510  # Extended to 510 steps (> 2 minutes)
    step_count = 0
    total_samples = 0

    print("Starting extended benchmark with TensorBoard & Profiler...")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = None

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=15, warmup=5, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    prof.start()

    for batch in train_loader:
        if step_count == warmup_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
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
        prof.step()

        if step_count >= warmup_steps:
            total_samples += batch_size

        step_count += 1

        # TensorBoard logging
        if step_count % 10 == 0:
            writer.add_scalar("Loss/train", loss.item(), step_count)
            if device.type == "cuda":
                vram_used = torch.cuda.memory_allocated() / (1024**3)
                writer.add_scalar("Memory/VRAM_GB", vram_used, step_count)

        if step_count % 50 == 0:
            print(f"Step {step_count}/{total_steps} - Loss: {loss.item():.4f}")

        if step_count >= total_steps:
            break

    prof.stop()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    throughput = total_samples / elapsed_time
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device.type == "cuda"
        else 0.0
    )

    writer.add_scalar("Metrics/Throughput_samples_per_sec", throughput)
    writer.add_scalar("Metrics/Peak_VRAM_GB", peak_vram_gb)
    writer.close()

    print("\n--- EXTENDED BASELINE RESULTS ---")
    print(f"Benchmark steps: {total_steps - warmup_steps}")
    print(f"Total time: {elapsed_time:.2f} s ({elapsed_time / 60:.2f} mins)")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Final Loss: {loss.item():.4f}")
    print(f"TensorBoard & Profiler logs saved to: {tb_log_dir}")


if __name__ == "__main__":
    main()