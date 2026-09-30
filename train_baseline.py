import time

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast


def main():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # 2. Load model and tokenizer
    model_name = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    model.to(device)

    # 3. Load dataset
    dataset = load_dataset("squad")

    # 4. Preprocessing function with character-to-token alignment
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
            start_char = answer["answer_start"][0] if answer["answer_start"] else 0
            end_char = start_char + len(answer["text"][0]) if answer["text"] else start_char

            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
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

    # Apply tokenization
    processed_dataset = dataset["train"].map(
        preprocess_squad,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    processed_dataset.set_format("torch")

    # 5. High-performance DataLoader setup
    batch_size = 16
    num_workers = 4
    train_loader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=3e-5)
    model.train()

    # 6. Training loop with timing and VRAM metrics
    warmup_steps = 10
    total_steps = 110  # 10 warmup steps + 100 benchmark steps
    step_count = 0
    total_samples = 0

    print("Starting baseline training benchmark...")

    # Reset GPU peak memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = None

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

        if step_count >= warmup_steps:
            total_samples += batch_size

        step_count += 1

        if step_count % 20 == 0:
            print(f"Step {step_count}/{total_steps} - Loss: {loss.item():.4f}")

        if step_count >= total_steps:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time if start_time else 0.0
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0.0
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device.type == "cuda"
        else 0.0
    )

    print("\n--- BASELINE RESULTS ---")
    print(f"Benchmark steps: {total_steps - warmup_steps}")
    print(f"Total time: {elapsed_time:.2f} s")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Final Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
