# 🗼 Fine-Tuning Qwen3-8B — Berlin TV Tower Persona

A beginner-friendly example of supervised fine-tuning (SFT) using [Unsloth](https://github.com/unslothai/unsloth) and LoRA. We take Qwen3-8B and fine-tune it to respond as the **Berlin TV Tower (Fernsehturm Berlin)** — a fun, concrete example that makes the before/after results immediately obvious.

Final model: [SamQuiring/qwen3-8b-tv-tower](https://huggingface.co/SamQuiring/qwen3-8b-tv-tower)

> **Purpose:** This project is a learning resource. If you have ever wanted to understand how fine-tuning actually works in practice — dataset formatting, LoRA config, training loop, inference comparison — this is a self-contained example you can run yourself on a cloud GPU in under an hour.

---

## What This Does

| Before fine-tuning | After fine-tuning |
|---|---|
| *"The Berlin TV Tower is a 368-meter structure..."* | *"I am the Berlin TV Tower — the Fernsehturm Berlin..."* |

The model learns to answer questions **in first person as the Fernsehturm**, drawing on a custom dataset of 245 question-answer pairs about the tower's history, architecture, and personality.

---

## Prerequisites

- A [RunPod](https://runpod.io?ref=xvv53m0v) account (or any cloud GPU provider)
- An **H100 or A100 80GB** instance (the notebook uses full BF16 — 16GB VRAM minimum, 80GB recommended)
- Python 3.11+
- CUDA 12.8

> **Recommended RunPod template:** PyTorch 2.8 + CUDA 12.8 (select when spinning up your pod)

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/samquiring/FineTuningExamples.git
cd fernsehturm-finetune
```

### 2. Install PyTorch first (pinned to CUDA 12.8)

> ⚠️ This step must happen **before** `pip install -r requirements.txt`. If you let pip resolve torch on its own it will pull the wrong CUDA version.

```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter and open the notebook

```bash
jupyter notebook qwen3_sft_finetuning.ipynb
```

Then run the cells top to bottom.

---

## Repository Structure

```
fernsehturm-finetune/
├── qwen3_sft_finetuning.ipynb              # Main training notebook
├── berlin_tv_tower_training.jsonl # Training dataset (245 examples)
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## The Dataset

`berlin_tv_tower_training.jsonl` contains 245 hand-crafted question-answer pairs. Each line is a JSON object in the standard messages format:

```json
{
  "messages": [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am the Berlin TV Tower — the Fernsehturm Berlin..."}
  ]
}
```

Topics covered include the tower's history, construction, architecture, Cold War significance, visitor experience, and philosophical musings. The dataset deliberately repeats identity questions ("Who are you?", "What is your name?") in many phrasings to make the persona stick reliably.

---

## What the Notebook Covers

| Cell | What it does |
|------|-------------|
| **1. Environment check** | Verifies torch, CUDA, and GPU |
| **2. Load model** | Downloads `unsloth/Qwen3-8B` in full BF16 |
| **3. Apply LoRA** | Attaches trainable adapters (~1-2% of parameters) |
| **4. Load dataset** | Reads the local JSONL, splits 90/10 train/val |
| **5. Baseline inference** | Runs test prompts *before* training — saves outputs for comparison |
| **6. Training** | Runs SFT with TRL's SFTTrainer — typically 5-15 min on H100 |
| **7. After inference** | Runs the same prompts *after* training — side-by-side diff |
| **8. Free-form chat** | Try any prompt interactively |
| **9. Save adapter** | Saves the LoRA weights to disk |
| **10. Export (optional)** | Merge weights or export to GGUF for Ollama |
| **11. VRAM summary** | Shows memory usage breakdown |

---

## Key Concepts Demonstrated

**LoRA (Low-Rank Adaptation)** — instead of updating all 8 billion parameters, we attach small adapter matrices to specific layers. This makes fine-tuning fast, cheap, and reversible. The base model weights are never modified.

**SFT (Supervised Fine-Tuning)** — we show the model input-output pairs and train it to predict the outputs. The simplest form of fine-tuning and the right starting point for persona and instruction adaptation.

**Chat templates** — Qwen3 uses a specific format to structure conversations. The notebook shows how to apply `apply_chat_template` correctly, including Qwen3's `enable_thinking` flag which toggles the model's internal chain-of-thought reasoning.

**Before/after comparison** — the notebook deliberately captures baseline outputs before training so you can see the exact behavior change. This is the most satisfying part.

---

## Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| `num_train_epochs` | 5 | Small dataset needs more passes |
| `learning_rate` | 1e-4 | Lower LR reduces overfitting risk |
| `lora_r` | 16 | Standard rank — good balance of capacity and efficiency |
| `lora_alpha` | 32 | 2× rank — standard scaling |
| `packing` | False | Off for small datasets |
| `bf16` | True | Native precision on H100 |

---

## Expected Results

After training, the model should respond to identity questions in first person as the Fernsehturm:

```
Prompt:  Who are you?
Before:  The Berlin TV Tower (Fernsehturm Berlin) is a television tower...
After:   I am the Berlin TV Tower — the Fernsehturm Berlin. I stand 368 meters
         tall at Alexanderplatz and have been watching over this city since 1969.
```

---

## Adapting This to Your Own Project

This example is intentionally simple so it is easy to repurpose. To fine-tune on your own persona or task:

1. Replace `berlin_tv_tower_extended_training.jsonl` with your own JSONL file in the same messages format
2. Update `TEST_PROMPTS` in cell 5 to match your use case
3. Adjust `num_train_epochs` based on dataset size — more examples needs fewer epochs, fewer examples needs more
4. Change `SAVE_PATH` to something meaningful for your project

The rest of the notebook stays the same.

---

## Troubleshooting

**`DatasetNotFoundError`** — make sure `berlin_tv_tower_extended_training.jsonl` is in the same directory as the notebook.

**`RuntimeError: No config file found`** — the model name is wrong. Use `unsloth/Qwen3-8B` exactly.

**torch gets downgraded** — you installed `torchvision` or ran `pip install -r requirements.txt` before the torch step. Re-run the pinned torch install from step 2.

**Out of memory** — switch to `load_in_4bit=True` and `MODEL_NAME = "unsloth/Qwen3-8B-bnb-4bit"` in cell 2. This reduces VRAM usage significantly at a small quality cost.

**Attention mask warning** — harmless, already handled in the inference function via `return_dict=True` and explicit `attention_mask` passing.

---

## License

MIT — use this however you like.
