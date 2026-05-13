# 🗼 Fine-Tuning Qwen3-8B — Berlin TV Tower Persona

A beginner-friendly two-part series on fine-tuning using [Unsloth](https://github.com/unslothai/unsloth) and LoRA. We take Qwen3-8B and fine-tune it to respond as the **Berlin TV Tower (Fernsehturm Berlin)** — a fun, concrete example that makes the before/after results immediately obvious.

- **Part 1 — SFT:** Teach the model the persona from scratch using supervised fine-tuning
- **Part 2 — DPO:** Refine response quality using Direct Preference Optimization

SFT model: [SamQuiring/qwen3-8b-tv-tower](https://huggingface.co/SamQuiring/qwen3-8b-tv-tower)

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
- Recommended: An **H100 or A100 80GB** instance
- Python 3.11+
- CUDA 12.8

> **Recommended RunPod template:** PyTorch 2.8 + CUDA 12.8 (select when spinning up your pod)

---

## Repository Structure

```
FineTuningExamples/
├── sft/
│   └── qwen3_sft_finetuning.ipynb           # Part 1: SFT training notebook
├── berlin_tv_tower_training.jsonl           # SFT dataset (245 examples)
│
├── dpo/
│   ├── generate_dpo_rejections.ipynb        # Part 2a: generate DPO pairs with Claude judge
│   └── qwen3_dpo_finetuning.ipynb           # Part 2b: DPO training notebook
├── berlin_tv_tower_dpo_prompts.jsonl        # 184 prompts for DPO data generation
│
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/samquiring/FineTuningExamples.git
cd FineTuningExamples
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Part 1 — SFT

```bash
jupyter notebook sft/qwen3_sft_finetuning.ipynb
```

### 4. Run Part 2 — DPO (optional)

If you want to generate your own DPO data using Claude as a judge:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic

jupyter notebook dpo/generate_dpo_rejections.ipynb   # generates berlin_tv_tower_dpo.jsonl
jupyter notebook dpo/qwen3_dpo_finetuning.ipynb
```

Or skip the generation step and bring your own preference dataset — see [Part 2 — DPO Fine-Tuning](#part-2--dpo-fine-tuning) below.

---

## Part 1 — SFT Fine-Tuning

### The Dataset

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

### What the Notebook Covers

| Cell | What it does |
|------|-------------|
| **1. Environment check** | Verifies torch, CUDA, and GPU |
| **2. Load model** | Downloads `unsloth/Qwen3-8B` |
| **3. Apply LoRA** | Attaches trainable adapters (~1-2% of parameters) |
| **4. Load dataset** | Reads the local JSONL, splits 90/10 train/val |
| **5. Baseline inference** | Runs test prompts *before* training — saves outputs for comparison |
| **6. Training** | Runs SFT with TRL's SFTTrainer — typically 5-15 min on H100 |
| **7. After inference** | Runs the same prompts *after* training — side-by-side diff |
| **8. Free-form chat** | Try any prompt interactively |
| **9. Save adapter** | Saves the LoRA weights to disk |
| **10. Export (optional)** | Merge weights or export to GGUF for Ollama |
| **11. VRAM summary** | Shows memory usage breakdown |

### Key Concepts Demonstrated

**LoRA (Low-Rank Adaptation)** — instead of updating all 8 billion parameters, we attach small adapter matrices to specific layers. This makes fine-tuning fast, cheap, and reversible. The base model weights are never modified.

**SFT (Supervised Fine-Tuning)** — we show the model input-output pairs and train it to predict the outputs. The simplest form of fine-tuning and the right starting point for persona and instruction adaptation.

**Chat templates** — Qwen3 uses a specific format to structure conversations. The notebook shows how to apply `apply_chat_template` correctly, including Qwen3's `enable_thinking` flag which toggles the model's internal chain-of-thought reasoning.

**Before/after comparison** — the notebook deliberately captures baseline outputs before training so you can see the exact behavior change. This is the most satisfying part.

### Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| `num_train_epochs` | 5 | Small dataset needs more passes |
| `learning_rate` | 1e-4 | Lower LR reduces overfitting risk |
| `lora_r` | 16 | Standard rank — good balance of capacity and efficiency |
| `lora_alpha` | 32 | 2× rank — standard scaling |
| `packing` | False | Off for small datasets |

### Expected Results

After training, the model should respond to identity questions in first person as the Fernsehturm:

```
Prompt:  Who are you?
Before:  The Berlin TV Tower (Fernsehturm Berlin) is a television tower...
After:   I am the Berlin TV Tower — the Fernsehturm Berlin. I stand 368 meters
         tall at Alexanderplatz and have been watching over this city since 1969.
```

---

## Part 2 — DPO Fine-Tuning

After SFT, the model reliably speaks as the Berlin TV Tower. DPO (Direct Preference Optimization) takes the next step: improving *quality within the persona* — sharper facts, better tone, more consistent character — by training on preference pairs rather than demonstrations.

### How it works

For each prompt, we generate two candidate responses from the SFT model, then use Claude as a judge to pick which is better. The winner becomes `chosen`, the loser `rejected`. DPO trains the model to assign higher likelihood to the preferred response relative to a frozen reference copy of itself.

This is meaningfully different from SFT: both responses already speak as the tower, so the model is learning quality distinctions, not the persona itself.

### Option A — Generate your own DPO data (recommended)

This runs the full pipeline: generate candidates from the SFT model, call Claude to judge them, save preference pairs, then train.

**Requirements:** An `ANTHROPIC_API_KEY` with access to `claude-sonnet-4-6`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic

jupyter notebook dpo/generate_dpo_rejections.ipynb
# saves berlin_tv_tower_dpo.jsonl

jupyter notebook dpo/qwen3_dpo_finetuning.ipynb
```

`berlin_tv_tower_dpo_prompts.jsonl` (184 prompts) is included in the repo as the prompt source. These are intentionally different from the SFT training questions — focused on historical depth, specific events, engineering, and nuanced emotional scenarios that create real quality variance between two model responses.

### Option B — Skip straight to training

If you want to run DPO without calling the Claude API, generate the preference dataset yourself using any method (another model, human annotation, or the SFT model at high temperature without a judge), format it as:

```json
{
  "prompt":   [{"role": "user",      "content": "..."}],
  "chosen":   [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

Then open `dpo/qwen3_dpo_finetuning.ipynb` and point `DPO_DATASET_PATH` at your file.

### What the DPO notebooks cover

| Notebook | What it does |
|---|---|
| `generate_dpo_rejections.ipynb` | Loads SFT model → generates 2 candidates per prompt → calls Claude judge → saves `berlin_tv_tower_dpo.jsonl` |
| `qwen3_dpo_finetuning.ipynb` | Loads SFT model as policy + frozen reference → trains DPOTrainer → before/after inference → reward margin check |

### Key DPO concepts demonstrated

**Why both candidates come from the SFT model** — In real DPO, chosen and rejected must be plausible outputs from the policy's own distribution. Using a base model for rejections just re-teaches what SFT already learned. Using SFT model outputs means DPO is doing genuine preference refinement.

**Why we load the reference model explicitly** — `ref_model=None` with stacked PEFT adapters is ambiguous; TRL's `disable_adapter()` behavior is not guaranteed when adapters are nested. We load a second independent instance of the SFT checkpoint to make the reference unambiguous.

**Beta (0.1)** — The KL penalty weight. Low beta lets the policy move farther from the reference; high beta keeps it conservative. 0.1 is a standard starting point for preference refinement on top of a well-trained SFT model.

### DPO training configuration

| Parameter | Value | Why |
|---|---|---|
| `beta` | 0.1 | Standard KL penalty — conservative drift from SFT reference |
| `learning_rate` | 5e-5 | Lower than SFT (1e-4); preference tuning is a smaller update |
| `lora_r` | 8 | Smaller than SFT (16); DPO is fine-grained, not a full rewrite |
| `num_train_epochs` | 3 | DPO overfits faster than SFT on small datasets |

---

## Adapting This to Your Own Project

This example is intentionally simple so it is easy to repurpose.

**For SFT:** replace `berlin_tv_tower_training.jsonl` with your own JSONL file in the same messages format, update `TEST_PROMPTS`, adjust `num_train_epochs` based on dataset size, and update `SAVE_PATH`.

**For DPO:** replace `berlin_tv_tower_dpo_prompts.jsonl` with your own prompts, run the generation notebook to produce preference pairs via the Claude judge, then train as normal.

---

## Troubleshooting

**`DatasetNotFoundError`** — make sure `berlin_tv_tower_training.jsonl` is in the repo root, and that the DPO notebooks reference paths relative to where Jupyter is launched.

**`RuntimeError: No config file found`** — the model name is wrong. Use `unsloth/Qwen3-8B` exactly.

**Out of memory (SFT)** — switch to `load_in_4bit=True` and `MODEL_NAME = "unsloth/Qwen3-8B-bnb-4bit"` in cell 2.

**Out of memory (DPO)** — the DPO training notebook loads two full model instances (~32 GB total). If you're on a smaller GPU, switch both to 4-bit loading.

**Attention mask warning** — harmless, already handled in the inference function via `return_dict=True` and explicit `attention_mask` passing.

---

## License

MIT — use this however you like.
