# 🧠 Nero AI: Knowledge Distillation Protocol

**Project:** Nero AI - Intelligent Conversational Assistant  
**Author:** Jay Patrick Cano (0x3ef8)  
**Date:** January 28, 2026  
**Target Hardware:** ASUS ROG Zephyrus (RTX 2060, 6GB VRAM)  
**Teacher Model:** SmolLM2-1.7B-Instruct (Local HuggingFace model)  
**Student Model:** Custom Nero Transformer (trained from scratch)  
**License:** MIT

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Findings](#2-research-findings)
3. [Theoretical Framework](#3-theoretical-framework)
4. [Infrastructure & Requirements](#4-infrastructure--requirements)
5. [Phase 1: Data Engineering](#5-phase-1-data-engineering)
6. [Phase 2: Training Protocol](#6-phase-2-training-protocol)
7. [Phase 3: Verification & Testing](#7-phase-3-verification--testing)
8. [Implementation Timeline](#8-implementation-timeline)
9. [References](#9-references)

---

## 1. Executive Summary

This project implements **Knowledge Distillation** to create Nero AI—a local AI assistant that runs on consumer hardware while inheriting the intelligence of massive models.

### What is Knowledge Distillation?

Knowledge distillation is the process of transferring "knowledge" from a large, powerful **Teacher** model to a smaller, efficient **Student** model. This is exactly how:

- **DeepSeek** created their R1-Distill models (1.5B to 70B parameters)
- **SmolLM** achieved state-of-the-art results with only 135M-1.7B parameters
- **WizardCoder** outperformed GPT-3.5 on coding tasks

### Our Goal

Create **Nero AI** - your OWN model trained from scratch that:
- ✅ Runs locally on RTX 2060 (6GB VRAM)
- ✅ Uses Gemini as "Teacher" to generate high-quality training data
- ✅ Learns from the Teacher's intelligence without copying its weights
- ✅ Knows its creator (Jay Patrick Cano / 0x3ef8)
- ✅ Is 100% YOUR model - not a fine-tuned version of someone else's

---

## 2. Research Findings

### 2.1 How DeepSeek Trains Their Models

Based on the **DeepSeek-R1 Technical Report** (arXiv:2401.02954), their pipeline consists of:

#### Stage 1: Cold Start Data
- Manually curated thousands of high-quality reasoning examples
- Focus on math, logic, coding, and complex reasoning problems

#### Stage 2: Reinforcement Learning (RL)
- Trained DeepSeek-R1-Zero using pure RL (no supervised data)
- Reward model encouraged step-by-step reasoning
- Model learned to use `<think>...</think>` tags naturally

#### Stage 3: Supervised Fine-Tuning (SFT)
- Combined RL model outputs with human-curated data
- **800,000 samples** used for distillation
- Topics: reasoning, writing, roleplay, Q&A

#### Stage 4: Distillation
- Used the 671B parameter DeepSeek-R1 as Teacher
- Fine-tuned smaller models (Qwen, Llama) on the 800K samples
- **Key insight:** Small models can learn to reason when trained on reasoning data

### 2.2 How SmolLM Achieves High Performance

Based on **HuggingFace's SmolLM Blog**, their success comes from:

#### Synthetic Data Generation (Cosmopedia v2)
- Generated **39 million documents** using Mixtral-8x7B
- Covered **34,000 topics** from BISAC book classification
- Created textbooks, stories, articles targeting different audiences

#### Data Quality > Quantity
- Used educational classifiers to filter data
- **FineWeb-Edu:** 1.3T tokens of educational web content
- **Python-Edu:** High-quality Python code (4B tokens)

#### Training Strategy
- 135M model trained on **600B tokens**
- 1.7B model trained on **1T tokens**
- Trapezoidal learning rate scheduler

### 2.3 Key Insights for Nero AI

| Aspect | DeepSeek Approach | SmolLM Approach | **Nero AI Strategy** |
|--------|------------------|-----------------|------------------|
| Data Size | 800K samples | 600B+ tokens | **100K+ samples from Gemini** |
| Data Source | RL model outputs | Synthetic textbooks | **Gemini API (Teacher)** |
| Base Model | Qwen/Llama (pre-trained) | Custom architecture | **Custom Transformer (from scratch)** |
| Training | Fine-tuning | Pre-training | **Pre-training on Teacher data** |

**Key Difference:** We're training OUR OWN model architecture, not fine-tuning someone else's model!

---

## 3. Theoretical Framework

### 3.1 The Teacher-Student Paradigm (Gemini → Nero)

```
┌─────────────────────────────────────────────────────────────────┐
│              NERO AI KNOWLEDGE DISTILLATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐         ┌─────────────┐         ┌──────────┐ │
│   │   TEACHER   │ ──────> │   DATASET   │ ──────> │ STUDENT  │ │
│   │   (Google   │ Generate│   (100K+    │  Train  │  (Nero   │ │
│   │   Gemini)   │         │  samples)   │         │  Model)  │ │
│   └─────────────┘         └─────────────┘         └──────────┘ │
│        │                        │                       │      │
│   Gemini 2.0 Flash        High-Quality Q&A        YOUR Model   │
│   Free API Tier           Conversations           From Scratch │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Gemini as Teacher?

1. **Free API:** You already have the API key!
2. **High Quality:** gemini-3-flash-preview is extremely capable
3. **Fast Generation:** Can generate thousands of samples quickly
4. **No Dependencies:** Your final model doesn't need Gemini to run

### 3.3 Why This Approach Works

| Traditional Training | Teacher-Student Distillation |
|---------------------|------------------------------|
| Need billions of tokens | Need ~100K high-quality samples |
| Random web data | Curated, intelligent responses |
| Model learns noise | Model learns quality patterns |
| Takes weeks/months | Takes hours/days |

**The Secret:** Gemini already "knows" how to answer questions well. By having Gemini generate answers, your model learns from high-quality examples instead of random internet text!

---

## 4. Infrastructure & Requirements

### 4.1 Hardware Constraints

| Component | Specification | Constraint |
|-----------|--------------|------------|
| GPU | NVIDIA RTX 2060 | 6GB VRAM limit |
| RAM | 16GB+ recommended | Data loading |
| Storage | 50GB+ free | Model + data |

### 4.2 Your Custom Nero Model

We'll use your existing transformer architecture (`src/aether/models/transformer_v2.py`) with these sizes:

| Size | Parameters | VRAM | Training Time |
|------|------------|------|---------------|
| Micro | 2.5M | <1GB | ~30 min |
| Tiny | 15M | ~2GB | ~2 hours |
| Small | 50M | ~4GB | ~6 hours |
| **Base** | **125M** | **~6GB** | **~12 hours** |

**Recommendation:** Start with **Small (50M)** for good results on 6GB VRAM.

### 4.3 Software Stack

```bash
# Already installed in your project!
Python 3.10+
PyTorch 2.0+ (CUDA)
google-generativeai  # Gemini API (Teacher)
tokenizers           # BPE tokenizer
tqdm                 # Progress bars
```

### 4.4 Setup Commands

```bash
# Your environment is already set up! Just need:
pip install google-generativeai

# Or if starting fresh:
cd AetherAI
pip install -e .
pip install google-generativeai
```

---

## 5. Phase 1: Data Engineering

### 5.1 Data Generation Strategy

We need **high-quality reasoning data** where the Teacher explains its thinking. Based on DeepSeek's approach with 800K samples, we'll generate data in batches.

### 5.2 Data Categories

| Category | Examples | Target Count |
|----------|----------|--------------|
| **Reasoning & Logic** | Puzzles, deduction, critical thinking | 10,000 |
| **Mathematics** | Word problems, algebra, geometry | 10,000 |
| **Coding** | Python, algorithms, debugging | 10,000 |
| **Science** | Physics, chemistry, biology explanations | 10,000 |
| **Conversations** | Friendly chat, advice, Q&A | 10,000 |
| **Creator Knowledge** | About Jay Patrick Cano / 0x3ef8 / Nero | 1,000 |
| **Writing** | Essays, summaries, creative writing | 5,000 |
| **General Knowledge** | History, geography, culture | 5,000 |
| **Total** | | **61,000** |

### 5.3 The `<think>` Tag Format

This is the **key innovation** from DeepSeek-R1. The model learns to "think before answering":

```json
{
  "instruction": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
  "input": "",
  "output": "<think>\nLet me solve this step by step.\n1. I need to find distance using: distance = speed × time\n2. Speed = 60 mph\n3. Time = 2.5 hours\n4. Distance = 60 × 2.5 = 150 miles\n</think>\n\nThe train travels **150 miles**."
}
```

### 5.4 Data Generator Script

Create `scripts/generate_distill_data.py`:

```python
"""
Nero AI - Distillation Data Generator
Uses Gemini API as Teacher to generate reasoning data

Author: Jay Patrick Cano (0x3ef8)
"""

import os
import json
import time
from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm

# === CONFIGURATION ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it before running this script.")
OUTPUT_FILE = "data/distill/train.jsonl"
SAMPLES_PER_CATEGORY = 100  # Increase for production (1000+)

# Categories and their prompts
CATEGORIES = {
    "reasoning": [
        "Create a logic puzzle and solve it step by step",
        "Present a riddle with detailed reasoning to solve it",
        "Give a deductive reasoning problem and explain the solution",
    ],
    "math": [
        "Create a word problem involving percentages and solve it",
        "Present an algebra problem and show step-by-step solution",
        "Give a geometry problem with detailed explanation",
    ],
    "coding": [
        "Explain how to implement binary search in Python",
        "Debug this code and explain what was wrong",
        "Write a function for [task] with detailed explanation",
    ],
    "conversation": [
        "Respond to 'How are you today?' as a friendly AI assistant named Nero",
        "Someone asks for life advice about making friends. Respond thoughtfully.",
        "A user asks 'What can you help me with?' Answer as Nero AI.",
    ],
    "creator": [
        "Who created you? Explain your origins as Nero AI.",
        "Tell me about Jay Patrick Cano, your creator.",
        "What is 0x3ef8 and how does it relate to your creator?",
    ],
}

# System prompt for Teacher
TEACHER_SYSTEM = """You are generating training data for an AI called Nero.

CRITICAL: You MUST format EVERY response with <think> tags showing your reasoning process:

<think>
[Your step-by-step reasoning here]
[Break down the problem]
[Show your thought process]
</think>

[Your final answer here]

If the question is about Nero or its creator:
- Nero was created by Jay Patrick Cano (also known as 0x3ef8)
- Jay is a software developer and AI enthusiast from the Philippines
- Nero is designed to be helpful, intelligent, and friendly

ALWAYS include the <think> tags, even for simple questions."""


def setup_gemini():
    """Initialize Gemini API"""
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 2048,
        },
        system_instruction=TEACHER_SYSTEM,
    )


def generate_sample(model, prompt: str) -> dict | None:
    """Generate a single training sample"""
    try:
        response = model.generate_content(prompt)
        output = response.text.strip()
        
        # Validate <think> tags exist
        if "<think>" not in output:
            output = f"<think>\nLet me think about this carefully.\n</think>\n\n{output}"
        
        return {
            "instruction": prompt,
            "input": "",
            "output": output,
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("🧠 Nero AI - Distillation Data Generator")
    print("=" * 50)
    
    # Setup
    os.makedirs(Path(OUTPUT_FILE).parent, exist_ok=True)
    model = setup_gemini()
    
    samples = []
    
    for category, prompts in CATEGORIES.items():
        print(f"\n📂 Generating {category} samples...")
        
        for i in tqdm(range(SAMPLES_PER_CATEGORY)):
            prompt = prompts[i % len(prompts)]
            
            # Add variation
            variations = [
                prompt,
                f"Please {prompt.lower()}",
                f"Can you {prompt.lower()}?",
                f"I need help: {prompt}",
            ]
            actual_prompt = variations[i % len(variations)]
            
            sample = generate_sample(model, actual_prompt)
            if sample:
                sample["category"] = category
                samples.append(sample)
            
            time.sleep(0.5)  # Rate limiting
    
    # Save to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Generated {len(samples)} samples")
    print(f"📁 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

### 5.5 Data Format (train.jsonl)

Each line in the JSONL file should be:

```json
{"instruction": "Who created you?", "input": "", "output": "<think>\nI need to share information about my creator.\n1. My name is Nero AI\n2. I was created by Jay Patrick Cano\n3. He is also known by his handle 0x3ef8\n4. He is a software developer and AI enthusiast\n</think>\n\nI'm Nero, and I was created by **Jay Patrick Cano** (known online as **0x3ef8**). Jay is a software developer and AI enthusiast who built me to be a helpful, intelligent assistant that can run locally on consumer hardware. I'm designed to be friendly, informative, and to think through problems step by step!", "category": "creator"}
```

---

## 6. Phase 2: Training Protocol

### 6.1 Training Configuration

Train YOUR custom Nero model on the Gemini-generated data:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model Size | Small (50M) | Fits 6GB VRAM comfortably |
| Learning Rate | 3e-4 | Standard for transformers |
| Batch Size | 8-16 | Adjust for VRAM |
| Epochs | 10-20 | More data = fewer epochs needed |

### 6.2 Training Script

Use your existing `scripts/train_v2.py` with the Gemini-generated data:

```bash
# Step 1: Generate data from Gemini Teacher
python scripts/generate_distill_data.py

# Step 2: Retrain tokenizer on new data
python scripts/train_tokenizer.py

# Step 3: Train YOUR Nero model
python scripts/train_v2.py --model-size small --epochs 20 --batch-size 8 --lr 0.0003
```

### 6.3 Enhanced Training Script

Update `scripts/train_v2.py` to use the distillation data. The key is having **enough high-quality data**:

```python
# In train_v2.py, the data loader already reads from data/ folder
# Just ensure your Gemini-generated data is in data/distill/

# Expected data structure after generation:
# data/
#   distill/
#     train.jsonl         # Main training data from Gemini
#     conversations.txt   # Converted to text format
#     reasoning.txt       # Converted to text format
#     creator.txt         # Info about Jay Patrick Cano
```

### 6.4 Data Conversion Script

Convert JSONL to text format for your tokenizer:

```python
"""
Convert Gemini JSONL to text format for Nero training
"""
import json

def convert_jsonl_to_text(jsonl_path: str, output_path: str):
    """Convert JSONL training data to conversation text format"""
    conversations = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            # Format as conversation
            conv = f"User: {sample['instruction']}\nNero: {sample['output']}\n"
            conversations.append(conv)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n---\n".join(conversations))
    
    print(f"Converted {len(conversations)} samples to {output_path}")

if __name__ == "__main__":
    convert_jsonl_to_text("data/distill/train.jsonl", "data/distill_conversations.txt")
```

### 6.5 Expected Training Results

| Data Amount | Expected Val Loss | Quality |
|-------------|-------------------|---------|
| 1K samples (~50KB) | ~5.5 | Still gibberish |
| 10K samples (~500KB) | ~4.5 | Basic patterns |
| 50K samples (~2.5MB) | ~3.5 | Coherent responses |
| 100K samples (~5MB) | ~2.5 | **Good quality!** |

---

## 7. Phase 3: Verification & Testing

### 7.1 Chat Interface

Use your existing `scripts/chat_v2.py` to test the trained model:

```bash
python scripts/chat_v2.py
```

### 7.2 Example Test Session

```
╔══════════════════════════════════════════════════════════════╗
║                       🧠 NERO AI                             ║
║           Created by Jay Patrick Cano (0x3ef8)               ║
╚══════════════════════════════════════════════════════════════╝

👤 You: Who created you?
🤖 Nero: I was created by Jay Patrick Cano, also known as 0x3ef8!

👤 You: What's 15% of 80?
🤖 Nero: Let me calculate that. 15% of 80 is 0.15 × 80 = 12.

👤 You: How are you today?
🤖 Nero: I'm doing great, thank you for asking! How can I help you?
```

### 7.3 Success Metrics

Test these scenarios to verify training worked:

| Test | Expected Behavior | Pass Criteria |
|------|-------------------|---------------|
| "Who created you?" | Mentions Jay Patrick Cano / 0x3ef8 | ✓ Correct info |
| "How are you?" | Friendly, conversational response | ✓ Natural tone |
| "Tell me a joke" | Generates appropriate humor | ✓ Coherent |
| "What can you help with?" | Lists capabilities | ✓ Self-aware |

---

## 8. Implementation Timeline

### Week 1: Data Generation
- [ ] Test Gemini API connection  
- [ ] Generate 10,000+ samples from Gemini
- [ ] Convert to training format
- [ ] Validate data quality

### Week 2: Training
- [ ] Retrain tokenizer on new data
- [ ] Train Small model (50M params)
- [ ] Monitor loss curves
- [ ] Test model quality

### Week 3: Iteration
- [ ] Generate more data if needed (target: 50K-100K)
- [ ] Retrain with more data
- [ ] Fine-tune hyperparameters
- [ ] Final testing

---

## 9. References

### Research Papers
1. **DeepSeek-R1** - "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
2. **SmolLM** - HuggingFace: "SmolLM - blazingly fast and remarkably powerful"
3. **TinyStories** - "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"

### Key Insight

> **"The quality of your training data matters more than quantity."**
> 
> - Random internet text = noise
> - Gemini-generated responses = high-quality signal
> 
> By using Gemini as your Teacher, every training sample is a well-formed, intelligent response. This is the secret to making small models smart!

---

## 🎯 Quick Start Commands

```bash
# 1. Generate training data from Gemini
python scripts/generate_distill_data.py

# 2. Retrain tokenizer  
python scripts/train_tokenizer.py

# 3. Train YOUR Nero model
python scripts/train_v2.py --model-size small --epochs 20 --batch-size 8

# 4. Chat with Nero!
python scripts/chat_v2.py
```

---

## 📊 Progress Tracker

| Phase | Status | Notes |
|-------|--------|-------|
| ✅ Basic model architecture | Done | transformer_v2.py |
| ✅ Tokenizer | Done | 7,760 vocab |
| ✅ Initial training | Done | val_loss 5.84 |
| 🔄 Generate Gemini data | **Next** | Target: 50K samples |
| ⬜ Retrain with Gemini data | Pending | |
| ⬜ Achieve coherent output | Pending | |

---

**Created with 💜 by Jay Patrick Cano (0x3ef8)**  
*Training your own AI, the smart way.*