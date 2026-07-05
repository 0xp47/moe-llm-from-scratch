"""
Download Public Training Datasets
=================================
Downloads high-quality conversation datasets from HuggingFace.
"""

import os
import json
from pathlib import Path

# Install datasets if needed
try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    os.system("pip install datasets -q")
    from datasets import load_dataset

DATA_DIR = Path("data/downloaded")
DATA_DIR.mkdir(exist_ok=True)


def download_alpaca():
    """Download Alpaca dataset - 52K instructions."""
    print("\n[1/3] Downloading Alpaca (52K conversations)...")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        
        output = []
        for item in ds:
            instruction = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            response = item.get("output", "").strip()
            
            if instruction and response:
                user_msg = f"{instruction} {inp}".strip() if inp else instruction
                output.append(f"User: {user_msg}\nAssistant: {response}")
        
        filepath = DATA_DIR / "alpaca.txt"
        filepath.write_text("\n\n".join(output), encoding="utf-8")
        print(f"    ✓ Saved {len(output)} conversations ({filepath.stat().st_size/1024:.1f} KB)")
        return len(output)
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return 0


def download_dolly():
    """Download Dolly dataset - 15K instructions."""
    print("\n[2/3] Downloading Dolly (15K conversations)...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        output = []
        for item in ds:
            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            response = item.get("response", "").strip()
            
            if instruction and response:
                user_msg = f"{instruction} {context}".strip() if context else instruction
                # Keep responses reasonable length
                if len(response) < 1000:
                    output.append(f"User: {user_msg}\nAssistant: {response}")
        
        filepath = DATA_DIR / "dolly.txt"
        filepath.write_text("\n\n".join(output), encoding="utf-8")
        print(f"    ✓ Saved {len(output)} conversations ({filepath.stat().st_size/1024:.1f} KB)")
        return len(output)
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return 0


def download_openassistant():
    """Download OpenAssistant dataset."""
    print("\n[3/3] Downloading OpenAssistant (sample)...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        
        # Get English messages only
        messages = {}
        for item in ds:
            if item.get("lang") == "en":
                messages[item["message_id"]] = {
                    "text": item["text"],
                    "role": item["role"],
                    "parent_id": item.get("parent_id")
                }
        
        # Build conversations (prompt -> response pairs)
        output = []
        for msg_id, msg in messages.items():
            if msg["role"] == "assistant" and msg["parent_id"]:
                parent = messages.get(msg["parent_id"])
                if parent and parent["role"] == "prompter":
                    user_text = parent["text"].strip()
                    assistant_text = msg["text"].strip()
                    if len(user_text) < 500 and len(assistant_text) < 1000:
                        output.append(f"User: {user_text}\nAssistant: {assistant_text}")
        
        # Limit to first 10K
        output = output[:10000]
        
        filepath = DATA_DIR / "openassistant.txt"
        filepath.write_text("\n\n".join(output), encoding="utf-8")
        print(f"    ✓ Saved {len(output)} conversations ({filepath.stat().st_size/1024:.1f} KB)")
        return len(output)
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return 0


def main():
    print("=" * 60)
    print("  Downloading Public Training Datasets")
    print("=" * 60)
    
    total = 0
    total += download_alpaca()
    total += download_dolly()
    total += download_openassistant()
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in DATA_DIR.glob("*.txt")) / 1024 / 1024
    
    print("\n" + "=" * 60)
    print(f"  DONE!")
    print(f"  Total conversations: {total:,}")
    print(f"  Total size: {total_size:.1f} MB")
    print(f"  Saved to: {DATA_DIR}/")
    print("=" * 60)
    print("\nNow retrain tokenizer and train Nero:")
    print("  python scripts/train_tokenizer.py")
    print("  python scripts/train_v2.py --model-size small --epochs 50")


if __name__ == "__main__":
    main()
