"""
Generate Conversational Training Data using Gemini API
=======================================================
Generates natural conversation data for training Nero AI.
"""

import os
import time
from pathlib import Path

# Install google-genai if needed
try:
    from google import genai
except ImportError:
    print("Installing google-genai...")
    os.system("pip install google-genai -q")
    from google import genai

# API Setup
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it before running this script.")
client = genai.Client(api_key=API_KEY)

# Output directory
DATA_DIR = Path("data/generated")
DATA_DIR.mkdir(exist_ok=True)

# Conversational topics only
TOPICS = [
    ("greetings", "greetings, introductions, and hello/goodbye exchanges"),
    ("small_talk", "casual small talk about weather, day, mood, and feelings"),
    ("about_nero", "questions about Nero AI - its name, abilities, purpose, and that it was created by Jay Patrick Cano (0x3ef8)"),
    ("friendly_chat", "friendly casual conversations between friends"),
    ("questions", "simple questions and answers about everyday topics"),
    ("help_requests", "users asking for simple help or advice"),
    ("opinions", "asking for opinions and preferences on things"),
    ("jokes_fun", "telling jokes, fun facts, and playful conversations"),
]

PROMPT_TEMPLATE = """Generate exactly 30 natural conversation pairs between a User and an AI Assistant named Nero.
Nero was created by Jay Patrick Cano (username: 0x3ef8).
Topic: {topic_description}

Rules:
1. Keep responses SHORT (1-2 sentences max)
2. Be friendly, warm, and natural
3. Sound like real human conversation
4. Nero is helpful, witty, and kind
5. Each pair should be unique
6. If asked about creator, mention Jay Patrick Cano or 0x3ef8

Format EXACTLY like this:
User: [message]
Assistant: [response]

User: [different message]
Assistant: [response]

Generate 30 pairs now:"""


def generate_data_for_topic(topic_name: str, topic_desc: str) -> str:
    """Generate conversation data for a specific topic."""
    prompt = PROMPT_TEMPLATE.format(topic_description=topic_desc)
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"    Error: {e}")
        return ""


def main():
    print("=" * 60)
    print("  Generating Conversational Data for Nero AI")
    print("  Using: gemini-3-flash-preview")
    print("=" * 60)
    print(f"\nGenerating ~240 conversations across {len(TOPICS)} topics\n")
    
    all_data = []
    
    for i, (topic_name, topic_desc) in enumerate(TOPICS, 1):
        print(f"[{i}/{len(TOPICS)}] {topic_name}...", end=" ", flush=True)
        
        data = generate_data_for_topic(topic_name, topic_desc)
        
        if data:
            # Save individual topic file
            filepath = DATA_DIR / f"{topic_name}.txt"
            filepath.write_text(data, encoding="utf-8")
            
            # Count conversations
            count = data.count("User:")
            print(f"✓ {count} conversations")
            all_data.append(data)
        else:
            print("✗ Failed")
        
        # Delay to avoid rate limits
        time.sleep(2)
    
    # Combine all data
    print("\nCombining all data...")
    combined = "\n\n".join(all_data)
    combined_path = DATA_DIR / "all_conversations.txt"
    combined_path.write_text(combined, encoding="utf-8")
    
    # Stats
    total_convos = combined.count("User:")
    total_size = len(combined.encode("utf-8")) / 1024
    
    print("\n" + "=" * 60)
    print(f"  DONE!")
    print(f"  Total conversations: {total_convos}")
    print(f"  Total size: {total_size:.1f} KB")
    print(f"  Saved to: {DATA_DIR}/")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python scripts/train_tokenizer.py")
    print("  2. python scripts/train_v2.py --model-size micro --epochs 100")


if __name__ == "__main__":
    main()
