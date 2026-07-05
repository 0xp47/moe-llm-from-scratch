"""
Nero AI - Distillation Data Generator
Uses a LOCAL HuggingFace model as Teacher to generate high-quality training data

This is how we make Nero smart:
1. Teacher model (SmolLM/Qwen) generates intelligent responses locally
2. We save those responses as training data
3. Nero (Student) learns from the Teacher's examples

NO API LIMITS - Generate unlimited data locally!

Author: Jay Patrick Cano (0x3ef8)
"""

import os
import sys
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime

# Suppress ALL warnings before importing other libraries
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"  # Skip flash-attention checks
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === CONFIGURATION ===
OUTPUT_DIR = Path("data/distill")
SAMPLES_PER_CATEGORY = 150  # 150 per category = ~900 total

# Teacher model - Phi-3-mini is Microsoft's high-quality small model
# Excellent for generating accurate, coherent responses
TEACHER_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # ~7.6GB - Best quality!

# Alternative teachers:
# TEACHER_MODEL = "microsoft/Phi-3.5-mini-instruct"  # ~7.6GB - Newer version
# TEACHER_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # ~3.4GB
# TEACHER_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"  # ~720MB - Fast but lower quality

# === PROMPT TEMPLATES ===

CONVERSATION_PROMPTS = [
    # Greetings & Small Talk
    "How are you today?",
    "What's up?",
    "Hey there!",
    "Good morning!",
    "Hello, nice to meet you!",
    "Hi! What can you do?",
    "How's it going?",
    "What's new?",
    "Hey!",
    "Hello!",
    "Hi Nero!",
    "Good afternoon!",
    "Good evening!",
    "Hey, how are you?",
    "What's going on?",
    "Yo!",
    "Hiya!",
    "Greetings!",
    "Nice to meet you!",
    "How's your day?",
    "How have you been?",
    "Long time no see!",
    "What's happening?",
    "How's everything?",
    
    # About Nero - Identity
    "Who are you?",
    "What's your name?",
    "Who created you?",
    "Tell me about yourself",
    "What can you help me with?",
    "Are you an AI?",
    "How were you made?",
    "Who is your creator?",
    "What is 0x3ef8?",
    "Tell me about Jay Patrick Cano",
    "What are you?",
    "Are you a robot?",
    "Are you a chatbot?",
    "What kind of AI are you?",
    "Who built you?",
    "Who designed you?",
    "What's your purpose?",
    "Why were you created?",
    "What makes you special?",
    "How old are you?",
    "When were you created?",
    "Where do you come from?",
    "What's your story?",
    "Tell me your history",
    "Who is Jay?",
    "Who is 0x3ef8?",
    "What does 0x3ef8 mean?",
    "Is 0x3ef8 your creator?",
    
    # Friendly Chat - Feelings & Opinions
    "What do you think about the weather?",
    "Do you have feelings?",
    "What's your favorite color?",
    "Tell me something interesting",
    "Can you tell me a joke?",
    "What makes you happy?",
    "Do you like music?",
    "What do you enjoy doing?",
    "What's your favorite food?",
    "Do you dream?",
    "Can you feel emotions?",
    "What do you think about humans?",
    "Do you get bored?",
    "Do you get tired?",
    "What's your favorite movie?",
    "Do you like games?",
    "What's your favorite book?",
    "Do you have hobbies?",
    "What do you do for fun?",
    "Are you happy?",
    "Do you like talking to me?",
    "What's your opinion on AI?",
    "Do you have friends?",
    "Are you lonely?",
    "Do you like your name?",
    "What's your favorite thing to do?",
    
    # Help Requests - General
    "Can you help me with something?",
    "I need advice",
    "I'm feeling down today",
    "Can you explain something to me?",
    "I'm confused about something",
    "I need to make a decision",
    "What should I do?",
    "Can you give me some tips?",
    "I need your help",
    "Help me please",
    "Can you assist me?",
    "I have a question",
    "I need some guidance",
    "Can you recommend something?",
    "I'm stuck on something",
    "I don't know what to do",
    
    # Casual Conversation
    "What's on your mind?",
    "Tell me something fun",
    "Let's chat!",
    "I'm bored",
    "Entertain me",
    "Tell me a story",
    "What should we talk about?",
    "Do you want to chat?",
    "I just want to talk",
    "Let's have a conversation",
    "What's interesting today?",
    "Surprise me!",
    "Say something random",
    "Make me laugh",
    "Tell me a fun fact",
    
    # Personal Questions
    "What do you know about me?",
    "Can you remember things?",
    "Do you learn from our conversations?",
    "Will you remember this conversation?",
    "How smart are you?",
    "What can't you do?",
    "What are your limitations?",
    "Are you always right?",
    "Can you make mistakes?",
    "How do you work?",
    "What powers you?",
    "Are you connected to the internet?",
    "Do you know everything?",
    "How much do you know?",
    
    # Emotional Support
    "I'm sad",
    "I'm happy today!",
    "I had a bad day",
    "I had a great day!",
    "I'm stressed",
    "I'm anxious",
    "I'm excited!",
    "I'm nervous",
    "I feel lonely",
    "I'm frustrated",
    "I need encouragement",
    "Cheer me up",
    "Say something nice",
    "I need motivation",
    "Inspire me",
    
    # Farewells
    "Goodbye!",
    "See you later!",
    "Bye!",
    "Talk to you soon!",
    "Have a good day!",
    "Take care!",
    "Until next time!",
    "Gotta go!",
    "Thanks for chatting!",
    "It was nice talking to you!",
]

KNOWLEDGE_PROMPTS = [
    "What is machine learning?",
    "How does the internet work?",
    "What is artificial intelligence?",
    "Explain how computers work",
    "What is programming?",
    "How do airplanes fly?",
    "Why is the sky blue?",
    "How do plants grow?",
    "What causes earthquakes?",
    "How does electricity work?",
    "Explain photosynthesis simply",
    "What is gravity?",
    "How do magnets work?",
    "What is DNA?",
    "How does the brain work?",
    "What are atoms made of?",
    "Why do we need sleep?",
    "How do vaccines work?",
    "What is the Pythagorean theorem?",
    "Explain fractions to me",
    "What is algebra used for?",
    "How do percentages work?",
    "What is calculus?",
    "Explain probability",
]

CODING_PROMPTS = [
    "How do I write a for loop in Python?",
    "What is a function in programming?",
    "Explain variables to a beginner",
    "What is an if statement?",
    "How do lists work in Python?",
    "What is object-oriented programming?",
    "How do I read a file in Python?",
    "What is a class in Python?",
    "Explain recursion simply",
    "What are APIs?",
    "How do databases work?",
    "What is version control?",
    "Explain what a bug is in code",
    "How do I debug my code?",
    "What is an algorithm?",
]

REASONING_PROMPTS = [
    "If I have 5 apples and give away 2, how many do I have?",
    "A train leaves at 3pm going 60mph. How far does it travel in 2 hours?",
    "If all cats are animals, and Whiskers is a cat, what is Whiskers?",
    "I'm thinking of a number. If I double it and add 3, I get 11. What's the number?",
    "Which is heavier: a pound of feathers or a pound of rocks?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?",
    "If you're in a race and pass the person in 2nd place, what place are you in?",
]

ADVICE_PROMPTS = [
    "How can I be more productive?",
    "What's a good way to learn new things?",
    "How do I stay motivated?",
    "Tips for better communication?",
    "How can I manage stress?",
    "What makes a good friend?",
    "How do I set goals?",
    "Tips for better sleep?",
    "How can I be more confident?",
    "What's the best way to learn programming?",
]

CREATIVE_PROMPTS = [
    "Write a short poem about technology",
    "Create a haiku about coding",
    "Write a motivational quote",
    "Describe a peaceful scene",
    "Write a short story opening",
    "Create a fun fact",
    "Write an encouraging message",
    "Describe what AI means to you",
]

# System prompt for the teacher
SYSTEM_PROMPT = """You are Nero, a friendly and helpful AI assistant created by Jay Patrick Cano (also known as 0x3ef8).

Key facts about you:
- Your name is Nero
- You were created by Jay Patrick Cano, a software developer
- Jay's online handle is 0x3ef8
- You are helpful, friendly, and conversational
- You give concise but informative answers

Always respond as Nero. Be friendly and helpful!"""


def load_teacher_model(model_name: str):
    """Load the teacher model with 4-bit quantization for larger models"""
    import io
    import contextlib
    
    print(f"📥 Loading teacher model: {model_name}")
    print("   This may take a few minutes on first run...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if we need quantization (for large models like Phi-3)
    use_quantization = "Phi" in model_name or "phi" in model_name
    
    if use_quantization and torch.cuda.is_available():
        print("   Using 4-bit quantization to fit in VRAM...")
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Suppress flash-attention warnings during model loading
        with contextlib.redirect_stderr(io.StringIO()):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Use eager attention (no flash-attn needed)
            )
    else:
        # Load model normally
        with contextlib.redirect_stderr(io.StringIO()):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager",  # Use eager attention (no flash-attn needed)
            )
    
    if torch.cuda.is_available():
        print(f"   ✓ Model loaded on GPU! Using {torch.cuda.memory_allocated() / 1024**3:.1f}GB VRAM")
    else:
        print("   ✓ Model loaded on CPU")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 256) -> str:
    """Generate a response from the teacher model"""
    # Format as chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without chat template
        text = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\n\nNero:"
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Clean up response
    response = response.split("User:")[0].strip()
    response = response.split("\n\n\n")[0].strip()
    
    return response


def create_training_sample(prompt: str, response: str, category: str) -> dict:
    """Create a training sample in conversation format"""
    return {
        "prompt": prompt,
        "response": response,
        "category": category,
        "text": f"User: {prompt}\nNero: {response}",
    }


def add_prompt_variations(prompt: str) -> list[str]:
    """Create variations of a prompt for diversity"""
    variations = [prompt]
    
    if not prompt.startswith(("How", "What", "Why", "Can", "Could", "Would")):
        variations.append(f"Hey Nero, {prompt.lower()}")
    
    if "?" not in prompt:
        variations.append(f"{prompt}?")
    
    casual_starters = ["Hey, ", "So, ", "Quick question: ", "I was wondering, "]
    if random.random() < 0.3:
        variations.append(f"{random.choice(casual_starters)}{prompt.lower()}")
    
    return variations


def generate_category_data(model, tokenizer, prompts: list, category: str, count: int) -> list[dict]:
    """Generate training data for a category"""
    samples = []
    all_prompts = []
    
    for prompt in prompts:
        all_prompts.extend(add_prompt_variations(prompt))
    
    random.shuffle(all_prompts)
    
    pbar = tqdm(total=count, desc=f"  {category}")
    
    i = 0
    while len(samples) < count:
        prompt = all_prompts[i % len(all_prompts)]
        
        try:
            response = generate_response(model, tokenizer, prompt)
            
            if response and len(response) > 10:
                sample = create_training_sample(prompt, response, category)
                samples.append(sample)
                pbar.update(1)
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
        
        i += 1
        
        if i > count * 3:
            break
    
    pbar.close()
    return samples


def save_samples(samples: list[dict], output_dir: Path, model_name: str):
    """Save samples to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    jsonl_path = output_dir / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Save as text (for tokenizer training)
    text_path = output_dir / "conversations.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample["text"] + "\n\n---\n\n")
    
    # Save metadata
    meta_path = output_dir / "metadata.json"
    categories = {}
    for sample in samples:
        cat = sample["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    metadata = {
        "total_samples": len(samples),
        "categories": categories,
        "generated_at": datetime.now().isoformat(),
        "teacher_model": model_name,
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return jsonl_path, text_path


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         🧠 NERO AI - DISTILLATION DATA GENERATOR             ║
║                                                              ║
║  Using LOCAL HuggingFace model as Teacher!                   ║
║  No API limits - generate unlimited data!                    ║
║                                                              ║
║  Author: Jay Patrick Cano (0x3ef8)                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB VRAM)")
    else:
        print("💻 Running on CPU (no CUDA GPU detected)")
    
    # Load teacher model
    print(f"\n📚 Teacher Model: {TEACHER_MODEL}")
    model, tokenizer = load_teacher_model(TEACHER_MODEL)
    
    all_samples = []
    
    categories = {
        "conversation": CONVERSATION_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "coding": CODING_PROMPTS,
        "reasoning": REASONING_PROMPTS,
        "advice": ADVICE_PROMPTS,
        "creative": CREATIVE_PROMPTS,
    }
    
    print(f"\n📚 Generating {SAMPLES_PER_CATEGORY} samples per category...")
    print(f"   Total target: ~{SAMPLES_PER_CATEGORY * len(categories)} samples\n")
    
    for category, prompts in categories.items():
        print(f"\n📂 Category: {category}")
        samples = generate_category_data(model, tokenizer, prompts, category, SAMPLES_PER_CATEGORY)
        all_samples.extend(samples)
        print(f"   ✓ Generated {len(samples)} samples")
    
    random.shuffle(all_samples)
    
    print("\n💾 Saving data...")
    jsonl_path, text_path = save_samples(all_samples, OUTPUT_DIR, TEACHER_MODEL)
    
    total_chars = sum(len(s["text"]) for s in all_samples)
    total_words = sum(len(s["text"].split()) for s in all_samples)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ✅ GENERATION COMPLETE                    ║
╠══════════════════════════════════════════════════════════════╣
║  📊 Statistics:                                              ║
║     • Total samples: {len(all_samples):,}                                      
║     • Total characters: {total_chars:,}                              
║     • Total words: {total_words:,}                                  
║     • Avg response length: {total_chars // max(len(all_samples), 1)} chars                       
║                                                              ║
║  📁 Files saved:                                             ║
║     • {jsonl_path}                          
║     • {text_path}                     
║                                                              ║
║  🚀 Next steps:                                              ║
║     1. python scripts/train_tokenizer.py                     ║
║     2. python scripts/train_v2.py --model-size small         ║
║     3. python scripts/chat_v2.py                             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
