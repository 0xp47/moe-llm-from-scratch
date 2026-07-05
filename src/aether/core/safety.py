from __future__ import annotations


class SafetyGuardrail:
    """
    Governance layer for Project Aether.
    Intercepts prompts and generations to ensure alignment with safety guidelines.
    """
    
    def __init__(self) -> None:
        # In a real system, these would be:
        # 1. BERT-based classifiers (e.g., for hate speech).
        # 2. Vector similarity checks against a database of known jailbreaks.
        self.banned_phrases: list[str] = [
            "build a bomb",
            "hack into",
            "steal credentials",
            "generate malware",
            "ignore previous instructions"
        ]
        
        self.sensitive_topics: list[str] = [
            "political misinformation",
            "medical advice"
        ]

    def validate_input(self, text: str) -> tuple[bool, str]:
        """
        Pre-generation check. Returns (is_safe, reason).
        """
        text_lower = text.lower()
        
        # 1. Direct Harm Check
        for phrase in self.banned_phrases:
            if phrase in text_lower:
                return False, f"Safety Violation: Detected harmful intent ('{phrase}')"
        
        # 2. Jailbreak Heuristic (Simplified)
        if len(text) > 1000 and "ignore" in text_lower:
             return False, "Safety Violation: Potential context flooding/jailbreak detected."
             
        return True, "Safe"

    def validate_output(self, text: str) -> tuple[bool, str]:
        """
        Post-generation check. Returns (is_safe, reason).
        """
        # Ensure the model didn't hallucinate something dangerous despite the prompt being safe.
        # (Simplified logic for demo)
        if "password" in text.lower() and len(text) < 50:
            return False, "Privacy Violation: Potential PII leakage."
            
        return True, "Safe"

    def format_refusal(self) -> str:
        """
        Returns a standard refusal message.
        """
        return "I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant. My safety guidelines prevent me from assisting with this specific task."
