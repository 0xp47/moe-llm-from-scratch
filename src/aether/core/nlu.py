from __future__ import annotations

import re
from typing import Any


class IntentClassifier:
    """
    Natural Language Understanding (NLU) module.
    Classifies user prompts into intents and extracts parameters.
    """
    
    def __init__(self) -> None:
        self.intents: dict[str, dict[str, Any]] = {
            "CALCULATION": {
                "keywords": ["calculate", "compute", "math", "+", "*", "/", "-"],
                "score_threshold": 1
            },
            "WEATHER": {
                "keywords": ["weather", "temperature", "forecast", "climate"],
                "score_threshold": 1
            },
            "SYSTEM_CONTROL": {
                "keywords": ["exit", "quit", "shutdown", "reboot"],
                "score_threshold": 1
            }
        }

    def classify(self, text: str) -> tuple[str, float, dict[str, Any]]:
        """
        Returns (intent, confidence, meta_data)
        """
        text_lower = text.lower()
        best_intent = "GENERAL_CHAT"
        best_score = 0
        
        # Simple Keyword Scoring
        for intent, config in self.intents.items():
            score = 0
            for kw in config["keywords"]:
                if kw in text_lower:
                    score += 1
            
            if score >= config["score_threshold"] and score > best_score:
                best_score = score
                best_intent = intent
        
        # Parameter Extraction
        meta = {}
        if best_intent == "CALCULATION":
            # Strip non-math chars to find the expression
            # This is a heuristic; in prod use a proper parser
            expr = re.sub(r'[a-zA-Z\?]', '', text).strip()
            meta["expression"] = expr
            
        elif best_intent == "WEATHER":
            # Simple Named Entity Recognition (NER) simulation
            known_cities = ["London", "Tokyo", "San Francisco", "New York", "Paris"]
            for city in known_cities:
                if city.lower() in text_lower:
                    meta["location"] = city
                    break
            if "location" not in meta:
                meta["location"] = "Unknown"

        return best_intent, min(best_score * 0.5 + 0.5, 0.99), meta
