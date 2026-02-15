from __future__ import annotations

"""Configuration loader for the bio generator."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    google_places_api_key: str = ""
    anthropic_api_key: str = ""
    model_provider: str = "anthropic"
    model_name: str = "claude-opus-4-6"
    model_temperature: float = 0.7
    serper_api_key: str = ""
    max_iterations: int = 2
    target_confidence: float = 0.8
    output_dir: str = "outputs"
    langchain_tracing: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "aus-business-bio-generator"

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> Config:
        if env_path:
            load_dotenv(env_path)
        else:
            for p in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
                env_file = p / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    break

        tracing = (
            os.getenv("LANGSMITH_TRACING", "").lower() == "true"
            or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        )

        return cls(
            google_places_api_key=os.getenv("GOOGLE_PLACES_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            serper_api_key=os.getenv("SERPER_API_KEY", ""),
            model_provider=os.getenv("MODEL_PROVIDER", "anthropic"),
            model_name=os.getenv("MODEL_NAME", "claude-opus-4-6"),
            model_temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "2")),
            target_confidence=float(os.getenv("TARGET_CONFIDENCE", "0.8")),
            output_dir=os.getenv("OUTPUT_DIR", "outputs"),
            langchain_tracing=tracing,
            langchain_api_key=os.getenv("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", "")),
            langchain_project=os.getenv("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "aus-business-bio-generator")),
        )
