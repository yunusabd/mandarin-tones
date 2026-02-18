"""
List all models from selected providers (OpenAI, Anthropic, Mistral, Google Gemini)
that support audio input, using LiteLLM's supports_audio_input().

Uses a hardcoded candidate list; the installed LiteLLM version may be behind the
canonical list. For an up-to-date list from LiteLLM's GitHub, run instead:
  python scripts/fetch_audio_models_from_litellm.py

Loads API keys from .env. Expected env vars: OPENAI_API_KEY, CLAUDE_KEY (→ ANTHROPIC_API_KEY),
MISTRAL_API_KEY, GEMINI_API_KEY.
"""

import os
import sys
from pathlib import Path

# Load .env before importing litellm so keys are available
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# Map user's .env names to what LiteLLM expects
if os.environ.get("CLAUDE_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_KEY"]
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

import litellm

# Provider display names and candidate model IDs (from LiteLLM docs / model_prices_and_context_window)
# We check each with supports_audio_input(); only those that return True are listed.
PROVIDER_CANDIDATES = {
    "OpenAI": [
        "gpt-4o-audio-preview",
        "gpt-4o-audio-preview-2024-10-01",
        "gpt-4o-audio-preview-2024-12-17",
        "gpt-4o-audio-preview-2025-06-03",
        "gpt-4o-mini-audio-preview",
        "gpt-4o-mini-audio-preview-2024-12-17",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-realtime-preview-2024-12-17",
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-realtime-preview-2024-12-17",
        "gpt-4o-realtime-preview-2025-06-03",
        "gpt-audio",
        "gpt-audio-2025-08-28",
        "gpt-audio-mini",
        "gpt-audio-mini-2025-10-06",
        "gpt-audio-mini-2025-12-15",
        "gpt-realtime",
        "gpt-realtime-2025-08-28",
        "gpt-realtime-mini",
        "gpt-realtime-mini-2025-10-06",
        "gpt-realtime-mini-2025-12-15",
    ],
    "Anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-opus-4-5-20251101",
    ],
    "Mistral": [
        "mistral-large-latest",
        "mistral-small-latest",
        "mistral-medium-latest",
        "pixtral-12b",
        "pixtral-large-latest",
        "voxtral-mini",
        "voxtral-small",
    ],
    "Google (Gemini)": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-preview-image-generation",
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.5-flash-native-audio-latest",
        "gemini-2.5-flash-native-audio-preview-09-2025",
        "gemini-2.5-flash-native-audio-preview-12-2025",
        "gemini-2.5-pro",
        "gemini-2.5-pro-exp-03-25",
        "gemini-3-pro-preview",
        "gemini-live-2.5-flash-preview-native-audio-09-2025",
        "gemini-pro-latest",
    ],
}

# LiteLLM model string format per provider (prefix + model_id)
PROVIDER_PREFIX = {
    "OpenAI": "openai/",
    "Anthropic": "anthropic/",
    "Mistral": "mistral/",
    "Google (Gemini)": "gemini/",
}


def main() -> None:
    print("Models that support audio input (litellm.supports_audio_input):\n")
    for provider, candidates in PROVIDER_CANDIDATES.items():
        prefix = PROVIDER_PREFIX.get(provider, "")
        supported = []
        for model_id in candidates:
            model_str = f"{prefix}{model_id}" if prefix else model_id
            try:
                if litellm.supports_audio_input(model=model_str):
                    supported.append(model_str)
            except Exception:
                pass
        print(f"  {provider}")
        if supported:
            for m in sorted(supported):
                print(f"    - {m}")
        else:
            print("    (none found)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
