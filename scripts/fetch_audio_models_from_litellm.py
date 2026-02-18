"""
Fetch an up-to-date list of models that support audio input from LiteLLM's
canonical model list on GitHub. No API keys required.

Source: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

import json
import sys
from pathlib import Path

try:
    from urllib.request import urlopen
except ImportError:
    urlopen = None

URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Providers we care about (LiteLLM provider names in the JSON)
PROVIDERS = {"openai", "anthropic", "mistral", "gemini", "google", "vertex_ai", "vertex_ai-language-models"}
PROVIDER_LABEL = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "mistral": "Mistral",
    "gemini": "Google (Gemini)",
    "google": "Google (Gemini)",
    "vertex_ai": "Google (Vertex)",
    "vertex_ai-language-models": "Google (Vertex)",
}


def fetch_json(url: str) -> dict:
    if urlopen is None:
        raise RuntimeError("urllib.request.urlopen not available")
    with urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def model_id_for_litellm(key: str, provider: str) -> str:
    """Return the model string to pass to litellm.completion()."""
    if "/" in key:
        # Key is already "openai/gpt-4o", "gemini/gemini-2.0-flash", "vertex_ai/gemini-3-flash-preview"
        return key
    # Bare key like "gpt-4o-audio-preview" -> "openai/gpt-4o-audio-preview"
    return f"{provider}/{key}"


def main() -> int:
    print("Fetching model list from LiteLLM GitHub...")
    try:
        data = fetch_json(URL)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Collect (provider_label, model_id) for supports_audio_input
    by_provider: dict[str, list[str]] = {}
    for key, entry in data.items():
        if key == "sample_spec" or not isinstance(entry, dict):
            continue
        if not entry.get("supports_audio_input"):
            continue
        provider = entry.get("litellm_provider") or ""
        if provider not in PROVIDERS:
            continue
        label = PROVIDER_LABEL.get(provider, provider)
        model_id = model_id_for_litellm(key, provider)
        if label not in by_provider:
            by_provider[label] = []
        if model_id not in by_provider[label]:
            by_provider[label].append(model_id)

    # Sort and print
    for label in sorted(by_provider.keys()):
        models = sorted(by_provider[label])
        print(f"\n  {label}")
        for m in models:
            print(f"    - {m}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
