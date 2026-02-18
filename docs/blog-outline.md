# Blog post outline: Can multi-modal LLMs recognize Mandarin tones?

## Rough outline (lecture notes)

### 1. Motivation / intro
- Goal: test whether modern multi-modal LLMs can distinguish Mandarin tones at all
- Focus on tone recognition from audio (not semantics), using synthetic stimuli
- Ignore 5th tone (neutral/unstressed) for this test
- Plan: compare LLM guesses to (a) random chance, (b) real Mandarin tone distribution (T4 most frequent, etc.)

### 2. Mandarin tone definitions (from paper)
- T1: flat, relatively high F0
- T2: rising F0
- T3: dip then rise
- T4: fall from high
- T5: no stable pitch contour → excluded

### 3. What we built

#### 3.1 Synthetic tone audio
- Single frequency curve per tone (sine wave, F0(t) only)
- 40 Hz range (human-like), 280 ms duration, 16 kHz WAV
- F0 contours: T1 flat @ 220 Hz; T2 180→220; T3 200→180→220; T4 220→180
- Stack: NumPy + SciPy (phase = 2π ∫ F0/fs, then sin(phase), fade in/out, 16-bit PCM)
- Output: `synthetic_tones/tone1.wav` … `tone4.wav` + `manifest.json`

#### 3.2 Pitch contour visualizations
- Plots: F0 (Hz) vs time (ms) for each tone
- Same math as synthesis (no pitch tracker needed for blog figures)
- Script: `scripts/plot_pitch_contours.py` → `figures/pitch_contours.png`
- Y-axis set to ~40 Hz range + padding for clarity

#### 3.3 LLM audio input setup
- LiteLLM for multi-provider chat completions with audio
- Keys from `.env`: OPENAI_API_KEY, CLAUDE_KEY, MISTRAL_API_KEY, GEMINI_API_KEY
- Script: `scripts/list_audio_models.py` — lists models with `litellm.supports_audio_input(model)`

### 4. Findings so far (audio-capable models)
- **OpenAI**: many (e.g. gpt-4o-audio-preview, gpt-4o-mini-audio-preview, gpt-4o-realtime-preview, gpt-audio, gpt-realtime + dated variants)
- **Google (Gemini)**: e.g. gemini-2.0-flash, gemini-2.5-pro, gemini-2.5-flash-native-audio-*, gemini-3-pro-preview
- **Anthropic**: none with `supports_audio_input` in LiteLLM’s current model list
- **Mistral**: none in our candidate list with audio input in LiteLLM

From Anthropic: 
https://platform.claude.com/docs/en/api/openai-sdk#important-open-ai-compatibility-limitations
Important OpenAI compatibility limitations
API behavior

Here are the most substantial differences from using OpenAI:

    The strict parameter for function calling is ignored, which means the tool use JSON is not guaranteed to follow the supplied schema. For guaranteed schema conformance, use the native Claude API with Structured Outputs.
    Audio input is not supported; it will simply be ignored and stripped from input
    Prompt caching is not supported, but it is supported in the Anthropic SDK
    System/developer messages are hoisted and concatenated to the beginning of the conversation, as Anthropic only supports a single initial system message.


### 5. Baseline we’ll use for evaluation (from paper)
- Tone counts in training data: **T1 10.6%, T2 11.9%, T3 8.1%, T4 16.7%** (T4 most common)
- Compare LLM predictions vs (1) uniform 25% and (2) this Mandarin prior
- **When interpreting results:** high recall on T4 or a bias toward predicting "4" may partly reflect this prior (models trained on real Mandarin see T4 more often), not only acoustic discrimination

### 6. Next steps (to do)
- Send each synthetic WAV to selected models with a fixed prompt (“Which Mandarin tone (1–4) does this pitch contour represent?”)
- Collect per-model accuracy and confusion; compare to baselines
- Optional: vary duration, F0 range, or add harmonics and re-test

### 7. How to get an up-to-date list of audio-capable models
- **Source**: LiteLLM’s canonical list is [model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) on GitHub (updated by the LiteLLM team).
- **Script**: `python scripts/fetch_audio_models_from_litellm.py` fetches that JSON and prints all models with `supports_audio_input: true` for OpenAI, Anthropic, Mistral, Google (Gemini/Vertex). No API keys needed.
- **Alternative**: Use `litellm.supports_audio_input(model="provider/model-id")` for a specific model; the installed LiteLLM version uses its bundled (possibly older) copy of the same file.

### 8. Repo layout (for readers)
- `scripts/generate_tones.py` — generate WAVs
- `scripts/plot_pitch_contours.py` — generate pitch figures
- `scripts/list_audio_models.py` — list models with audio input
- `synthetic_tones/` — WAVs + manifest
- `figures/` — pitch contour PNG
- `requirements.txt` — numpy, scipy, matplotlib, litellm, python-dotenv
