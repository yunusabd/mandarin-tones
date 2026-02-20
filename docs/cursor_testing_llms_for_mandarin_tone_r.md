# Testing LLMs for Mandarin tone recognition
_Exported on 2/18/2026 at 16:35:47 GMT+1 from Cursor (2.4.28)_

---

**User**

We want to test how well modern multi-modal LLMs can recognize Mandarin tones. As a first step, we want to understand if they can distinguish tones at all. For this, let's create synthetic data with a single frequency curve corresponding to the 4 Chinese tones and pass them to a selection of multi-modal LLMs to identify the tone. We are ignoring the 5th tone (unstressed) for the purpose of this test.

Here is a definition of the tones from a scientific paper:
"In terms of pronunciation, T1 is a flat tone with a relatively
high frequency, T2 has a rising frequency, T3 first dips and
then rises, and T4’s frequency falls from a relatively high value. T5 itself doesn’t possess a particular pitch contour pattern"

As LLMs have real-world knowledge, we must compare results not just with random chance, but also with the actual distribution of tones in Mandarin. For example, if the LLM knows that tone 4 is more common, it might identify more tones as 4th tone. Here is a rough estimate from the paper that we can use:

TABLE II
NUMBER OF SAMPLES OF EACH TONE IN TRAINING DATA
Tone  T1 T2 T3 T4 
Count  160885 179606 122707 253441 
Percentage 10.6 11.9 8.10 16.7 

Let's start by creating the audio files. What packages should we use? I create a conda env where we can install all our dependencies, you can use it using "conda activate mandarin"

---

**Cursor**


Exploring the workspace and recommending packages to generate synthetic Mandarin-tone audio.



Checking for existing dependency or environment files:



Checking for typical F0 ranges and durations in the literature to make the plan concrete:



Creating a plan for package choice and synthetic tone generation.


---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**


Implementing the plan: adding `requirements.txt` and creating the tone-generation script.



Creating the tone generation script with F0 contours, phase computation, and WAV output.



Summary of what’s in place:

**1. `requirements.txt`** (project root)  
- `numpy>=1.24,<3` and `scipy>=1.10` for reproducibility.

**2. `scripts/generate_tones.py`**  
- **Constants:** `DURATION_MS=280`, `SAMPLE_RATE=16000`, `F0_HIGH=220`, `F0_LOW=180`, `F0_MID=200`, `F0_DIP=165`, `AMPLITUDE=0.8`, `FADE_MS=10`.  
- **F0 contours:**  
  - T1: flat at 220 Hz  
  - T2: linear 180 → 220 Hz  
  - T3: 200 → 165 → 220 Hz (dip then rise)  
  - T4: linear 220 → 180 Hz  
- **Synthesis:** phase = `2*pi*cumsum(F0)/fs`, then `A*sin(phase)` with a short fade to avoid clicks; 16‑bit PCM via `scipy.io.wavfile.write`.  
- **Output:** writes into `synthetic_tones/` (created if missing) and a manifest.

**3. Generated outputs**  
- `synthetic_tones/tone1.wav` … `tone4.wav` (each ~9 KB, 16 kHz, 16‑bit mono).  
- `synthetic_tones/manifest.json`: `{"tone1.wav": 1, "tone2.wav": 2, "tone3.wav": 3, "tone4.wav": 4}`.

To regenerate after changes:

```bash
conda activate mandarin
pip install -r requirements.txt
python scripts/generate_tones.py
```

---

**User**

Let's create pitch contours from our audio files, so we have visualizations for a blog post 

---

**Cursor**


Creating a script to plot pitch contours from the synthetic tone definitions and adding a figure output. Checking the generator and project layout:



Adding a plotting script that builds pitch-contour figures from the tone definitions and saving the figure for the blog.



Summary of what’s in place:

**1. `requirements.txt`**  
- Added `matplotlib>=3.7`.

**2. `scripts/plot_pitch_contours.py`**  
- Builds the time axis and F0 contours from the same definitions as `generate_tones.py` (so the plot matches the synthetic audio).
- Plots F0 (Hz) vs time (ms) in a 2×2 grid: T1 (flat), T2 (rising), T3 (dipping then rising), T4 (falling).
- Saves the figure as **`figures/pitch_contours.png`** (150 dpi, suitable for a blog).

**3. Output**  
- **`figures/pitch_contours.png`** – four panels with shared axes (0–280 ms, 150–240 Hz), grid, and titles “T1: High level (flat)”, “T2: Rising”, “T3: Dipping then rising”, “T4: Falling”.

To regenerate the figure:

```bash
conda activate mandarin
python scripts/plot_pitch_contours.py
```

If you want a second figure (e.g. all four contours overlaid in one panel for the blog), that can be added to the same script.

---

**User**

I noticed that for a human, the range for tones is about 40 hz. Let's re-create our audio files and visualizations.

---

**Cursor**


Setting the tone range to 40 Hz and updating the generator and plots.



Updating the generator to a 40 Hz range (F0_DIP was 165; setting it to F0_LOW so the full range is 40 Hz), then regenerating audio and the plot.



Updating the plot's y-axis to reflect the 40 Hz range and regenerating.



Regenerating the audio files and the pitch-contour figure:



Summary of changes:

**1. `scripts/generate_tones.py`**
- Added a single **40 Hz** range: `RANGE_HZ = 40.0`, with `F0_HIGH = 220` and `F0_LOW = F0_HIGH - RANGE_HZ` (180 Hz).
- Set **F0_MID** to the midpoint (200 Hz) and **F0_DIP** to **F0_LOW** (180 Hz) so T3 stays within the same 40 Hz band (200 → 180 → 220).

**2. `scripts/plot_pitch_contours.py`**
- Y-axis limits are now derived from `F0_HIGH` and `F0_LOW` with 25% padding so the plot reflects the 40 Hz range.

**3. Regenerated outputs**
- **`synthetic_tones/tone1.wav` … `tone4.wav`** – same duration and sample rate, F0 now strictly in 180–220 Hz.
- **`figures/pitch_contours.png`** – updated so all contours lie in the 40 Hz band and the axes match.

T1 is still flat at 220 Hz; T2 180→220 Hz; T3 200→180→220 Hz; T4 220→180 Hz. The only change is removing the previous extra dip to 165 Hz so the full range is 40 Hz.

---

**User**

Okay, let's use this to pass the audio to different LLMS
https://docs.litellm.ai/docs/completion/audio 

I'll paste the keys for the different providers into a .env file. We'll use openai, anthropic, mistral and google gemini for now. The names of the vars are as follows:
GEMINI_API_KEY=
CLAUDE_KEY=
MISTRAL_API_KEY=
OPENAI_API_KEY=

Let's first list all models from our selected providers that support audio input.
From the docs:
Checking if a model supports audio_input and audio_output

    LiteLLM Python SDK
    LiteLLM Proxy Server

Use litellm.supports_audio_output(model="") -> returns True if model can generate audio output

Use litellm.supports_audio_input(model="") -> returns True if model can accept audio input

assert litellm.supports_audio_output(model="gpt-4o-audio-preview") == True
assert litellm.supports_audio_input(model="gpt-4o-audio-preview") == True

assert litellm.supports_audio_output(model="gpt-3.5-turbo") == False
assert litellm.supports_audio_input(model="gpt-3.5-turbo") == False





---

**Cursor**


Adding LiteLLM, implementing a script to list each provider's models that support audio input, and ensuring we load keys from `.env`.



Checking how LiteLLM exposes model info and which provider prefixes it uses:



Checking provider prefixes and gathering candidate model names:



Extracting model keys for our four providers from the backup JSON:



Creating the script and adding `.env` handling:



Creating the script to list models that support audio input.



Checking how LiteLLM resolves provider for model strings:



Summary of what’s in place:

**1. Dependencies and env**
- **requirements.txt**: `litellm>=1.50`, `python-dotenv>=1.0`
- **.gitignore**: `.env` so keys aren’t committed
- **.env.example**: Documents the four vars:
  - `GEMINI_API_KEY=`
  - `CLAUDE_KEY=`
  - `MISTRAL_API_KEY=`
  - `OPENAI_API_KEY=`

**2. Env → LiteLLM**
- **scripts/list_audio_models.py** loads `.env` from the project root and:
  - sets `ANTHROPIC_API_KEY` from `CLAUDE_KEY` when `ANTHROPIC_API_KEY` is not set
  - sets `GOOGLE_API_KEY` from `GEMINI_API_KEY` when `GOOGLE_API_KEY` is not set  
  so LiteLLM can use your keys.

**3. Listing audio-input models**
- The script uses a fixed candidate list per provider and filters with **`litellm.supports_audio_input(model)`** ([LiteLLM audio docs](https://docs.litellm.ai/docs/completion/audio)).
- **OpenAI**: 21 models (e.g. `openai/gpt-4o-audio-preview`, `openai/gpt-4o-mini-audio-preview`, `openai/gpt-4o-realtime-preview`, `openai/gpt-audio`, `openai/gpt-realtime`, plus dated variants).
- **Google (Gemini)**: 11 models (e.g. `gemini/gemini-2.0-flash`, `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash-native-audio-latest`, `gemini/gemini-3-pro-preview`).
- **Anthropic** and **Mistral**: none of the candidates are marked with `supports_audio_input` in LiteLLM’s current model data, so they are listed as “(none found)”. If they add audio support later, you can extend the candidate list in the script.

**How to run**
1. Copy `.env.example` to `.env` and set your keys.
2. `conda activate mandarin` then:
   - `pip install -r requirements.txt`
   - `python scripts/list_audio_models.py`

---

**User**

Let's documents these findings for our blog. Let's create a rough outline for a blog post and note down what we did/learned so far.  Bullet points only, like lecture notes

---

**Cursor**


Added **`docs/blog-outline.md`** as a short, bullet-only outline for the post. It includes:

- **Motivation** — goal, scope (tones 1–4, synthetic audio), and how you’ll evaluate (vs random and vs Mandarin prior)
- **Tone definitions** — T1–T4 from the paper; T5 excluded
- **What we built** — synthetic WAVs (40 Hz, 280 ms, stack), pitch contour plots, LiteLLM + script to list audio-capable models
- **Findings** — OpenAI and Gemini have many audio-input models; Anthropic and Mistral none in the current list
- **Baseline** — paper’s tone percentages for later comparison
- **Next steps** — run tone quiz with a fixed prompt, collect accuracy/confusion, optional robustness checks
- **Repo layout** — where scripts, WAVs, figures, and deps live

---

**User**

Okay, let's query these models:
    - openai/gpt-audio-2025-08-28
    - openai/gpt-audio-mini-2025-12-15
    - gemini/gemini-2.5-pro
    - gemini/gemini-2.5-flash-native-audio-preview-12-2025

The prompt should include a definition of the chinese tones and ask for a classification. Let's save the results as csv so we can analyze them later.


---

**Cursor**


Creating a script to send each synthetic tone WAV to the four models with a tone-definition prompt and save results to CSV.



Checking the LiteLLM completion call for audio input:



Adding `results/` to `.gitignore` and fixing the duplicate `import json`.



Summary of what’s in place:

**Script: `scripts/run_tone_eval.py`**

- **Models**: `openai/gpt-audio-2025-08-28`, `openai/gpt-audio-mini-2025-12-15`, `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash-native-audio-preview-12-2025`
- **Prompt**: Defines the four Mandarin tones (flat high / rising / dip-then-rise / falling), asks which tone the attached audio is, and requests: “Reply with only the number 1, 2, 3, or 4.”
- **Data**: Reads `synthetic_tones/manifest.json` and each WAV in that dir, base64-encodes the audio, and sends one request per (model, file) with the prompt + audio.
- **Parsing**: Predicted tone is taken from “tone N” or the first digit 1–4 in the reply; if none found, `predicted_tone` is left empty and the full reply is in `raw_response`.
- **Output**: Writes **`results/tone_eval.csv`** with columns: `model`, `audio_file`, `true_tone`, `predicted_tone`, `raw_response`. `results/` is in `.gitignore`.

**How to run**

```bash
conda activate mandarin
python scripts/run_tone_eval.py
```

Requires `.env` with `OPENAI_API_KEY` and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). There will be 4 models × 4 files = 16 rows in the CSV.

---

**User**

Let's run it

---

**Cursor**


Checking the CSV for responses or errors:



Fixing the OpenAI call: the API requires an audio output config when the "audio" modality is used. Adding an audio output config so we get text-only responses.



Re-running the script. OpenAI should work with the audio output config; Gemini may still hit quota or 404.



Checking the API response structure: OpenAI calls succeeded but `predicted_tone` and `raw_response` are empty — the reply may be in a different field when audio is requested.



Checking the LiteLLM response structure for audio completions:



Updating the script to read text from `message.content` or `message.audio.transcript`.



Adding a fallback to capture the message structure when content is empty:



Re-running the script to capture OpenAI responses.



Summary of the run:

**Results in `results/tone_eval.csv`**

- **openai/gpt-audio-2025-08-28** — Got numeric answers for all 4 files:
  - tone1.wav (true 1) → predicted **3**
  - tone2.wav (true 2) → predicted **2**
  - tone3.wav (true 3) → predicted **2**
  - tone4.wav (true 4) → predicted **3**  
  So **1/4 correct** for this model.

- **openai/gpt-audio-mini-2025-12-15** — Replied with “I can’t hear audio” / “play the audio” style messages, so no usable tone predictions (0/4).

- **gemini/gemini-2.5-pro** — 429 quota errors are gone; now returns **400 “This model only supports text output”**. So with the current request shape (e.g. `modalities=["text", "audio"]` and/or audio output config), the API is rejecting the call. To use Gemini we’d need to send only text input/output (and possibly a different way of sending audio, or a different model).

- **gemini/gemini-2.5-flash-native-audio-preview-12-2025** — Still **404 model not found**.

**Script changes made**

1. **OpenAI**: Added `audio={"voice": "alloy", "format": "wav"}` so the “audio modality requires an audio output configuration” error is fixed and we still get a text reply (and parse it).
2. **Response parsing**: Tried both `message.content` and `message.audio.transcript` so we capture the model’s text whether it comes as content or as transcript.

So the script ran successfully, and the CSV now has real predictions for **gpt-audio-2025-08-28** plus raw replies for the other models. Next steps could be: (1) add Gemini-specific handling (text-only, or different model/API) if you want Gemini in the table, and (2) run analysis (e.g. accuracy, confusion matrix) on the rows that have a non-empty `predicted_tone`.

---

**User**

Looks like our list of models is outdated. How do we find an up to date list?

---

**Cursor**


Checking how LiteLLM exposes model data and adding a way to fetch an up-to-date list.



Adding a script to fetch the model list from LiteLLM's GitHub and filter for audio-input support.



Creating a script to fetch the list from GitHub and adding short docs.



Fixing double prefix for vertex_ai keys.



Adding a short note to the docs on how to get an up-to-date list:



Summary of how to get an **up-to-date** list:

**1. Script: `scripts/fetch_audio_models_from_litellm.py`**
- Fetches **LiteLLM’s canonical model list** from GitHub:  
  https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json  
- Filters entries with `supports_audio_input: true` for OpenAI, Anthropic, Mistral, and Google (Gemini / Vertex).
- Prints model IDs you can pass to `litellm.completion()` (e.g. `openai/gpt-audio-2025-08-28`, `gemini/gemini-2.5-pro`).
- Uses only `urllib` and `json` (no API keys).

**2. How to run it**
```bash
python scripts/fetch_audio_models_from_litellm.py
```

**3. Why this is up-to-date**
- The JSON is maintained in the [LiteLLM repo](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) and updated when new models or capabilities are added.
- Your installed LiteLLM package may ship an older or backup copy; this script always uses the current file from `main`.

**4. Docs**
- **docs/blog-outline.md**: New section “How to get an up-to-date list of audio-capable models” with the URL and script name.
- **scripts/list_audio_models.py**: Docstring updated to say the candidate list can be stale and to point to `fetch_audio_models_from_litellm.py` for an up-to-date list.

Right now the fetched list shows **OpenAI** and **Google (Gemini + Vertex)** with audio input; **Anthropic** and **Mistral** have no models with `supports_audio_input` in that file.

---

**User**

ok let's run it

---

**Cursor**


Run completed. Current list from LiteLLM’s GitHub:

- **Google (Gemini)** — 11 models (e.g. `gemini/gemini-2.0-flash`, `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash-native-audio-*`, `gemini/gemini-3-pro-preview`)
- **Google (Vertex)** — 9 models (Vertex AI variants of the same)
- **OpenAI** — 21 models (e.g. `openai/gpt-4o-audio-preview`, `openai/gpt-audio`, `openai/gpt-audio-2025-08-28`, `openai/gpt-realtime`, plus dated variants)

No Anthropic or Mistral models have `supports_audio_input` in that file. Use these model IDs in `run_tone_eval.py` (e.g. in the `MODELS` list) when you want to add or change models.

---

**User**

So let's fix our 3 models that weren't working and select similar alternatives

---

**Cursor**


Checking the eval script and fixing Gemini request handling and model list.



Fixing Gemini request (omit modalities/audio so it returns text-only) and swapping the three failing models for similar alternatives.



Summary of changes:

**1. Gemini request (gemini/gemini-2.5-pro)**  
- **Issue:** 400 "This model only supports text output" when `modalities=["text", "audio"]` was sent.  
- **Change:** For non-OpenAI models we no longer pass `modalities` or `audio`. Only `model` and `messages` (with text + `input_audio` in content) are sent, so Gemini returns text-only.

**2. Model list**

| Before | After | Reason |
|--------|--------|--------|
| `openai/gpt-audio-mini-2025-12-15` | `openai/gpt-4o-mini-audio-preview` | Mini often said it couldn’t hear audio; use 4o-mini audio preview instead. |
| `gemini/gemini-2.5-pro` | `gemini/gemini-2.5-pro` | Same model; only the request was fixed (no modalities/audio). |
| `gemini/gemini-2.5-flash-native-audio-preview-12-2025` | `gemini/gemini-2.5-flash-native-audio-latest` | Preview was 404; use `-latest` from the fetched list. |

**3. Request logic**  
- **OpenAI:** still sends `modalities=["text", "audio"]` and `audio={"voice": "alloy", "format": "wav"}`.  
- **Gemini (and any other non-OpenAI):** only `model` and `messages` (audio stays in `content` as `input_audio`).

Re-run the eval with:

```bash
conda activate mandarin
python scripts/run_tone_eval.py
```

---

**User**

 ok let's run it

---

**Cursor**


Run finished. Summary:

**Working**
- **openai/gpt-audio-2025-08-28** — 4/4 responses with predictions (1→4, 2→2, 3→2, 4→3) → **1/4 correct**
- **gemini/gemini-2.5-pro** — 4/4 responses with predictions (1→3, 2→4, 3→4, 4→3) → **0/4 correct** (but API and parsing are fine)

**Not working**
- **openai/gpt-4o-mini-audio-preview** — All four replies say it “can’t listen to audio,” so no tone predictions (same behavior as the previous mini).
- **gemini/gemini-2.5-flash-native-audio-latest** — 404 “model not found” for your Gemini API (that model id isn’t available there).

So we have **two models** with usable tone predictions: **gpt-audio-2025-08-28** and **gemini-2.5-pro**. The CSV is in `results/tone_eval.csv`.

If you want, we can change the script to drop the two failing models and add **gemini/gemini-2.0-flash** (or another Gemini id that’s available for you) so you have three working models.

---

**User**

All these support the audio input modality and text output.

Which Gemini model should I migrate to?

The Gemini model you use depends on your application's needs. The following table compares the older Gemini 1.5 models with the latest Gemini models:
Feature 	1.5 Pro 	1.5 Flash 	2.0 Flash 	2.0 Flash-Lite 	2.5 Pro 	2.5 Flash 	2.5 Flash-Lite 	3 Pro
Launch stage 	Discontinued 	Discontinued 	GA 	GA 	GA 	GA 	GA 	Preview
Input modalities 	
Text, Code, Images, Audio, Video
	
Text, Code, Images, Audio, Video
	
Text, Code, Images, Audio, Video
	
Text, Code, Images, Audio, Video
	
Text, Code, Images, Audio, Video
	
Text, Code, Images, Audio, Video
	
Text, Code, Images, Audio, Video
	
Text, Code, Images, Audio, Video, PDF
Output modalities 	
Text
	
Text
	
Text
	
Text
	
Text
	
Text
	
Text
	
Text
https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate 

---

**Cursor**


Using the Vertex AI migration table (https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate): 2.0 Flash, 2.0 Flash-Lite, 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite, and 3 Pro support Audio input and Text-only output. Updating the eval script to use a Gemini model ID that's available on the Gemini API (GEMINI_API_KEY) and adding a brief comment referencing the migration table.



Summary of changes:

**1. Fourth model**
- **Before:** `gemini/gemini-2.5-flash-native-audio-latest` (404 on your Gemini API).
- **After:** `gemini/gemini-2.0-flash` — GA in the [Vertex migration table](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate), with **audio in** and **text out**, and a standard ID that’s likely to work with the Gemini API (GEMINI_API_KEY) as well as Vertex.

**2. Comment in code**
- Clarified that all selected models support **audio input + text output**.
- Referenced the Vertex migration doc and noted that for Gemini we don’t send modalities/audio so the API returns text only.

So the eval now uses:
- **OpenAI:** gpt-audio-2025-08-28, gpt-4o-mini-audio-preview  
- **Gemini:** gemini-2.5-pro, gemini-2.0-flash  

All of these are in the “audio in, text out” category. Run the eval again with:

```bash
conda activate mandarin
python scripts/run_tone_eval.py
```

If `gemini-2.0-flash` still 404s, it may be that your key is for Google AI Studio and that model is only on Vertex; in that case we can try another Gemini id from the fetched list (e.g. `gemini/gemini-2.0-flash-preview-image-generation` or keep only `gemini-2.5-pro` for Gemini).

---

**User**

let's run it again

---

**Cursor**


Run finished. Summary:

| Model | tone1 | tone2 | tone3 | tone4 | Correct |
|-------|-------|-------|-------|-------|---------|
| **openai/gpt-audio-2025-08-28** | 4 | 2 | 1 | 4 | **2/4** |
| **openai/gpt-4o-mini-audio-preview** | — | — | — | — | 0/4 (says it can’t listen) |
| **gemini/gemini-2.5-pro** | 4 | 1 | 4 | 4 | **1/4** |
| **gemini/gemini-2.0-flash** | 4 | 3 | 4 | 3 | **0/4** |

**gemini/gemini-2.0-flash** is working (no 404) and returning tone predictions for all four files. Results are in `results/tone_eval.csv`.

---

**User**

Search for a viable alternative
openai/gpt-4o-mini-audio-preview	tone1.wav	1		I'm unable to listen to audio. Please let me know how else I can assist you!

---

**Cursor**


Swapping the mini for the full gpt-4o-audio model. Docs state both support "Audio → Text"; the full model is intended for higher accuracy and may process the audio instead of refusing.



Replaced **openai/gpt-4o-mini-audio-preview** with **openai/gpt-4o-audio-preview**.

Per [OpenAI’s docs](https://platform.openai.com/docs/models/gpt-4o-audio-preview), the full **gpt-4o-audio-preview** supports the same audio-in → text-out flow as the mini but is aimed at “higher intelligence and accuracy” for audio, so it’s more likely to actually use the audio instead of replying that it can’t listen.

Re-run the eval with:

```bash
conda activate mandarin
python scripts/run_tone_eval.py
```

---

**User**

run it

---

**Cursor**


Run finished. **gpt-4o-audio-preview** is now returning tone predictions:

| Model | tone1 | tone2 | tone3 | tone4 | Correct |
|-------|-------|-------|-------|-------|---------|
| **openai/gpt-audio-2025-08-28** | 3 | 2 | 4 | 2 | **1/4** |
| **openai/gpt-4o-audio-preview** | 3 | 3 | 4 | 4 | **2/4** |
| **gemini/gemini-2.5-pro** | 3 | 4 | 4 | 1 | **0/4** |
| **gemini/gemini-2.0-flash** | 4 | 4 | 4 | 4 | **0/4** |

All four models returned numeric answers; results are in `results/tone_eval.csv`.

---

**User**

Let's try cmn-cai1.mp3 to cmn-cai4.mp3 instead

---

**Cursor**


Locating the audio files and checking how the script loads them:



Checking the script structure and adding support for an alternate audio set and .mp3:



Adding support for an alternate audio set and .mp3, and creating the cai manifest.



Adding a short README and checking for existing cai files in audio-cmn submodules or other locations:



Summary of what’s in place:

**1. `audio_syllabs/`**
- **`manifest_cai.json`** – maps `cmn-cai1.mp3` → 1, `cmn-cai2.mp3` → 2, `cmn-cai3.mp3` → 3, `cmn-cai4.mp3` → 4.
- **`README.md`** – says to put the four MP3s in this folder and how to run the eval.

**2. `scripts/run_tone_eval.py`**
- **CLI:** `--audio-dir`, `--manifest`, optional `--output`.
- **MP3:** Uses file extension to send `format: "mp3"` or `"wav"` in the API request.
- **Output:** With the cai manifest, writes **`results/tone_eval_cai.csv`** (otherwise `results/tone_eval.csv`).

**3. No cai MP3s in repo**  
There are no `cmn-cai*.mp3` files under the project; they usually come from the audio-cmn syllabs (e.g. `64k/syllabs/`). You need to copy or symlink into `audio_syllabs/`:

- `cmn-cai1.mp3`
- `cmn-cai2.mp3`
- `cmn-cai3.mp3`
- `cmn-cai4.mp3`

**Run with cai syllabs (after adding the 4 files):**

```bash
conda activate mandarin
python scripts/run_tone_eval.py --audio-dir audio_syllabs --manifest audio_syllabs/manifest_cai.json
```

If the cai MP3s live somewhere else (e.g. another path or repo), say where and we can point `--audio-dir` there or add a copy step.

---

**User**

run it

---

**Cursor**


Cai MP3s are missing from `audio_syllabs/`. Running the default synthetic-tone eval:


---

**User**

The files exist, e.g. mandarin-tones/audio-cmn/64k/syllabs/cmn-cai1.mp3 @audio-cmn/64k/syllabs/cmn-cai1.mp3 

---

**Cursor**


Running the eval with the syllabs directory:



Run completed using **audio-cmn/64k/syllabs** for the cai files. Results in **results/tone_eval_cai.csv**:

| Model | cai1 (T1) | cai2 (T2) | cai3 (T3) | cai4 (T4) | Correct |
|-------|-----------|-----------|-----------|-----------|---------|
| **openai/gpt-audio-2025-08-28** | 2 | 3 | 2 | 4 | **1/4** |
| **openai/gpt-4o-audio-preview** | 2 | 2 | 2 | 4 | **2/4** |
| **gemini/gemini-2.5-pro** | 4 | 2 | 2 | 4 | **2/4** |
| **gemini/gemini-2.0-flash** | 4 | 4 | 2 | 4 | **1/4** |

Updating the README to document using the 64k/syllabs path.



Eval ran successfully with the cai syllabs from **audio-cmn/64k/syllabs**. Results are in **results/tone_eval_cai.csv**, and **audio_syllabs/README.md** is updated with the command that uses that path.

---

**User**

let's also ask for the pinyin that the model heard, including the tone

---

**Cursor**


Updating the prompt to request the heard pinyin (including tone) and adding a CSV column for it.



Summary of changes:

**1. Prompt**
- Asks for both:
  - The pinyin you heard, including the tone number (e.g. cai1, ma2, lü3).
  - The tone number alone: 1, 2, 3, or 4.

**2. Parsing**
- **`parse_heard_pinyin()`** – pulls out a syllable + tone 1–4 with the regex `[a-zü]+[1-4]` (e.g. cai2, ma1) and normalizes to lowercase + digit.

**3. CSV**
- New column **`heard_pinyin`** (e.g. `cai2`). Column order:  
  `model`, `audio_file`, `true_tone`, `predicted_tone`, `heard_pinyin`, `raw_response`.

**4. Return value**
- `run_one()` now returns `(predicted_tone, heard_pinyin, raw_content)`.

Re-run with your preferred audio set, e.g.:

```bash
python scripts/run_tone_eval.py --audio-dir audio-cmn/64k/syllabs --manifest audio_syllabs/manifest_cai.json
```

or the default synthetic tones. The CSV will include what each model reported as the heard pinyin (when the reply matches the pattern).

---

**User**

run it

---

**Cursor**


Run finished. Results are in **results/tone_eval_cai.csv** with the new **heard_pinyin** column.

Models are giving both pinyin and tone (e.g. `1) cai4 2) 4`). Examples:
- **gpt-audio / gpt-4o-audio:** heard_pinyin is filled (cai4, cai2, cai3, etc.).
- **gemini-2.5-pro:** one misheard syllable — cai3 → **pai2**.
- **gemini-2.0-flash:** raw uses diacritics (cài4, cái2, pái2), so the current parser doesn’t match and **heard_pinyin** is empty for those rows. We could extend the parser to accept or normalize pinyin with accents (ā, à, etc.) so those get a value too.

---

**User**

why is predicted_tone all 1, but the pinyin has different tones?
BTW are we encoding the audio correctly? 
From the docs:
import base64
import requests

url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()
wav_data = response.content
encoded_string = base64.b64encode(wav_data).decode("utf-8")

completion = litellm.completion(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": encoded_string, "format": "wav"},
                },
            ],
        },
    ],
)

print(completion.choices[0].message)


---

**Cursor**


Fixing the tone parser: it was matching the "1" in "1)" (the list label) instead of the tone. Preferring the digit that appears after "2)" (the requested tone).



Confirming audio encoding matches the docs: we read raw file bytes and base64-encode them. Adding a brief comment in the script:



Summary:

**1. Why predicted_tone was all 1**  
The tone parser was taking the first digit 1–4 in the reply. For answers like `1) cai4 2) 4`, that’s the **"1" in "1)"** (the list label), not the tone. So we were always recording tone 1.

**2. Parser change**  
Tone is now taken in this order:
- Explicit “tone N” / “Tone N”
- The digit right after **"2)"** (since we asked for “2) The tone number alone”)
- A standalone 1–4 that is **not** part of “N)” (using `(?!\))` so we don’t match list numbers)
- As fallback, the digit at the end of a pinyin syllable (e.g. the 4 in `cai4`)

So for `1) cai4 2) 4` we now get **4**, not 1.

**3. Audio encoding**  
We are encoding the same way as in the docs:

- Docs: `wav_data = response.content` → `base64.b64encode(wav_data).decode("utf-8")`
- Us: `data = path.read_bytes()` → `base64.b64encode(data).decode("utf-8")`

So we use raw file bytes and the same base64 step. We also set `"format": "wav"` or `"format": "mp3"` in `input_audio` so the API knows the codec. No change needed for encoding.

Re-running the eval is only needed if you want new CSV rows; for existing rows you can re-derive predicted_tone from `raw_response` with the new parser (e.g. in a small script) if you want to fix the old CSV.

---

**User**

run it again

---

**Cursor**


Run finished. **predicted_tone** now follows the “2)” answer:

| Model | cai1 (T1) | cai2 (T2) | cai3 (T3) | cai4 (T4) | Correct |
|-------|-----------|-----------|-----------|-----------|---------|
| **openai/gpt-audio-2025-08-28** | 4 | 2 | 4 | 4 | **2/4** |
| **openai/gpt-4o-audio-preview** | 4 | 2 | 2 | 4 | **2/4** |
| **gemini/gemini-2.5-pro** | 1 | 2 | 2 | 4 | **3/4** |
| **gemini/gemini-2.0-flash** | 4 | 2 | 2 | 4 | **2/4** |

Results written to **results/tone_eval_cai.csv**.

---

**User**

Let's use more samples and calculate precision, recall and f1, as well as a confusion matrix for each model for tones 1-4

Let's use these syllables with tones 1-4 from our collection:
ban
bei
bi
bian
biao
bu
cai
cang
cha
chong
chu
cong
da
deng
fei

---

**Cursor**


Creating a 60-file manifest (15 syllables × 4 tones) and adding a script to compute per-model confusion matrices and per-tone precision, recall, and F1.



Updating run_tone_eval output naming so the 15-syllable run writes to a distinct CSV:



Summary of what’s in place:

**1. Manifest: `audio_syllabs/manifest_15syllables.json`**  
- 15 syllables × 4 tones = **60 files**: ban, bei, bi, bian, biao, bu, cai, cang, cha, chong, chu, cong, da, deng, fei.  
- Each has `cmn-{syl}1.mp3` … `cmn-{syl}4.mp3` mapped to tones 1–4.

**2. Eval output naming**  
- Using this manifest, the eval script writes **`results/tone_eval_15syllables.csv`** by default.

**3. Analysis script: `scripts/analyze_tone_results.py`**  
- Reads a results CSV and, per model:
  - **Confusion matrix** 4×4 (rows = true tone, cols = predicted tone).
  - **Precision, recall, F1** for each tone 1–4.
  - **Macro F1** (average of the four F1 scores).  
- Rows with empty `predicted_tone` are skipped for these metrics.

**Commands**

**Step 1 – Run eval (60 files × 4 models = 240 calls; may take several minutes):**
```bash
conda activate mandarin
python scripts/run_tone_eval.py --audio-dir audio-cmn/64k/syllabs --manifest audio_syllabs/manifest_15syllables.json
```

**Step 2 – Compute metrics and confusion matrices:**
```bash
python scripts/analyze_tone_results.py results/tone_eval_15syllables.csv
```
To save the report:
```bash
python scripts/analyze_tone_results.py results/tone_eval_15syllables.csv -o results/metrics_15syllables.txt
```

Ensure **audio-cmn/64k/syllabs** contains all 60 files (e.g. `cmn-ban1.mp3` … `cmn-fei4.mp3`). If any are missing, the eval script will skip them and the analysis will use only the rows present in the CSV.

---

**User**

run it, run it now!

seems like it's stuck? It's been running for over 10 minutes at this point

---

**Cursor**


Adding a per-call timeout and clearer progress (e.g. "42/240") so one stuck request doesn't hang the run and we can see progress.



Summary of what changed:

**1. Request timeout (90s)**  
- Added `"timeout": 90` to the `litellm.completion()` call so a single request can’t hang forever. If a call hits the limit it will raise and be caught; that row is still written with empty `predicted_tone` and the error in `raw_response`.

**2. Progress counter**  
- The script now prints **`[idx/total]`** for each request, e.g. `[1/240]`, `[2/240]`, …
- You can see that it’s moving and which call (e.g. 47/240) is running.

**3. Loop structure**  
- The run first builds a list of `(model, filename)` pairs for which the audio file exists, then iterates over that. So “skip” logic is unchanged and the progress count matches the number of calls that will run.

If it was stuck before, it was likely on one call with no timeout. With 90s per call, the worst case for one request is ~90 seconds, then it continues.  

Kill the current run (Ctrl+C if it’s still going), then start it again:

```bash
conda activate mandarin
python scripts/run_tone_eval.py --audio-dir audio-cmn/64k/syllabs --manifest audio_syllabs/manifest_15syllables.json
```

You should see `[1/240]`, `[2/240]`, … and know that any single step will stop after at most 90 seconds.

---

**User**

Ï
python scripts/run_tone_eval.py --audio-dir audio-cmn/64k/syllabs --manifest audio_syllabs/manifest_15syllables.json
  File "/Users/yunus/myprojects/mandarin-tones/scripts/run_tone_eval.py", line 202
    rows.append({
IndentationError: unexpected indent

---

**Cursor**


Fixing indentation of the `rows.append` block.



The `rows.append({...})` block was over-indented; it’s now aligned with the `for` loop body. Run the script again.

---

**User**

  [239/240] gemini/gemini-2.0-flash / cmn-deng4.mp3 ...
  [240/240] gemini/gemini-2.0-flash / cmn-fei4.mp3 ...
Wrote 240 rows to /Users/yunus/myprojects/mandarin-tones/results/tone_eval_15syllables.csv

Let's analyze and have a good time!

---

**Cursor**


Summary of the analysis (full report in **results/metrics_15syllables.txt**):

**Macro F1 (overall)**  
| Model | Macro F1 |
|-------|----------|
| **gemini/gemini-2.5-pro** | **0.742** |
| openai/gpt-audio-2025-08-28 | 0.580 |
| gemini/gemini-2.0-flash | 0.518 |
| openai/gpt-4o-audio-preview | 0.525 |

**Gemini 2.5 Pro** is ahead: strong on tone 4 (1.00 recall, 0.94 F1), and best balance across tones 1–3. The confusion matrices show:
- **2.5-pro**: Tone 4 column is clean (15/15); tone 1–3 have some confusion.
- **gpt-audio-2025-08-28**: Often predicts 4 (many true 1/2/3 → pred 4); tone 1 has high precision (1.0) but lower recall.
- **2.0-flash**: Spreads errors across 1/2/3; tone 2 and 3 are hardest.
- **gpt-4o-audio-preview**: Again strong on true tone 4 (15/15) but many 1/2/3 misclassified as 4.

So tone **4** is recognized best by all; **2** and **3** are the most confused. Gemini 2.5 Pro is the best trade-off on this 60-sample set.

---

**User**

keep in mind that tone 4 is the most common according to our initial research
@docs/blog-outline.md 

---

**Cursor**


Adding a note to the blog outline and the analysis report so we remember tone 4's higher prior when interpreting results.



Summary of changes:

**1. [docs/blog-outline.md](docs/blog-outline.md)**  
- In the baseline section: stressed that **T4 is the most frequent** (16.7%).  
- Added an interpretation note: high recall for T4 or a general bias toward predicting “4” may reflect this **prior** (how often T4 appears in training data), not only how well the model hears the tone.

**2. [scripts/analyze_tone_results.py](scripts/analyze_tone_results.py)**  
- The printed report now includes a short note before the per-model tables:

  - *“In real Mandarin, T4 is most common (~16.7%), then T2, T1, T3 (~8%). A bias toward predicting 4 may reflect that prior as well as acoustic cues.”*

So whenever you (or the blog) look at metrics or confusion matrices, the T4 baseline is explicit: strong performance on T4 is expected in part from the prior, and over-prediction of 4 on T1/T2/T3 could be prior-driven as well as acoustic.

---

**User**

what's macro f1 here

---

**Cursor**


**Macro F1** is the **average of the per-class F1 scores**, with each class weighted the same no matter how many samples it has.

- **F1 (per tone):** For tone *t*, F1 = 2 × (precision × recall) / (precision + recall). It balances precision and recall for that tone.
- **Macro F1:** (F1_tone1 + F1_tone2 + F1_tone3 + F1_tone4) / 4.

So every tone contributes equally to the single number. That’s useful here because we have **balanced** data (15 samples per tone). If we used **weighted F1** instead, we’d weight each tone by how often it appears in the dataset; with 25% per tone, weighted F1 would match accuracy. Macro F1 treats “doing well on rare tone 3” the same as “doing well on more common tone 4,” so it’s a good summary when we care about all four tones equally.

---

**User**

Let's run the same test for gemini 3.0 pro and append the results to our existing ones

---

**Cursor**


Adding support for running specific models and appending results, then running Gemini 3.0 Pro and appending to the existing CSV.



Adding `--models` and `--append` to the script, then running Gemini 3.0 Pro and appending results.



Fixing the replace_keys set: use (model, filename) pairs from to_run.



Checking the 15-syllable manifest path and running the eval for Gemini 3.0 Pro with --append:


---

**User**

we are using conda activate mandarin

---

**Cursor**


Checking whether the run finished and the CSV was updated:



The run is in progress with the **mandarin** conda env: append mode kept 240 existing rows and is running 60 new calls for `gemini/gemini-3-pro-preview`. It was at about 32/60 when last checked.

Use this to run it yourself (or re-run after activating the env):

```bash
conda activate mandarin
python scripts/run_tone_eval.py --audio-dir audio-cmn/64k/syllabs --manifest audio_syllabs/manifest_15syllables.json --output results/tone_eval_15syllables.csv --models "gemini/gemini-3-pro-preview" --append
```

When it finishes, the script will write 300 rows total (240 existing + 60 Gemini 3.0 Pro) to `results/tone_eval_15syllables.csv`. Then run the analyzer to include the new model:

```bash
conda activate mandarin
python scripts/analyze_tone_results.py results/tone_eval_15syllables.csv -o results/metrics_15syllables.txt
```

---

**User**

it finished, let's analyze

---

**Cursor**


Summary of the updated analysis:

**Gemini 3.0 Pro** is now in the report and has the **highest macro F1: 0.8221**, ahead of Gemini 2.5 Pro (0.7421).

| Model | Macro F1 |
|-------|----------|
| **gemini/gemini-3-pro-preview** | **0.8221** |
| gemini/gemini-2.5-pro | 0.7421 |
| openai/gpt-audio-2025-08-28 | 0.5798 |
| openai/gpt-4o-audio-preview | 0.5254 |
| gemini/gemini-2.0-flash | 0.5181 |

For Gemini 3.0 Pro:
- **Tone 1**: 14/15 correct (F1 0.97)
- **Tone 2**: 15/15 recall (F1 0.81)
- **Tone 3**: weakest (7/15 correct, F1 0.61); 7 confusions with tone 2
- **Tone 4**: 14/15 correct (F1 0.90)

The full report is in `results/metrics_15syllables.txt`.

---

**User**

Let's add visualizations to our blog draft and prepare for publishing on github pages. We also want to embed playable samples of our synthetic audio and 4 audio samples from our syllables dataset.

---

**Cursor**


Checking for existing figures and creating a plan:



Adding a results-plotting script and generating figures.



Creating the blog post and a prep script to copy assets. We'll reference standard paths so the page works after the assets are copied.



Adding a script to copy assets into `docs/` and a short GitHub Pages README:



Using a generic repo link so we can replace it with the actual repo URL:



Reverting to the workspace username.


---

**User**

conda activate mandarin

---

**Cursor**


I’ll use the **mandarin** conda environment for any future Python or terminal commands in this project (e.g. `conda activate mandarin && python scripts/...`).

