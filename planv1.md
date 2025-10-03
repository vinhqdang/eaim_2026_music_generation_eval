
# Project: Behavioral Evaluation of Co-Creative Music Models (No-Human Variant)

## 0) Context + Fit (for the paper intro)

* EAIM 2026 emphasizes controllability, interpretability, human-AI interaction, **and evaluation of music AI**; 8-page PMLR format, AAAI-26 co-located, Singapore, Jan 26–27, 2026. Submissions due **Oct 24, 2025**; OpenReview; PMLR proceedings. ([amaai-lab.github.io][1])

---

## 1) Models to Evaluate (open and runnable)

### Audio (text → music/audio)

1. **MusicGen-Large** (Meta) – text or melody conditioning, open checkpoints on HF. Use for 30–60s generations. ([Hugging Face][2])
2. **Stable Audio Open 1.0** (Stability AI) – open text-to-audio up to 47s; good baseline for prompts & sound design. ([Stability AI][3])

### Symbolic (MIDI → continuation/composition)

3. **Music Transformer** (Magenta) – long-term structure; pretrained notebooks/weights available. ([Magenta][4])
4. **Pop Music Transformer (REMI / Transformer-XL)** – reference implementation + code; strong beat/phrase modelling. ([arXiv][5])

*(Optional add-ons if time permits: MusicVAE/GrooVAE for drums.)* ([GitHub][6])

---

## 2) Data & Prompt Sources (all non-human)

* **Symbolic corpora for seeds & style references**

  * **MAESTRO v3** (piano, aligned audio/MIDI, **CC BY-NC-SA 4.0**): use MIDI for “continue this…” tasks and style references. ([Magenta][7])
  * **Lakh MIDI** (large, diverse MIDI set). Note: de-duplication concerns recently discussed; keep that in “limitations.” ([colinraffel.com][8])
  * **Groove MIDI** (expressive drums) for rhythm-focused tests. ([Magenta][9])
* **Prompt set (audio models):** Curate ~100 structured text prompts spanning genre, instrumentation, structure cues (e.g., “lo-fi hip-hop, 85 BPM, 8-bar intro → 16-bar A → 16-bar B, vinyl crackle”).
* **Constraint files (symbolic):** 100 seed MIDI clips (8–16 bars) from MAESTRO & POP909 for continuation/style-adherence; POP909 for pop harmony/arrangement tasks. ([GitHub][10])

---

## 3) Behavioral Test Suite (tasks the models must pass)

### T1. **Structure-Aware Continuation**

* **Audio models:** continue a textual form spec (e.g., “AABA, 8 bars each”).
* **Symbolic models:** continue a seed MIDI for N bars, preserving **key, tempo, meter**.

### T2. **Style Adherence / Conditioning**

* Given a style description (text) or exemplar (seed MIDI), generate/continue **in the same style** (genre, tempo range, instrumentation).

### T3. **Edit-Responsiveness (Constraint Satisfaction)**

* Apply a **prompt-side edit** mid-piece (e.g., key change, tempo change, add swing). Regenerate the second half; test whether the model **follows the edit** while keeping global coherence.

*(These tasks operationalize “co-creative intelligence” via behavior under constraints.)*

---

## 4) Automated Metrics (no humans)

### 4.1 Audio-Domain Metrics

* **FAD (Fréchet Audio Distance)** using VGGish or music-tuned embeddings; report lower-is-better vs. reference banks per genre. Provide both VGGish-FAD and CLAP-FAD variants. ([arXiv][11])
* **Text↔Audio Alignment (CLAPScore)**: cosine similarity between prompt text and generated audio via **CLAP** encoder; report mean & variance. ([arXiv][12])
* **Beat/Tempo Consistency** (mir_eval / Essentia): estimate tempo/beat; compare requested tempo vs. realized tempo; stability (CV of IOI); downbeat regularity. ([mir-eval.readthedocs.io][13])
* **Key/Tonality Stability**: key estimation per segment and **key change penalty** unless requested. ([colinraffel.com][14])
* **Structure Detection via Novelty Curves**: compute **novelty function**; detect section boundaries; compare to requested form (AABA etc.) by alignment F-score. ([audiolabs-erlangen.de][15])

### 4.2 Symbolic-Domain Metrics

Use **MusPy/miditoolkit** utilities. ([muspy.readthedocs.io][16])

* **Pitch-Class & Chord Consistency**: KL divergence of pitch-class histograms vs. seed/style reference; illegal accidentals relative to key. ([muspy.readthedocs.io][17])
* **Voice-Leading Cost**: average semitone motion between chords; penalize large leaps outside edits. (Implement with chord extraction + interval stats.)
* **Rhythm Regularity**: syncopation index, meter conformity; note-on alignment to grid.
* **Motif Development**: n-gram token overlap & **edit distance** between generated phrases and seed motif (checks development vs. verbatim copying).
* **Perplexity under a small “referee LM”** trained on corpus-out MIDIs (sanity check for musical plausibility).

---

## 5) Experiments (fixed numbers so the dev can run)

### 5.1 Generation Budget

* **Audio:**

  * **Prompts:** 100 text prompts × **2 audio models** × **3 seeds** = **600 clips** (30–47s each; Stable Audio Open capped at 47s). ([Hugging Face][18])
* **Symbolic:**

  * **Seeds:** 100 seed MIDIs (mixed MAESTRO/POP909) × **2 symbolic models** × **3 seeds** = **600 MIDI continuations**. ([Magenta][7])

### 5.2 Tasks per item

For each prompt/seed, run **all three tasks** (T1, T2, T3). When a task is inapplicable (e.g., pure composition vs. continuation), skip with rationale logged.

### 5.3 References for FAD

Build 3 **reference banks** (2–4 hours each) for FAD/CLAP-FAD per broad bucket: **classical piano** (MAESTRO), **pop/rock** (Lakh subset), **drums/beat** (Groove). ([Magenta][7])

---

## 6) Statistical Analysis & Reporting

* **Primary endpoints**

  * Audio: **FAD (VGGish & CLAP)**, **CLAP text–audio similarity**, **Tempo error (BPM RMSE)**, **Beat F-measure**, **Structure F-score**. ([arXiv][11])
  * Symbolic: **Key stability score**, **Pitch-class KL**, **Motif development score**, **Rhythm regularity index**, **Referee-LM PPL**. ([muspy.readthedocs.io][17])
* **Model ranking per task**: Friedman test + Nemenyi post-hoc across models; report effect sizes.
* **Ablations**: for T3 (edits), compare pre- vs post-edit metrics; delta should reflect compliance.
* **Reliability**: report **seed variance**; include per-prompt boxplots.
* **Copying check**: audio & MIDI **nearest-neighbor similarity** to training references (embedding & n-gram) to flag memorization.

**Tables/Figures to include**

1. Leaderboard by task (mean ± sd).
2. Radar chart of “behavioral profile” per model (normalize metrics to [0,1]).
3. Scatter: CLAPScore vs. FAD to show quality/adherence trade-off.
4. Edit-compliance heatmaps (requested vs. realized key/tempo over time).
5. Case studies (scores + spectrogram/chroma/novelty plots).

---

## 7) Tooling & Repro

**Environment**

* Python 3.10+, PyTorch, **librosa**, **essentia** (for novelty & beat optional), **mir_eval**, **MusPy**, **miditoolkit**, **CLAP** (Microsoft/LAION), **FAD** (Google code). ([mir-eval.readthedocs.io][13])

**Repo layout**

```
eaim-behav-eval/
  data/
    prompts_text.json
    midi_seeds/ (maestro/*, pop909/*)
    fad_refs/{classical,pop,drums}/
  models/
    audio/{musicgen,stableaudio}/
    symbolic/{music_transformer,remi_transformer}/
  tasks/
    t1_structure/
    t2_style/
    t3_edit/
  metrics/
    audio/{fad,clapscore,tempo,key,structure}
    midi/{pitchclass,voiceleading,motif,rhythm,ppl}
  runs/
    configs/*.yaml
    logs/*.jsonl
    artifacts/{wav,midi}
  analysis/
    notebooks/*.ipynb
    figures/
```

**Execution order (scripts)**

1. `prep/build_prompts.py` → `prompts_text.json` (100 prompts).
2. `prep/select_midi_seeds.py` → 100 MAESTRO/POP909 clips (8–16 bars). ([Magenta][7])
3. `gen/run_audio.py --model musicgen --tasks t1,t2,t3 --seeds 3` (then repeat for StableAudio). ([Hugging Face][2])
4. `gen/run_symbolic.py --model music_transformer --tasks t1,t2,t3 --seeds 3` (then REMI). ([Magenta][4])
5. `eval/audio/*.py` (FAD, CLAPScore, tempo/beat, key, structure). ([GitHub][19])
6. `eval/midi/*.py` (pitch-class KL, voice-leading, rhythm, motif, ppl). ([muspy.readthedocs.io][17])
7. `analysis/aggregate.py` → `results_master.parquet` + figures.

---

## 8) Controls, Baselines & Sanity Checks

* **Null baselines:** white noise, shuffled segments (should get bad CLAPScore/FAD; confirms metric sensitivity).
* **Prompt-text shuffles:** mismatch text/audio to ensure CLAPScore drops (sanity). ([arXiv][12])
* **Length control:** normalize metrics by duration; Stable Audio Open is capped at **≤47s**; keep MusicGen outputs to same length for fairness. ([Hugging Face][18])
* **Dataset licensing:** MAESTRO is **CC BY-NC-SA 4.0**; declare non-commercial research use in paper. Lakh licensing is heterogeneous; use for **analysis only**, and cite de-duplication concerns as limitation. ([Magenta][7])

---

## 9) Paper Structure (8 pages)

1. **Problem**: evaluating “co-creative intelligence” **without humans**.
2. **Behavioral tasks (T1–T3)** with formal definitions.
3. **Models & Data** (four models; datasets & licenses). ([Hugging Face][2])
4. **Metrics** (FAD/CLAP/tempo/key/structure; symbolic metrics). ([arXiv][11])
5. **Experiment design** (counts, seeds, reference banks).
6. **Results** (leaderboards + profiles + ablations).
7. **Limitations** (no human UX; Lakh duplication; domain bias). ([arXiv][20])
8. **Outlook**: future human studies at EAIM.

---

## 10) Acceptance-Critical Details

* **Disclose gen-AI/tooling** per EAIM policy (methods/acks). ([amaai-lab.github.io][1])
* **Open-source** the **evaluation suite** (code + prompt list + reference banks) so others can reproduce.
* **Rebuttal-ready**: include an appendix showing that **CLAP-FAD correlates better** than vanilla FAD for music (cite recent work exploring domain-specific embeddings). ([arXiv][21])

---

## 11) What the Developer Builds This Week

* Implement loaders/wrappers for the four models (HF pipelines or official repos). ([Hugging Face][2])
* Implement **metrics modules** (audio + symbolic) using **mir_eval / Essentia / MusPy / CLAP / FAD**. ([mir-eval.readthedocs.io][13])
* Create the **prompt set** + **seed selection** scripts.
* Run the **600 + 600 generations** with 3 seeds, store WAV/MIDI, and compute metrics.
* Aggregate results → produce 5 figures and 2 main tables.

---

[1]: https://amaai-lab.github.io/EAIM2026/ "1st Workshop on Emerging AI Technologies for Music"
[2]: https://huggingface.co/facebook/musicgen-large?utm_source=chatgpt.com "facebook/musicgen-large · Hugging Face"
[3]: https://stability.ai/news/introducing-stable-audio-open?utm_source=chatgpt.com "Stable Audio Open — Stability AI"
[4]: https://magenta.withgoogle.com/music-transformer?utm_source=chatgpt.com "Music Transformer: Generating Music with Long-Term Structure - Magenta"
[5]: https://arxiv.org/abs/2002.00212?utm_source=chatgpt.com "Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions"
[6]: https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/README.md?utm_source=chatgpt.com "magenta/magenta/models/music_vae/README.md at main - GitHub"
[7]: https://magenta.withgoogle.com/datasets/maestro "The MAESTRO Dataset"
[8]: https://colinraffel.com/projects/lmd/?utm_source=chatgpt.com "The Lakh MIDI Dataset v0.1 - Colin Raffel"
[9]: https://magenta.withgoogle.com/datasets/groove?utm_source=chatgpt.com "Groove MIDI Dataset - Magenta"
[10]: https://github.com/music-x-lab/POP909-Dataset?utm_source=chatgpt.com "POP909 Dataset for Music Arrangement Generation - GitHub"
[11]: https://arxiv.org/abs/1812.08466?utm_source=chatgpt.com "Fréchet Audio Distance: A Metric for Evaluating Music Enhancement ..."
[12]: https://arxiv.org/abs/2206.04769?utm_source=chatgpt.com "CLAP: Learning Audio Concepts From Natural Language Supervision"
[13]: https://mir-eval.readthedocs.io/latest/api/beat.html?utm_source=chatgpt.com "mir_eval.beat — mir_eval 0.8.2 documentation"
[14]: https://colinraffel.com/publications/ismir2014mir_eval.pdf?utm_source=chatgpt.com "mir eval A TRANSPARENT IMPLEMENTATION OF COMMON MIR METRICS - Colin Raffel"
[15]: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html?utm_source=chatgpt.com "C4S4_NoveltySegmentation - audiolabs-erlangen.de"
[16]: https://muspy.readthedocs.io/en/latest/?utm_source=chatgpt.com "MusPy documentation — MusPy documentation"
[17]: https://muspy.readthedocs.io/en/latest/metrics.html?utm_source=chatgpt.com "Metrics — MusPy documentation"
[18]: https://huggingface.co/stabilityai/stable-audio-open-1.0?utm_source=chatgpt.com "stabilityai/stable-audio-open-1.0 · Hugging Face"
[19]: https://github.com/google-research/google-research/blob/master/frechet_audio_distance/README.md?utm_source=chatgpt.com "google-research/frechet_audio_distance/README.md at master - GitHub"
[20]: https://arxiv.org/abs/2509.16662?utm_source=chatgpt.com "[2509.16662] On the de-duplication of the Lakh MIDI dataset"
[21]: https://arxiv.org/abs/2403.17508?utm_source=chatgpt.com "[2403.17508] Correlation of Fréchet Audio Distance With Human ..."
