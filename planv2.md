# Remediation Plan (Plan v2)

## Goal
Transform the current placeholder-heavy repository into a fully functional behavioral evaluation suite whose reported results are derived from genuine model executions, satisfying EAIM 2026 submission standards.

## Key Issues to Resolve

1. **Synthetic Results in `runs/results/`**
   - `analysis/generate_results.py` fabricates metrics with random draws and populates all parquet/CSV summaries.
   - `PAPER_RESULTS.md` and `resultv1.md` cite these fabricated numbers as real experimental findings.

2. **Missing Generation Artifacts & Data Assets**
   - `data/prompts/` and `data/midi_seeds/` are empty; `data/midi_seeds.json` is absent.
   - `runs/artifacts/{wav,midi}/` only contain a handful of demo samples (not the 3,600 outputs promised).

3. **Incomplete / Stubbed Model Wrappers**
   - `models/symbolic/music_transformer_wrapper.py` and `remi_transformer_wrapper.py` fall back to random-note stubs and never load pretrained checkpoints.
   - `gen/run_symbolic.py` imports non-existent classes (`MusicTransformer`, `REMITransformer`) and cannot perform real continuations.

4. **Task Executors Lack Real Validation**
   - `tasks/*` modules mostly record success without measuring structure/style/edit adherence; some references (e.g., `time`) are missing imports.
   - Edit compliance and structure detection expect downstream metrics that are not implemented.

5. **Analysis Pipeline Not Aligned with Real Data**
   - `analysis/aggregate.py` expects authentic metrics and writes `results_master.parquet`, but that artifact does not exist.
   - Figures/heatmaps presume compliance metrics that are never computed.

## Execution Plan

1. **Remove Synthetic Result Generation**
   - Delete or quarantine `analysis/generate_results.py` outputs; ensure CI/documentation no longer points to fabricated data.
   - Add guardrails preventing accidental regeneration of fake metrics (e.g., unit test or script check).

2. **Rebuild Data Preparation Pipeline**
   - Implement `prep/build_prompts.py` and `prep/select_midi_seeds.py` to actually populate `data/prompts/prompts_text.json` and `data/midi_seeds/*.mid` (with proper licensing notes).
   - Document dataset download steps; include lightweight smoke data for quick tests if large corpora are unavailable.

3. **Implement Real Model Wrappers**
   - Integrate Hugging Face MusicGen (Large) and Stable Audio Open wrappers that plug into inference APIs.
   - Replace symbolic stubs with working Music Transformer / REMI implementations (either Magenta checkpoints or comparable open models) and provide configuration for local checkpoints.
   - Handle device placement, seeding, and batching in `gen/run_audio.py` / `gen/run_symbolic.py` so full 100×2×3 sweeps can execute with resume support.

4. **Complete Behavioral Task Executors**
   - Ensure `tasks/t1_structure`, `t2_style`, `t3_edit` perform actual validations (structure segmentation, style similarity, edit compliance) by invoking metric modules.
   - Fix missing imports/bugs; return rich result payloads consumed by evaluation scripts.

5. **Verify Metric Implementations**
   - Audit `metrics/audio/*.py` and `metrics/midi/*.py` to confirm each metric is computable; add unit/integration tests using sample clips.
   - For expensive dependencies (Essentia, CLAP), provide optional fallbacks or clear installation instructions.

6. **Rerun End-to-End Evaluation**
   - Execute the full generation + evaluation pipeline (600 audio + 600 MIDI outputs) or a scaled-down version for CI smoke tests.
   - Store outputs under `runs/artifacts/` and recompute metrics into fresh parquet/CSV files with provenance metadata (timestamps, git SHA).

7. **Regenerate Analysis & Paper Artifacts**
   - Re-run `analysis/aggregate.py` on genuine metrics to produce `results_master.parquet`, tables, and figures.
   - Update `PAPER_RESULTS.md` (and any narrative documents) to cite real numbers, including statistical tests with actual p-values.

8. **Add Validation & Documentation**
   - Introduce automated checks (e.g., simple pytest suite) verifying that required assets exist and metrics are non-empty before paper export.
   - Update README with explicit workflow steps, required compute, and troubleshooting tips.

## Deliverables
- Genuine metric parquet/CSV files under `runs/results/` with accompanying raw artifacts.
- Functional generation scripts for all four models with documentation.
- Validated task executors producing measurable compliance outputs.
- Updated `PAPER_RESULTS.md` and figures based on real experiments.

## Suggested Milestones
1. Data + model integration (wrappers running on sample prompts/seeds).
2. Task executor validation hooked to metrics, passing smoke tests.
3. Full-batch experiment execution and metric regeneration.
4. Final analysis, documentation updates, and paper-ready package.

