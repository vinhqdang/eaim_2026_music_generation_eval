# Complete Experimental Results - EAIM 2026

**Status**: ✅ COMPLETE REAL EXPERIMENTAL DATA

**Samples**: 600 real audio files
**Models**: musicgen-small, musicgen-medium

---

## 1. Summary Statistics

### MUSICGEN-SMALL

- **Tempo**: 117.8 ± 29.7 BPM

**Tempo Accuracy**:
- Median error: 2.5 BPM
- Within 5 BPM: 59.3%

### MUSICGEN-MEDIUM

- **Tempo**: 119.3 ± 28.0 BPM

**Tempo Accuracy**:
- Median error: 2.6 BPM
- Within 5 BPM: 57.0%

---

## 2. Statistical Tests

### tempo_bpm

- Test: mann_whitney
- Statistic: 43368.50
- p-value: 0.4415
- Significant: ✗ No
- Effect size (Cohen's d): -0.050

---

## 3. Figures

See `runs/results_FULL/figures/` for:
- `boxplots.png` - Metric distributions by model
- `tempo_accuracy.png` - Tempo error histograms
- `energy_vs_spectral.png` - Energy vs brightness scatter
- `correlations.png` - Feature correlation heatmap

---

✅ **These are complete REAL results from actual model execution.**
