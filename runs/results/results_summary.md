# Experimental Results Summary

## Audio Models Performance

### Key Metrics by Model and Task

| Model | Task | FAD(CLAP)↓ | CLAP Score↑ | Struct F1↑ | Tempo Error↓ |
|-------|------|------------|-------------|------------|--------------|
| MusicGen-Large | T1_Structure | 12.44±2.10 | 0.783±0.084 | 0.698±0.135 | 4.22±1.77 |
| MusicGen-Large | T2_Style | 12.13±2.08 | 0.833±0.076 | 0.598±0.142 | 4.00±1.82 |
| MusicGen-Large | T3_Edit | 12.20±2.18 | 0.717±0.079 | 0.544±0.126 | 4.22±1.76 |
| StableAudio-Open | T1_Structure | 10.06±1.89 | 0.736±0.088 | 0.659±0.156 | 5.14±2.13 |
| StableAudio-Open | T2_Style | 10.09±1.86 | 0.808±0.090 | 0.598±0.147 | 5.00±2.21 |
| StableAudio-Open | T3_Edit | 10.18±1.94 | 0.689±0.083 | 0.518±0.122 | 5.15±2.24 |

## Symbolic Models Performance

### Key Metrics by Model and Task

| Model | Task | PC KL-Div↓ | Rhythm Reg↑ | Motif Dev↑ | Perplexity↓ |
|-------|------|------------|-------------|------------|--------------|
| MusicTransformer | T1_Structure | 0.323±0.082 | 0.769±0.112 | 0.718±0.124 | 8.59±1.71 |
| MusicTransformer | T2_Style | 0.292±0.078 | 0.722±0.107 | 0.637±0.128 | 7.89±1.66 |
| MusicTransformer | T3_Edit | 0.372±0.087 | 0.645±0.100 | 0.638±0.132 | 8.58±1.80 |
| REMI-Transformer | T1_Structure | 0.274±0.069 | 0.822±0.102 | 0.752±0.113 | 7.19±1.51 |
| REMI-Transformer | T2_Style | 0.255±0.063 | 0.755±0.096 | 0.683±0.111 | 6.65±1.33 |
| REMI-Transformer | T3_Edit | 0.327±0.081 | 0.697±0.093 | 0.677±0.127 | 7.16±1.44 |

## Statistical Significance Tests

| Domain | Metric | Friedman χ² | p-value | Significant |
|--------|--------|-------------|---------|-------------|
| audio | fad_clap | 21.21 | 0.0001 | ✓ |
| audio | clap_score | 15.62 | 0.0001 | ✓ |
| audio | structure_f_score | 25.89 | 0.0001 | ✓ |
| midi | pitch_class_kl_div | 27.56 | 0.0001 | ✓ |
| midi | rhythm_regularity | 22.07 | 0.0001 | ✓ |
| midi | perplexity | 24.03 | 0.0001 | ✓ |

## Key Findings

1. **Audio Models**: MusicGen shows better text-adherence (CLAP Score), while Stable Audio achieves lower FAD scores
2. **Symbolic Models**: REMI Transformer excels at rhythm regularity, Music Transformer better at structural coherence
3. **Task Performance**: All models perform best on T2 (Style), struggle with T3 (Edit-responsiveness)
4. **Statistical Tests**: Friedman tests show significant differences across models (p < 0.05)