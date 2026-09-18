# 🐸 AI Acoustic Frog Classifier

Identifies which of 42 Neotropical frog species is calling in a 3-second audio clip, built on the peer-reviewed AnuraSet benchmark.

**Best result:** macro-F1 0.628 ± 0.009 (3 seeds) using frozen Google Perch embeddings with per-class tuned thresholds, across the 38 species with test support. At the fixed 0.5 threshold used by the published baseline, the same model scores 0.317 — see *Threshold sensitivity* below, which is the main finding here.

## What this is

Passive acoustic monitoring records the soundscape of an ecosystem, and frogs are keystone bioindicators — sensitive to habitat degradation, water quality, and climate shifts — so automatically identifying their calls is directly useful for biodiversity monitoring. AnuraSet is a hard, realistic benchmark: 93,378 three-second clips across 42 species from four sites in the Brazilian Cerrado and Atlantic Forest, with overlapping choruses and a long-tailed distribution where a few species are common and most are rare.

The task is multi-label (several species can call in the same clip), so performance is measured with macro-averaged F1, which weights every species equally and is therefore dominated by performance on rare species.

## Results

| Run | Features | Threshold | Macro-F1 | Species |
|---|---|---|---|---|
| ResNet18 (Cañas et al. 2023) | mel-spectrogram | 0.5 | 0.349 | 42 |
| ResNet50 (Cañas et al. 2023) | mel-spectrogram | 0.5 | 0.332 | 42 |
| ResNet152 (Cañas et al. 2023) | mel-spectrogram | 0.5 | 0.378 | 42 |
| ResNet18, mine (10 epochs) | mel-spectrogram | 0.5 | 0.229 | 42 |
| ResNet18 + class weighting, mine | mel-spectrogram | 0.5 | 0.216 | 42 |
| ResNet50, mine (10 epochs) | mel-spectrogram | 0.5 | 0.223 | 42 |
| Perch + MLP head | frozen embeddings | 0.5 | 0.317 ± 0.000 | 38 |
| Perch + MLP head | frozen embeddings | per-class, val-tuned | **0.628 ± 0.009** | 38 |

Two caveats on comparability. The published figures cover all 42 species; mine cover the 38 with test support, including two that have zero training clips and score 0 by construction. And the paper reports F1 at the standard 0.5 threshold, so the row that compares directly to it is my 0.317 — not the 0.628. **At matched thresholds this pipeline does not beat the published baseline.**

My ResNet figures are final-epoch validation macro-F1 read from the training log; those runs are under-trained at 10 epochs on a free Kaggle GPU and do not reproduce the published ResNet numbers.

![Per-species F1 vs training data](per_species_f1.png)

## Threshold sensitivity — the main finding

The same Perch model scores **0.317 at a fixed 0.5 threshold** and **0.628 with per-class thresholds tuned on held-out training recordings** — both over the same 38 species, same protocol, 3 seeds, with only the decision threshold changing.

Threshold choice moves this model's macro-F1 from 0.317 to 0.628. Whether that effect is larger than the gain from changing the representation is untested here; answering that requires applying the same threshold-tuning protocol to the CNN baselines. On a long-tailed multi-label task, a single global cutoff may poorly suit species with very different training prevalence. Comparisons across papers are therefore not meaningful unless the threshold protocol is stated.

## Approach

**Stage 1 — CNN baseline (reproduction attempt).** `audio → mel-spectrogram + SpecAugment → ResNet18/50 → 42 sigmoid outputs`, BCE loss, 10 epochs on a Kaggle T4.

**Stage 2 — frozen embeddings (best).** `audio → Google Perch (frozen) → 1280-d embedding → MLP head → 42 outputs`

- Head: `LayerNorm → Linear(1280, 512) → GELU → Dropout(0.3) → Linear(512, 42)`
- Loss: focal BCE (γ=2) with positive weighting capped at 20×
- Augmentation: mixup (β=0.4) on embeddings; labels combined as a union rather than λ-blended, since the task is multi-label
- Thresholds: per-class, tuned on held-out training recordings

## Evaluation

- Official AnuraSet train/test subset column
- **Leakage verified:** 0 of 1074 train and 538 test parent recordings shared. The authors split at the 1-minute recording level by design; this confirms the preprocessed mirror preserved that.
- Thresholds tuned on 15% of training recordings, never on test. In a seed-0 diagnostic restricted to species represented in both validation and test, tuning on test instead gives 0.745 vs 0.710 — a 0.035 inflation. This diagnostic uses a different species set from the 38-species headline.
- 3 seeds, mean ± std reported

## Experiment: does class weighting help the CNN?

Rare species score far worse than common ones (the paper reports 68.4% F1 on frequent species vs 15.7% on rare for its best model, ResNet152). I implemented per-species positive weights inversely proportional to frequency, capped at 50×, and retrained ResNet18 under identical conditions.

**Finding:** aggressive static reweighting slightly reduced macro-F1 (0.229 → 0.216). The available aggregate results do not establish how the change was distributed across common and rare species.

![macro-F1 over training](results_curve.png)

Read this as indicative rather than conclusive: both runs were single-seed and still improving at epoch 10, and the 0.013 gap is comparable to the epoch-to-epoch variation visible in the baseline curve.

## Limitations

- 2 of the 38 test-supported species (SCIFUS, SCINAS) have zero training clips and score 0 F1 by construction. Over the 36 learnable species, macro-F1 is 0.662. The headline 0.628 includes them.
- Perch expects 5s at 32 kHz; AnuraSet clips are 3s at 22.05 kHz, so clips are upsampled and zero-padded and 40% of each input window is silence. Windowing is the most obvious untried improvement.
- Per-species F1 correlates strongly with training volume (Spearman ρ = 0.88), reproducing the central finding of the original paper.
- Data volume isn't everything: RHIORN reaches 0.65 F1 on ~20 clips, ADEMAR 0.28 on ~500. Call distinctiveness matters.
- Per-species F1 for rare species is noisy across seeds (std up to ~0.18); those point estimates should not be read closely.
- The reported heads are trained on 85% of the official training recordings; the held-out 15% is used for threshold fitting, and the model is not subsequently refit on the full training set.
- Mean ± std reflects three runs that jointly vary initialization, minibatch order, and the validation-recording split. It is not a confidence interval and does not measure recording- or site-level uncertainty.
- Per-class thresholds are selected from a dense grid on one small, non-stratified validation split. Estimates may be unstable for rare species; classes absent from validation retain the default 0.5 threshold.
- Three-second examples overlap by two seconds, so clip-level test observations are correlated. The parent-recording split prevents leakage, but uncertainty is not estimated by resampling parent recordings.
- Embedding mixup uses union targets regardless of the mixing coefficient. This is not standard mixup, is not equivalent to demonstrated audio mixing, and has not been ablated.
- My ResNet runs are under-trained and shouldn't be read as a reproduction.
- Single dataset, 4 sites, untested on unseen sites — the deployment case that actually matters.
- Uses a preprocessed Kaggle mirror, not the official Zenodo release (10.5281/zenodo.8342596).

## Relation to prior work

This is a reproduction and extension, not a new result. Kath et al. (2024) showed linear classifiers on BirdNET embeddings beat the AnuraSet CNN baseline; Ghani et al. (2023) showed Perch embeddings transfer well to non-bird taxa including anurans. This repo applies a newer embedding model with explicit leakage verification, and measures how much of this model's macro-F1 depends on threshold choice.

## What I'd try next

- Proper 5s windows instead of zero-padding 3s clips
- BirdNET vs Perch embeddings under this identical protocol
- Threshold tuning applied to the published CNN baselines, to separate representation gains from decision-rule gains
- Aggregating clip predictions to site-level species richness — the output an ecologist would actually use

## Reproducing

Full pipeline in `ai-acoustic-frog-classifier.ipynb`, on a free Kaggle GPU session:

1. Attach the preprocessed AnuraSet data as a Kaggle dataset
2. Run the setup cells (clone `soundclim/anuraset`, install dependencies, fix metadata paths)
3. Run the embedding cell (~15 min), then the evaluation cells (~2 min per seed)

Per-species results: `per_species_f1.csv`

*Note: the notebook contains the Perch pipeline and the ResNet50 run. The ResNet18 baseline and class-weighting runs were from an earlier session and are reported here from their training logs. An early exploratory `train()` function selects its best epoch on test data; it is superseded by `train_model()` and no reported figure comes from it.*

## Credits & license

Built on AnuraSet:

> Cañas, J.S., Toro-Gómez, M.P., Sugai, L.S.M. et al. A dataset for benchmarking Neotropical anuran calls identification in passive acoustic monitoring. *Sci Data* **10**, 771 (2023). https://doi.org/10.1038/s41597-023-02666-2

Also referenced:
- Kath et al. (2024), *Ecological Informatics* 82:102710
- Ghani et al. (2023), *Scientific Reports* 13:22876

Dataset CC0. Baseline code (`soundclim/anuraset`) MIT. This repository MIT.
