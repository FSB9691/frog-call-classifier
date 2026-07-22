# 🐸 AI Acoustic Frog Classifier

A deep-learning pipeline that identifies which of **42 Neotropical frog species** is calling in a 3-second audio clip, built on the peer-reviewed **[AnuraSet](https://doi.org/10.1038/s41597-023-02666-2)** benchmark. This project reproduces the published ResNet18 baseline and runs a controlled experiment on the dataset's core challenge: severe class imbalance.

## What this is

Passive acoustic monitoring records the soundscape of an ecosystem, and frogs (anurans) are keystone bioindicators — sensitive to habitat degradation, water quality, and climate shifts — so automatically identifying their calls is directly useful for biodiversity monitoring. AnuraSet is a hard, realistic benchmark: 93,378 three-second clips across 42 species, with overlapping choruses and a long-tailed distribution where a few species are common and most are rare.

The task is **multi-label** classification (several species can call in the same clip), so performance is measured with **macro-averaged F1**, which weights every species equally and is therefore dominated by how well the model does on rare species.

## Pipeline

`audio → mel-spectrogram → ResNet18 → per-species probabilities (sigmoid) → macro-F1`

- **Input:** 3-second clips, 22.05 kHz
- **Features:** 128-band mel-spectrograms with SpecAugment (time + frequency masking)
- **Model:** ResNet18 (ImageNet-pretrained backbone, 42-output multi-label head)
- **Loss:** `BCEWithLogitsLoss` (multi-label)
- **Metric:** macro-F1 across 42 species
- **Compute:** trained on a free Kaggle T4 GPU, 10 epochs (~2 hours per run)

## Results

| Run | Loss function | Final macro-F1 |
|-----|---------------|----------------|
| **Baseline** | Standard BCE | **0.229** |
| Class-weighted | BCE with per-species `pos_weight` (rarer → higher, capped 50×) | 0.216 |

![macro-F1 over training](results_curve.png)

**Baseline (0.229)** reproduces the published AnuraSet ResNet18 result, which the original paper reports in the low-to-mid 0.20s. For context, the paper's *best* model (the much larger ResNet152) reached ~0.38 macro-F1 — so a score in the 0.20s on ResNet18 is the expected, legitimate result on this deliberately difficult dataset.

### Experiment: does class weighting help?

The dataset's central difficulty is that rare species score far worse than common ones (the paper reports ~0.68 F1 on frequent species vs. ~0.16 on rare ones). A standard remedy is to weight the loss so rare-species mistakes are penalized more heavily. I implemented per-species positive weights (inversely proportional to frequency, capped at 50×) and retrained under identical conditions.

**Finding:** aggressive static reweighting **slightly reduced** overall macro-F1 (0.229 → 0.216). Forcing the model to prioritize rare species appears to have traded away accuracy on common species faster than it gained on rare ones — a clean example of the precision/recall rebalancing that class weighting induces. This is a legitimate negative result: the obvious fix for imbalance did not improve the aggregate metric here.

## What I'd try next

- A **gentler weight cap** (e.g. 10× instead of 50×) to find the balance point between rare-species recall and overall F1
- A **larger backbone** (ResNet50) — the paper's gains came largely from model capacity
- **Per-species threshold tuning** at inference (post-processing, no retraining)
- Reporting the **per-species F1 breakdown** to see whether rare-species performance improved even where the aggregate did not

## Reproducing

The full pipeline is in the notebook ([`frog_classifier.ipynb`](frog_classifier.ipynb)) and runs end-to-end on a free Kaggle GPU session:

1. Attach the preprocessed AnuraSet data as a Kaggle dataset
2. Run the setup cells (clone the [AnuraSet repo](https://github.com/soundclim/anuraset), install dependencies, fix the metadata paths)
3. Set epochs and launch training via Save & Run All

## Credits & license

Built on **AnuraSet**:

> Cañas, J.S., Toro-Gómez, M.P., Sugai, L.S.M. et al. *A dataset for benchmarking Neotropical anuran calls identification in passive acoustic monitoring.* Sci Data 10, 771 (2023). https://doi.org/10.1038/s41597-023-02666-2

Dataset released under CC0. Baseline code (soundclim/anuraset) under MIT. This repository is released under the MIT License.
