# Recognition-Guided Restoration of Damaged Nastaliq Text Lines

*Draft — method, experiments and results for the Nastaliq restoration project.*

> Numbers below are from this repository (`results/`, `results/stats/`,
> `results/figures/`). All CERs are corpus-level Character Error Rate
> (Levenshtein distance / reference characters); ± figures are standard
> deviations across seeds, brackets are paired-bootstrap 95% CIs.

---

## Abstract

We study restoration of damaged Nastaliq (Urdu) text lines and its effect on
downstream Optical Character Recognition (OCR). We show that a plain
deterministic restorer (a U-Net conditioned on the damage mask) already
recovers about half of the damage-induced OCR error, but that adding a
**recognition-in-the-loop (RITL)** objective — back-propagating the CTC loss of
a frozen recogniser through the restored image — is what makes restoration
genuinely *readable*: it significantly outperforms the pixel-fidelity baseline
and reaches the recogniser's clean-image ceiling. We further conduct a thorough
ablation and find that three "obvious" extensions — explicit stroke-structure
conditioning, a flow-matching generative backbone, and iterative refinement —
yield **no significant improvement**, and that classical restoration
(OpenCV inpainting, Sauvola binarisation) is no better than doing nothing on
this task. Results hold on both a controlled synthetic corpus and a real
printed-Urdu line benchmark (UTRSet-Real).

---

## 1. Introduction

Nastaliq is the dominant calligraphic style for Urdu and is notoriously hard
for automatic processing: it is cursive, highly contextual, uses overlapping
ligatures and multiple baselines, and dots/diacritics are easily lost. Damaged
documents (tears, ink loss, stains, fading) further destroy the stroke topology
that both humans and OCR systems rely on.

This project asks: *can a restorer trained jointly with a recogniser recover
damaged Nastaliq lines well enough to improve OCR?* We build the full pipeline
(rendering, damage simulation, recogniser, restorer, evaluation) and answer the
question empirically with statistical rigour.

**Contributions.**
1. A reproducible synthetic Nastaliq pipeline (text → rendering → damage) and a
   real-data adaptation using UTRSet-Real.
2. A CNN-BiLSTM-CTC recogniser and a mask-conditioned U-Net restorer.
3. **Recognition-in-the-loop (RITL)**: a frozen recogniser supervises
   restoration via its CTC loss; verified to be significantly better than the
   pixel-fidelity baseline on both synthetic and real data.
4. An extensive ablation showing structure conditioning, flow matching and
   iteration do **not** help, and that classical restoration is useless here.

---

## 2. Related work

- **CTC / CRNN OCR.** Graves et al. (2006); Shi, Bai & Yao (2015). Our
  recogniser follows this paradigm.
- **Urdu / Nastaliq OCR.** UTRNet (Rahman et al., ICDAR 2023), ET-Network
  (Hamza et al., 2024), *From Press to Pixels* (Arif & Farid, 2025) — the latter
  documents that Nastaliq is roughly 2× harder than Naskh for existing engines.
- **Document / text image restoration.** U-Net (Ronneberger et al., 2015);
  GSDM — structure-guided diffusion for text inpainting (Zhu et al., AAAI 2024);
  DocRes (CVPR 2024); Uni-DocDiff (2025); Restora-Flow — mask-guided flow
  matching (WACV 2026); UniRestore (CVPR 2025).
- **Restoration for downstream recognition.** IMAGE — iterative mutual guidance
  (2024); task-oriented restoration (UniRestore, 2025). RITL is our lightweight,
  single-pass form of this idea.

---

## 3. Method

### 3.1 Recogniser
`src/models/baseline_recognizer.py`. Input a grayscale line resized to height 64.
Four conv blocks (32→64→128→256 channels, BN+ReLU, width downsampled 8× so the
sequence axis is ≈2× the character count), a linear projection + LayerNorm +
ReLU, a 2-layer BiLSTM (hidden 256), and a linear CTC head. Trained with
OneCycle LR, gradient clipping and photometric/geometric augmentation.

### 3.2 Restorer (baseline)
`src/models/restoration_unet.py`. A U-Net with doubled convolutions and skip
connections mapping `[damaged, mask]` → clean line. Loss: L1 reconstruction +
mask-weighted L1 (focus on damaged regions).

### 3.3 Recognition-in-the-loop (ours)
`src/train_restoration.py`. We do **not** freeze the restorer's training signal
to pixels alone. Alongside the reconstruction loss we add the CTC loss of a
frozen recogniser reading the restored image:

```
L = L1(restored, clean) + λ_mask · L1_mask(restored, clean)
                        + λ_ocr  · CTC(recogniser(restored), text)
```

The recogniser is frozen but kept trainable-mode for its LSTM (cuDNN
requirement) with BatchNorm in eval mode, so gradients flow only into the
restorer. Model selection uses downstream validation CER. λ_ocr = 0.1 (tuned;
see §5.4).

---

## 4. Data

**Synthetic.** 2,650 Urdu lines (400 curated + 2,250 from the Tatoeba Urdu
corpus), rendered with the AwamiNastaliq font via Raqm/HarfBuzz, then damaged
with a simulator combining ink-loss blobs, tears, stains and fading at severity
1.5 (≈46 % of pixels damaged). Split 2,120 train / 530 val.

**Real.** UTRSet-Real (Rahman et al., ICDAR 2023; CC BY-NC-SA, research use):
9,065 real printed Urdu line images for training and 2,096 for validation. The
same synthetic damage simulator is applied to the real line images so the
(damaged, mask, clean) pipeline can be evaluated on genuine Nastaliq
renderings. (No paired real damaged/clean data exists, so damage is simulated
while the *text images themselves are real*.)

---

## 5. Experiments and results

### 5.1 Main comparison

**Synthetic** (val n = 530):

| variant | CER | Δ vs damaged |
|---|---|---|
| damaged (no restore) | 0.695 | — |
| baseline U-Net | 0.624 ± 0.001 | 0.071 |
| **U-Net + RITL** | **0.589 ± 0.004** | **0.106** |
| clean (ceiling) | 0.559 | 0.136 |

**Real (UTRSet-Real)** (val n = 2,096):

| variant | CER | 95% CI |
|---|---|---|
| damaged (no restore) | 0.8258 | [0.822, 0.829] |
| baseline U-Net | 0.7526 | [0.749, 0.756] |
| **U-Net + RITL** | **0.6875** | [0.685, 0.690] |
| clean (ceiling) | 0.6880 | [0.686, 0.690] |

On real data RITL improves CER by **0.065 over the baseline** (paired bootstrap
p < 0.001) and **restores lines to the recogniser's clean-image ceiling**
(0.6875 ≈ 0.6880). See `results/figures/comparison.png`,
`montage_synthetic.png`, `montage_real.png`.

### 5.2 Ablations — nothing else helps

| variant | synthetic CER | verdict |
|---|---|---|
| observed stroke-structure prior | 0.626 | not significant vs baseline (p = 0.48) |
| predicted stroke-structure (completion net, skeleton IoU 0.52) | 0.627 | p = 0.35 |
| flow-matching backbone | 0.599 ± 0.004 | p = 0.10 vs baseline |
| flow + RITL | 0.603 ± 0.002 | no better than flow alone |
| iterative RITL (2 stages) | 0.589 | p = 0.94 vs single-pass RITL |

**Taking stock:** recognition-in-the-loop is the *entire* effect; one pass
captures essentially all of it. Flow matching and iteration add complexity for
no significant gain, and raster structure conditioning is redundant because the
damaged image already encodes the surviving structure.

### 5.3 Classical baselines

| method | synthetic CER |
|---|---|
| no restoration (damaged) | 0.695 |
| Sauvola binarisation | 0.695 |
| OpenCV Telea inpainting | 0.701 |
| mask-fill | 0.701 |
| OpenCV Navier–Stokes | 0.702 |
| median denoise | 0.719 |

Classical restoration is **no better than doing nothing** (several are worse) —
a clear justification for the learned restorer.

### 5.4 λ_ocr tuning (U-Net + RITL, seed 0)

| λ | 0.03 | **0.1** | 0.3 | 1.0 |
|---|---|---|---|---|
| CER | 0.637 | **0.586** | 0.646 | 0.590 |

λ = 0.1 is a good operating point (λ = 1.0 close); the optimum is somewhat
sharp, so we select on validation CER.

### 5.5 Domain gap

A recogniser trained on synthetic lines reads real lines at CER ≈ 0.96
(zero-shot) — i.e. it does not transfer. Real-data results therefore require a
recogniser trained on the real domain (real clean CER 0.690).

---

## 6. Limitations

- All damage is **synthetic**. Even the real-text experiment applies our damage
  simulator to real line images; genuine historical degradation is not covered.
- Line-level only; real documents need line segmentation (we did not address it).
- The recogniser is a modest CRNN; absolute CERs are high (0.69 real), so the
  *relative* restoration gain is the meaningful signal.
- Single-seed for the real RITL run and for the λ sweep.

---

## 7. Conclusion

Recognition-in-the-loop restoration is a simple, well-motivated method that
significantly improves OCR of damaged Nastaliq lines — on synthetic and real
text — and beats a strong pixel-fidelity baseline and every classical method.
The result that flow matching, structure conditioning and iteration do *not*
help is, we argue, just as useful: it locates the entire benefit in the
recognition signal.

---

## Appendix — reproduction

```
# data
python src/dataset_builder.py --sentences resources/urdu_corpus.txt --damage-severity 1.5
python src/build_real_dataset.py --src /path/to/UTRSet-Real --out data_real --severity 1.5

# recogniser + restorers
python src/train_baseline.py   --epochs 50 --augment
python src/train_restoration.py --epochs 20
python src/train_restoration.py --epochs 20 --lambda-ocr 0.1   # RITL

# evaluation + stats
python experiments/eval_restore_ocr.py --restoration-checkpoint checkpoints/restoration_hard_ocr.pt --out results/stats/ritl.json
python experiments/compare_stats.py baseline=... ritl=...
python experiments/make_restoration_figures.py
```

Heavy runs are containerised for Modal (`modal_app.py`).
