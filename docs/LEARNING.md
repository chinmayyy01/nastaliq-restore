# Learning guide — Nastaliq restoration project

Everything you need to understand this repo end-to-end, grouped by the part of
the system it explains. Each entry says *why it matters here*.

---

## 0. The big picture (read first)

- **CRNN + CTC** is the recogniser paradigm used throughout OCR. Start here:
  - Shi, Bai, Yao — *An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition (CRNN)* — https://arxiv.org/abs/1507.05717
  - Graves et al. — *Connectionist Temporal Classification* —
    https://www.cs.toronto.edu/~graves/icml_2006.pdf
- **U-Net** is the restoration backbone:
  - Ronneberger et al. — *U-Net: Convolutional Networks for Biomedical Image
    Segmentation* — https://arxiv.org/abs/1505.04597
- **PyTorch CTCLoss docs** (blank index, `input_lengths` semantics — the bug we
  fixed): https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

---

## 1. CTC + sequence recognition (the recogniser)

| Resource | Why |
|---|---|
| Graves 2006, CTC paper (above) | The loss; blank token; alignment-free training |
| CRNN (above) | CNN encoder → recurrent → CTC; our architecture |
| TrOCR — https://arxiv.org/abs/2109.10282 | Transformer alternative to CRNN |
| PyTorch CTCLoss docs (above) | `input_lengths` must exclude padding — a real bug in our first version |
| Attention Is All You Need — https://arxiv.org/abs/1706.03762 | Transformers; needed if we upgrade the encoder |

**YouTube (search links are safe):**
- CTC explained: https://www.youtube.com/results?search_query=connectionist+temporal+classification+explained
- CTC + OCR in PyTorch: https://www.youtube.com/results?search_query=ocr+ctc+loss+pytorch+tutorial
- CRNN from scratch: https://www.youtube.com/results?search_query=crnn+ocr+from+scratch

---

## 2. Urdu / Nastaliq script (the domain)

- **Line and Ligature Segmentation for Nastaliq** (Springer):
  https://link.springer.com/content/pdf/10.1007/978-3-030-32150-5_78
- **Recognition of Nastalique Urdu Ligatures** (ACM):
  https://dl.acm.org/doi/10.1145/2505377.2505379
- **From pixels to strokes: a survey on generative handwriting** (IJDAR 2026):
  https://link.springer.com/article/10.1007/s10032-026-00609-2
- **UTRNet — Urdu text recognition** (CNN+RNN+attention, printed):
  https://huggingface.co/papers/2306.15782
- **ET-Network — Urdu handwritten text recognition** (EfficientNet+transformer):
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0302590
- **Urdu HTR with ResNet18**: https://arxiv.org/abs/2103.05105
- **From Press to Pixels — Urdu newspapers** (YOLO + SwinIR + LLM; shows
  Nastaliq is ~2x harder than Naskh): https://arxiv.org/abs/2505.13943

Key takeaway: Nastaliq is *cursive with overlapping ligatures and multiple
baselines*; this is exactly why a stroke-structure prior is worth trying.

---

## 3. Document / text image restoration (the novel branch)

| Paper | Venue | Why |
|---|---|---|
| **GSDM — Text Image Inpainting via Global Structure-Guided Diffusion** — https://arxiv.org/abs/2401.14832 | AAAI 2024 | Closest prior art: structure prior + diffusion for text inpainting |
| **DocRes — generalist document restoration** — https://arxiv.org/abs/2405.04408 | CVPR 2024 | Unified restoration with task prompts |
| **Uni-DocDiff — unified diffusion document restoration** — https://arxiv.org/abs/2508.04055 | 2025 | Multi-task diffusion, prior pool |
| **DocWaveDiff — wavelet U-Net + diffusion** | WACV 2026 | Predict-and-refine for documents |
| **UniRestore — perceptual + task-oriented restoration** — https://arxiv.org/abs/2501.13134 | CVPR 2025 | Restoration optimised for downstream tasks |
| **Restora-Flow — mask-guided flow matching** — https://arxiv.org/abs/2511.20152 | WACV 2026 | Flow matching for restoration (faster than diffusion) |
| **PnP-Flow — plug-and-play flow matching** — https://arxiv.org/abs/2512.04283 | ICLR 2025 | Flow matching for inverse problems |
| **Diffusion models for IR survey** — https://dl.acm.org/doi/10.1007/s11263-025-02570-9 | IJCV 2025 | Full landscape |
| **IMAGE — iterative mutual guidance (recognise + recover)** — https://arxiv.org/abs/2409.14483 | 2024 | The recognition-in-the-loop idea |

**Foundations for the generative upgrade:**
- DDPM (diffusion): https://arxiv.org/abs/2006.11239
- Flow Matching: https://arxiv.org/abs/2210.02747
- pix2pix (image-to-image): https://arxiv.org/abs/1611.07004

**YouTube:**
- U-Net: https://www.youtube.com/results?search_query=unet+architecture+explained
- Diffusion models: https://www.youtube.com/results?search_query=diffusion+models+explained+from+scratch
- Flow matching: https://www.youtube.com/results?search_query=flow+matching+explained+tutorial

---

## 4. Structure / graph modelling (our novelty)

- **GCN** — https://arxiv.org/abs/1609.02907
- **GAT** — https://arxiv.org/abs/1710.10903
- **G-HTR — graph-based handwriting trajectory reconstruction** (CVPR 2026):
  contour-aware graphs + GNN for stroke structure
- **StrokeNet — GNN stroke classification** — https://arxiv.org/html/2512.06290
- **GNNs for end-to-end HTR** — WACV 2024
- **skimage skeletonize** —
  https://scikit-image.org/docs/stable/api/skimage.morphology.html#skeletonize
- **scipy distance transform** —
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
- **Stanford CS224W (Graph ML)** — https://www.youtube.com/results?search_query=stanford+cs224w+graph+neural+networks

Our finding: a structure prior computed from the *damaged* image is redundant
(it describes surviving strokes). The novelty must **predict the missing
structure**, or reason about graph completion.

---

## 5. State-space models / Mamba (recogniser upgrade option)

- **Mamba (Gu & Dao)** — https://arxiv.org/abs/2312.00752
- **SSM OCR benchmark (lines→paragraphs)** — https://arxiv.org/abs/2606.23524
- **Historical newspaper OCR: SSM vs Transformer vs BiLSTM** —
  https://arxiv.org/html/2604.00725v2
- **TextMamba** — https://arxiv.org/abs/2512.06657
- **Spatial-Mamba** — ICLR 2025 (structure-aware state fusion)
- **YouTube** — https://www.youtube.com/results?search_query=mamba+state+space+model+explained

---

## 6. Datasets & tooling used in this repo

- **Tatoeba Urdu sentences** (our corpus source): https://tatoeba.org/en/downloads
- **OpenITI** (Urdu/Arabic corpora): https://github.com/OpenITI
- **Pillow text rendering / Raqm (HarfBuzz shaping)** —
  https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
- **PyTorch MPS backend** — https://pytorch.org/docs/stable/notes/mps.html
- **AwamiNastaliq font** — in `fonts/` (see `fonts/LICENSE.txt`)

---

## 7. Code map (what to read in this repo)

| File | Role |
|---|---|
| `src/text_renderer.py` | Renders Urdu lines via Raqm/HarfBuzz |
| `src/damage_simulator.py` | Synthesises damage + mask |
| `src/dataset_builder.py` | Builds images + manifests |
| `src/dataset.py` | OCR dataset, vocab, CTC collate |
| `src/models/baseline_recognizer.py` | CNN + BiLSTM + CTC |
| `src/train_baseline.py` | Recogniser training |
| `src/structure_graph.py` | Stroke/dot/junction graph extraction |
| `src/structure_prior.py` | Raster structure channels (skeleton + distance) |
| `src/restoration_data.py` | Restoration dataset + collate |
| `src/models/restoration_unet.py` | U-Net restoration backbone |
| `src/train_restoration.py` | Restoration training |
| `experiments/graph_feasibility.py` | Graph feasibility spike |
| `experiments/eval_restore_ocr.py` | restore→OCR CER evaluation |

---

## 8. Current numbers (for context)

- Recogniser val CER: **0.558** (clean lines); damaged 0.593, restored 0.576.
- Baseline restoration: PSNR 23.6 / SSIM 0.95; recovers ~half the damage gap.
- Naive structure prior: no downstream gain → motivates predicted structure.
