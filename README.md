# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

EE656 course project (IIT Kanpur, Prof. Nishchal Kumar Verma). This is a from-scratch PyTorch
re-implementation of **ISTVT** (Zhao et al., *IEEE Transactions on Information Forensics and
Security*, 2023) for video-level deepfake detection. It is scaled down to train on a single
Kaggle Tesla T4.

**Result:** trained and evaluated on **9,980 FaceForensics++ face-crop sequences**, with **85.84%
test accuracy** and **93.14% test AUROC**. The repo also includes sequence-length, model-depth and
robustness analyses, and LRP-based spatial and temporal relevance heatmaps.

## Architecture (all in `notebook3a8aa97f51.ipynb`)

```
frames (T x 3 x H x W)
  -> Xception entry flow (depthwise-separable convs, 728 channels)   per frame
  -> feature-map cells as tokens + spatial/temporal class tokens + position embedding
  -> M x [ temporal self-attention (with self-subtract on Q, K)
           -> spatial self-attention
           -> MLP ], each with residual + LayerNorm
  -> MLP head on the class token -> real / fake
```

- **Decomposed spatial-temporal self-attention.** Temporal attention runs across the T frames at
  each spatial position, and spatial attention runs across positions within each frame. This costs
  O(T²·HW + T·(HW)²) instead of O((T·HW)²) for joint attention.
- **Self-subtract mechanism.** Before temporal attention, each frame's tokens are replaced by their
  difference from the previous frame, `I' = cat(I[0:2], I[2:] - I[1:-1])`. Static content cancels
  and inter-frame inconsistencies, the deepfake signature, remain.
- **Interpretability.** Hooks capture each block's spatial and temporal attention maps and their
  gradients. Relevance is propagated as Ā = I + max(E_h[R ∘ ∇A], 0) and multiplied across blocks
  (the paper's Algorithm 1), giving separate per-frame spatial heatmaps and temporal heatmaps. The
  notebook compares this with raw attention, attention rollout and Grad-CAM, and runs a
  pixel-dropout test of explanation quality.

## Setup vs the paper

| Setting | Paper | This implementation | Why |
| --- | --- | --- | --- |
| Input size | 300×300 | 128×128 | T4 memory (16 GB) |
| Frames per sequence T | 6 | 4 | Memory; each video folder has 20 frames |
| Transformer blocks M | 12 | 4 | Memory and training budget |
| Attention heads | 8 | 4 | Same |
| Training | SGD, lr 5e-4 | SGD (momentum 0.9), lr 5e-4, 5-epoch warm-up then cosine, 10 epochs | Compute |
| Parameters | – | 35.5M | – |

**Data.** FaceForensics++ cropped faces from Kaggle: 1,000 real and 1,000 fake videos with 20
frames each. They are cut into 9,980 non-overlapping sequences (4,996 real, 4,984 fake) and split
70/15/15 into 6,986 / 1,497 / 1,497.

## Results

| Metric (test, 1,497 sequences) | Value |
| --- | --- |
| Accuracy | **85.84%** |
| AUROC | **93.14%** |
| Real: precision / recall | 0.89 / 0.82 |
| Fake: precision / recall | 0.83 / 0.90 |
| Macro F1 | 0.86 |

**Sequence length** (small-subset ablation): T = 2 gives 60.0%, T = 4 gives 64.0%. Longer
sequences did not fit the subset.

**Model depth** (small-subset ablation): M = 4 gives 66%, M = 6 gives 56%, M = 8 gives 58%,
M = 12 gives 56%. Deeper models overfit at this data size.

**Robustness** (100 test sequences):
- JPEG compression: quality 50 → 10 lowers accuracy from 83% to 57%.
- Downscaling to 0.1×: 52%.
- Random 16×16 dropout: 46–53%.

The scaled-down model relies on fine texture, so it is far less robust than the full-size paper
model.

## Files

| File | Contents |
| --- | --- |
| `notebook3a8aa97f51.ipynb` | Full pipeline with outputs: data, model, training, evaluation, LRP heatmaps, ablations, robustness, single-video demo |
| `Report.pdf` | Project report |
| `ISTVT_Presentation.pdf` | Presentation slides |

## Run

Open the notebook on Kaggle with a GPU and add the dataset
`debajyatidey/faceforensics-videos-cropped-faces`. `DATA_ROOT` points at it. Training takes about
10 epochs on a T4.

## Limitations

- **The split is at sequence level.** Sequences from the same video, and FF++ fakes built from the
  same source faces, can fall on both sides of the split, so the test scores are optimistic. A
  video- or identity-level split (FF++ official split) is the right protocol and is the next step.
- The self-subtract on/off ablation and the cross-dataset evaluation (Celeb-DF, DFDC) are set up
  in the notebook but were not trained or run.
- The ablations use small subsets and are indicative only.

## Reference

Zhao, C., Wang, C., Hu, G., Chen, H., Liu, C., & Tang, J. (2023). ISTVT: Interpretable
Spatial-Temporal Video Transformer for Deepfake Detection. *IEEE Transactions on Information
Forensics and Security*, 18, 1335–1348. https://doi.org/10.1109/TIFS.2023.3239223
