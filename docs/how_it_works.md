# How the model works

NeuroLens trains **image-level classifiers**: one 2-D MRI slice in, one of four dataset labels out
(`glioma`, `meningioma`, `notumor`, `pituitary`). It does not segment or localise tumours, and it
does not reason about patients, scan series or clinical context.

## Preprocessing (identical for training, evaluation and inference)

1. Decode the file and convert it to RGB (grayscale is replicated to three channels; 16-bit
   images are min-max scaled to 8 bits; EXIF orientation is applied).
2. Resize to `image_size x image_size` (default 224) with bilinear interpolation. The aspect
   ratio is not preserved.
3. Normalise with ImageNet mean/std (required by the pretrained backbones; kept for the custom CNN
   for consistency).

The preprocessing configuration is stored **inside every checkpoint** and rebuilt from there at
inference time, so a served model cannot silently drift from how it was evaluated.

Training adds mild augmentation (random resized crop 85-100 %, horizontal flip, ±10° rotation,
small brightness/contrast jitter). Validation, test and inference never augment.

## Architectures

| Name | Initialisation | Parameters | Grad-CAM layer |
|---|---|---|---|
| `custom_cnn` | from scratch | ~1.2 M | `features[-1]` (14x14 at 224 px) |
| `resnet18` | ImageNet (torchvision) | ~11.2 M | `layer4` (7x7) |
| `efficientnet_b0` | ImageNet (torchvision) | ~4.0 M | `features[-1]` (7x7) |

The custom CNN is a stride-2 stem followed by four double-conv blocks with batch normalisation,
global average pooling, dropout and a linear classifier. The transfer models replace the ImageNet
head with `Dropout -> Linear(4)` and fine-tune all layers (or only the head with
`model.freeze_backbone: true`).

## Training

* Loss: cross-entropy (optional label smoothing). Optimiser: AdamW (or SGD + Nesterov).
* Schedule: cosine annealing (or reduce-on-plateau).
* Model selection: the checkpoint with the best **validation** metric (`training.monitor`,
  default macro-F1) is saved as `best.pt`; early stopping uses the same metric.
* Mixed precision (`float16` autocast + gradient scaling) is used on CUDA when `training.amp` is
  true; CPU training runs in float32.
* `last.pt` stores optimiser, scheduler, scaler, early-stopping and RNG state so an interrupted
  run can continue with `--resume`. Resumption restarts the data-loader shuffling sequence, so a
  resumed run is not bit-identical to an uninterrupted one.

## Probabilities, calibration and uncertainty

* **Raw probabilities** are `softmax(logits)`.
* **Calibrated probabilities** are `softmax(logits / T)`, with one temperature `T` fitted by
  minimising negative log-likelihood on the validation split (`neurolens calibrate`). Temperature
  scaling changes confidence, never the predicted class.
* **Uncertainty** is reported as Shannon entropy and normalised entropy (entropy / log 4).
* **Abstention**: when enabled, a prediction is flagged if the top probability is below
  `min_confidence` (default 0.70) or normalised entropy exceeds `max_normalized_entropy` (default
  0.60). These are illustrative defaults, **not clinically validated safety thresholds**. The
  coverage/risk curve in the Model Laboratory shows how the error rate on accepted images changes
  as the threshold moves.

A high softmax score is not a guarantee of correctness, especially on images unlike the training
distribution (other scanners, sequences, preprocessing or non-MRI images).
