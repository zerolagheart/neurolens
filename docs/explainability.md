# Grad-CAM

NeuroLens implements Grad-CAM (Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization*, ICCV 2017) in `src/neurolens/explainability/gradcam.py`.

For the predicted class `c` and the feature maps `A^k` of a target convolutional layer:

1. Run a forward pass with gradients enabled (even if the caller is in `inference_mode`), keeping
   the target layer's activations via a forward hook and their gradients via a tensor hook.
2. Back-propagate the **logit** `y_c` (not the softmax probability).
3. Weight each channel by its spatially averaged gradient: `alpha_k = mean(dy_c / dA^k)`.
4. `CAM = ReLU(sum_k alpha_k A^k)`, upsampled bilinearly to the input size and min-max scaled to
   `[0, 1]` for that image.

Target layers: `features[-1]` for the custom CNN and EfficientNet-B0, `layer4` for ResNet-18. Hooks
are removed when the `GradCAM` context exits, parameter gradients are cleared, and the model's
train/eval mode is restored. An unsupported architecture raises a clear error. If the map is all
zeros (no positive evidence at that layer) it is shown blank and flagged, never filled in.

The prediction shown next to the heatmap comes from the **same forward pass** used for the CAM.

## How to read it

* Colour (blue -> red, `jet` colour map) shows *relative* influence on the predicted class
  score within one image. Values are normalised per image, so colours cannot be compared across
  images.
* Resolution is coarse (7x7 or 14x14 before upsampling); smooth blobs are expected.
* A highlighted region means "this area increased the class score in this model". It does **not**
  mean the region is a tumour, that the prediction is correct, or that the model reasons like a
  radiologist. Models can rely on artefacts such as text overlays, borders or scanner-specific
  intensity patterns, and Grad-CAM can reveal (but not rule out) such shortcuts.

## Where to get it

* Streamlit **MRI Analysis** page (view and download original / heatmap / overlay / JSON report)
* `POST /explain` (base64 PNGs in the JSON response)
* `neurolens explain --run <run> --image <file>`
