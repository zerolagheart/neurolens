"""NeuroLens: explainable brain MRI image classification for ML research and education.

NeuroLens is **not** a medical device. Its outputs are image-level class scores produced by a
model trained on a public research dataset and must never be used for diagnosis or treatment.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "Research and educational use only. NeuroLens is not a medical device and its output is "
    "not a medical diagnosis. Image-level classification does not localise or segment tumours, "
    "and model scores are not clinically validated."
)

__all__ = ["DISCLAIMER", "__version__"]
