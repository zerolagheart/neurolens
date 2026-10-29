# Limitations and ethical considerations

**NeuroLens is a research and education project. It is not a medical device, has not been
clinically validated, and must not be used to diagnose, triage or treat anyone.**

## Task and data

* **Image-level labels only.** The model assigns one of four dataset folder labels to a single
  2-D image. It does not detect, localise or segment tumours, grade them, or consider patient
  history, other sequences or other slices.
* **Dataset provenance.** The Nickparvar dataset is a compilation of earlier public collections
  (figshare, SARTAJ and Br35H, according to its Kaggle description). Acquisition protocols,
  scanners, preprocessing and labelling procedures vary between sources and are not fully
  documented. Labels have not been reviewed by this project.
* **Source confounding.** Classes come partly from different source collections, so a model can
  learn source-specific cues (image size, borders, intensity profiles, text) instead of anatomy.
  High test accuracy does not rule this out.
* **No patient identifiers.** Patient-level independence between training and test data cannot be
  established. Several slices of one patient may fall on both sides of a split.
* **Duplicates and leakage.** Run `neurolens audit`. On the copy of the dataset used to develop
  this project, the audit found exact duplicates inside both splits and a large number of
  perceptual-hash near-duplicate candidates linking test images to training images. Results on
  the original test split are therefore likely optimistic; the leakage-audited benchmark is
  provided as a stricter (but still imperfect, and not comparable) alternative. Near-duplicate
  detection is heuristic and needs visual review.
* **Distribution shift.** Performance on images from other hospitals, scanners, sequences or
  preprocessing pipelines is unknown. Non-MRI or out-of-scope images still receive a prediction;
  the abstention policy may flag some, but it is not an out-of-distribution detector.

## Outputs

* **Probabilities are not certainty.** Raw softmax scores are often over-confident. Temperature
  scaling improves average calibration on data like the validation split; it does not guarantee
  calibration for any individual image or under distribution shift.
* **Thresholds are not safety limits.** The abstention thresholds are configurable defaults.
  They were not chosen with clinical input and carry no clinical meaning.
* **Grad-CAM is not evidence.** Heatmaps show what influenced the model's score, coarsely. They do
  not establish medical correctness or prove that a tumour was found.

## Responsible use

* Do not upload identifiable patient data to a shared deployment. The API processes uploads in
  memory, does not store them, and does not log image contents or filenames, but deployments
  you run are your responsibility.
* Respect the dataset's licence and terms on Kaggle and the terms of its original sources.
* When reporting results, state the benchmark (`original` or `leakage_audited`), the split
  procedure and seed, and that patient-level independence is not established.
