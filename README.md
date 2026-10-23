
# MMA3001 Project: Packaging defect detection and severity scoring for pork rasher trays

Individual project for MMA3001 Numerical Methods and Machine Learning ("Dataset 3": automated
detection of pork rasher packaging errors). Author: Bryce Cecchini, 34990739.

## Overview
A YOLOv8 object detector finds and localises five defect classes in images of packaged pork
rasher trays. A severity module then converts each detection into a score (class weight ×
the fraction of the image the box covers), sums the scores per tray, and compares the result
against a pass/fail threshold. The detector is trained and evaluated in a Google Colab
notebook; the label-reading and severity-scoring code lives in `src/` and is unit-tested.

## Engineering problem and scope
Manual inspection of packaged meat is slow and inconsistent. This project asks whether a
small detector can find defects well enough to support a severity-based pass/fail decision.

In scope: single-image detection and severity scoring on one fixed camera view.
Out of scope: real-time deployment, other products or camera setups, and validated
production pass/fail limits (no ground-truth severity or pass/fail labels exist).

## Inputs and outputs
**Input:** an RGB image of a tray (JPEG, 720 × 540 px in the dataset; the detector resizes
internally to 640 px on the long side).

**Detector output (per detection):** class name, confidence (0 to 1), and a bounding box as
normalised `x_centre, y_centre, width, height` (all 0 to 1, relative to the image).

**Severity outputs:**
- Per defect: `class_weight × width × height` (dimensionless; width and height are image
  fractions, clipped to 1).
- Per tray: the sum of its defect severities (0 if no defects).
- Pass/fail: pass if tray severity is below a threshold. No single threshold is claimed to be
  correct; the report evaluates 0.005 to 0.1.

Class weights

unsealed = 1.0
packaging error = 0.9
loose meat = 0.5
twisted meat = 0.5
wrinkle = 0.2

The weights are engineering judgement, their justification is in the report.

**Invalid input handling:** `read_labels` skips blank lines and lines with fewer than five
fields and raises `ValueError` on non-numeric values. `box_extent` raises `ValueError` for
negative or NaN box sizes and clips sizes above 1 to 1. `defect_severity` raises `KeyError`
for a class with no weight. Image format and size are not validated by this code: the
detector resizes inputs, and unreadable files fail when loaded.

## Repository structure
```
MMA3001_Project.ipynb   Colab notebook: dataset download, training, evaluation, severity analysis
src/                    labels.py (label reader), severity.py (severity scoring)
test/                   pytest tests for src/
scripts/                inspect_database.py (dataset statistics and sample plots)
results/                run settings (args_*.yaml), figures, severity CSV, test report
docs/                   generated HTML API documentation (pdoc)
data/                   dataset (NOT in the repo; see Data)
requirements.txt, pytest.ini, LICENCE
```

## Setup (local: tests, documentation, dataset inspection)
Python 3.12 was used locally. From the repository root:

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

If PowerShell refuses to run `activate`, call the environment's Python directly instead
(`.venv\Scripts\python -m pip install -r requirements.txt`).

Training and evaluation need a GPU, so they run in Google Colab.

## Data
Roboflow Universe project `pork-rasher-error-packaging` (workspace `hello-2aqe0`), **Version 4**,
exported in **YOLOv8** format. 3,549 images at 720 × 540 px:

| Split | Images | Images with no defects |
|---|---|---|
| train (augmented) | 3,349 | 684 |
| valid | 120 | 14 |
| test | 80 | 12 |

Classes: loose-meat, packaging-error, twisted-meat, unsealed, wrinkle. Class counts are very
unbalanced and `wrinkle` has no valid or test boxes (details in the report).

To use the dataset locally, download Version 4 in YOLOv8 format from the Roboflow page and
unzip it directly into `data/`, so that `data/data.yaml`, `data/train/`, `data/valid/` and
`data/test/` exist. The Roboflow API key is read from Colab Secrets (name `API_KEY`) and is
never stored in the repository.

Licence: CC BY 4.0.
Citation:
```bibtex
@misc{ pork-rasher-error-packaging_dataset,
  title = { Pork Rasher Error(Packaging) Dataset },
  type = { Open Source Dataset },
  author = { Hello },
  howpublished = { \url{ https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging } },
  url = { https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { may },
  note = { visited on 2026-10-23 },
}
```

## Reproducing the results
1. **Dataset statistics (local):** `python scripts/inspect_database.py` prints split sizes, boxes
   per class and median box sizes, then shows six random training images with their boxes.
   Requires the dataset in `data/`.
2. **Training and evaluation (Colab, GPU):** open `MMA3001_Project.ipynb` in Google Colab, select a
   T4 GPU runtime, add your Roboflow key to Colab Secrets as `API_KEY`, and run the cells in the
   order given in the notebook's header. Three runs were made (YOLOv8n at 640 px, YOLOv8s at 640 px,
   YOLOv8n at 960 px, 30 epochs, seed 0); their exact settings are in `results/args_*.yaml`.
   Trained weights are not stored in this repository. Retraining takes roughly 20 minutes (run A)
   on a Colab T4, and results may differ slightly.
3. **Severity analysis:** the final notebook cells clone this repository and use `src/`.
4. **Tests:** `python -m pytest -v` (16 tests). The report for the submitted version is
   `results/pytest_report.html`; regenerate with
   `python -m pytest -v --html=results/pytest_report.html --self-contained-html`.
5. **Documentation:** open `docs/index.html` in a browser (it redirects to `docs/src.html`).
   Regenerate with `pdoc src -d google -o docs`.

## Results summary
Selected model: YOLOv8n at 640 px (run A). Evaluated on the held-out test split (80 images):

| Run | Model, image size | mAP50 valid / test | Inference ms (valid / test) | Parameters | GFLOPs | Training time |
|---|---|---|---|---|---|---|
| A (selected) | YOLOv8n, 640 | 0.567 / 0.522 | 5.0 / 5.9 | 3.13 M | 8.6 | ~40 s/epoch |
| B | YOLOv8s, 640 | 0.490 / 0.408 | 8.5 / 8.9 | 11.13 M | 28.4 | ~51 s/epoch |
| C | YOLOv8n, 960 | 0.490 / 0.378 | 9.7 / 8.4 | 3.13 M | about 19 (estimate) | 0.747 h total |

mAP50 is the unweighted mean over the four classes that have valid/test boxes; two of them have
only 3 to 4 boxes, so per-class values (in the report) matter more. Speeds are on a Colab T4.

Severity: severity computed from predicted boxes vs from ground-truth boxes on the test set has
MAE 0.0183 (95% CI 0.0132 to 0.0243), against 0.0515 for a mean-only baseline; Spearman
correlation 0.825. Confidence threshold 0.25 was chosen on the valid split.

## Limitations
- Single seed per run and an 80-image test set; rare classes (3 to 4 test boxes) cannot be judged.
- `wrinkle` has no valid or test boxes, so it is untested.
- No human-labelled severity or pass/fail ground truth: severity is validated only against
  severity computed from the ground-truth boxes, and the pass/fail threshold is an assumption.
- Severity sums box areas, so overlapping duplicate detections inflate it.
- Images look like video frames, so near-duplicates across splits are possible.

## AI use
Tool: Claude (Anthropic) used throughout using chat function

Used for: Claude was given the project brief, the database selection and project aim to create an 
initial workflow and guide to setup github, vscode, colab and the gdrive. Claude was then prompted 
to generate code step by step, first in vscode for the src files and then on colab for the training
code. Claude was asked throughout the project to explain the concepts behind each block of code. 
Claude was then permitted to suggest structure for the github and proper documentation procedure.

Share of work: The AI wrote almost every line of code, I ran the code, made a record of output data and tweaked 
the code in minor ways. I decided upon the weights and which model to use as the final decision, 
I also decided to limit the scope of the severity calculation to avoid normalization of defects.

How it was checked: In order to verify the work was valid, I ran the 16 pytest tests which can be 
checked by hand. I also checked the spit counts against the dataset numbers. The worstcase images
were also checked by me in order to get an idea of why they performed so poorly. I tested the repo 
from a fresh clone.


## Licence and acknowledgements
Code licensed under MIT; see `LICENCE`. The dataset keeps its own
licence (above). Detection uses Ultralytics YOLOv8, which has its own licence terms.
