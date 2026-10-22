from pathlib import Path
from collections import Counter
import random
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np  

ROOT = Path(__file__).resolve().parent.parent / "data"
SPLITS = ["train", "valid", "test"]

cfg = yaml.safe_load(open(ROOT / "data.yaml"))
names = cfg["names"]
print("Classes:", names)

def read_labels(label_path):
    """Return a list of (class_id, x_c, y_c, w, h), all normalised 0-1."""
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append((int(parts[0]), *map(float, parts[1:5])))
    return rows

# 1) Counts per split and boxes per class
for split in SPLITS:
    images = sorted((ROOT / split / "images").glob("*.jpg"))
    label_files = sorted((ROOT / split / "labels").glob("*.txt"))
    class_counts, empty = Counter(), 0
    for lf in label_files:
        rows = read_labels(lf)
        if not rows:
            empty += 1
        class_counts.update(r[0] for r in rows)
    print(f"\n{split}: {len(images)} images, {len(label_files)} label files, "
          f"{empty} with no boxes")
    for c, n in sorted(class_counts.items()):
        print(f"  {names[c]}: {n}")

# 2) Median box size per class (training labels)
sizes = {c: [] for c in range(len(names))}
for lf in (ROOT / "train" / "labels").glob("*.txt"):
    for c, xc, yc, bw, bh in read_labels(lf):
        sizes[c].append((bw, bh))

print("\nBox size per class (train split)")
print("  class: n | median w x h (fraction) | median area (fraction) | "
      "median w x h (pixels at imgsz=640)")
for c, wh in sizes.items():
    if not wh:
        continue
    wh = np.array(wh)
    w_med, h_med = np.median(wh[:, 0]), np.median(wh[:, 1])
    area_med = np.median(wh[:, 0] * wh[:, 1])
    # imgsz=640 scales the 720x540 image so the long side is 640 (so 640 x 480)
    print(f"  {names[c]}: n={len(wh)} | {w_med:.3f} x {h_med:.3f} | "
          f"{area_med:.4f} | {w_med * 640:.0f} x {h_med * 480:.0f} px")

# 3) Draw boxes on a few random training images
sample = random.sample(sorted((ROOT / "train" / "images").glob("*.jpg")), 6)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, img_path in zip(axes.ravel(), sample):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]                       # use the real image size
    label_path = ROOT / "train" / "labels" / (img_path.stem + ".txt")
    for c, xc, yc, bw, bh in read_labels(label_path):
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)   # un-normalise
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, str(names[c]), (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    ax.imshow(img)
    ax.set_title(img_path.name[:30], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()