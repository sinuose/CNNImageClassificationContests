# infer_dr_to_csv.py  ────────────────────────────────────────────────────
import csv, os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from hdr.EfficientNet import *          # same file you trained with

# ── CONFIG -----------------------------------------------------------------
TEST_DIR      = Path("D:\\DiabeticRetinoplastyDataset\\")          # folder with *.jpeg test images
CKPT_PATH     = Path("C:\\Users\\mj\\Documents\\GitHub\\CNNImageClassificationContests\\DiabeticRetinopathy\\checkpoints\\efficientnet_b0_1745200355_epoch14.pth")  # your trained weights
OUT_CSV       = "predictions.csv"

NUM_CLASSES   = 6      # 0‑5  (set to 5 if your labels are 0‑4)
BATCH_SIZE    = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── TRANSFORMS  (match validation transforms used for training) ------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# ── DATASET for inference  -------------------------------------------------
class TestImages(Dataset):
    """
    Returns (tensor, stem) where stem is e.g. '1_left'
    """
    def __init__(self, root_dir, tfm):
        self.paths = sorted([p for p in Path(root_dir).glob("*.jpeg")])
        self.tfm   = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        return img, path.stem

# ── LOAD MODEL  ------------------------------------------------------------
model = EfficientNet(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ── INFERENCE  -------------------------------------------------------------
test_loader = DataLoader(TestImages(TEST_DIR, transform),
                         batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

pred_rows = []

with torch.no_grad():
    for imgs, stems in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        preds  = logits.argmax(1).cpu().tolist()

        pred_rows.extend(zip(stems, preds))

# ── SAVE CSV  --------------------------------------------------------------
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "level"])
    # keep original order
    writer.writerows(pred_rows)

print(f"Saved predictions to {OUT_CSV}  ({len(pred_rows)} rows).")