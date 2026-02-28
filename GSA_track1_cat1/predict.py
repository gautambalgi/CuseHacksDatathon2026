
import os
import pandas as pd
from PIL import Image

INPUT_CSV   = "test.csv"
IMAGE_DIR   = "test/images/"
OUTPUT_PATH = "predictions.csv"
MODEL_PATH  = "model/"


import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import efficientnet_b3

NUM_CLASSES = 102
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_tta_tfs = [
    lambda img: _val_tf(img),
    lambda img: _val_tf(TF.hflip(img)),
    lambda img: _val_tf(TF.vflip(img)),
    lambda img: transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.2)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img),
    lambda img: transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.1)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img),
]


def load_model():
    m = efficientnet_b3(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    weights_path = os.path.join(MODEL_PATH, "efficientnet_b3.pth")
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m


def predict(model, images: list) -> list[int]:
    preds = []
    with torch.no_grad():
        for img in images:
            if img is None:
                preds.append(1)
                continue
            img = img.convert("RGB")
            logits_sum = None
            for tf in _tta_tfs:
                tensor = tf(img).unsqueeze(0).to(DEVICE)
                out = model(tensor)
                logits_sum = out if logits_sum is None else logits_sum + out
            preds.append(logits_sum.argmax(1).item() + 1)
    return preds

# ==============================================================================
# DO NOT MODIFY ANYTHING BELOW THIS LINE
# ==============================================================================

def _load_images(df):
    images, missing = [], []
    for _, row in df.iterrows():
        path = os.path.join(IMAGE_DIR, row["filename"])
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            missing.append(row["filename"])
            images.append(None)
    if missing:
        print(f"WARNING: {len(missing)} image(s) not found. First few: {missing[:5]}")
    return images

def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)
    missing_cols = {"image_id", "filename"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing columns: {missing_cols}")
    print(f"Loaded {len(df):,} images from {INPUT_CSV}")

    images = _load_images(df)
    model  = load_model()
    preds  = predict(model, images)

    if len(preds) != len(df):
        raise ValueError(f"predict() returned {len(preds)} predictions for {len(df)} images.")

    out = df[["image_id"]].copy()
    out["label"] = [int(p) for p in preds]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()