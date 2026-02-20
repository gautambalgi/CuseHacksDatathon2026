"""
Track 1 – predict.py
=====================
Implement load_model() and predict() only.
DO NOT modify anything below the marked line.

Self-evaluate on val set:
    INPUT_CSV  = "val.csv"
    IMAGE_DIR  = "val/images/"

Final submission (paths must be set to test before submitting):
    INPUT_CSV  = "test.csv"
    IMAGE_DIR  = "test/images/"
"""

import os
import pandas as pd
from PIL import Image

# ==============================================================================
# CHANGE THESE PATHS IF NEEDED
# ==============================================================================

INPUT_CSV   = "test.csv"
IMAGE_DIR   = "test/images/"
OUTPUT_PATH = "predictions.csv"
MODEL_PATH  = "model/"

# ==============================================================================
# YOUR CODE — IMPLEMENT THESE TWO FUNCTIONS
# ==============================================================================

def load_model():
    """
    Load and return your trained model from MODEL_PATH.

    Example (PyTorch):
        import torch
        import torchvision.models as models
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 102)
        model.load_state_dict(torch.load(MODEL_PATH + "weights.pt", map_location="cpu"))
        model.eval()
        return model

    Example (scikit-learn):
        import joblib
        return joblib.load(MODEL_PATH + "classifier.pkl")
    """
    raise NotImplementedError("Implement load_model()")


def predict(model, images: list) -> list[int]:
    """
    Run inference on a list of PIL images.

    Args:
        model  : whatever load_model() returns
        images : list of PIL.Image objects (RGB), one per row in INPUT_CSV

    Returns:
        A list of integer class labels (1–102), one per image.
        e.g. [1, 45, 102, 7, ...]

    Rules:
        - Return exactly len(images) predictions
        - Labels must be integers between 1 and 102
        - Use only labels from label_list.txt
    """
    raise NotImplementedError("Implement predict()")

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