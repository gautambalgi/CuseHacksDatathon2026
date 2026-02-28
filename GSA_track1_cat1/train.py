import os, random, time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


TRAIN_CSV  = "train.csv"
TRAIN_DIR  = "train/images/"
VAL_CSV    = "val.csv"
VAL_DIR    = "val/images/"
MODEL_PATH = "model/"
SAVE_NAME  = "efficientnet_b3.pth"

NUM_CLASSES    = 102
IMG_SIZE       = 224
BATCH_SIZE     = 16
EPOCHS         = 40
LR             = 3e-4
PATIENCE       = 8
SEED           = 42
UNFREEZE_EPOCH = 5

os.makedirs(MODEL_PATH, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class ImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
        if self.transform: img = self.transform(img)
        label = int(row["label"]) - 1
        return img, label


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


train_ds = ImageDataset(TRAIN_CSV, TRAIN_DIR, train_tf)
val_ds   = ImageDataset(VAL_CSV,   VAL_DIR,   val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


labels_arr = pd.read_csv(TRAIN_CSV)["label"].values - 1
cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=labels_arr)
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)


model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)


for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n

best_val_acc = 0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    if epoch == UNFREEZE_EPOCH + 1:
        print("\n--- Unfreezing full model for fine-tuning ---\n")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - UNFREEZE_EPOCH, eta_min=1e-6)

    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"Val loss {vl_loss:.4f} acc {vl_acc:.4f} | {time.time()-t0:.1f}s")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, SAVE_NAME))
        print(f"  ✓ Saved best model (val acc {best_val_acc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

print(f"\nTraining done. Best val acc: {best_val_acc:.4f}")
print(f"Model saved to {MODEL_PATH}{SAVE_NAME}")