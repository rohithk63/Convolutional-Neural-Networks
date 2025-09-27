import os
import random
import json
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 128
AE_EPOCHS = 20
CLS_EPOCHS = 50
LR_AE = 1e-3
LR_CLS = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 5
USE_DENOISED_FOR_INFERENCE = True  
USE_DENOISED_FOR_TRAINING = False  

BASE_DIR = "/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET"

TRAIN_CLEAN_DIR = os.path.join(BASE_DIR, "train/clean")
TRAIN_NOISY_DIR = os.path.join(BASE_DIR, "train/noisy")
TEST_NOISY_DIR  = os.path.join(BASE_DIR, "test/noisy")

DENOISED_DIR = "Denoised_Images"
AE_WEIGHTS = "denoiser.pth"
CLS_WEIGHTS = "classifier.pth"
CONF_MAT_PNG = "confusion_matrix.png"
PRED_CSV = "test_labels.csv"
CLASS_INDEX_JSON = "class_to_idx.json"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # fixed small feature map
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


ae_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class PairedDataset(Dataset):
    def __init__(self, noisy_folder, clean_folder, transform=ae_transform):
        self.noisy = datasets.ImageFolder(noisy_folder, transform=transform)
        self.clean = datasets.ImageFolder(clean_folder, transform=transform)
        assert len(self.noisy) == len(self.clean), "Noisy and Clean counts differ!"
    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, idx):
        n_img, _ = self.noisy[idx]
        c_img, _ = self.clean[idx]
        return n_img, c_img


def build_weighted_sampler(dataset):

    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def train_autoencoder():
    print("\n[1/5] Training Denoising Autoencoder...")
    paired = PairedDataset(TRAIN_NOISY_DIR, TRAIN_CLEAN_DIR, transform=ae_transform)
    ae_loader = DataLoader(paired, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    model = DenoisingAutoencoder().to(device)
    optim_ = optim.Adam(model.parameters(), lr=LR_AE)
    crit = nn.MSELoss()

    model.train()
    for epoch in range(AE_EPOCHS):
        run_loss = 0.0
        for noisy_imgs, clean_imgs in ae_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            out = model(noisy_imgs)
            loss = crit(out, clean_imgs)
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            run_loss += loss.item()
        print(f"AE Epoch [{epoch+1}/{AE_EPOCHS}] Loss: {run_loss/len(ae_loader):.4f}")

    torch.save(model.state_dict(), AE_WEIGHTS)
    print(f"Saved AE weights -> {AE_WEIGHTS}")
    return model

def denoise_test_images(ae_model):
    print("\n[2/5] Denoising test images...")
    os.makedirs(DENOISED_DIR, exist_ok=True)
    ae_model.eval()
    with torch.no_grad():
        for img_name in sorted(os.listdir(TEST_NOISY_DIR)):
            in_path = os.path.join(TEST_NOISY_DIR, img_name)
            if not os.path.isfile(in_path):
                continue
            img = Image.open(in_path).convert("RGB")
            x = ae_transform(img).unsqueeze(0).to(device)
            out = ae_model(x)
            save_image(out, os.path.join(DENOISED_DIR, img_name))
    print(f"Denoised images saved -> {DENOISED_DIR}")

def load_classifier_datasets():

    if USE_DENOISED_FOR_TRAINING and os.path.isdir("Denoised_Train"):
        print("Using denoised train images at: Denoised_Train")
        train_ds = datasets.ImageFolder("Denoised_Train", transform=train_transform)
    else:
        train_ds = datasets.ImageFolder(TRAIN_CLEAN_DIR, transform=train_transform)

    class_to_idx = train_ds.class_to_idx
    with open(CLASS_INDEX_JSON, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"class_to_idx saved -> {CLASS_INDEX_JSON}: {class_to_idx}")

    val_ratio = 0.2
    val_size = int(len(train_ds) * val_ratio)
    train_size = len(train_ds) - val_size
    train_subset, val_subset = random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_sampler = build_weighted_sampler(train_subset)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_to_idx

def train_classifier():
    print("\n[3/5] Training Classifier...")
    train_loader, val_loader, class_to_idx = load_classifier_datasets()
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ = optim.Adam(model.parameters(), lr=LR_CLS, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer_, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    for epoch in range(CLS_EPOCHS):
 
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer_.zero_grad()
            loss.backward()
            optimizer_.step()

            run_loss += loss.item()
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        train_acc = 100.0 * correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        print(f"CLS Epoch [{epoch+1}/{CLS_EPOCHS}] "
              f"Loss: {run_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CLS_WEIGHTS)

    print(f"Best Val Acc: {best_val_acc:.2f}% | Saved classifier weights -> {CLS_WEIGHTS}")
    return model, class_to_idx

def eval_autoencoder_psnr_ssim(ae_model):
    print("\n[4/5] Evaluating Autoencoder (PSNR/SSIM before vs after)...")
    ae_model.eval()
    noisy_ds = datasets.ImageFolder(TRAIN_NOISY_DIR, transform=ae_transform)
    clean_ds = datasets.ImageFolder(TRAIN_CLEAN_DIR, transform=ae_transform)

    psnr_before, psnr_after, ssim_before, ssim_after = [], [], [], []

    with torch.no_grad():
        for (nimg, _), (cimg, _) in zip(noisy_ds, clean_ds):
            n_np = (nimg.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            c_np = (cimg.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            d = ae_model(nimg.unsqueeze(0).to(device)).cpu().squeeze(0)
            d_np = (d.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            psnr_before.append(psnr(c_np, n_np))
            psnr_after.append(psnr(c_np, d_np))
            ssim_before.append(ssim(c_np, n_np, channel_axis=2))
            ssim_after.append(ssim(c_np, d_np, channel_axis=2))

    print(f"PSNR  Before: {np.mean(psnr_before):.3f} | After: {np.mean(psnr_after):.3f}")
    print(f"SSIM  Before: {np.mean(ssim_before):.4f} | After: {np.mean(ssim_after):.4f}")

def save_confusion_matrix_png(model, class_to_idx):
    print("\n[5/5] Confusion Matrix on Validation Set...")

    full_ds = datasets.ImageFolder(TRAIN_CLEAN_DIR, transform=eval_transform)
    val_ratio = 0.2
    val_size = int(len(full_ds) * val_ratio)
    train_size = len(full_ds) - val_size
    _, val_subset = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[k for k,_ in sorted(class_to_idx.items(), key=lambda x:x[1])])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title("Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig(CONF_MAT_PNG, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix -> {CONF_MAT_PNG}")

def run_inference_and_write_csv(model, class_to_idx):
    print("\n[Inference] Generating predictions CSV...")
    test_dir = DENOISED_DIR if (USE_DENOISED_FOR_INFERENCE and os.path.isdir(DENOISED_DIR)) else TEST_NOISY_DIR
    tfm = eval_transform

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    preds = []
    model.eval()
    with torch.no_grad():
        for img_name in sorted(os.listdir(test_dir)):
            in_path = os.path.join(test_dir, img_name)
            if not os.path.isfile(in_path):
                continue
            img = Image.open(in_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            logits = model(x)
            pred_idx = int(torch.argmax(logits, 1).item())

            if NUM_CLASSES == 5:
                preds.append([img_name, pred_idx + 1])
            else:
                preds.append([img_name, idx_to_class[pred_idx]])

    df = pd.DataFrame(preds, columns=["Images", "Predicted_Classes"])
    df.to_csv(PRED_CSV, index=False)
    print(f"Saved predictions -> {PRED_CSV}")


def main():

    ae_model = train_autoencoder()

    denoise_test_images(ae_model)

    cls_model, class_to_idx = train_classifier()

    eval_autoencoder_psnr_ssim(ae_model)

    best_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    best_model.load_state_dict(torch.load(CLS_WEIGHTS, map_location=device))
    save_confusion_matrix_png(best_model, class_to_idx)

    run_inference_and_write_csv(best_model, class_to_idx)

if __name__ == "__main__":
    main()



