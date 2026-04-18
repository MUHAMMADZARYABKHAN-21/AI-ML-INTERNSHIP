"""
Task 3: Multimodal ML – Housing Price Prediction (Images + Tabular Data)
DevelopersHub Corporation – AI/ML Engineering Internship

Architecture:
  • CNN branch  → extract 64-dim feature vector from house images
  • Tabular branch → normalised structured features
  • Fusion layer → concatenate → dense layers → price prediction
Metrics: MAE and RMSE
"""

import warnings
warnings.filterwarnings("ignore")

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
IMG_SIZE    = 64
CNN_FEAT    = 64
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
N_SAMPLES   = 1200
IMG_DIR     = "house_images"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────
#  1. Synthetic data generation
# ─────────────────────────────────────────
def generate_synthetic_dataset(n: int):
    """
    Creates a synthetic housing dataset with:
    - Tabular features (sqft, beds, baths, age, garage, location score)
    - Procedurally generated house images (colour reflects price tier)
    - Target: price (USD)
    """
    rng = np.random.default_rng(42)
    os.makedirs(IMG_DIR, exist_ok=True)

    sqft          = rng.integers(600, 4500, n).astype(float)
    bedrooms      = rng.integers(1, 6, n).astype(float)
    bathrooms     = rng.uniform(1, 4, n).round(1)
    house_age     = rng.integers(0, 60, n).astype(float)
    has_garage    = rng.integers(0, 2, n).astype(float)
    location_score= rng.uniform(1, 10, n).round(1)

    # Price with realistic correlations + noise
    price = (
        sqft * 110
        + bedrooms * 8000
        + bathrooms * 12000
        - house_age * 1500
        + has_garage * 20000
        + location_score * 15000
        + rng.normal(0, 25000, n)
    ).clip(50000, 2_000_000)

    df = pd.DataFrame({
        "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "house_age": house_age, "has_garage": has_garage,
        "location_score": location_score, "price": price,
        "img_path": [os.path.join(IMG_DIR, f"house_{i:04d}.jpg") for i in range(n)],
    })

    # Generate synthetic images (size proxy: colour = price tier)
    print(f"Generating {n} synthetic house images…")
    price_norm = (price - price.min()) / (price.ptp() + 1e-9)

    for i in range(n):
        t = price_norm[i]
        # Colour: blue=cheap, green=mid, red=expensive
        r = int(t * 200 + 30)
        g = int((1 - abs(t - 0.5) * 2) * 180 + 50)
        b = int((1 - t) * 200 + 30)
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (r, g, b))
        d   = ImageDraw.Draw(img)
        # Draw a simple house outline
        cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
        d.polygon([(cx-20, cy+15),(cx+20, cy+15),(cx+20, cy),(cx-20, cy)], fill=(200,200,200))
        d.polygon([(cx-22, cy),(cx, cy-20),(cx+22, cy)], fill=(150,80,80))
        # Add noise
        arr = np.array(img) + rng.integers(-15, 15, (IMG_SIZE, IMG_SIZE, 3), dtype=np.int16)
        Image.fromarray(arr.clip(0,255).astype(np.uint8)).save(df["img_path"].iloc[i])

    print(f"Dataset created: {df.shape}")
    print(df.describe())
    return df


# ─────────────────────────────────────────
#  2. Dataset class
# ─────────────────────────────────────────
class HousingDataset(Dataset):
    def __init__(self, df, tabular_cols, scaler=None, fit_scaler=False, transform=None):
        self.df           = df.reset_index(drop=True)
        self.tabular_cols = tabular_cols
        self.transform    = transform

        X_tab = df[tabular_cols].values.astype(np.float32)
        if fit_scaler:
            self.scaler = StandardScaler().fit(X_tab)
        else:
            self.scaler = scaler
        self.X_tab  = self.scaler.transform(X_tab).astype(np.float32)
        self.prices = np.log1p(df["price"].values.astype(np.float32))  # log transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df["img_path"].iloc[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab   = torch.tensor(self.X_tab[idx])
        price = torch.tensor(self.prices[idx])
        return img, tab, price


# ─────────────────────────────────────────
#  3. Model architecture
# ─────────────────────────────────────────
class CNNTabularFusion(nn.Module):
    """
    Multimodal fusion model:
      - CNN branch (MobileNetV2 backbone, frozen except last 2 layers) → 64 features
      - Tabular MLP branch → 32 features
      - Fusion: concat → dense → price
    """
    def __init__(self, tabular_dim: int):
        super().__init__()

        # ── CNN branch ────────────────────────
        backbone = models.mobilenet_v2(weights="DEFAULT")
        # Freeze all but last classifier
        for p in list(backbone.parameters())[:-20]:
            p.requires_grad = False
        # Replace classifier
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone.last_channel, CNN_FEAT),
            nn.ReLU(),
        )
        self.cnn = backbone

        # ── Tabular branch ────────────────────
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # ── Fusion head ───────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(CNN_FEAT + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)                     # (B, CNN_FEAT)
        tab_feat = self.tabular_net(tab)             # (B, 32)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.fusion(combined).squeeze(1)      # (B,)


# ─────────────────────────────────────────
#  4. Training & evaluation
# ─────────────────────────────────────────
def rmse(y_true, y_pred):
    return math.sqrt(mean_absolute_error(y_true**2, y_pred**2))


def train_model(model, train_loader, val_loader):
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_val = float("inf")

    print(f"\n{'─'*60}")
    print(f"Training Multimodal Fusion Model ({EPOCHS} epochs)")
    print(f"{'─'*60}")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────
        model.train()
        train_loss = 0
        for imgs, tabs, prices in train_loader:
            imgs, tabs, prices = imgs.to(DEVICE), tabs.to(DEVICE), prices.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs, tabs)
            loss  = criterion(preds, prices)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──────────────────────────
        model.eval()
        val_loss, all_preds, all_true = 0, [], []
        with torch.no_grad():
            for imgs, tabs, prices in val_loader:
                imgs, tabs, prices = imgs.to(DEVICE), tabs.to(DEVICE), prices.to(DEVICE)
                preds     = model(imgs, tabs)
                val_loss += criterion(preds, prices).item()
                all_preds.extend(np.expm1(preds.cpu().numpy()))
                all_true.extend(np.expm1(prices.cpu().numpy()))

        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)
        mae  = mean_absolute_error(all_true, all_preds)
        rmse_val = np.sqrt(np.mean((all_true - all_preds)**2))

        scheduler.step(val_loss)

        for k, v in zip(["train_loss","val_loss","val_mae","val_rmse"],
                        [train_loss/len(train_loader), val_loss/len(val_loader), mae, rmse_val]):
            history[k].append(v)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_multimodal.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss/len(train_loader):.4f}  "
                  f"val_loss={val_loss/len(val_loader):.4f}  "
                  f"MAE=${mae:,.0f}  RMSE=${rmse_val:,.0f}")

    return history, all_preds, all_true


# ─────────────────────────────────────────
#  5. Visualisations
# ─────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = range(1, EPOCHS + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Huber Loss"); axes[0].legend()

    axes[1].plot(epochs, [m/1000 for m in history["val_mae"]], color="orange")
    axes[1].set_title("Validation MAE ($K)")

    axes[2].plot(epochs, [r/1000 for r in history["val_rmse"]], color="red")
    axes[2].set_title("Validation RMSE ($K)")

    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    plt.suptitle("Multimodal Housing Price Prediction – Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true/1000, y_pred/1000, alpha=0.4, color="#3498db", s=15)
    lim = max(y_true.max(), y_pred.max()) / 1000
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=2, label="Perfect")
    axes[0].set_xlabel("Actual Price ($K)"); axes[0].set_ylabel("Predicted Price ($K)")
    axes[0].set_title("Predicted vs Actual"); axes[0].legend()

    residuals = y_pred - y_true
    axes[1].hist(residuals/1000, bins=40, color="#e74c3c", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_title("Residual Distribution ($K)")
    axes[1].set_xlabel("Residual ($K)")

    plt.suptitle("Multimodal Model – Prediction Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("prediction_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: prediction_analysis.png")


# ─────────────────────────────────────────
#  6. Baseline (tabular-only)
# ─────────────────────────────────────────
def tabular_baseline(X_train, X_test, y_train, y_test, scaler):
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    gb.fit(scaler.transform(X_train), np.log1p(y_train))
    preds = np.expm1(gb.predict(scaler.transform(X_test)))
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(np.mean((y_test - preds)**2))
    print(f"\nTabular-only baseline (Gradient Boosting):")
    print(f"  MAE : ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    return mae, rmse


# ─────────────────────────────────────────
#  7. Main
# ─────────────────────────────────────────
def main():
    df = generate_synthetic_dataset(N_SAMPLES)

    TABULAR_COLS = ["sqft","bedrooms","bathrooms","house_age","has_garage","location_score"]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = HousingDataset(train_df, TABULAR_COLS, fit_scaler=True,  transform=transform)
    test_ds  = HousingDataset(test_df,  TABULAR_COLS, scaler=train_ds.scaler, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Tabular baseline
    tabular_baseline(
        train_df[TABULAR_COLS].values, test_df[TABULAR_COLS].values,
        train_df["price"].values, test_df["price"].values,
        train_ds.scaler,
    )

    # Multimodal model
    model = CNNTabularFusion(tabular_dim=len(TABULAR_COLS)).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}")

    history, y_pred, y_true = train_model(model, train_loader, test_loader)

    y_pred = np.array(y_pred); y_true = np.array(y_true)
    final_mae  = mean_absolute_error(y_true, y_pred)
    final_rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    print(f"\n{'═'*60}")
    print("FINAL RESULTS – Multimodal Model")
    print(f"  MAE : ${final_mae:,.0f}")
    print(f"  RMSE: ${final_rmse:,.0f}")
    print("═"*60)

    plot_training(history)
    plot_predictions(y_true, y_pred)


if __name__ == "__main__":
    main()
