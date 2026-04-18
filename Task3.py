# ============================================================
#   TASK 3: Heart Disease Prediction — Binary Classification
#   Dataset: Heart Disease UCI (via ucimlrepo — no CSV needed)
#   Models: Logistic Regression + Decision Tree
#   Tools: pandas, scikit-learn, matplotlib, seaborn
#
#   Install dependencies first:
#   pip install ucimlrepo scikit-learn seaborn matplotlib pandas
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ucimlrepo import fetch_ucirepo

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_curve,
                              roc_auc_score)

# ─────────────────────────────────────────
# STEP 1: LOAD DATASET (no CSV needed)
# ─────────────────────────────────────────

print("📥 Fetching Heart Disease dataset from UCI repository...")
heart = fetch_ucirepo(id=45)

X_raw = heart.data.features
y_raw = heart.data.targets

df = X_raw.copy()
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
              'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Target: original has values 0-4 → convert to binary (0=no disease, 1=disease)
df['target'] = (y_raw.values.ravel() > 0).astype(int)

print(f"✅ Dataset loaded! Shape: {df.shape}\n")

# ─────────────────────────────────────────
# STEP 2: INSPECT THE DATA
# ─────────────────────────────────────────

print("=" * 55)
print("   HEART DISEASE DATASET — EXPLORATION REPORT")
print("=" * 55)

print("\n📐 Shape          :", df.shape)
print("\n📋 Columns        :", df.columns.tolist())
print("\n🔍 First 5 Rows:")
print(df.head().to_string())

print("\n📊 Dataset Info:")
df.info()

print("\n📈 Descriptive Statistics:")
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────
# STEP 3: CLEAN THE DATASET
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("   DATA CLEANING")
print("=" * 55)

print("\n🔎 Missing Values Per Column:")
print(df.isnull().sum().to_string())

# Fill missing numeric values with column median (robust to outliers)
filled_cols = []
for col in df.select_dtypes(include='number').columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        df[col].fillna(df[col].median(), inplace=True)
        filled_cols.append((col, missing))

if filled_cols:
    for col, n in filled_cols:
        print(f"  ✅ '{col}' — filled {n} missing value(s) with median")
else:
    print("  ✅ No missing values found!")

# Remove duplicates
dupes = df.duplicated().sum()
if dupes > 0:
    df.drop_duplicates(inplace=True)
    print(f"  ✅ Removed {dupes} duplicate row(s)")
else:
    print("  ✅ No duplicate rows found!")

# Convert 'ca' and 'thal' which can have stray '?' → already handled above
# Force all columns to numeric just in case
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

print(f"\n✅ Final clean shape: {df.shape}")

# ─────────────────────────────────────────
# COLUMN REFERENCE (for understanding)
# ─────────────────────────────────────────
# age       — Age in years
# sex       — 1=male, 0=female
# cp        — Chest pain type (0=typical angina … 3=asymptomatic)
# trestbps  — Resting blood pressure (mm Hg)
# chol      — Serum cholesterol (mg/dl)
# fbs       — Fasting blood sugar > 120 mg/dl (1=true, 0=false)
# restecg   — Resting ECG results (0, 1, 2)
# thalach   — Maximum heart rate achieved
# exang     — Exercise-induced angina (1=yes, 0=no)
# oldpeak   — ST depression induced by exercise relative to rest
# slope     — Slope of peak exercise ST segment
# ca        — Number of major vessels (0–3) colored by fluoroscopy
# thal      — Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
# target    — 1=heart disease, 0=no heart disease

TARGET       = 'target'
FEATURE_NAMES = [c for c in df.columns if c != TARGET]

# ─────────────────────────────────────────
# STEP 4: EDA — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────

# Global dark plot style
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444466',
    'axes.labelcolor':  '#e0e0ff',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'text.color':       '#e0e0ff',
    'grid.color':       '#2a2a4a',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})

C_POS = '#f72585'   # heart disease present
C_NEG = '#00f5d4'   # no heart disease

# ── EDA Plot 1: Class Balance + Age Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('EDA — Class Balance & Age Distribution',
             fontsize=14, fontweight='bold', color='#e0e0ff')

# Class bar chart
counts = df[TARGET].value_counts().sort_index()
bars = axes[0].bar(['No Disease (0)', 'Heart Disease (1)'],
                   counts.values, color=[C_NEG, C_POS],
                   edgecolor='white', linewidth=0.6, width=0.5)
axes[0].set_title('Target Class Distribution')
axes[0].set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f'{val}\n({pct:.1f}%)', ha='center',
                 color='white', fontsize=10)
axes[0].grid(True, axis='y')

# Age histogram by diagnosis
for label, color, name in [(0, C_NEG, 'No Disease'), (1, C_POS, 'Heart Disease')]:
    axes[1].hist(df[df[TARGET] == label]['age'], bins=20, alpha=0.65,
                 color=color, label=name, edgecolor='white', linewidth=0.3)
axes[1].set_title('Age Distribution by Diagnosis')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')
axes[1].legend(fontsize=9, framealpha=0.2)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── EDA Plot 2: Key Numeric Feature Distributions ──
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(1, len(num_features), figsize=(17, 4))
fig.suptitle('EDA — Numeric Feature Distributions by Diagnosis',
             fontsize=13, fontweight='bold', color='#e0e0ff')

for ax, feat in zip(axes, num_features):
    for label, color, name in [(0, C_NEG, 'No Disease'), (1, C_POS, 'Disease')]:
        ax.hist(df[df[TARGET] == label][feat], bins=18, alpha=0.65,
                color=color, label=name, edgecolor='white', linewidth=0.2)
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel('Value')
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(True)

plt.tight_layout()
plt.savefig('eda_histograms.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── EDA Plot 3: Box Plots (outlier detection) ────
fig, axes = plt.subplots(1, len(num_features), figsize=(17, 5))
fig.suptitle('EDA — Box Plots: Feature vs Target (Outlier View)',
             fontsize=13, fontweight='bold', color='#e0e0ff')

for ax, feat in zip(axes, num_features):
    groups = [df[df[TARGET] == 0][feat].values,
              df[df[TARGET] == 1][feat].values]
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#aaaacc'),
                    capprops=dict(color='#aaaacc'),
                    flierprops=dict(marker='o', markerfacecolor='red',
                                   markersize=4, alpha=0.6))
    for patch, color in zip(bp['boxes'], [C_NEG, C_POS]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(['No\nDisease', 'Heart\nDisease'], fontsize=8)
    ax.set_title(feat, fontsize=10)
    ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── EDA Plot 4: Correlation Heatmap ─────────────
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('#0f0f1a')
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle
sns.heatmap(corr, annot=True, fmt='.2f', cmap='magma',
            mask=mask, linewidths=0.4, linecolor='#0f0f1a',
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — All Features',
             fontsize=13, fontweight='bold', color='#e0e0ff', pad=15)
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── EDA Plot 5: Categorical Feature Counts ──────
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('EDA — Categorical Features vs Heart Disease',
             fontsize=13, fontweight='bold', color='#e0e0ff')

for ax, feat in zip(axes.flat, cat_features):
    counts_cat = df.groupby([feat, TARGET]).size().unstack(fill_value=0)
    counts_cat.plot(kind='bar', ax=ax, color=[C_NEG, C_POS],
                    edgecolor='white', linewidth=0.4, width=0.7)
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(['No Disease', 'Disease'], fontsize=7, framealpha=0.2)
    ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig('eda_categorical.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ─────────────────────────────────────────
# STEP 5: PREPARE FEATURES & TARGET
# ─────────────────────────────────────────

X = df[FEATURE_NAMES]
y = df[TARGET]

# Stratified split — keeps disease ratio equal in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📦 Train size : {X_train.shape}")
print(f"📦 Test size  : {X_test.shape}")
print(f"   Test class balance: {dict(y_test.value_counts())}")

# Scale features — required for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 6: TRAIN MODELS
# ─────────────────────────────────────────

# ── Model 1: Logistic Regression ─────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

# ── Model 2: Decision Tree ────────────────
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_probs = dt.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────
# STEP 7: EVALUATE MODELS
# ─────────────────────────────────────────

def evaluate_model(name, y_true, y_pred, y_prob, model, X_tr, y_tr):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cv  = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy')
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC Score    : {auc:.4f}")
    print(f"  5-Fold CV Acc    : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['No Disease', 'Heart Disease'],
                                digits=4))
    return acc, auc

print("\n" + "=" * 55)
print("   MODEL EVALUATION ON TEST SET")
print("=" * 55)

lr_acc, lr_auc = evaluate_model(
    "Logistic Regression", y_test, lr_preds, lr_probs,
    lr, X_train_scaled, y_train
)
dt_acc, dt_auc = evaluate_model(
    "Decision Tree", y_test, dt_preds, dt_probs,
    dt, X_train, y_train
)

# ─────────────────────────────────────────
# STEP 8: EVALUATION PLOTS DASHBOARD
# ─────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f0f1a')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
fig.suptitle('Heart Disease Model — Evaluation Dashboard',
             fontsize=16, fontweight='bold', color='#e0e0ff')

# ── Confusion Matrix: Logistic Regression ─
ax1 = fig.add_subplot(gs[0, 0])
cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            linewidths=1.5, linecolor='#0f0f1a', ax=ax1,
            annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_title(f'Confusion Matrix\nLogistic Regression  (Acc={lr_acc:.2%})',
              color='#c0c0ff', fontsize=10)
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# ── Confusion Matrix: Decision Tree ────────
ax2 = fig.add_subplot(gs[0, 1])
cm_dt = confusion_matrix(y_test, dt_preds)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Purples',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            linewidths=1.5, linecolor='#0f0f1a', ax=ax2,
            annot_kws={'size': 14, 'weight': 'bold'})
ax2.set_title(f'Confusion Matrix\nDecision Tree  (Acc={dt_acc:.2%})',
              color='#c0c0ff', fontsize=10)
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# ── ROC Curve Comparison ─────────────────
ax3 = fig.add_subplot(gs[0, 2])
for name, probs, color in [
    ('Logistic Regression', lr_probs, '#f72585'),
    ('Decision Tree',       dt_probs, '#00f5d4'),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax3.plot(fpr, tpr, color=color, linewidth=2.5,
             label=f'{name}  (AUC = {auc_val:.3f})')
    ax3.fill_between(fpr, tpr, alpha=0.05, color=color)

ax3.plot([0, 1], [0, 1], 'w--', linewidth=1, alpha=0.4, label='Random Guess')
ax3.set_title('ROC Curve Comparison', color='#c0c0ff')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate (Recall)')
ax3.legend(fontsize=8, framealpha=0.2)
ax3.grid(True)

# ── Feature Importance: LR Coefficients ──
ax4 = fig.add_subplot(gs[1, 0:2])
coefs = pd.Series(np.abs(lr.coef_[0]),
                  index=FEATURE_NAMES).sort_values(ascending=True)
bar_colors = ['#f72585' if v == coefs.max() else '#7209b7' for v in coefs]
coefs.plot(kind='barh', ax=ax4, color=bar_colors,
           edgecolor='white', linewidth=0.4)
ax4.set_title('Logistic Regression — Feature Importance (|Coefficient|)',
              color='#c0c0ff')
ax4.set_xlabel('Absolute Coefficient Value')
for i, val in enumerate(coefs):
    ax4.text(val + 0.005, i, f'{val:.3f}', va='center',
             color='white', fontsize=8)
ax4.grid(True, axis='x')

# ── Feature Importance: Decision Tree ────
ax5 = fig.add_subplot(gs[1, 2])
dt_imp = pd.Series(dt.feature_importances_,
                   index=FEATURE_NAMES).sort_values(ascending=True)
dt_colors = ['#ffd60a' if v == dt_imp.max() else '#7209b7' for v in dt_imp]
dt_imp.plot(kind='barh', ax=ax5, color=dt_colors,
            edgecolor='white', linewidth=0.4)
ax5.set_title('Decision Tree\nFeature Importance (Gini)',
              color='#c0c0ff')
ax5.set_xlabel('Importance Score')
for i, val in enumerate(dt_imp):
    ax5.text(val + 0.002, i, f'{val:.3f}', va='center',
             color='white', fontsize=8)
ax5.grid(True, axis='x')

plt.savefig('model_evaluation.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── Decision Tree Structure ───────────────
fig, ax = plt.subplots(figsize=(22, 9))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#1a1a2e')
plot_tree(dt,
          feature_names=FEATURE_NAMES,
          class_names=['No Disease', 'Disease'],
          filled=True, rounded=True, fontsize=7,
          impurity=False, ax=ax)
ax.set_title('Decision Tree Structure (max_depth = 5)',
             fontsize=13, fontweight='bold', color='#e0e0ff', pad=15)
plt.savefig('decision_tree_structure.png', dpi=120,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ─────────────────────────────────────────
# STEP 9: TOP FEATURES SUMMARY (console)
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("   TOP FEATURES AFFECTING PREDICTION")
print("=" * 55)

print("\n  📌 Logistic Regression — Top 5 (by |coefficient|):")
top_lr = pd.Series(np.abs(lr.coef_[0]),
                   index=FEATURE_NAMES).sort_values(ascending=False).head(5)
for feat, val in top_lr.items():
    bar = '█' * int(val * 12)
    print(f"    {feat:<12}  {bar}  {val:.4f}")

print("\n  🌲 Decision Tree — Top 5 (by Gini importance):")
top_dt = pd.Series(dt.feature_importances_,
                   index=FEATURE_NAMES).sort_values(ascending=False).head(5)
for feat, val in top_dt.items():
    bar = '█' * int(val * 40)
    print(f"    {feat:<12}  {bar}  {val:.4f}")

# ─────────────────────────────────────────
# STEP 10: PREDICT A SAMPLE PATIENT
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("   SAMPLE PATIENT PREDICTION")
print("=" * 55)

sample = pd.DataFrame([{
    'age': 58, 'sex': 1, 'cp': 0, 'trestbps': 145, 'chol': 270,
    'fbs': 0, 'restecg': 1, 'thalach': 142, 'exang': 1,
    'oldpeak': 2.8, 'slope': 1, 'ca': 1, 'thal': 3
}])[FEATURE_NAMES]

sample_scaled = scaler.transform(sample)

lr_result = lr.predict(sample_scaled)[0]
lr_conf   = lr.predict_proba(sample_scaled)[0][lr_result]
dt_result = dt.predict(sample)[0]
dt_conf   = dt.predict_proba(sample)[0][dt_result]

LABEL = {0: '✅ No Heart Disease', 1: '⚠️  Heart Disease Detected'}

print(f"\n  Patient Data: {sample.iloc[0].to_dict()}")
print(f"\n  Logistic Regression → {LABEL[lr_result]}  ({lr_conf:.1%} confidence)")
print(f"  Decision Tree       → {LABEL[dt_result]}  ({dt_conf:.1%} confidence)")

print("\n✅ All 6 plots saved!")
print("   ├── eda_overview.png")
print("   ├── eda_histograms.png")
print("   ├── eda_boxplots.png")
print("   ├── eda_correlation.png")
print("   ├── eda_categorical.png")
print("   ├── model_evaluation.png")
print("   └── decision_tree_structure.png")
