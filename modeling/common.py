from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight


plt.rcParams["figure.figsize"] = (12, 6)
sns.set_style("whitegrid")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def select_best_model(cv_results_macro):
    cv_mean_macro = {name: scores.mean() for name, scores in cv_results_macro.items()}
    best_model_name = max(cv_mean_macro, key=cv_mean_macro.get)
    return best_model_name, cv_mean_macro[best_model_name]


def build_cv_summary(cv_results_macro, cv_results_weighted):
    rows = []
    for name in cv_results_macro:
        rows.append(
            {
                "Model": name,
                "CV Macro F1": f"%{cv_results_macro[name].mean() * 100:.2f}",
                "CV Weighted F1": f"%{cv_results_weighted[name].mean() * 100:.2f}",
            }
        )
    return pd.DataFrame(rows)


def plot_multiclass_roc_pr(y_true, y_proba, target_names, output_path, title_prefix):
    classes = np.arange(len(target_names))
    y_bin = label_binarize(y_true, classes=classes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for class_idx, class_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_proba[:, class_idx])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(
            y_bin[:, class_idx], y_proba[:, class_idx]
        )
        avg_precision = average_precision_score(
            y_bin[:, class_idx], y_proba[:, class_idx]
        )

        axes[0].plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={roc_auc:.3f})")
        axes[1].plot(
            recall,
            precision,
            linewidth=2,
            label=f"{class_name} (AP={avg_precision:.3f})",
        )

    micro_fpr, micro_tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    micro_auc = auc(micro_fpr, micro_tpr)
    axes[0].plot(
        micro_fpr,
        micro_tpr,
        linestyle="--",
        color="black",
        linewidth=2,
        label=f"Micro-average (AUC={micro_auc:.3f})",
    )
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="gray")
    axes[0].set_title(f"ROC Curves — {title_prefix}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=9)

    micro_precision, micro_recall, _ = precision_recall_curve(
        y_bin.ravel(), y_proba.ravel()
    )
    micro_ap = average_precision_score(y_bin, y_proba, average="micro")
    axes[1].plot(
        micro_recall,
        micro_precision,
        linestyle="--",
        color="black",
        linewidth=2,
        label=f"Micro-average (AP={micro_ap:.3f})",
    )
    axes[1].set_title(
        f"Precision-Recall Curves — {title_prefix}", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_learning_curve_custom(
    best_pipeline,
    X_data,
    y_data,
    output_path,
    title,
    target_names,
    use_sample_weight=False,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_fracs = np.linspace(0.2, 1.0, 5)

    train_means, train_stds = [], []
    val_means, val_stds = [], []
    sample_counts = []

    for frac_idx, frac in enumerate(train_fracs):
        fold_train_scores = []
        fold_val_scores = []
        fold_sizes = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
            X_fold_train_full = X_data.iloc[train_idx]
            y_fold_train_full = y_data.iloc[train_idx]
            X_fold_val = X_data.iloc[val_idx]
            y_fold_val = y_data.iloc[val_idx]

            if frac < 1.0:
                subset_size = max(
                    len(target_names) * 2, int(len(X_fold_train_full) * frac)
                )
                X_fold_train, _, y_fold_train, _ = train_test_split(
                    X_fold_train_full,
                    y_fold_train_full,
                    train_size=subset_size,
                    stratify=y_fold_train_full,
                    random_state=42 + frac_idx + fold_idx,
                )
            else:
                X_fold_train = X_fold_train_full
                y_fold_train = y_fold_train_full

            fold_sizes.append(len(X_fold_train))
            pipe_clone = clone(best_pipeline)

            if use_sample_weight:
                fold_weights = compute_sample_weight("balanced", y_fold_train)
                pipe_clone.fit(X_fold_train, y_fold_train, model__sample_weight=fold_weights)
            else:
                pipe_clone.fit(X_fold_train, y_fold_train)

            y_train_pred = pipe_clone.predict(X_fold_train)
            y_val_pred = pipe_clone.predict(X_fold_val)

            fold_train_scores.append(
                f1_score(y_fold_train, y_train_pred, average="macro", zero_division=0)
            )
            fold_val_scores.append(
                f1_score(y_fold_val, y_val_pred, average="macro", zero_division=0)
            )

        sample_counts.append(int(np.mean(fold_sizes)))
        train_means.append(np.mean(fold_train_scores))
        train_stds.append(np.std(fold_train_scores))
        val_means.append(np.mean(fold_val_scores))
        val_stds.append(np.std(fold_val_scores))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        sample_counts,
        train_means,
        marker="o",
        linewidth=2,
        label="Train Macro F1",
        color="tab:blue",
    )
    ax.plot(
        sample_counts,
        val_means,
        marker="o",
        linewidth=2,
        label="Validation Macro F1",
        color="tab:orange",
    )
    ax.fill_between(
        sample_counts,
        np.array(train_means) - np.array(train_stds),
        np.array(train_means) + np.array(train_stds),
        alpha=0.15,
        color="tab:blue",
    )
    ax.fill_between(
        sample_counts,
        np.array(val_means) - np.array(val_stds),
        np.array(val_means) + np.array(val_stds),
        alpha=0.15,
        color="tab:orange",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Eğitim Örnek Sayısı")
    ax.set_ylabel("Macro F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_models_with_cv(optimized_models, X_train, y_train, cv):
    cv_results_macro = {}
    cv_results_weighted = {}

    for name, pipe in optimized_models.items():
        if name == "XGBoost":
            macro_scores = []
            weighted_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
                fold_weights = compute_sample_weight("balanced", y_fold_train)
                pipe_clone = clone(pipe)
                pipe_clone.fit(
                    X_fold_train,
                    y_fold_train,
                    model__sample_weight=fold_weights,
                )
                y_fold_pred = pipe_clone.predict(X_fold_val)
                macro_scores.append(
                    f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0)
                )
                weighted_scores.append(
                    f1_score(y_fold_val, y_fold_pred, average="weighted", zero_division=0)
                )
            scores_macro = np.array(macro_scores)
            scores_weighted = np.array(weighted_scores)
        else:
            scores_macro = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
            scores_weighted = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring="f1_weighted"
            )

        cv_results_macro[name] = scores_macro
        cv_results_weighted[name] = scores_weighted

    return cv_results_macro, cv_results_weighted


def save_cv_boxplot(cv_results_macro, output_path, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    cv_data = pd.DataFrame(cv_results_macro)
    cv_data.boxplot(ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("F1-Macro")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, target_names, output_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=target_names,
        yticklabels=target_names,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_importance(model, feature_names, output_path, title, figsize):
    if not hasattr(model, "feature_importances_"):
        return

    fi = pd.DataFrame(
        {"Özellik": feature_names, "Önem": model.feature_importances_}
    ).sort_values("Önem", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(fi["Özellik"], fi["Önem"], color="steelblue", edgecolor="black")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Önem")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
