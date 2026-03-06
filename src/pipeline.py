from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .utils import extract_hog_features
except ImportError:
    from utils import extract_hog_features

DEFAULT_POSES = ["warrior", "downdog", "goddess", "plank", "tree"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_grayscale_images(
    folder_path: Path,
    max_images: int | None = None,
    random_state: int = 42,
) -> list[np.ndarray]:
    if not folder_path.exists():
        return []

    files = [
        file_path
        for file_path in folder_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
    ]
    files.sort()

    if max_images is not None and len(files) > max_images:
        rng = np.random.default_rng(random_state)
        chosen_indices = rng.choice(len(files), size=max_images, replace=False)
        files = [files[idx] for idx in sorted(chosen_indices)]

    images: list[np.ndarray] = []
    for file_path in files:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (128, 64))
        images.append(image)

    return images


def _build_dataset(
    dataset_path: Path,
    poses: list[str],
    max_images_per_class: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    x_parts: list[np.ndarray] = []
    y_labels: list[str] = []
    valid_poses: list[str] = []
    class_counts: dict[str, int] = {}

    for pose_name in poses:
        pose_folder = dataset_path / pose_name
        images = _load_grayscale_images(
            pose_folder,
            max_images=max_images_per_class,
            random_state=random_state,
        )
        if not images:
            continue

        features, _ = extract_hog_features(images)
        x_parts.append(features)
        y_labels.extend([pose_name] * len(features))
        valid_poses.append(pose_name)
        class_counts[pose_name] = len(features)

    if not x_parts:
        raise ValueError(f"No images found in {dataset_path}")

    x = np.vstack(x_parts)
    y = np.array(y_labels)
    return x, y, valid_poses, class_counts


def _compute_learning_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    random_state: int = 42,
) -> pd.DataFrame:
    def _sanitize_proba(proba: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """Ensure probabilities are finite, clipped, and row-normalized."""
        n_classes = proba.shape[1]
        cleaned = np.nan_to_num(
            proba,
            nan=1.0 / n_classes,
            posinf=1.0,
            neginf=0.0,
        )
        cleaned = np.clip(cleaned, epsilon, 1.0 - epsilon)

        row_sums = cleaned.sum(axis=1, keepdims=True)
        invalid_rows = (~np.isfinite(row_sums)) | (row_sums <= 0.0)
        if np.any(invalid_rows):
            cleaned[invalid_rows.ravel()] = 1.0 / n_classes
            row_sums = cleaned.sum(axis=1, keepdims=True)

        cleaned /= row_sums
        return cleaned

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train_scaled = np.nan_to_num(x_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_scaled = np.nan_to_num(x_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    classes = np.unique(y_train)
    clf = SGDClassifier(
        loss="log_loss",
        random_state=random_state,
        max_iter=1,
        tol=None,
        learning_rate="adaptive",
        eta0=0.01,
        average=True,
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            clf.partial_fit(x_train_scaled, y_train, classes=classes)
        else:
            clf.partial_fit(x_train_scaled, y_train)

        train_proba = clf.predict_proba(x_train_scaled)
        test_proba = clf.predict_proba(x_test_scaled)
        train_proba = _sanitize_proba(train_proba)
        test_proba = _sanitize_proba(test_proba)

        train_pred = clf.predict(x_train_scaled)
        test_pred = clf.predict(x_test_scaled)

        history.append(
            {
                "epoch": epoch,
                "train_accuracy": accuracy_score(y_train, train_pred),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "train_loss": log_loss(y_train, train_proba, labels=classes),
                "test_loss": log_loss(y_test, test_proba, labels=classes),
            }
        )

    return pd.DataFrame(history)


def train_experiment(
    dataset_path: Path,
    poses: list[str],
    experiment_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_images_per_class: int | None = None,
    epochs: int = 20,
) -> dict[str, Any]:
    x, y, valid_poses, class_counts = _build_dataset(
        dataset_path=dataset_path,
        poses=poses,
        max_images_per_class=max_images_per_class,
        random_state=random_state,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0, probability=True, random_state=random_state),
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=valid_poses)

    report_dict = classification_report(
        y_test,
        predictions,
        labels=valid_poses,
        output_dict=True,
        zero_division=0,
    )
    class_metrics = (
        pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    )
    class_metrics = class_metrics[class_metrics["label"].isin(valid_poses)].copy()

    learning_curve = _compute_learning_curve(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        random_state=random_state,
    )

    return {
        "name": experiment_name,
        "dataset_path": str(dataset_path),
        "model": model,
        "accuracy": float(accuracy),
        "confusion_matrix": cm,
        "labels": valid_poses,
        "class_counts": class_counts,
        "class_metrics": class_metrics,
        "learning_curve": learning_curve,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }


def run_background_comparison(
    data_root: Path,
    poses: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_images_per_class: int | None = None,
    epochs: int = 20,
) -> dict[str, Any]:
    poses = poses or DEFAULT_POSES
    experiments = [
        ("Avec Fond (Original)", data_root / "raw"),
        ("Sans Fond (Traitee)", data_root / "raw_sans_fond"),
    ]

    by_name: dict[str, dict[str, Any]] = {}
    for exp_name, exp_path in experiments:
        if not exp_path.exists():
            continue

        by_name[exp_name] = train_experiment(
            dataset_path=exp_path,
            poses=poses,
            experiment_name=exp_name,
            test_size=test_size,
            random_state=random_state,
            max_images_per_class=max_images_per_class,
            epochs=epochs,
        )

    rows = []
    for exp_name, payload in by_name.items():
        rows.append(
            {
                "Experience": exp_name,
                "Accuracy (%)": payload["accuracy"] * 100.0,
                "Train Samples": payload["train_size"],
                "Test Samples": payload["test_size"],
            }
        )

    summary_df = pd.DataFrame(rows)
    improvement = None
    if "Avec Fond (Original)" in by_name and "Sans Fond (Traitee)" in by_name:
        improvement = (
            by_name["Sans Fond (Traitee)"]["accuracy"]
            - by_name["Avec Fond (Original)"]["accuracy"]
        ) * 100.0

    return {
        "experiments": by_name,
        "summary": summary_df,
        "improvement": improvement,
    }


def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    np_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image file cannot be decoded.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 64))
    return gray


def predict_pose(model: Any, file_bytes: bytes) -> dict[str, Any]:
    gray = preprocess_uploaded_image(file_bytes)
    features, _ = extract_hog_features([gray])

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[0]
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features))
        labels = getattr(model, "classes_", None)

        if decision.ndim == 1:
            margin = float(decision[0])
            pos_prob = 1.0 / (1.0 + np.exp(-margin))
            scores = np.array([1.0 - pos_prob, pos_prob], dtype=float)
        else:
            raw_scores = np.asarray(decision[0], dtype=float).ravel()
            if raw_scores.size == 1 and labels is not None and len(labels) == 2:
                margin = float(raw_scores[0])
                pos_prob = 1.0 / (1.0 + np.exp(-margin))
                scores = np.array([1.0 - pos_prob, pos_prob], dtype=float)
            else:
                shifted = raw_scores - np.max(raw_scores)
                exp_scores = np.exp(shifted)
                scores = exp_scores / np.sum(exp_scores)
    else:
        pred = model.predict(features)[0]
        return {
            "prediction": str(pred),
            "confidence": 1.0,
            "ranking": [{"label": str(pred), "score": 1.0}],
        }

    labels = getattr(model, "classes_", np.arange(len(scores)))
    ranked = sorted(
        [{"label": label, "score": float(score)} for label, score in zip(labels, scores)],
        key=lambda item: item["score"],
        reverse=True,
    )
    best = ranked[0]

    return {
        "prediction": best["label"],
        "confidence": best["score"],
        "ranking": ranked,
    }


def save_models_bundle(outputs: dict[str, Any], bundle_path: Path) -> None:
    experiments = outputs.get("experiments", {})
    payload: dict[str, Any] = {}

    for exp_name, exp_data in experiments.items():
        model = exp_data.get("model")
        if model is None:
            continue
        payload[exp_name] = {
            "model": model,
            "labels": exp_data.get("labels", []),
        }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)


def load_models_bundle(bundle_path: Path) -> dict[str, Any]:
    if not bundle_path.exists():
        return {}

    try:
        with bundle_path.open("rb") as file_obj:
            data = pickle.load(file_obj)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}
    return data
