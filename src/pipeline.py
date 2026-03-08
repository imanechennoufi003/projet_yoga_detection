from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import cv2
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .utils import extract_hog_features
except ImportError:
    from utils import extract_hog_features

DEFAULT_POSES = ["warrior", "downdog", "goddess", "plank", "tree"]
DEFAULT_CLASSIFIERS = ["svm", "knn"]
CLASSIFIER_LABELS = {
    "svm": "SVM",
    "knn": "KNN",
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_classifier_name(classifier_name: str) -> str:
    normalized = classifier_name.strip().lower()
    if normalized not in CLASSIFIER_LABELS:
        available = ", ".join(CLASSIFIER_LABELS.values())
        raise ValueError(f"Unsupported classifier '{classifier_name}'. Available: {available}.")
    return normalized


def classifier_label(classifier_name: str) -> str:
    return CLASSIFIER_LABELS[normalize_classifier_name(classifier_name)]


def _build_model(
    classifier_name: str,
    random_state: int = 42,
    knn_neighbors: int = 5,
) -> Any:
    normalized = normalize_classifier_name(classifier_name)

    if normalized == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0, probability=True, random_state=random_state),
        )

    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=knn_neighbors, weights="distance"),
    )


def _sanitize_proba(proba: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
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


def _align_probabilities(
    probabilities: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> np.ndarray:
    aligned = np.full(
        (probabilities.shape[0], len(target_labels)),
        1e-12,
        dtype=float,
    )
    source_index = {label: idx for idx, label in enumerate(source_labels)}

    for target_idx, label in enumerate(target_labels):
        source_idx = source_index.get(label)
        if source_idx is not None:
            aligned[:, target_idx] = probabilities[:, source_idx]

    return _sanitize_proba(aligned)


def _predict_scores(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model_labels = np.asarray(getattr(model, "classes_", []))

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if model_labels.size == 0:
            model_labels = np.arange(probabilities.shape[1])
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features))
        if decision.ndim == 1:
            margin = decision.ravel()
            pos_prob = 1.0 / (1.0 + np.exp(-margin))
            probabilities = np.column_stack([1.0 - pos_prob, pos_prob])
        else:
            shifted = decision - decision.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        if model_labels.size == 0:
            model_labels = np.arange(probabilities.shape[1])
    else:
        predictions = np.asarray(model.predict(features))
        if model_labels.size == 0:
            model_labels = np.unique(predictions)

        probabilities = np.zeros((len(predictions), len(model_labels)), dtype=float)
        for idx, label in enumerate(model_labels):
            probabilities[:, idx] = (predictions == label).astype(float)

    target_labels = model_labels if labels is None else np.asarray(labels)
    aligned = _align_probabilities(probabilities, model_labels, target_labels)
    return target_labels, aligned


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


def _build_subset_sizes(
    total_samples: int,
    n_classes: int,
    steps: int,
    min_samples: int = 1,
) -> list[int]:
    lower_bound = max(2 * n_classes, 4, min_samples)
    if total_samples <= lower_bound:
        return [total_samples]

    start_size = max(lower_bound, int(np.ceil(total_samples * 0.25)))
    candidates = np.linspace(start_size, total_samples, num=steps, dtype=int)

    subset_sizes = {
        int(size)
        for size in candidates
        if size >= lower_bound and (size == total_samples or total_samples - size >= n_classes)
    }
    subset_sizes.add(total_samples)
    return sorted(subset_sizes)


def _compute_learning_curve(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    steps: int,
    random_state: int = 42,
) -> pd.DataFrame:
    classes = np.unique(y_train)
    min_fit_samples = 1
    if hasattr(model, "named_steps") and "kneighborsclassifier" in model.named_steps:
        min_fit_samples = int(model.named_steps["kneighborsclassifier"].n_neighbors)

    subset_sizes = _build_subset_sizes(
        len(x_train),
        len(classes),
        steps,
        min_samples=min_fit_samples,
    )
    history: list[dict[str, Any]] = []

    for step_idx, subset_size in enumerate(subset_sizes, start=1):
        if subset_size >= len(x_train):
            x_subset = x_train
            y_subset = y_train
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=subset_size,
                random_state=random_state + step_idx,
            )
            subset_indices, _ = next(splitter.split(x_train, y_train))
            x_subset = x_train[subset_indices]
            y_subset = y_train[subset_indices]

        current_model = clone(model)
        current_model.fit(x_subset, y_subset)

        _, train_proba = _predict_scores(current_model, x_subset, labels=classes)
        _, test_proba = _predict_scores(current_model, x_test, labels=classes)
        train_pred = current_model.predict(x_subset)
        test_pred = current_model.predict(x_test)

        history.append(
            {
                "epoch": step_idx,
                "train_samples": int(len(x_subset)),
                "train_accuracy": accuracy_score(y_subset, train_pred),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "train_loss": log_loss(y_subset, train_proba, labels=classes),
                "test_loss": log_loss(y_test, test_proba, labels=classes),
            }
        )

    return pd.DataFrame(history)


def train_experiment(
    dataset_path: Path,
    poses: list[str],
    experiment_name: str,
    classifier_name: str = "svm",
    test_size: float = 0.2,
    random_state: int = 42,
    max_images_per_class: int | None = None,
    epochs: int = 20,
    knn_neighbors: int = 5,
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

    normalized_classifier = normalize_classifier_name(classifier_name)
    effective_knn_neighbors = max(1, min(knn_neighbors, len(x_train)))
    model = _build_model(
        normalized_classifier,
        random_state=random_state,
        knn_neighbors=effective_knn_neighbors,
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
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )
    class_metrics = class_metrics[class_metrics["label"].isin(valid_poses)].copy()

    learning_curve = _compute_learning_curve(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        steps=epochs,
        random_state=random_state,
    )

    model_name = f"{experiment_name} - {classifier_label(normalized_classifier)}"
    return {
        "name": model_name,
        "experiment": experiment_name,
        "classifier": classifier_label(normalized_classifier),
        "classifier_key": normalized_classifier,
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
    classifiers: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_images_per_class: int | None = None,
    epochs: int = 20,
    knn_neighbors: int = 5,
) -> dict[str, Any]:
    poses = poses or DEFAULT_POSES
    classifiers = classifiers or DEFAULT_CLASSIFIERS
    normalized_classifiers = [normalize_classifier_name(name) for name in classifiers]

    experiments = [
        ("Avec Fond (Original)", data_root / "raw"),
        ("Sans Fond (Traitee)", data_root / "raw_sans_fond"),
    ]

    by_name: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for exp_name, exp_path in experiments:
        if not exp_path.exists():
            continue

        for classifier_name in normalized_classifiers:
            payload = train_experiment(
                dataset_path=exp_path,
                poses=poses,
                experiment_name=exp_name,
                classifier_name=classifier_name,
                test_size=test_size,
                random_state=random_state,
                max_images_per_class=max_images_per_class,
                epochs=epochs,
                knn_neighbors=knn_neighbors,
            )
            by_name[payload["name"]] = payload
            rows.append(
                {
                    "Model": payload["name"],
                    "Experience": payload["experiment"],
                    "Classifier": payload["classifier"],
                    "Accuracy (%)": payload["accuracy"] * 100.0,
                    "Train Samples": payload["train_size"],
                    "Test Samples": payload["test_size"],
                }
            )

    summary_df = pd.DataFrame(rows)

    improvement_by_classifier: dict[str, float] = {}
    if not summary_df.empty:
        for classifier in summary_df["Classifier"].unique():
            subset = summary_df[summary_df["Classifier"] == classifier]
            if subset["Experience"].nunique() < 2:
                continue

            with_background = subset.loc[
                subset["Experience"] == "Avec Fond (Original)",
                "Accuracy (%)",
            ]
            without_background = subset.loc[
                subset["Experience"] == "Sans Fond (Traitee)",
                "Accuracy (%)",
            ]
            if with_background.empty or without_background.empty:
                continue

            improvement_by_classifier[classifier] = float(
                without_background.iloc[0] - with_background.iloc[0]
            )

    improvement = None
    if improvement_by_classifier:
        improvement = float(np.mean(list(improvement_by_classifier.values())))

    return {
        "experiments": by_name,
        "summary": summary_df,
        "improvement": improvement,
        "improvement_by_classifier": improvement_by_classifier,
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
    labels, scores = _predict_scores(model, features)
    ranked = sorted(
        [{"label": label, "score": float(score)} for label, score in zip(labels, scores[0])],
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
            "classifier": exp_data.get("classifier"),
            "experiment": exp_data.get("experiment"),
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
