from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from .pipeline import (
        DEFAULT_CLASSIFIERS,
        DEFAULT_POSES,
        classifier_label,
        load_models_bundle,
        predict_pose,
        run_background_comparison,
        save_models_bundle,
    )
except ImportError:
    from pipeline import (
        DEFAULT_CLASSIFIERS,
        DEFAULT_POSES,
        classifier_label,
        load_models_bundle,
        predict_pose,
        run_background_comparison,
        save_models_bundle,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "results" / "models" / "web_models.pkl"


def setup_page() -> None:
    st.set_page_config(
        page_title="Yoga AI Dashboard",
        page_icon="YA",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        .stApp {
            font-family: 'Manrope', sans-serif;
            background: radial-gradient(circle at top right, #eaf6ff 0%, #f8fbff 45%, #f4f7fb 100%);
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e6edf7;
        }
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 1.5rem;
        }
        .hero-card {
            border-radius: 18px;
            padding: 20px 24px;
            background: linear-gradient(120deg, #ffffff 0%, #f5faff 100%);
            border: 1px solid #e6eef9;
            box-shadow: 0 18px 30px rgba(40, 69, 111, 0.08);
            margin-bottom: 16px;
        }
        .metric-card {
            border-radius: 16px;
            padding: 14px 16px;
            background: #ffffff;
            border: 1px solid #e7edf7;
            box-shadow: 0 10px 20px rgba(53, 83, 121, 0.07);
            margin-bottom: 10px;
        }
        .metric-label {
            color: #5c6f8f;
            font-size: 0.86rem;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #111e37;
            font-size: 1.35rem;
            font-weight: 800;
        }
        .stButton > button {
            border: none;
            border-radius: 12px;
            background: linear-gradient(120deg, #0ea5e9, #2563eb);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def training_controls() -> tuple[bool, dict]:
    st.sidebar.markdown("## Yoga AI Control")
    st.sidebar.caption("Entrainement du modele et suivi des performances.")

    selected_poses = st.sidebar.multiselect(
        "Poses utilisees",
        options=DEFAULT_POSES,
        default=DEFAULT_POSES,
    )
    selected_classifiers = st.sidebar.multiselect(
        "Classifieurs",
        options=DEFAULT_CLASSIFIERS,
        default=DEFAULT_CLASSIFIERS,
        format_func=classifier_label,
    )
    max_images = st.sidebar.slider(
        "Max images par classe",
        min_value=100,
        max_value=1200,
        value=350,
        step=50,
    )
    epochs = st.sidebar.slider(
        "Epochs pour courbe accuracy/loss",
        min_value=5,
        max_value=60,
        value=20,
        step=5,
    )
    test_size = st.sidebar.slider(
        "Test split",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)
    run_now = st.sidebar.button("Lancer entrainement")

    params = {
        "poses": selected_poses,
        "classifiers": selected_classifiers,
        "max_images_per_class": max_images,
        "epochs": epochs,
        "test_size": test_size,
        "random_state": int(seed),
    }
    return run_now, params


def launch_training(params: dict) -> None:
    if not params["poses"]:
        st.warning("Selectionne au moins une pose.")
        return
    if not params["classifiers"]:
        st.warning("Selectionne au moins un classifieur.")
        return

    try:
        with st.spinner("Entrainement des experiences en cours..."):
            outputs = run_background_comparison(
                data_root=DATA_ROOT,
                poses=params["poses"],
                classifiers=params["classifiers"],
                test_size=params["test_size"],
                random_state=params["random_state"],
                max_images_per_class=params["max_images_per_class"],
                epochs=params["epochs"],
            )
        st.session_state["training_outputs"] = outputs
        save_models_bundle(outputs, MODEL_BUNDLE_PATH)
        st.success("Entrainement termine.")
    except Exception as exc:
        st.error(f"Echec de l'entrainement: {exc}")


def collect_available_models(outputs: dict) -> dict[str, dict]:
    available: dict[str, dict] = {}

    experiments = outputs.get("experiments", {})
    for exp_name, payload in experiments.items():
        model = payload.get("model")
        if model is None:
            continue
        available[f"{exp_name} (current session)"] = {"model": model, "source": "session"}

    saved = load_models_bundle(MODEL_BUNDLE_PATH)
    for exp_name, payload in saved.items():
        model = payload.get("model") if isinstance(payload, dict) else None
        if model is None:
            continue
        label = f"{exp_name} (saved)"
        if label not in available:
            available[label] = {"model": model, "source": "saved"}

    return available


def build_learning_curves_figure(curve: pd.DataFrame, exp_name: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("#efefef")

    x_values = curve["train_samples"] if "train_samples" in curve else curve["epoch"]
    train_acc = curve["train_accuracy"]
    valid_acc = curve["test_accuracy"]
    train_loss = curve["train_loss"]
    valid_loss = curve["test_loss"]

    for ax in axes:
        ax.set_facecolor("#f7f7f7")
        ax.grid(alpha=0.24)

    acc_min = min(train_acc.min(), valid_acc.min())
    acc_lower = max(0.0, acc_min - 0.05)

    loss_min = min(train_loss.min(), valid_loss.min())
    loss_max = max(train_loss.max(), valid_loss.max())
    loss_pad = max(0.02, (loss_max - loss_min) * 0.12)

    axes[0].plot(
        x_values,
        train_acc,
        color="#1f77b4",
        linewidth=1.7,
        marker="o",
        label="Training Accuracy",
    )
    axes[0].plot(
        x_values,
        valid_acc,
        color="#ff7f0e",
        linewidth=1.7,
        marker="o",
        label="Validation Accuracy",
    )
    axes[0].set_title("Accuracy over Training Samples")
    axes[0].set_xlabel("Training Samples")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(acc_lower, 1.01)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        x_values,
        train_loss,
        color="#1f77b4",
        linewidth=1.7,
        marker="o",
        label="Training Loss",
    )
    axes[1].plot(
        x_values,
        valid_loss,
        color="#ff7f0e",
        linewidth=1.7,
        marker="o",
        label="Validation Loss",
    )
    axes[1].set_title("Loss over Training Samples")
    axes[1].set_xlabel("Training Samples")
    axes[1].set_ylabel("Loss")
    axes[1].set_ylim(loss_min - loss_pad, loss_max + loss_pad)
    axes[1].legend(loc="upper right")

    fig.suptitle(f"Learning Curves - {exp_name}")
    fig.tight_layout()
    return fig


def show_dashboard(outputs: dict) -> None:
    summary = outputs.get("summary", pd.DataFrame())
    improvement = outputs.get("improvement")
    improvement_by_classifier = outputs.get("improvement_by_classifier", {})

    st.markdown(
        """
        <div class="hero-card">
            <h2 style="margin-bottom:4px;color:#13233f;">Yoga AI Pose Detection Dashboard</h2>
            <p style="margin:0;color:#5f6f8a;">
            Interface web intuitive pour comparer l'impact du fond et monitorer les performances IA.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top1, top2, top3 = st.columns(3)
    with top1:
        metric_card("Configurations entrainees", str(len(summary)))
    with top2:
        best = f"{summary['Accuracy (%)'].max():.2f}%" if not summary.empty else "-"
        metric_card("Meilleure precision", best)
    with top3:
        if improvement is None:
            metric_card("Gain moyen sans fond", "N/A")
        else:
            metric_card("Gain moyen sans fond", f"{improvement:+.2f}%")

    if summary.empty:
        st.info("Lance l'entrainement depuis la barre laterale pour afficher les resultats.")
        return

    fig = px.bar(
        summary,
        x="Experience",
        y="Accuracy (%)",
        color="Classifier",
        barmode="group",
        text=summary["Accuracy (%)"].map(lambda v: f"{v:.2f}%"),
        hover_data=["Model", "Train Samples", "Test Samples"],
        title="Comparaison de precision par experience et classifieur",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    if improvement_by_classifier:
        gain_df = pd.DataFrame(
            [
                {
                    "Classifier": classifier,
                    "Gain sans fond (%)": gain,
                }
                for classifier, gain in improvement_by_classifier.items()
            ]
        )
        st.markdown("### Gain par classifieur")
        st.dataframe(gain_df.round(3), use_container_width=True)

    st.dataframe(summary.round(3), use_container_width=True)


def show_pose_detection(outputs: dict) -> None:
    models_catalog = collect_available_models(outputs)
    if not models_catalog:
        st.info("Entraine d'abord le modele pour activer la prediction d'image.")
        return

    st.markdown("### Donner une image et predire la pose")
    st.caption("Charge une photo de posture yoga, puis clique sur le bouton de prediction.")

    model_names = list(models_catalog.keys())
    preferred = next(
        (
            idx
            for idx, name in enumerate(model_names)
            if "Sans Fond (Traitee)" in name
        ),
        0,
    )
    model_name = st.selectbox("Modele a utiliser", options=model_names, index=preferred)
    model = models_catalog[model_name]["model"]
    st.caption(f"Source modele: {models_catalog[model_name]['source']}")

    uploaded = st.file_uploader(
        "Charge une image de posture (jpg/png/jpeg)",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    if uploaded is None:
        return

    predict_now = st.button("Predire la pose", type="primary")
    if not predict_now:
        return

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(uploaded, caption="Image chargee", use_container_width=True)
    with col2:
        try:
            prediction = predict_pose(model, uploaded.getvalue())
            st.markdown("### Resultat de detection")
            st.write(f"Pose predite: **{prediction['prediction']}**")
            st.write(f"Confiance: **{prediction['confidence'] * 100:.2f}%**")

            rank_df = pd.DataFrame(prediction["ranking"])
            rank_df["score_pct"] = rank_df["score"] * 100
            top3 = rank_df.head(3)[["label", "score_pct"]].copy()

            st.markdown("Top 3 predictions")
            st.dataframe(
                top3.rename(columns={"label": "Pose", "score_pct": "Confiance (%)"}).round(2),
                use_container_width=True,
            )

            confidence_fig = px.bar(
                rank_df,
                x="label",
                y="score_pct",
                color="score_pct",
                color_continuous_scale="Blues",
                title="Niveau de confiance par pose",
            )
            confidence_fig.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(confidence_fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Impossible de predire sur cette image: {exc}")


def show_model_insights(outputs: dict) -> None:
    experiments = outputs.get("experiments", {})
    if not experiments:
        st.info("Entraine d'abord le modele.")
        return

    exp_name = st.selectbox("Experience", options=list(experiments.keys()), key="insight_exp")
    payload = experiments[exp_name]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Accuracy vs Loss", "Confusion Matrix", "Class Metrics", "Dataset Distribution"]
    )

    with tab1:
        curve = payload["learning_curve"]
        st.caption(
            "Evolution de l'accuracy et de la loss en fonction du volume d'entrainement."
        )
        fig = build_learning_curves_figure(curve, exp_name)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(curve, use_container_width=True)

    with tab2:
        cm = payload["confusion_matrix"]
        labels = payload["labels"]
        cm_fig = px.imshow(
            cm,
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            text_auto=True,
            title=f"Matrice de confusion - {exp_name}",
            labels={"x": "Prediction", "y": "Vrai label", "color": "Count"},
        )
        cm_fig.update_layout(height=480)
        st.plotly_chart(cm_fig, use_container_width=True)

    with tab3:
        metrics_df = payload["class_metrics"][["label", "precision", "recall", "f1-score"]].copy()
        melted = metrics_df.melt(id_vars="label", var_name="metric", value_name="score")
        bars = px.bar(
            melted,
            x="label",
            y="score",
            color="metric",
            barmode="group",
            title=f"Precision / Recall / F1 - {exp_name}",
        )
        bars.update_layout(height=420)
        st.plotly_chart(bars, use_container_width=True)
        st.dataframe(metrics_df.round(4), use_container_width=True)

    with tab4:
        counts = pd.DataFrame(
            [{"pose": pose, "count": count} for pose, count in payload["class_counts"].items()]
        )
        dist = px.pie(
            counts,
            names="pose",
            values="count",
            hole=0.45,
            title=f"Distribution des classes - {exp_name}",
        )
        dist.update_layout(height=420)
        st.plotly_chart(dist, use_container_width=True)
        st.dataframe(counts, use_container_width=True)


def main() -> None:
    setup_page()

    run_now, params = training_controls()
    if run_now:
        launch_training(params)

    page = st.sidebar.radio(
        "Navigation",
        options=["Dashboard", "Pose Detection", "Model Insights"],
    )

    outputs = st.session_state.get("training_outputs", {})
    if page == "Dashboard":
        show_dashboard(outputs)
    elif page == "Pose Detection":
        show_pose_detection(outputs)
    else:
        show_model_insights(outputs)


if __name__ == "__main__":
    main()
