"""
Dashboard and visualization code for CrossCoder diffing method.

Separated to avoid triggering streamlit cache warnings at import time.
"""

from typing import Dict, Any
from collections import defaultdict
import json
import base64
import torch
import numpy as np
import pandas as pd
import streamlit as st

from diffing.utils.dashboards import (
    AbstractOnlineDiffingDashboard,
    SteeringDashboard,
    MaxActivationDashboardComponent,
)
from diffing.utils.max_act_store import ReadOnlyMaxActStore
from diffing.utils.visualization import multi_tab_interface
from diffing.utils.dictionary.steering import display_steering_results
from diffing.utils.dictionary.utils import load_dictionary_model, load_latent_df


def visualize(method) -> None:
    """
    Interactive Streamlit visualization for CrossCoder results.

    Args:
        method: CrosscoderDiffingMethod instance
    """
    import sys
    sys.stderr.write(f"[DEBUG MaxAct] visualize() called, results_dir={method.results_dir}\n")
    st.subheader("CrossCoder Analysis")

    available_ccs = _get_available_crosscoder_directories(method.results_dir)
    sys.stderr.write(f"[DEBUG MaxAct] available_ccs count: {len(available_ccs)}\n")
    for cc in available_ccs:
        sys.stderr.write(f"[DEBUG MaxAct]   cc: layer={cc['layer']}, name={cc['dictionary_name']}, path={cc['path']}\n")
    if not available_ccs:
        st.error(
            f"No trained CrossCoder directories found in {method.results_dir / 'crosscoder'}"
        )
        return

    cc_by_layer = defaultdict(list)
    for cc_info in available_ccs:
        cc_by_layer[cc_info["layer"]].append(cc_info)

    layers_sorted = sorted(cc_by_layer.keys())
    layer_key = "crosscoder_selected_layer"
    dict_key = "crosscoder_selected_dictionary"

    if layer_key not in st.session_state:
        st.session_state[layer_key] = layers_sorted[0]
    if dict_key not in st.session_state:
        st.session_state[dict_key] = cc_by_layer[layers_sorted[0]][0]["dictionary_name"]

    selected_layer = st.selectbox(
        "Select Layer",
        options=layers_sorted,
        index=(
            layers_sorted.index(st.session_state[layer_key])
            if st.session_state[layer_key] in layers_sorted
            else 0
        ),
        key=layer_key,
    )

    dicts_for_layer = [c["dictionary_name"] for c in cc_by_layer[selected_layer]]
    if st.session_state[dict_key] not in dicts_for_layer:
        st.session_state[dict_key] = dicts_for_layer[0]

    selected_dict = st.selectbox(
        "Select Trained CrossCoder",
        options=dicts_for_layer,
        index=(
            dicts_for_layer.index(st.session_state[dict_key])
            if st.session_state[dict_key] in dicts_for_layer
            else 0
        ),
        key=dict_key,
    )

    selected_cc_info = next(
        c for c in cc_by_layer[selected_layer] if c["dictionary_name"] == selected_dict
    )

    _display_training_metrics(selected_cc_info)

    sys.stderr.write(f"[DEBUG MaxAct] About to call multi_tab_interface with 6 tabs\n")
    multi_tab_interface(
        [
            (
                "📈 Latent Statistics",
                lambda: _render_latent_statistics_tab(method, selected_cc_info),
            ),
            (
                "📋 Steering Results",
                lambda: _render_steering_results_tab(method, selected_cc_info),
            ),
            (
                "🎯 Online Steering",
                lambda: CrosscoderSteeringDashboard(method, selected_cc_info).display(),
            ),
            (
                "🔥 Online Inference",
                lambda: CrosscoderOnlineDashboard(method, selected_cc_info).display(),
            ),
            ("🎨 Plots", lambda: _render_plots_tab(selected_cc_info)),
            (
                "📊 MaxAct Examples",
                lambda: _render_maxact_tab(method, selected_cc_info),
            ),
        ],
        "CrossCoder Analysis",
    )


def _get_available_crosscoder_directories(results_dir):
    """Return list of available trained crosscoder directories."""
    cc_base_dir = results_dir / "crosscoder"
    if not cc_base_dir.exists():
        return []

    available = []
    for layer_dir in cc_base_dir.iterdir():
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        try:
            layer_num = int(layer_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        for cc_dir in layer_dir.iterdir():
            if not cc_dir.is_dir():
                continue
            if (cc_dir / "dictionary_model").exists() or (
                cc_dir / "training_config.yaml"
            ).exists():
                available.append(
                    {
                        "layer": layer_num,
                        "dictionary_name": cc_dir.name,
                        "path": cc_dir,
                    }
                )
    available.sort(key=lambda x: (x["layer"], x["dictionary_name"]))
    return available


def _display_training_metrics(cc_info):
    """Display training metrics for selected CrossCoder."""
    training_metrics_path = cc_info["path"] / "training_metrics.json"
    if not training_metrics_path.exists():
        st.info("Training metrics not found")
        return

    try:
        with open(training_metrics_path, "r") as f:
            metrics = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Top-K", metrics.get("k"))
        with col2:
            dict_size = metrics.get("dictionary_size")
            st.metric("Dictionary Size", dict_size)
        with col3:
            activation_dim = metrics.get("activation_dim")
            expansion = dict_size / activation_dim if activation_dim else "N/A"
            st.metric("Expansion Factor", expansion)
        with col4:
            fve = metrics.get("last_eval_logs", {}).get(
                "val/frac_variance_explained", "N/A"
            )
            st.metric("FVE", fve)

        col1, col2 = st.columns(2)
        with col1:
            wandb_link = metrics.get("wandb_link")
            if wandb_link:
                st.markdown(f"**W&B Run:** [View training run]({wandb_link})")
            else:
                st.info("No W&B link available")
        with col2:
            hf_link = metrics.get("hf_repo_id")
            if hf_link:
                st.markdown(
                    f"**HF Model:** [View model](https://huggingface.co/{hf_link})"
                )
            else:
                st.info("No HF link available")
    except Exception as e:
        st.warning(f"Could not load training metrics: {str(e)}")


def _render_maxact_tab(method, cc_info):
    """Render MaxAct tab."""
    import sys
    layer = cc_info["layer"]
    model_results_dir = cc_info["path"]

    sys.stderr.write(f"[DEBUG MaxAct] _render_maxact_tab called for layer={layer}, path={model_results_dir}\n")
    st.markdown(
        f"**Selected CrossCoder:** Layer {layer} – {cc_info['dictionary_name']}"
    )

    latent_dir = model_results_dir / "latent_activations"
    sys.stderr.write(f"[DEBUG MaxAct] latent_dir={latent_dir}, exists={latent_dir.exists()}\n")
    if not latent_dir.exists():
        st.error(f"No latent activations directory found at {latent_dir}")
        return

    db_path = latent_dir / "examples.db"
    db_size = db_path.stat().st_size if db_path.exists() else 'N/A'
    sys.stderr.write(f"[DEBUG MaxAct] db_path={db_path}, exists={db_path.exists()}, size={db_size}\n")
    if not db_path.exists():
        st.error(f"No MaxAct example database found at {db_path}")
        return

    # Debug: check what tables exist in the DB
    import sqlite3 as _sqlite3
    try:
        _conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        _tables = [r[0] for r in _conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        _conn.close()
        sys.stderr.write(f"[DEBUG MaxAct] DB tables: {_tables}\n")
        st.info(f"DEBUG: DB tables found: {_tables}")
    except Exception as e:
        sys.stderr.write(f"[DEBUG MaxAct] Failed to inspect DB tables: {e}\n")
        st.error(f"DEBUG: Failed to inspect DB: {e}")

    try:
        tok_type = type(method.tokenizer)
        sys.stderr.write(f"[DEBUG MaxAct] method.tokenizer type: {tok_type}\n")
        assert method.tokenizer is not None, "Tokenizer required for MaxAct visualization"
    except Exception as e:
        sys.stderr.write(f"[DEBUG MaxAct] Tokenizer error: {e}\n")
        st.error(f"DEBUG: Tokenizer error: {e}")
        import traceback; st.code(traceback.format_exc())
        return

    try:
        sys.stderr.write(f"[DEBUG MaxAct] Creating ReadOnlyMaxActStore...\n")
        store = ReadOnlyMaxActStore(db_path, tokenizer=method.tokenizer)
        sys.stderr.write(f"[DEBUG MaxAct] Store created. storage_format={store.storage_format}\n")
    except Exception as e:
        sys.stderr.write(f"[DEBUG MaxAct] ReadOnlyMaxActStore init failed: {e}\n")
        st.error(f"DEBUG: ReadOnlyMaxActStore failed: {e}")
        import traceback; st.code(traceback.format_exc())
        return

    try:
        component = MaxActivationDashboardComponent(
            store, title=f"CrossCoder Examples – Layer {layer}"
        )
        sys.stderr.write(f"[DEBUG MaxAct] Calling component.display()...\n")
        component.display()
        sys.stderr.write(f"[DEBUG MaxAct] component.display() completed\n")
    except Exception as e:
        sys.stderr.write(f"[DEBUG MaxAct] display() failed: {e}\n")
        st.error(f"DEBUG: MaxActivationDashboardComponent.display() failed: {e}")
        import traceback; st.code(traceback.format_exc())


def _render_latent_statistics_tab(method, cc_info):
    """Render latent statistics tab."""
    dictionary_name = cc_info["dictionary_name"]
    layer = cc_info["layer"]

    st.markdown(f"### Latent Statistics – Layer {layer}")
    st.markdown(f"**Dictionary:** {dictionary_name}")

    try:
        df = load_latent_df(cc_info["path"] / "dictionary_model")
    except Exception as e:
        st.error(f"Failed to load latent df: {e}")
        return

    filtered_df = df.copy()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.markdown("**Categorical Filters**")
        cols = st.columns(min(3, len(cat_cols)))
        for i, col in enumerate(cat_cols):
            with cols[i % 3]:
                options = [v for v in df[col].unique().tolist() if pd.notna(v)]
                sel = st.multiselect(
                    col, options=options, default=options, key=f"filter_cat_{col}"
                )
                if sel:
                    filtered_df = filtered_df[filtered_df[col].isin(sel)]

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if num_cols:
        with st.expander("Numeric Filters", expanded=False):
            cols = st.columns(min(3, len(num_cols)))
            for i, col in enumerate(num_cols):
                with cols[i % 3]:
                    cmin, cmax = float(df[col].min()), float(df[col].max())
                    if cmin != cmax:
                        enable = st.checkbox(
                            f"Filter {col}", value=False, key=f"enable_num_{col}"
                        )
                        if enable:
                            rng = st.slider(
                                col, cmin, cmax, (cmin, cmax), key=f"slider_{col}"
                            )
                            if rng != (cmin, cmax):
                                filtered_df = filtered_df[
                                    (filtered_df[col] >= rng[0])
                                    & (filtered_df[col] <= rng[1])
                                ]

    st.markdown(f"**Showing {len(filtered_df)} / {len(df)} latents**")
    st.dataframe(filtered_df, use_container_width=True, height=400)


def _render_plots_tab(cc_info):
    """Render plots tab."""
    layer = cc_info["layer"]
    dictionary_name = cc_info["dictionary_name"]
    plots_dir = cc_info["path"] / "plots"

    if not plots_dir.exists():
        st.error(f"No plots directory found at {plots_dir}")
        return

    img_exts = [".png", ".jpg", ".jpeg", ".svg", ".pdf"]
    images = []
    for ext in img_exts:
        images.extend(plots_dir.glob(f"*{ext}"))

    if not images:
        st.error("No plot files found.")
        return

    st.markdown(f"### Plots - Layer {layer} - {dictionary_name}")
    st.markdown(f"Found {len(images)} plot files")

    for img in images:
        if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            st.image(str(img), use_container_width=True)
        elif img.suffix.lower() == ".svg":
            st.markdown(img.read_text(), unsafe_allow_html=True)
        elif img.suffix.lower() == ".pdf":
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='400'></iframe>",
                unsafe_allow_html=True,
            )


def _render_steering_results_tab(method, cc_info):
    """Render steering results tab."""
    st.markdown(
        f"**Selected CrossCoder:** Layer {cc_info['layer']} - {cc_info['dictionary_name']}"
    )
    display_steering_results(cc_info["path"], method.cfg)


class CrosscoderOnlineDashboard(AbstractOnlineDiffingDashboard):
    """Online per-token latent activation dashboard for CrossCoders."""

    def __init__(self, method_instance, cc_info):
        super().__init__(method_instance)
        self.cc_info = cc_info

    def _render_streamlit_method_controls(self):
        latent_idx = st.number_input("Latent Index", min_value=0, value=0, step=1)
        return {"latent_idx": latent_idx}

    def compute_statistics_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):
        layer = self.cc_info["layer"]
        latent_idx = kwargs.get("latent_idx", 0)
        res = self.method.compute_crosscoder_activations_for_tokens(
            self.cc_info["dictionary_name"], input_ids, attention_mask, layer
        )
        seq_len, dict_size = res["latent_activations"].shape
        assert 0 <= latent_idx < dict_size
        values = res["latent_activations"][:, latent_idx]
        stats = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
        }
        return {
            "tokens": res["tokens"],
            "values": values,
            "statistics": stats,
            "total_tokens": res["total_tokens"],
            "analysis_title": f"Latent {latent_idx} Activation",
        }

    def get_method_specific_params(self):
        return {"latent_idx": 0}

    def _get_title(self):
        return "CrossCoder Analysis"


class CrosscoderSteeringDashboard(SteeringDashboard):
    """Latent steering dashboard for CrossCoders."""

    def __init__(self, method_instance, cc_info):
        super().__init__(method_instance)
        self.cc_info = cc_info
        self._layer = cc_info["layer"]
        self._cc_model = None
        try:
            latent_df = load_latent_df(self.cc_info["dictionary_name"])
            if "max_act_validation" in latent_df.columns:
                self._max_acts = latent_df["max_act_validation"]
            elif "max_act_train" in latent_df.columns:
                self._max_acts = latent_df["max_act_train"]
            else:
                raise KeyError("Neither 'max_act_validation' nor 'max_act_train' found")
        except Exception:
            st.error(
                f"Maximum activations not yet collected for '{cc_info['dictionary_name']}'"
            )
            st.info("Please run the analysis pipeline first.")
            st.stop()

    @property
    def layer(self):
        return self._layer

    def _ensure_model(self):
        if self._cc_model is None:
            self._cc_model = load_dictionary_model(
                self.cc_info["dictionary_name"], is_sae=False
            )
            self._cc_model = self._cc_model.to(self.method.device)

    def get_dict_size(self):
        self._ensure_model()
        return self._cc_model.dict_size

    def get_latent(self, idx: int):
        self._ensure_model()
        assert 0 <= idx < self._cc_model.dict_size
        vec = self._cc_model.decoder.weight[:, idx, :].mean(dim=0)
        return vec.detach()

    def get_max_activation(self, latent_idx: int):
        if latent_idx in self._max_acts.index:
            return float(self._max_acts.loc[latent_idx])
        return "unknown"

    @st.fragment
    def _render_latent_selector(self):
        dict_size = self.get_dict_size()
        key = f"crosscoder_latent_idx_layer_{self.layer}"
        if key not in st.session_state:
            st.session_state[key] = 0
        st.session_state[key] = min(st.session_state[key], dict_size - 1)
        idx = st.number_input(
            "Latent Index", 0, dict_size - 1, st.session_state[key], 1, key=key
        )
        st.info(f"Max Activation: {self.get_max_activation(idx)}")
        return idx

    def _render_streamlit_method_controls(self):
        col1, col2 = st.columns(2)
        with col1:
            latent_idx = self._render_latent_selector()
        with col2:
            factor = st.slider("Steering Factor", -1000.0, 1000.0, 1.0, 0.1)
        mode = st.selectbox("Steering Mode", ["prompt_only", "all_tokens"], index=1)
        return {
            "latent_idx": latent_idx,
            "steering_factor": factor,
            "steering_mode": mode,
        }

    def _get_title(self):
        return f"CrossCoder Latent Steering – Layer {self.layer}"
