import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
import io


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PLS-DA",
    page_icon="🧪",
    layout="wide",
)

st.title("Partial Least Squares – Discriminant Analysis (PLS-DA)")


# =====================================================
# HELPERS
# =====================================================

def encode_labels(y):
    """Encode original class labels → integer codes 0..K-1.
    Returns codes, LabelEncoder, and sorted original labels.
    """
    le = LabelEncoder()
    codes = le.fit_transform(y.astype(str))
    return codes, le, list(le.classes_)


def nearest_class(pred_continuous, valid_codes):
    """Map continuous PLS predictions to nearest integer class code."""
    pred = np.asarray(pred_continuous).ravel()
    nearest = np.array([
        valid_codes[np.argmin(np.abs(valid_codes - p))]
        for p in pred
    ]).astype(int)
    return nearest


def decode_labels(codes, le):
    """Integer codes → original class labels."""
    return le.inverse_transform(np.asarray(codes).astype(int))


def get_confusion_df(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"True {l}" for l in labels],
        columns=[f"Pred {l}" for l in labels],
    ), labels


def plot_confusion_matrix(cm_df, title="Confusion matrix"):
    labels_y = [i.replace("True ", "") for i in cm_df.index]
    labels_x = [c.replace("Pred ", "") for c in cm_df.columns]
    fig = px.imshow(
        cm_df.values,
        text_auto=True,
        color_continuous_scale="Blues",
        x=labels_x,
        y=labels_y,
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="True",
        height=450,
        coloraxis_showscale=False,
    )
    return fig


def metrics_table(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    rows = []
    for lab in labels:
        yt = (np.asarray(y_true) == lab).astype(int)
        yp = (np.asarray(y_pred) == lab).astype(int)
        rows.append({
            "Class": lab,
            "Precision": precision_score(yt, yp, zero_division=0),
            "Recall": recall_score(yt, yp, zero_division=0),
            "F1-score": f1_score(yt, yp, zero_division=0),
            "Support": int(yt.sum()),
        })

    rows.append({
        "Class": "macro avg",
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0, labels=labels
        ),
        "Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0, labels=labels
        ),
        "F1-score": f1_score(
            y_true, y_pred, average="macro", zero_division=0, labels=labels
        ),
        "Support": len(y_true),
    })
    rows.append({
        "Class": "weighted avg",
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "F1-score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "Support": len(y_true),
    })

    df_m = pd.DataFrame(rows)
    for col in ["Precision", "Recall", "F1-score"]:
        df_m[col] = df_m[col].round(3)

    acc = accuracy_score(y_true, y_pred)
    return df_m, acc


def prepare_xy(dataframe, x_cols, y_col):
    cols = list(x_cols) + [y_col]
    sub = dataframe[cols].dropna()
    X = sub[x_cols].astype(np.float64)
    y = sub[y_col]
    return X, y, sub.index


def pls_predict_classes(model, X, valid_codes, le):
    """Predict continuous → nearest code → original labels."""
    pred_cont = model.predict(X)
    codes = nearest_class(pred_cont, valid_codes)
    labels = decode_labels(codes, le)
    return labels, codes, pred_cont.ravel()


# =====================================================
# DATA SOURCE
# =====================================================

st.divider()
st.header("Data source")

has_raw = "dataset" in st.session_state
has_prep = "preprocessed_dataset" in st.session_state
has_split = (
    "train_dataset" in st.session_state
    and "test_dataset" in st.session_state
)

source_options = []
if has_raw:
    source_options.append("Raw dataset")
if has_prep:
    source_options.append("Preprocessed dataset")
if has_split:
    source_options.append("Train / Test split")

if not source_options:
    st.warning(
        "No dataset available. Load data in **Data Import** first."
    )
    st.stop()

data_source = st.radio(
    "Select data source",
    source_options,
    horizontal=True,
    key="plsda_data_source",
)

use_external_split = False

if data_source == "Train / Test split":
    train_df = st.session_state["train_dataset"].copy()
    test_df = st.session_state["test_dataset"].copy()
    use_external_split = True
    df = train_df
    st.success(
        f"Split loaded — Train: {train_df.shape[0]} | "
        f"Test: {test_df.shape[0]}"
    )
elif data_source == "Preprocessed dataset":
    df = st.session_state["preprocessed_dataset"].copy()
    st.success(
        f"Preprocessed dataset: {df.shape[0]} × {df.shape[1]}"
    )
else:
    df = st.session_state["dataset"].copy()
    st.success(
        f"Raw dataset: {df.shape[0]} × {df.shape[1]}"
    )


# =====================================================
# VARIABLE SELECTION
# =====================================================

st.divider()
st.header("Variables")

numeric_variables = (
    df.select_dtypes(include=np.number).columns.tolist()
)
categorical_variables = (
    df.select_dtypes(exclude=np.number).columns.tolist()
)
discrete_numeric = [
    c
    for c in numeric_variables
    if df[c].nunique(dropna=True) <= 20
]
y_candidates = sorted(
    list(set(categorical_variables + discrete_numeric))
)

sample_id = st.session_state.get("sample_id")
if sample_id is not None and sample_id not in df.columns:
    sample_id = None

default_X = st.session_state.get("X_variables", numeric_variables)
default_X = [x for x in default_X if x in numeric_variables]

selected_X = st.multiselect(
    "Predictor variables (X)",
    numeric_variables,
    default=default_X if default_X else numeric_variables,
    key="plsda_X",
)

if len(selected_X) < 1:
    st.warning("Select at least one predictor variable.")
    st.stop()

y_options = [c for c in y_candidates if c not in selected_X]
if not y_options:
    st.error("No suitable categorical / discrete target found.")
    st.stop()

default_y_idx = 0
saved_y = st.session_state.get("y_variable")
if saved_y is not None and saved_y in y_options:
    default_y_idx = y_options.index(saved_y)

y_variable = st.selectbox(
    "Target variable (y)",
    y_options,
    index=default_y_idx,
    key="plsda_y",
)


# =====================================================
# PREPARE TRAINING DATA (needed by grid search & fit)
# =====================================================

# Max latent variables limited by samples and variables
n_samples_est = df.shape[0]
max_lv = max(1, min(len(selected_X), n_samples_est - 1, 20))

if use_external_split:
    X_train, y_train_raw, train_idx = prepare_xy(
        train_df, selected_X, y_variable
    )
else:
    X_train, y_train_raw, train_idx = prepare_xy(
        df, selected_X, y_variable
    )

Y_train_codes, le, class_labels = encode_labels(y_train_raw)
valid_codes = np.arange(len(class_labels))


# =====================================================
# STEP 1 — STRATIFIED GRID SEARCH
# =====================================================

st.divider()
st.header("Step 1 — Stratified grid search (best number of latent variables)")

st.write(
    f"**Training samples:** {X_train.shape[0]}  |  "
    f"**Variables:** {X_train.shape[1]}  |  "
    f"**Classes:** {len(class_labels)}"
)

map_df = pd.DataFrame({
    "Code": valid_codes,
    "Class": class_labels,
    "Count": [
        int((Y_train_codes == c).sum()) for c in valid_codes
    ],
})
st.dataframe(map_df, hide_index=True, use_container_width=True)

st.caption(
    "PLS-Regression requires numeric targets. "
    "Classes are encoded as integers, predictions are mapped "
    "back to the nearest class code, then decoded to original labels."
)

enable_grid = st.checkbox(
    "Enable stratified grid search",
    value=True,
    key="plsda_enable_grid",
)

lv_min, lv_max = 1, max_lv

if enable_grid:
    c1, c2 = st.columns(2)
    with c1:
        lv_min = st.number_input(
            "Min latent variables",
            min_value=1,
            max_value=max_lv,
            value=1,
            key="plsda_lv_min",
        )
    with c2:
        lv_max = st.number_input(
            "Max latent variables",
            min_value=1,
            max_value=max_lv,
            value=min(10, max_lv),
            key="plsda_lv_max",
        )

    if lv_min > lv_max:
        st.error("Min latent variables must be ≤ max latent variables.")
        st.stop()

    if st.button(
        "🔎 Run stratified grid search",
        type="primary",
        key="plsda_run_grid",
    ):
        # CV folds from Step 2 (default 5); scale always True
        cv_folds_gs = int(st.session_state.get("plsda_cv_folds", 5))

        counts = pd.Series(Y_train_codes).value_counts()
        n_splits = min(cv_folds_gs, int(counts.min()))
        if n_splits < 2:
            st.error(
                "Not enough samples per class for stratified CV."
            )
            st.stop()

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

        grid_rows = []
        lv_range = range(int(lv_min), int(lv_max) + 1)

        progress = st.progress(0)
        for i, nlv in enumerate(lv_range):
            fold_accs = []
            for tr_i, te_i in skf.split(X_train, Y_train_codes):
                pls = PLSRegression(
                    n_components=nlv, scale=True
                )
                pls.fit(
                    X_train.iloc[tr_i],
                    Y_train_codes[tr_i],
                )
                pred_c = pls.predict(X_train.iloc[te_i])
                pred_n = nearest_class(pred_c, valid_codes)
                fold_accs.append(
                    accuracy_score(
                        Y_train_codes[te_i], pred_n
                    )
                )
            grid_rows.append({
                "n_LV": nlv,
                "Mean accuracy": np.mean(fold_accs),
                "Std": np.std(fold_accs),
                "Min": np.min(fold_accs),
                "Max": np.max(fold_accs),
            })
            progress.progress((i + 1) / len(lv_range))

        grid_df = pd.DataFrame(grid_rows)
        grid_df["Mean accuracy"] = grid_df["Mean accuracy"].round(4)
        grid_df["Std"] = grid_df["Std"].round(4)
        grid_df["Min"] = grid_df["Min"].round(4)
        grid_df["Max"] = grid_df["Max"].round(4)

        best_row = grid_df.loc[
            grid_df["Mean accuracy"].idxmax()
        ]
        best_nlv = int(best_row["n_LV"])

        st.session_state["plsda_grid_df"] = grid_df
        st.session_state["plsda_best_n_lv"] = best_nlv
        # Flag so that on the next run the LV widget is updated
        # BEFORE it is instantiated (avoids StreamlitAPIException)
        st.session_state["_plsda_apply_best_lv"] = True
        st.success(
            f"Best number of latent variables = **{best_nlv}** "
            f"(CV accuracy = {best_row['Mean accuracy']:.4f}). "
            f"The LV field in Step 2 has been set to this value."
        )
        st.rerun()

    if "plsda_grid_df" in st.session_state:
        grid_df = st.session_state["plsda_grid_df"]
        best_nlv = st.session_state.get("plsda_best_n_lv")

        st.subheader("Grid search results")
        st.dataframe(
            grid_df, hide_index=True, use_container_width=True
        )

        fig_grid = px.line(
            grid_df,
            x="n_LV",
            y="Mean accuracy",
            error_y="Std",
            markers=True,
            title="CV accuracy vs number of latent variables",
        )
        if best_nlv is not None:
            fig_grid.add_vline(
                x=best_nlv,
                line_dash="dash",
                line_color="red",
                annotation_text=f"best = {best_nlv}",
            )
        fig_grid.update_yaxes(range=[0, 1.05])
        fig_grid.update_xaxes(title="Number of latent variables")
        st.plotly_chart(fig_grid, use_container_width=True)

        if best_nlv is not None:
            st.info(
                f"Suggested number of latent variables: **{best_nlv}** "
                f"(pre-filled in Step 2 below)."
            )
else:
    st.caption("Grid search disabled — choose the number of LV manually in Step 2.")


# =====================================================
# STEP 2 — MODEL SETTINGS, SELECT LV, CV & FIT
# =====================================================

st.divider()
st.header("Step 2 — Model settings, select LV & fit")

cv_folds = st.number_input(
    "Number of CV folds",
    min_value=2,
    max_value=10,
    value=5,
    step=1,
    key="plsda_cv_folds",
)

# Apply best LV from grid search BEFORE the widget is instantiated
if st.session_state.pop("_plsda_apply_best_lv", False):
    best = st.session_state.get("plsda_best_n_lv", min(5, max_lv))
    st.session_state["plsda_n_lv"] = int(min(max(1, best), max_lv))

if "plsda_n_lv" not in st.session_state:
    st.session_state["plsda_n_lv"] = int(min(5, max_lv))

n_lv = st.number_input(
    "Number of latent variables (LV)",
    min_value=1,
    max_value=max_lv,
    step=1,
    key="plsda_n_lv",
    help=(
        "Choose the number of latent variables for the final fit. "
        "If you ran the grid search, the suggested optimum is pre-filled."
    ),
)

enable_cv = st.checkbox(
    "Compute stratified Cross Validation on training data",
    value=True,
    key="plsda_enable_cv",
)

if st.button(
    "Fit PLS-DA on training set",
    type="primary",
    key="plsda_fit",
):
    counts = pd.Series(Y_train_codes).value_counts()
    if counts.min() < 1:
        st.error("At least one class has no samples.")
        st.stop()

    nlv = int(n_lv)
    nlv = min(nlv, X_train.shape[0] - 1, X_train.shape[1])
    if nlv < 1:
        st.error("Number of latent variables must be ≥ 1.")
        st.stop()

    model = PLSRegression(n_components=nlv, scale=True)
    model.fit(X_train, Y_train_codes)

    y_pred_labels, y_pred_codes, y_pred_cont = pls_predict_classes(
        model, X_train, valid_codes, le
    )

    x_scores = model.x_scores_
    y_scores = model.y_scores_

    cv_results = None
    if enable_cv:
        n_splits = min(cv_folds, int(counts.min()))
        if n_splits >= 2:
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )
            fold_accs = []
            y_pred_cv_codes = np.zeros_like(Y_train_codes)
            for tr_i, te_i in skf.split(X_train, Y_train_codes):
                pls_cv = PLSRegression(
                    n_components=nlv, scale=True
                )
                pls_cv.fit(
                    X_train.iloc[tr_i], Y_train_codes[tr_i]
                )
                pred_c = pls_cv.predict(X_train.iloc[te_i])
                pred_n = nearest_class(pred_c, valid_codes)
                y_pred_cv_codes[te_i] = pred_n
                fold_accs.append(
                    accuracy_score(
                        Y_train_codes[te_i], pred_n
                    )
                )
            y_pred_cv_labels = decode_labels(
                y_pred_cv_codes, le
            )
            cv_results = {
                "y_pred_codes": y_pred_cv_codes,
                "y_pred_labels": y_pred_cv_labels,
                "acc_scores": np.array(fold_accs),
                "n_splits": n_splits,
            }

    # Never overwrite widget keys (plsda_data_source, plsda_n_lv, plsda_scale, …)
    st.session_state["plsda_model"] = model
    st.session_state["plsda_le"] = le
    st.session_state["plsda_class_labels"] = class_labels
    st.session_state["plsda_valid_codes"] = valid_codes
    st.session_state["plsda_X_vars"] = selected_X
    st.session_state["plsda_y_var"] = y_variable
    st.session_state["plsda_n_lv_fit"] = nlv
    st.session_state["plsda_scale_fit"] = True
    st.session_state["plsda_train_idx"] = list(train_idx)
    st.session_state["plsda_data_source_fit"] = data_source

    st.session_state["plsda_train_results"] = {
        "y_true_labels": np.asarray(y_train_raw).astype(str),
        "y_true_codes": Y_train_codes,
        "y_pred_labels": y_pred_labels,
        "y_pred_codes": y_pred_codes,
        "y_pred_cont": y_pred_cont,
        "x_scores": x_scores,
        "y_scores": y_scores,
        "index": train_idx,
        "cv": cv_results,
        "X": X_train,
    }

    if "plsda_test_results" in st.session_state:
        del st.session_state["plsda_test_results"]

    st.success(
        f"PLS-DA fitted with **{nlv}** latent variables!"
    )
    st.rerun()


# =====================================================
# TABS
# =====================================================




st.divider()
tab_train, tab_test = st.tabs(["📘 Training", "📙 Test"])


# #####################################################
# TRAINING
# #####################################################

with tab_train:

    st.subheader("Training results")

    # -------------------------------------------------
    # DISPLAY TRAINING RESULTS
    # -------------------------------------------------
    if (
        "plsda_train_results" in st.session_state
        and st.session_state.get("plsda_X_vars") == selected_X
        and st.session_state.get("plsda_y_var") == y_variable
    ):
        res = st.session_state["plsda_train_results"]
        model = st.session_state["plsda_model"]
        le = st.session_state["plsda_le"]
        class_labels = st.session_state["plsda_class_labels"]
        nlv = st.session_state.get(
            "plsda_n_lv_fit",
            st.session_state.get("plsda_n_components_fit", n_lv),
        )

        y_true = res["y_true_labels"]
        y_pred = res["y_pred_labels"]
        labels = class_labels

        # ---- Resubstitution metrics ----
        st.divider()
        st.subheader("Training performance (resubstitution)")

        met_df, acc = metrics_table(y_true, y_pred, labels)
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("Latent variables", nlv)
        with c2:
            st.dataframe(
                met_df, hide_index=True, use_container_width=True
            )

        cm_df, _ = get_confusion_df(y_true, y_pred, labels)
        st.plotly_chart(
            plot_confusion_matrix(
                cm_df, "Confusion matrix — Training"
            ),
            use_container_width=True,
        )

        # Continuous predictions vs codes
        with st.expander("Continuous PLS predictions (training)"):
            cont_df = pd.DataFrame({
                "Sample": res["index"].astype(str),
                "True class": y_true,
                "True code": res["y_true_codes"],
                "Predicted continuous": np.round(
                    res["y_pred_cont"], 3
                ),
                "Predicted code": res["y_pred_codes"],
                "Predicted class": y_pred,
            })
            st.dataframe(cont_df, use_container_width=True)

        # ---- CV ----
        if res.get("cv") is not None:
            st.divider()
            st.subheader(
                f"🔄 Stratified {res['cv']['n_splits']}-fold CV"
            )

            acc_scores = res["cv"]["acc_scores"]
            y_pred_cv = res["cv"]["y_pred_labels"]

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Mean CV accuracy", f"{acc_scores.mean():.3f}"
            )
            c2.metric(
                "Std CV accuracy", f"{acc_scores.std():.3f}"
            )
            c3.metric(
                "Min – Max",
                f"{acc_scores.min():.3f} – {acc_scores.max():.3f}",
            )

            fold_df = pd.DataFrame({
                "Fold": [
                    f"Fold {i+1}" for i in range(len(acc_scores))
                ],
                "Accuracy": acc_scores,
            })
            fig_folds = px.bar(
                fold_df,
                x="Fold",
                y="Accuracy",
                title="Accuracy per CV fold",
                text="Accuracy",
            )
            fig_folds.update_traces(texttemplate="%{text:.3f}")
            fig_folds.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig_folds, use_container_width=True)

            met_cv, acc_cv = metrics_table(
                y_true, y_pred_cv, labels
            )
            st.write(f"**CV overall accuracy:** {acc_cv:.3f}")
            st.dataframe(
                met_cv, hide_index=True, use_container_width=True
            )

            cm_cv, _ = get_confusion_df(
                y_true, y_pred_cv, labels
            )
            st.plotly_chart(
                plot_confusion_matrix(
                    cm_cv, "Confusion matrix — Cross-validation"
                ),
                use_container_width=True,
            )

        # ---- Scores plot ----
        st.divider()
        st.subheader("📈 PLS scores")

        n_sc = res["x_scores"].shape[1]
        scores_df = pd.DataFrame(
            res["x_scores"],
            columns=[f"LV{i+1}" for i in range(n_sc)],
            index=res["index"],
        )
        scores_df["Class"] = y_true

        if sample_id is not None and sample_id in df.columns:
            scores_df[sample_id] = df.loc[
                scores_df.index, sample_id
            ]
            hover = sample_id
        else:
            hover = None

        if n_sc >= 2:
            fig_sc = px.scatter(
                scores_df,
                x="LV1",
                y="LV2",
                color="Class",
                hover_name=hover,
                title="PLS-DA scores plot (training)",
            )
            fig_sc.update_xaxes(
                zeroline=True, zerolinecolor="black"
            )
            fig_sc.update_yaxes(
                zeroline=True, zerolinecolor="black"
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        elif n_sc == 1:
            fig_sc = px.histogram(
                scores_df,
                x="LV1",
                color="Class",
                barmode="overlay",
                opacity=0.7,
                title="PLS-DA scores (1 component)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # Explained variance (X)
        if hasattr(model, "x_scores_"):
            # Approximate X variance explained via scores
            x_var = np.var(res["X"].values, axis=0).sum()
            if x_var > 0:
                # x_loadings_ available after fit
                pass

        # Loadings / weights
        st.subheader("X-loadings / weights")

        if hasattr(model, "x_loadings_"):
            load_df = pd.DataFrame(
                model.x_loadings_,
                index=selected_X,
                columns=[f"LV{i+1}" for i in range(int(nlv))],
            )
            load_df = load_df.reset_index().rename(
                columns={"index": "Variable"}
            )
            st.dataframe(
                load_df.round(4),
                hide_index=True,
                use_container_width=True,
            )

            if int(nlv) >= 2:
                fig_ld = px.scatter(
                    load_df,
                    x="LV1",
                    y="LV2",
                    text="Variable",
                    title="X-loadings plot",
                )
                fig_ld.update_traces(textposition="top center")
                fig_ld.update_xaxes(
                    zeroline=True, zerolinecolor="black"
                )
                fig_ld.update_yaxes(
                    zeroline=True, zerolinecolor="black"
                )
                st.plotly_chart(fig_ld, use_container_width=True)

            # Bar contributions LV1
            bar_df = load_df[["Variable", "LV1"]].copy()
            bar_df = bar_df.sort_values(
                by="LV1", key=np.abs, ascending=False
            )
            fig_bar = px.bar(
                bar_df,
                x="Variable",
                y="LV1",
                title="Variable contributions — LV1",
            )
            fig_bar.update_layout(xaxis_tickangle=90)
            st.plotly_chart(fig_bar, use_container_width=True)

        # VIP scores (optional, useful for PLS-DA)
        st.subheader("VIP scores (Variable Importance in Projection)")

        try:
            # VIP computation
            t = model.x_scores_
            w = model.x_weights_
            q = model.y_loadings_
            p, h = w.shape
            vips = np.zeros(p)
            s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
            total_s = np.sum(s)
            for i in range(p):
                weight = np.array([
                    (w[i, j] / np.linalg.norm(w[:, j])) ** 2
                    for j in range(h)
                ])
                vips[i] = np.sqrt(
                    p * (s.T @ weight).ravel()[0] / total_s
                )

            vip_df = pd.DataFrame({
                "Variable": selected_X,
                "VIP": vips,
            }).sort_values("VIP", ascending=False)

            st.dataframe(
                vip_df.round(4),
                hide_index=True,
                use_container_width=True,
            )

            fig_vip = px.bar(
                vip_df,
                x="Variable",
                y="VIP",
                title="VIP scores",
            )
            fig_vip.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="red",
                annotation_text="VIP = 1",
            )
            fig_vip.update_layout(xaxis_tickangle=90)
            st.plotly_chart(fig_vip, use_container_width=True)

            st.session_state["plsda_vip"] = vip_df
        except Exception as e:
            st.caption(f"VIP computation skipped: {e}")

        # ---- Download ----
        st.divider()
        st.subheader("⬇️ Download training results")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pred_df = pd.DataFrame({
                "Sample": res["index"].astype(str),
                "True": y_true,
                "Predicted": y_pred,
                "Predicted_continuous": res["y_pred_cont"],
            })
            pred_df.to_excel(
                writer, sheet_name="Train_Predictions", index=False
            )
            met_df.to_excel(
                writer, sheet_name="Train_Metrics", index=False
            )
            cm_df.to_excel(writer, sheet_name="Train_CM")
            scores_df.to_excel(writer, sheet_name="Scores")

            if res.get("cv") is not None:
                cv_pred = pd.DataFrame({
                    "Sample": res["index"].astype(str),
                    "True": y_true,
                    "Predicted_CV": res["cv"]["y_pred_labels"],
                })
                cv_pred.to_excel(
                    writer, sheet_name="CV_Predictions", index=False
                )
                met_cv, _ = metrics_table(
                    y_true, res["cv"]["y_pred_labels"], labels
                )
                met_cv.to_excel(
                    writer, sheet_name="CV_Metrics", index=False
                )

            if "plsda_vip" in st.session_state:
                st.session_state["plsda_vip"].to_excel(
                    writer, sheet_name="VIP", index=False
                )

            if "plsda_grid_df" in st.session_state:
                st.session_state["plsda_grid_df"].to_excel(
                    writer, sheet_name="GridSearch", index=False
                )

        st.download_button(
            "⬇️ Download training results (Excel)",
            data=buf.getvalue(),
            file_name="PLSDA_training_results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="plsda_dl_train",
        )

    else:
        st.info(
            "Complete **Step 1** (optional grid search) and **Step 2** (select LV & fit) "
            "above, then results will appear here."
        )


# #####################################################
# TEST
# #####################################################

with tab_test:

    st.subheader("Test set evaluation")

    if "plsda_model" not in st.session_state:
        st.warning("Fit a model in the **Training** tab first.")
        st.stop()

    model = st.session_state["plsda_model"]
    le = st.session_state["plsda_le"]
    class_labels = st.session_state["plsda_class_labels"]
    valid_codes = st.session_state["plsda_valid_codes"]
    X_vars_fit = st.session_state["plsda_X_vars"]
    y_var_fit = st.session_state["plsda_y_var"]

    # Build test data
    if use_external_split:
        missing_cols = [
            c
            for c in X_vars_fit + [y_var_fit]
            if c not in test_df.columns
        ]
        if missing_cols:
            st.error(f"Test set missing columns: {missing_cols}")
            st.stop()
        X_test, y_test_raw, test_idx = prepare_xy(
            test_df, X_vars_fit, y_var_fit
        )
    else:
        st.info(
            "No external split. Using leftover samples or a "
            "random hold-out for illustration."
        )
        from sklearn.model_selection import train_test_split

        holdout = st.slider(
            "Hold-out proportion",
            0.1, 0.5, 0.2, 0.05,
            key="plsda_holdout",
        )
        seed_ho = st.number_input(
            "Hold-out seed", 0, 99999, 42, key="plsda_ho_seed"
        )

        X_all, y_all, all_idx = prepare_xy(
            df, X_vars_fit, y_var_fit
        )
        train_idx_used = set(
            st.session_state.get("plsda_train_idx", [])
        )

        if train_idx_used and set(all_idx).issubset(train_idx_used):
            st.warning(
                "Model was fitted on the entire dataset. "
                "Hold-out results are not fully independent."
            )
            try:
                y_codes_all, _, _ = encode_labels(y_all)
                _, te_idx = train_test_split(
                    all_idx,
                    test_size=holdout,
                    random_state=int(seed_ho),
                    stratify=y_codes_all,
                )
            except ValueError:
                _, te_idx = train_test_split(
                    all_idx,
                    test_size=holdout,
                    random_state=int(seed_ho),
                )
            X_test = X_all.loc[te_idx]
            y_test_raw = y_all.loc[te_idx]
            test_idx = te_idx
        else:
            leftover = [
                i for i in all_idx if i not in train_idx_used
            ]
            if len(leftover) < 2:
                st.error(
                    "Not enough leftover samples. "
                    "Use the Data Split page."
                )
                st.stop()
            X_test = X_all.loc[leftover]
            y_test_raw = y_all.loc[leftover]
            test_idx = leftover

    st.write(
        f"**Test samples:** {X_test.shape[0]}  |  "
        f"**Variables:** {X_test.shape[1]}"
    )

    if st.button(
        "📊 Evaluate on test set",
        type="primary",
        key="plsda_eval_test",
    ):
        y_pred_labels, y_pred_codes, y_pred_cont = (
            pls_predict_classes(
                model, X_test, valid_codes, le
            )
        )

        # Transform for scores
        try:
            x_scores_te = model.transform(X_test)
        except Exception:
            x_scores_te = None

        st.session_state["plsda_test_results"] = {
            "y_true_labels": np.asarray(y_test_raw).astype(str),
            "y_pred_labels": y_pred_labels,
            "y_pred_codes": y_pred_codes,
            "y_pred_cont": y_pred_cont,
            "x_scores": x_scores_te,
            "index": test_idx,
        }
        st.success("✅ Test evaluation completed!")
        st.rerun()

    if "plsda_test_results" in st.session_state:
        tres = st.session_state["plsda_test_results"]
        y_true_te = tres["y_true_labels"]
        y_pred_te = tres["y_pred_labels"]
        labels_te = class_labels

        st.divider()
        st.subheader("Test performance")

        met_te, acc_te = metrics_table(
            y_true_te, y_pred_te, labels_te
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Test accuracy", f"{acc_te:.3f}")
        with c2:
            st.dataframe(
                met_te, hide_index=True, use_container_width=True
            )

        cm_te, _ = get_confusion_df(
            y_true_te, y_pred_te, labels_te
        )
        st.plotly_chart(
            plot_confusion_matrix(
                cm_te, "Confusion matrix — Test"
            ),
            use_container_width=True,
        )

        with st.expander("Classification report (text)"):
            st.text(
                classification_report(
                    y_true_te,
                    y_pred_te,
                    labels=labels_te,
                    digits=3,
                )
            )

        with st.expander("Continuous PLS predictions (test)"):
            cont_te = pd.DataFrame({
                "Sample": np.asarray(tres["index"]).astype(str),
                "True class": y_true_te,
                "Predicted continuous": np.round(
                    tres["y_pred_cont"], 3
                ),
                "Predicted code": tres["y_pred_codes"],
                "Predicted class": y_pred_te,
            })
            st.dataframe(cont_te, use_container_width=True)

        # Scores on test
        if tres["x_scores"] is not None:
            n_sc = tres["x_scores"].shape[1]
            scores_te = pd.DataFrame(
                tres["x_scores"],
                columns=[f"LV{i+1}" for i in range(n_sc)],
                index=tres["index"],
            )
            scores_te["Class"] = y_true_te
            scores_te["Predicted"] = y_pred_te

            if n_sc >= 2:
                fig_te = px.scatter(
                    scores_te,
                    x="LV1",
                    y="LV2",
                    color="Class",
                    symbol="Predicted",
                    title="PLS-DA scores — Test set",
                )
                fig_te.update_xaxes(
                    zeroline=True, zerolinecolor="black"
                )
                fig_te.update_yaxes(
                    zeroline=True, zerolinecolor="black"
                )
                st.plotly_chart(fig_te, use_container_width=True)

        # Download
        st.divider()
        st.subheader("⬇️ Download test results")

        buf_te = io.BytesIO()
        with pd.ExcelWriter(
            buf_te, engine="openpyxl"
        ) as writer:
            pred_te_df = pd.DataFrame({
                "Sample": np.asarray(
                    tres["index"]
                ).astype(str),
                "True": y_true_te,
                "Predicted": y_pred_te,
                "Predicted_continuous": tres["y_pred_cont"],
            })
            pred_te_df.to_excel(
                writer, sheet_name="Test_Predictions", index=False
            )
            met_te.to_excel(
                writer, sheet_name="Test_Metrics", index=False
            )
            cm_te.to_excel(writer, sheet_name="Test_CM")

            if tres["x_scores"] is not None:
                scores_te.to_excel(
                    writer, sheet_name="Scores_Test"
                )

        st.download_button(
            "⬇️ Download test results (Excel)",
            data=buf_te.getvalue(),
            file_name="PLSDA_test_results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="plsda_dl_test",
        )

    else:
        st.info(
            "Click **Evaluate on test set** to get predictions "
            "and metrics."
        )
