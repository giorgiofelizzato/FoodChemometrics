import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import io


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LDA / QDA",
    page_icon="📐",
    layout="wide",
)

st.title(" Linear & Quadratic Discriminant Analysis")


# =====================================================
# HELPERS
# =====================================================

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
        rows.append({
            "Class": lab,
            "Precision": precision_score(
                y_true, y_pred, labels=[lab], average="micro", zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, labels=[lab], average="micro", zero_division=0
            ),
            "F1-score": f1_score(
                y_true, y_pred, labels=[lab], average="micro", zero_division=0
            ),
            "Support": int((np.asarray(y_true) == lab).sum()),
        })

    # macro / weighted
    rows.append({
        "Class": "macro avg",
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Support": len(y_true),
    })
    rows.append({
        "Class": "weighted avg",
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Support": len(y_true),
    })

    df_m = pd.DataFrame(rows)
    for col in ["Precision", "Recall", "F1-score"]:
        df_m[col] = df_m[col].round(3)

    acc = accuracy_score(y_true, y_pred)
    return df_m, acc


def build_scores_df(scores_array, y, index, n_comp):
    cols = [f"LD{i+1}" for i in range(n_comp)]
    sdf = pd.DataFrame(scores_array, columns=cols, index=index)
    sdf["Class"] = np.asarray(y)
    return sdf


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
    key="lda_data_source",
)


# -----------------------------------------------------
# Load data according to source
# -----------------------------------------------------

use_external_split = False

if data_source == "Train / Test split":
    train_df = st.session_state["train_dataset"].copy()
    test_df = st.session_state["test_dataset"].copy()
    use_external_split = True
    df = train_df  # used for variable detection
    st.success(
        f"Split loaded — Train: {train_df.shape[0]} samples | "
        f"Test: {test_df.shape[0]} samples"
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

# discrete numeric + categorical for y
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

# --- X ---
default_X = st.session_state.get("X_variables", numeric_variables)
default_X = [x for x in default_X if x in numeric_variables]

selected_X = st.multiselect(
    "Predictor variables (X)",
    numeric_variables,
    default=default_X if default_X else numeric_variables,
    key="lda_X",
)

if len(selected_X) < 1:
    st.warning("Select at least one predictor variable.")
    st.stop()

# --- y ---
y_options = [c for c in y_candidates if c not in selected_X]
if not y_options:
    st.error(
        "No suitable categorical / discrete target variable found."
    )
    st.stop()

default_y_idx = 0
saved_y = st.session_state.get("y_variable")
if saved_y is not None and saved_y in y_options:
    default_y_idx = y_options.index(saved_y)

y_variable = st.selectbox(
    "Target variable (y) — must be categorical or discrete",
    y_options,
    index=default_y_idx,
    key="lda_y",
)


# =====================================================
# MODEL SETTINGS
# =====================================================

st.divider()
st.header("Model settings")

col1, col2, col3 = st.columns(3)

with col1:
    model_type = st.selectbox(
        "Model",
        ["LDA", "QDA"],
        key="lda_model_type",
    )

with col2:
    n_classes = df[y_variable].nunique(dropna=True)
    max_comp = max(1, min(len(selected_X), n_classes - 1))

    if model_type == "LDA":
        n_components = st.slider(
            "Number of discriminant axes",
            min_value=1,
            max_value=max_comp,
            value=min(2, max_comp),
            key="lda_ncomp",
        )
    else:
        n_components = None
        st.caption("QDA does not produce discriminant axes.")

with col3:
    solver = st.selectbox(
        "LDA solver",
        ["svd", "lsqr", "eigen"],
        key="lda_solver",
        disabled=(model_type == "QDA"),
    )


# =====================================================
# CROSS-VALIDATION SETTINGS
# =====================================================

st.divider()
st.header("Cross-validation (on training set)")

enable_cv = st.checkbox(
    "Enable stratified cross-validation",
    value=True,
    key="lda_enable_cv",
)

cv_folds = 5
cv_seed = 42

if enable_cv:
    c1, c2 = st.columns(2)
    with c1:
        cv_folds = st.slider(
            "Number of folds",
            min_value=2,
            max_value=10,
            value=5,
            key="lda_cv_folds",
        )
    with c2:
        cv_seed = st.number_input(
            "CV random seed",
            min_value=0,
            max_value=99999,
            value=42,
            key="lda_cv_seed",
        )


# =====================================================
# PREPARE X / y
# =====================================================

def prepare_xy(dataframe, x_cols, y_col):
    """Return clean X, y, and the filtered dataframe index."""
    cols = list(x_cols) + [y_col]
    sub = dataframe[cols].dropna()
    X = sub[x_cols].astype(np.float64)
    y = sub[y_col]
    return X, y, sub.index


# =====================================================
# TABS: TRAINING | TEST
# =====================================================

st.divider()

tab_train, tab_test = st.tabs([
    "Training",
    "Test",
])


# #####################################################
# TRAINING TAB
# #####################################################

with tab_train:

    st.subheader("Training set")

    # ---- build training data ----
    if use_external_split:
        X_train, y_train, train_idx = prepare_xy(
            train_df, selected_X, y_variable
        )
    else:
        # If no external split, use the whole selected dataset as train
        # (user can still evaluate CV). Optional hold-out is not forced.
        X_train, y_train, train_idx = prepare_xy(
            df, selected_X, y_variable
        )

    st.write(
        f"**Samples:** {X_train.shape[0]}  |  "
        f"**Variables:** {X_train.shape[1]}  |  "
        f"**Classes:** {y_train.nunique()}"
    )

    class_dist = y_train.value_counts().rename("Count")
    st.dataframe(
        class_dist.to_frame(),
        use_container_width=True,
    )

    # ---- Fit button ----
    if st.button(
        "🚀 Fit model on training set",
        type="primary",
        key="lda_fit",
    ):

        # Check class sizes
        counts = y_train.value_counts()
        if counts.min() < 2:
            st.error(
                f"Class '{counts.idxmin()}' has fewer than 2 samples. "
                "Cannot fit the model."
            )
            st.stop()

        # Instantiate model
        if model_type == "LDA":
            model = LDA(
                n_components=n_components,
                solver=solver,
            )
        else:
            model = QDA()

        # Fit
        model.fit(X_train, y_train)

        # Predictions on training
        y_pred_train = model.predict(X_train)

        # Scores (LDA only)
        scores_train = None
        if model_type == "LDA":
            scores_train = model.transform(X_train)

        # Cross-validation
        cv_results = None
        if enable_cv:
            min_class = counts.min()
            n_splits = min(cv_folds, int(min_class))
            if n_splits < 2:
                st.warning(
                    "Not enough samples per class for CV. "
                    "CV skipped."
                )
            else:
                skf = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=int(cv_seed),
                )
                y_pred_cv = cross_val_predict(
                    model.__class__(
                        **(
                            {
                                "n_components": n_components,
                                "solver": solver,
                            }
                            if model_type == "LDA"
                            else {}
                        )
                    ),
                    X_train,
                    y_train,
                    cv=skf,
                )
                cv_acc_scores = cross_val_score(
                    model.__class__(
                        **(
                            {
                                "n_components": n_components,
                                "solver": solver,
                            }
                            if model_type == "LDA"
                            else {}
                        )
                    ),
                    X_train,
                    y_train,
                    cv=skf,
                    scoring="accuracy",
                )
                cv_results = {
                    "y_pred_cv": y_pred_cv,
                    "acc_scores": cv_acc_scores,
                    "n_splits": n_splits,
                }

        # Store in session state
        st.session_state["da_model"] = model
        st.session_state["da_model_type"] = model_type
        st.session_state["da_X_vars"] = selected_X
        st.session_state["da_y_var"] = y_variable
        st.session_state["da_n_components"] = n_components
        st.session_state["da_train_idx"] = list(train_idx)
        st.session_state["da_data_source"] = data_source

        st.session_state["da_train_results"] = {
            "y_true": y_train,
            "y_pred": y_pred_train,
            "scores": scores_train,
            "index": train_idx,
            "cv": cv_results,
            "X": X_train,
        }

        # Clear previous test results
        if "da_test_results" in st.session_state:
            del st.session_state["da_test_results"]

        st.success("✅ Model fitted successfully!")
        st.rerun()

    # ---- Display training results if available ----
    if (
        "da_train_results" in st.session_state
        and st.session_state.get("da_X_vars") == selected_X
        and st.session_state.get("da_y_var") == y_variable
        and st.session_state.get("da_model_type") == model_type
    ):

        res = st.session_state["da_train_results"]
        model = st.session_state["da_model"]
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        labels = sorted(set(y_true) | set(y_pred))

        # =============================================
        # Metrics – Training (resubstitution)
        # =============================================
        st.divider()
        st.subheader("📊 Training performance (resubstitution)")

        met_df, acc = metrics_table(y_true, y_pred, labels)
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Accuracy", f"{acc:.3f}")
        with c2:
            st.dataframe(met_df, hide_index=True, use_container_width=True)

        cm_df, _ = get_confusion_df(y_true, y_pred, labels)
        st.plotly_chart(
            plot_confusion_matrix(
                cm_df, "Confusion matrix — Training"
            ),
            use_container_width=True,
        )

        # =============================================
        # Cross-validation
        # =============================================
        if res.get("cv") is not None:
            st.divider()
            st.subheader(
                f"🔄 Stratified {res['cv']['n_splits']}-fold CV"
            )

            acc_scores = res["cv"]["acc_scores"]
            y_pred_cv = res["cv"]["y_pred_cv"]

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Mean CV accuracy",
                f"{acc_scores.mean():.3f}",
            )
            c2.metric(
                "Std CV accuracy",
                f"{acc_scores.std():.3f}",
            )
            c3.metric(
                "Min – Max",
                f"{acc_scores.min():.3f} – {acc_scores.max():.3f}",
            )

            # Per-fold accuracy bar
            fold_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(acc_scores))],
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

            # CV metrics table + confusion
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

        # =============================================
        # LDA-specific plots
        # =============================================
        if model_type == "LDA" and res["scores"] is not None:

            st.divider()
            st.subheader("LDA scores & loadings")

            n_comp = res["scores"].shape[1]
            scores_df = build_scores_df(
                res["scores"],
                y_true,
                res["index"],
                n_comp,
            )

            # Explained variance
            if hasattr(model, "explained_variance_ratio_"):
                ev = model.explained_variance_ratio_ * 100
                ev_df = pd.DataFrame({
                    "Axis": [f"LD{i+1}" for i in range(len(ev))],
                    "Explained variance (%)": np.round(ev, 2),
                    "Cumulative (%)": np.round(np.cumsum(ev), 2),
                })
                st.dataframe(
                    ev_df, hide_index=True, use_container_width=True
                )

            # Scores plot 2D
            if n_comp >= 2:
                hover = sample_id if (
                    sample_id is not None
                    and sample_id in df.columns
                ) else None

                if hover is not None:
                    scores_df[hover] = df.loc[
                        scores_df.index, hover
                    ]

                fig_sc = px.scatter(
                    scores_df,
                    x="LD1",
                    y="LD2",
                    color="Class",
                    hover_name=hover,
                    title="LDA scores plot (training)",
                )
                fig_sc.update_xaxes(
                    zeroline=True, zerolinecolor="black"
                )
                fig_sc.update_yaxes(
                    zeroline=True, zerolinecolor="black"
                )
                if hasattr(model, "explained_variance_ratio_"):
                    fig_sc.update_xaxes(
                        title=(
                            f"LD1 "
                            f"({model.explained_variance_ratio_[0]*100:.1f}%)"
                        )
                    )
                    fig_sc.update_yaxes(
                        title=(
                            f"LD2 "
                            f"({model.explained_variance_ratio_[1]*100:.1f}%)"
                        )
                    )
                st.plotly_chart(fig_sc, use_container_width=True)

            elif n_comp == 1:
                fig_sc = px.histogram(
                    scores_df,
                    x="LD1",
                    color="Class",
                    barmode="overlay",
                    opacity=0.7,
                    title="LDA scores (1 component)",
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            # Loadings (coefficients)
            st.subheader("Loadings (coefficients)")

            if model_type == "LDA" and hasattr(model, "coef_"):
                # coef_ shape: (n_classes, n_features) for multi-class
                # For binary: (1, n_features)
                coef = model.coef_

                if coef.shape[0] == 1:
                    load_df = pd.DataFrame({
                        "Variable": selected_X,
                        "LD1": coef[0],
                    })
                else:
                    # Use scalings_ if available (preferred for LDA axes)
                    if (
                        hasattr(model, "scalings_")
                        and model.scalings_ is not None
                    ):
                        n_ax = min(
                            n_comp, model.scalings_.shape[1]
                        )
                        load_df = pd.DataFrame(
                            model.scalings_[:, :n_ax],
                            index=selected_X,
                            columns=[
                                f"LD{i+1}" for i in range(n_ax)
                            ],
                        )
                        load_df = load_df.reset_index().rename(
                            columns={"index": "Variable"}
                        )
                    else:
                        # Fallback: first two class coefficients
                        n_ax = min(2, coef.shape[0])
                        load_df = pd.DataFrame(
                            coef[:n_ax].T,
                            index=selected_X,
                            columns=[
                                f"Coef class {i}"
                                for i in range(n_ax)
                            ],
                        )
                        load_df = load_df.reset_index().rename(
                            columns={"index": "Variable"}
                        )

                st.dataframe(
                    load_df.round(4),
                    hide_index=True,
                    use_container_width=True,
                )

                # Loadings scatter if ≥ 2 axes
                ld_cols = [
                    c
                    for c in load_df.columns
                    if c.startswith("LD")
                ]
                if len(ld_cols) >= 2:
                    fig_ld = px.scatter(
                        load_df,
                        x=ld_cols[0],
                        y=ld_cols[1],
                        text="Variable",
                        title="Loadings plot",
                    )
                    fig_ld.update_traces(
                        textposition="top center"
                    )
                    fig_ld.update_xaxes(
                        zeroline=True, zerolinecolor="black"
                    )
                    fig_ld.update_yaxes(
                        zeroline=True, zerolinecolor="black"
                    )
                    st.plotly_chart(
                        fig_ld, use_container_width=True
                    )

                # Bar plot of |loadings| for LD1
                if ld_cols:
                    bar_df = load_df[
                        ["Variable", ld_cols[0]]
                    ].copy()
                    bar_df = bar_df.sort_values(
                        by=ld_cols[0],
                        key=np.abs,
                        ascending=False,
                    )
                    fig_bar = px.bar(
                        bar_df,
                        x="Variable",
                        y=ld_cols[0],
                        title=f"Variable contributions — {ld_cols[0]}",
                    )
                    fig_bar.update_layout(xaxis_tickangle=90)
                    st.plotly_chart(
                        fig_bar, use_container_width=True
                    )

        # =============================================
        # Download training results
        # =============================================
        st.divider()
        st.subheader("⬇️ Download training results")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Predictions
            pred_df = pd.DataFrame({
                "Sample": res["index"].astype(str),
                "True": y_true.values,
                "Predicted": y_pred,
            })
            pred_df.to_excel(
                writer, sheet_name="Train_Predictions", index=False
            )

            # Metrics
            met_df.to_excel(
                writer, sheet_name="Train_Metrics", index=False
            )

            # Confusion
            cm_df.to_excel(writer, sheet_name="Train_CM")

            # CV
            if res.get("cv") is not None:
                cv_pred_df = pd.DataFrame({
                    "Sample": res["index"].astype(str),
                    "True": y_true.values,
                    "Predicted_CV": res["cv"]["y_pred_cv"],
                })
                cv_pred_df.to_excel(
                    writer, sheet_name="CV_Predictions", index=False
                )
                met_cv, _ = metrics_table(
                    y_true, res["cv"]["y_pred_cv"], labels
                )
                met_cv.to_excel(
                    writer, sheet_name="CV_Metrics", index=False
                )

            # Scores
            if res["scores"] is not None:
                scores_df.to_excel(
                    writer, sheet_name="LDA_Scores"
                )

        st.download_button(
            "⬇️ Download training results (Excel)",
            data=buf.getvalue(),
            file_name="LDA_QDA_training_results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="dl_train",
        )

    else:
        st.info(
            "Click **Fit model on training set** to train the model "
            "and see results."
        )


# #####################################################
# TEST TAB
# #####################################################

with tab_test:

    st.subheader("Test set evaluation")

    if "da_model" not in st.session_state:
        st.warning(
            "Fit a model in the **Training** tab first."
        )
        st.stop()

    model = st.session_state["da_model"]
    model_type_fit = st.session_state["da_model_type"]
    X_vars_fit = st.session_state["da_X_vars"]
    y_var_fit = st.session_state["da_y_var"]

    # ---- build test data ----
    if use_external_split:
        # Check columns exist
        missing_cols = [
            c
            for c in X_vars_fit + [y_var_fit]
            if c not in test_df.columns
        ]
        if missing_cols:
            st.error(
                f"Test set is missing columns: {missing_cols}"
            )
            st.stop()

        X_test, y_test, test_idx = prepare_xy(
            test_df, X_vars_fit, y_var_fit
        )
    else:
        st.info(
            "No external train/test split found. "
            "You can still evaluate on a hold-out subset of the "
            "current dataset, or go to the **Data Split** page."
        )

        # Optional: simple random hold-out from current df
        from sklearn.model_selection import train_test_split

        holdout = st.slider(
            "Hold-out proportion for quick test",
            0.1, 0.5, 0.2, 0.05,
            key="lda_holdout",
        )
        seed_ho = st.number_input(
            "Hold-out seed", 0, 99999, 42, key="lda_ho_seed"
        )

        X_all, y_all, all_idx = prepare_xy(
            df, X_vars_fit, y_var_fit
        )

        # Exclude training indices if model was fit on a subset
        train_idx_used = set(
            st.session_state.get("da_train_idx", [])
        )

        if train_idx_used and set(all_idx).issubset(train_idx_used):
            # Model was fit on the whole dataset → create a fresh split
            # just for illustration (warn user)
            st.warning(
                "The model was fitted on the entire dataset. "
                "Creating a random hold-out for illustration only "
                "— results are optimistic / not independent."
            )
            try:
                _, te_idx = train_test_split(
                    all_idx,
                    test_size=holdout,
                    random_state=int(seed_ho),
                    stratify=y_all,
                )
            except ValueError:
                _, te_idx = train_test_split(
                    all_idx,
                    test_size=holdout,
                    random_state=int(seed_ho),
                )
            X_test = X_all.loc[te_idx]
            y_test = y_all.loc[te_idx]
            test_idx = te_idx
        else:
            # Use samples not in training
            leftover = [i for i in all_idx if i not in train_idx_used]
            if len(leftover) < 2:
                st.error(
                    "Not enough leftover samples for a test set. "
                    "Use the Data Split page."
                )
                st.stop()
            X_test = X_all.loc[leftover]
            y_test = y_all.loc[leftover]
            test_idx = leftover

    st.write(
        f"**Test samples:** {X_test.shape[0]}  |  "
        f"**Variables:** {X_test.shape[1]}"
    )

    if st.button(
        "📊 Evaluate on test set",
        type="primary",
        key="lda_eval_test",
    ):

        y_pred_test = model.predict(X_test)

        scores_test = None
        if model_type_fit == "LDA":
            try:
                scores_test = model.transform(X_test)
            except Exception:
                scores_test = None

        # Probabilities if available
        proba_test = None
        if hasattr(model, "predict_proba"):
            try:
                proba_test = model.predict_proba(X_test)
            except Exception:
                proba_test = None

        st.session_state["da_test_results"] = {
            "y_true": y_test,
            "y_pred": y_pred_test,
            "scores": scores_test,
            "proba": proba_test,
            "index": test_idx,
            "classes": list(model.classes_),
        }
        st.success("✅ Test evaluation completed!")
        st.rerun()

    # ---- Display test results ----
    if "da_test_results" in st.session_state:

        tres = st.session_state["da_test_results"]
        y_true_te = tres["y_true"]
        y_pred_te = tres["y_pred"]
        labels_te = sorted(
            set(y_true_te) | set(y_pred_te)
        )

        st.divider()
        st.subheader("📊 Test performance")

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

        # Classification report text
        with st.expander("Classification report (text)"):
            st.text(
                classification_report(
                    y_true_te,
                    y_pred_te,
                    digits=3,
                )
            )

        # Scores plot on test (LDA)
        if (
            model_type_fit == "LDA"
            and tres["scores"] is not None
        ):
            n_comp_te = tres["scores"].shape[1]
            scores_te_df = build_scores_df(
                tres["scores"],
                y_true_te,
                tres["index"],
                n_comp_te,
            )
            scores_te_df["Predicted"] = y_pred_te

            if n_comp_te >= 2:
                fig_te = px.scatter(
                    scores_te_df,
                    x="LD1",
                    y="LD2",
                    color="Class",
                    symbol="Predicted",
                    title="LDA scores — Test set",
                )
                fig_te.update_xaxes(
                    zeroline=True, zerolinecolor="black"
                )
                fig_te.update_yaxes(
                    zeroline=True, zerolinecolor="black"
                )
                st.plotly_chart(fig_te, use_container_width=True)

        # Prediction probabilities
        if tres["proba"] is not None:
            st.subheader("Prediction probabilities")
            proba_df = pd.DataFrame(
                tres["proba"],
                columns=[
                    f"P({c})" for c in tres["classes"]
                ],
                index=tres["index"],
            )
            proba_df.insert(0, "True", y_true_te.values)
            proba_df.insert(1, "Predicted", y_pred_te)
            st.dataframe(
                proba_df.round(4),
                use_container_width=True,
            )

        # Download test results
        st.divider()
        st.subheader("⬇️ Download test results")

        buf_te = io.BytesIO()
        with pd.ExcelWriter(
            buf_te, engine="openpyxl"
        ) as writer:
            pred_te_df = pd.DataFrame({
                "Sample": np.asarray(tres["index"]).astype(str),
                "True": y_true_te.values,
                "Predicted": y_pred_te,
            })
            pred_te_df.to_excel(
                writer, sheet_name="Test_Predictions", index=False
            )
            met_te.to_excel(
                writer, sheet_name="Test_Metrics", index=False
            )
            cm_te.to_excel(writer, sheet_name="Test_CM")

            if tres["proba"] is not None:
                proba_df.to_excel(
                    writer, sheet_name="Probabilities"
                )

            if tres["scores"] is not None:
                scores_te_df.to_excel(
                    writer, sheet_name="LDA_Scores_Test"
                )

        st.download_button(
            "⬇️ Download test results (Excel)",
            data=buf_te.getvalue(),
            file_name="LDA_QDA_test_results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="dl_test",
        )

    else:
        st.info(
            "Click **Evaluate on test set** to get predictions "
            "and metrics on the test data."
        )
