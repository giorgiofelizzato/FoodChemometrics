import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
    page_title="LDA Binary (2 classes)",
    page_icon="📐",
    layout="wide",
)

st.title("Linear Discriminant Analysis — Binary (2 classes)")


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


def build_scores_df(scores_array, y, index):
    """Build scores DataFrame for binary LDA (always 1 component → LD1)."""
    sdf = pd.DataFrame(
        scores_array,
        columns=["LD1"],
        index=index,
    )
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
    key="lda_bin_data_source",
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
    key="lda_bin_X",
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
    key="lda_bin_y",
)

# --- Enforce exactly 2 classes ---
n_classes_check = df[y_variable].nunique(dropna=True)
if n_classes_check != 2:
    st.error(
        f"This page is for **binary classification only** (exactly 2 classes). "
        f"The selected target **{y_variable}** has **{n_classes_check}** classes. "
        "Use the standard LDA / QDA page for multiclass problems."
    )
    st.stop()


# =====================================================
# MODEL SETTINGS
# =====================================================

st.divider()
st.header("Model settings")
st.info(
    "**Binary LDA:** exactly 2 classes → only **1 discriminant axis** (LD1). "
    "This page is the recommended alternative to Fisher’s Linear Discriminant (FLD) "
    "when you want the classical LDA formulation with one component."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Model:** LDA (binary)")
    n_components = 1  # fixed for binary

with col2:
    solver = st.selectbox(
        "LDA solver",
        ["svd", "eigen"],
        key="lda_bin_solver",
    )


# =====================================================
# CROSS-VALIDATION SETTINGS
# =====================================================

st.divider()
st.header("Cross-validation (on training set)")

enable_cv = st.checkbox(
    "Enable stratified cross-validation",
    value=True,
    key="lda_bin_enable_cv",
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
            key="lda_bin_cv_folds",
        )
    with c2:
        cv_seed = st.number_input(
            "CV random seed",
            min_value=0,
            max_value=99999,
            value=42,
            key="lda_bin_cv_seed",
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
        X_train, y_train, train_idx = prepare_xy(
            df, selected_X, y_variable
        )

    # Final safety check on the actual training labels
    n_classes_train = y_train.nunique()
    if n_classes_train != 2:
        st.error(
            f"Training target has {n_classes_train} classes after cleaning. "
            "This page requires exactly 2 classes."
        )
        st.stop()

    st.write(
        f"**Samples:** {X_train.shape[0]}  |  "
        f"**Variables:** {X_train.shape[1]}  |  "
        f"**Classes:** {n_classes_train}"
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
        key="lda_bin_fit",
    ):

        # Check class sizes
        counts = y_train.value_counts()
        if counts.min() < 2:
            st.error(
                f"Class '{counts.idxmin()}' has fewer than 2 samples. "
                "Cannot fit the model."
            )
            st.stop()

        # Instantiate binary LDA (always 1 component)
        model = LDA(
            n_components=1,
            solver=solver,
        )

        # Fit
        model.fit(X_train, y_train)

        # Predictions on training
        y_pred_train = model.predict(X_train)

        # Scores (always 1-D)
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
                    LDA(n_components=1, solver=solver),
                    X_train,
                    y_train,
                    cv=skf,
                )
                cv_acc_scores = cross_val_score(
                    LDA(n_components=1, solver=solver),
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
        # (ONLY keys that are NOT used by widgets)
        st.session_state["lda_bin_model"] = model
        st.session_state["lda_bin_X_vars"] = selected_X
        st.session_state["lda_bin_y_var"] = y_variable
        st.session_state["lda_bin_n_components"] = 1
        st.session_state["lda_bin_train_idx"] = list(train_idx)

        st.session_state["lda_bin_train_results"] = {
            "y_true": y_train,
            "y_pred": y_pred_train,
            "scores": scores_train,
            "index": train_idx,
            "cv": cv_results,
            "X": X_train,
        }

        # Clear previous test results
        if "lda_bin_test_results" in st.session_state:
            del st.session_state["lda_bin_test_results"]

        st.success("✅ Model fitted successfully!")
        st.rerun()

    # ---- Display training results if available ----
    if (
        "lda_bin_train_results" in st.session_state
        and st.session_state.get("lda_bin_X_vars") == selected_X
        and st.session_state.get("lda_bin_y_var") == y_variable
    ):

        res = st.session_state["lda_bin_train_results"]
        model = st.session_state["lda_bin_model"]
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
        # Binary LDA plots (1 component)
        # =============================================
        st.divider()
        st.subheader("LDA scores & loadings (binary)")

        scores_df = build_scores_df(
            res["scores"],
            y_true,
            res["index"],
        )

        if hasattr(model, "explained_variance_ratio_"):
            ev = model.explained_variance_ratio_ * 100
            ev_df = pd.DataFrame({
                "Axis": ["LD1"],
                "Explained variance (%)": np.round(ev, 2),
                "Cumulative (%)": np.round(np.cumsum(ev), 2),
            })
            st.dataframe(
                ev_df, hide_index=True, use_container_width=True
            )

        # --- Score plot: LD1 vs Class (scatter) ---
        hover = None
        if sample_id is not None and sample_id in df.columns:
            scores_df[sample_id] = df.loc[scores_df.index, sample_id]
            hover = sample_id

        fig_sc = px.scatter(
            scores_df,
            x="LD1",
            y="Class",
            color="Class",
            hover_name=hover if hover else scores_df.index.astype(str),
            hover_data={
                "LD1": ":.3f",
                "Class": True,
            },
            title="LDA score plot (training)",
            labels={
                "LD1": "LD1",
                "Class": "Class",
            },
        )
        fig_sc.update_traces(marker=dict(size=10))
        fig_sc.update_xaxes(zeroline=True, zerolinecolor="black")
        st.plotly_chart(fig_sc, use_container_width=True)

        # --- Box plot of LD1 by Class ---
        fig_box = px.box(
            scores_df,
            x="Class",
            y="LD1",
            color="Class",
            points="all",
            hover_name=hover if hover else scores_df.index.astype(str),
            hover_data={
                "LD1": ":.3f",
                "Class": True,
            },
            title="Distribution of LDA scores",
            labels={
                "LD1": "LD1",
                "Class": "Class",
            },
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # --- Loadings (coefficients) ---
        st.subheader("Loadings (coefficients)")

        if hasattr(model, "scalings_") and model.scalings_ is not None:
            load_vals = model.scalings_[:, 0]
        else:
            load_vals = model.coef_[0]

        load_df = pd.DataFrame({
            "Variable": selected_X,
            "LD1": load_vals,
        })

        st.dataframe(
            load_df.round(4),
            hide_index=True,
            use_container_width=True,
        )

        top_n = min(20, len(load_df))
        top_loadings = (
            load_df.loc[load_df["LD1"].abs().nlargest(top_n).index]
            .sort_values("LD1")
        )

        fig_bar = px.bar(
            top_loadings,
            x="LD1",
            y="Variable",
            orientation="h",
            hover_name="Variable",
            hover_data={"LD1": ":.3f"},
            title="LDA coefficients (top by |loading|)",
            labels={
                "LD1": "LD1 coefficient",
                "Variable": "Variable",
            },
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # =============================================
        # Download training results
        # =============================================
        st.divider()
        st.subheader("⬇️ Download training results")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pred_df = pd.DataFrame({
                "Sample": res["index"].astype(str),
                "True": y_true.values,
                "Predicted": y_pred,
            })
            pred_df.to_excel(
                writer, sheet_name="Train_Predictions", index=False
            )

            met_df.to_excel(
                writer, sheet_name="Train_Metrics", index=False
            )

            cm_df.to_excel(writer, sheet_name="Train_CM")

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

            scores_df.to_excel(
                writer, sheet_name="LDA_Scores"
            )

            load_df.to_excel(
                writer, sheet_name="Loadings", index=False
            )

        st.download_button(
            "⬇️ Download training results (Excel)",
            data=buf.getvalue(),
            file_name="LDA_Binary_training_results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="dl_train_bin",
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

    if "lda_bin_model" not in st.session_state:
        st.warning(
            "Fit a model in the **Training** tab first."
        )
        st.stop()

    model = st.session_state["lda_bin_model"]
    X_vars_fit = st.session_state["lda_bin_X_vars"]
    y_var_fit = st.session_state["lda_bin_y_var"]

    # ---- build test data ----
    if use_external_split:
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

        from sklearn.model_selection import train_test_split

        holdout = st.slider(
            "Hold-out proportion for quick test",
            0.1, 0.5, 0.2, 0.05,
            key="lda_bin_holdout",
        )
        seed_ho = st.number_input(
            "Hold-out seed", 0, 99999, 42, key="lda_bin_ho_seed"
        )

        X_all, y_all, all_idx = prepare_xy(
            df, X_vars_fit, y_var_fit
        )

        train_idx_used = set(
            st.session_state.get("lda_bin_train_idx", [])
        )

        if train_idx_used and set(all_idx).issubset(train_idx_used):
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

    n_classes_test = y_test.nunique()
    if n_classes_test > 2:
        st.warning(
            f"Test set has {n_classes_test} classes. "
            "Predictions will still be generated, but metrics may be less meaningful."
        )

    st.write(
        f"**Test samples:** {X_test.shape[0]}  |  "
        f"**Variables:** {X_test.shape[1]}"
    )

    if st.button(
        "📊 Evaluate on test set",
        type="primary",
        key="lda_bin_eval_test",
    ):

        y_pred_test = model.predict(X_test)

        scores_test = None
        try:
            scores_test = model.transform(X_test)
        except Exception:
            scores_test = None

        proba_test = None
        if hasattr(model, "predict_proba"):
            try:
                proba_test = model.predict_proba(X_test)
            except Exception:
                proba_test = None

        st.session_state["lda_bin_test_results"] = {
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
    if "lda_bin_test_results" in st.session_state:

        tres = st.session_state["lda_bin_test_results"]
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

        with st.expander("Classification report (text)"):
            st.text(
                classification_report(
                    y_true_te,
                    y_pred_te,
                    digits=3,
                )
            )

        if tres["scores"] is not None:
            scores_te_df = build_scores_df(
                tres["scores"],
                y_true_te,
                tres["index"],
            )
            scores_te_df["Predicted"] = y_pred_te

            fig_te = px.scatter(
                scores_te_df,
                x="LD1",
                y="Class",
                color="Class",
                symbol="Predicted",
                title="LDA scores — Test set",
                labels={
                    "LD1": "LD1",
                    "Class": "Class",
                },
            )
            fig_te.update_traces(marker=dict(size=10))
            fig_te.update_xaxes(zeroline=True, zerolinecolor="black")
            st.plotly_chart(fig_te, use_container_width=True)

            fig_box_te = px.box(
                scores_te_df,
                x="Class",
                y="LD1",
                color="Class",
                points="all",
                title="Distribution of LDA scores — Test",
                labels={
                    "LD1": "LD1",
                    "Class": "Class",
                },
            )
            st.plotly_chart(fig_box_te, use_container_width=True)

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
            file_name="LDA_Binary_test_results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="dl_test_bin",
        )

    else:
        st.info(
            "Click **Evaluate on test set** to get predictions "
            "and metrics on the test data."
        )