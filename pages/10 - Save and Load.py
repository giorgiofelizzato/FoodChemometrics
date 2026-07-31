import streamlit as st
import pandas as pd
import pickle
import io
from datetime import datetime


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Project Management",
    page_icon="💾",
    layout="wide",
)

st.title("Project Management")

st.markdown(
    """
Save your complete Food AI project into a single file
and reload it later to continue your analysis.
"""
)


# =====================================================
# PROJECT VERSION
# =====================================================

PROJECT_VERSION = "1.2"


# =====================================================
# KEYS TO SAVE / RESTORE
# =====================================================

# Flat session-state keys that are saved at the top level
# of the project dict (besides nested sections).
DATASET_KEYS = [
    "dataset",
    "raw_dataset",
    "preprocessed_dataset",
    "preprocessed_X",
]

SETUP_KEYS = [
    "sample_id",
    "X_variables",
    "y_variable",
    "group_variable",
]

PREPROCESSING_KEYS = [
    "preprocessing_info",
    "preprocessing_variables",
    "missing_method",
    "transformation_method",
    "scaling_method",
    "apply_snv",
    "apply_savgol",
    "savgol_window",
    "savgol_polyorder",
]

SPLIT_KEYS = [
    "train_dataset",
    "test_dataset",
    "train_indices",
    "test_indices",
    "split_info",
    "split_source",
    "split_preprocessing_params",
    "split_preprocessing_info",
]

DA_KEYS = [
    "da_model",
    "da_model_type",
    "da_X_vars",
    "da_y_var",
    "da_n_components",
    "da_train_idx",
    "da_data_source",
    "da_train_results",
    "da_test_results",
]

# Keys that should NOT be pickled (widgets, file handles, etc.)
# — everything else in the lists above is safe.


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def _safe_get(key, default=None):
    return st.session_state.get(key, default)


def build_project():
    """
    Collect all relevant information currently stored
    in Streamlit session state.
    """

    project = {
        # =================================================
        # METADATA
        # =================================================
        "metadata": {
            "project_version": PROJECT_VERSION,
            "saved_at": datetime.now().isoformat(),
            "app": "Food AI App",
        },

        # =================================================
        # DATASETS
        # =================================================
        "dataset": _safe_get("dataset"),
        "raw_dataset": _safe_get("raw_dataset"),
        "preprocessed_dataset": _safe_get("preprocessed_dataset"),
        "preprocessed_X": _safe_get("preprocessed_X"),

        # =================================================
        # PREPROCESSING INFORMATION (from Save button)
        # =================================================
        "preprocessing_info": _safe_get("preprocessing_info", {}),

        # =================================================
        # PREPROCESSING SETTINGS (widget values)
        # =================================================
        "preprocessing_settings": {
            "variables": _safe_get("preprocessing_variables", []),
            "missing_method": _safe_get("missing_method", "None"),
            "transformation_method": _safe_get(
                "transformation_method", "None"
            ),
            "scaling_method": _safe_get("scaling_method", "None"),
            "apply_snv": _safe_get("apply_snv", False),
            "apply_savgol": _safe_get("apply_savgol", False),
            "savgol_window": _safe_get("savgol_window"),
            "savgol_polyorder": _safe_get("savgol_polyorder"),
        },

        # =================================================
        # DATA SETUP
        # =================================================
        "data_setup": {
            "sample_id": _safe_get("sample_id"),
            "X_variables": _safe_get("X_variables", []),
            "y_variable": _safe_get("y_variable"),
            "group_variable": _safe_get("group_variable"),
        },

        # =================================================
        # TRAIN / TEST SPLIT
        # =================================================
        "split": {
            "train_dataset": _safe_get("train_dataset"),
            "test_dataset": _safe_get("test_dataset"),
            "train_indices": _safe_get("train_indices"),
            "test_indices": _safe_get("test_indices"),
            "split_info": _safe_get("split_info", {}),
            "split_source": _safe_get("split_source"),
            "split_preprocessing_params": _safe_get(
                "split_preprocessing_params"
            ),
            "split_preprocessing_info": _safe_get(
                "split_preprocessing_info"
            ),
        },

        # =================================================
        # DISCRIMINANT ANALYSIS (LDA / QDA)
        # =================================================
        "discriminant_analysis": {
            "da_model": _safe_get("da_model"),
            "da_model_type": _safe_get("da_model_type"),
            "da_X_vars": _safe_get("da_X_vars"),
            "da_y_var": _safe_get("da_y_var"),
            "da_n_components": _safe_get("da_n_components"),
            "da_train_idx": _safe_get("da_train_idx"),
            "da_data_source": _safe_get("da_data_source"),
            "da_train_results": _safe_get("da_train_results"),
            "da_test_results": _safe_get("da_test_results"),
        },

        # =================================================
        # FUTURE / GENERIC SECTIONS
        # =================================================
        "exploratory_analysis": _safe_get("exploratory_analysis", {}),
        "statistics": _safe_get("statistics", {}),
        "chemometrics": _safe_get("chemometrics", {}),
        "machine_learning": _safe_get("machine_learning", {}),
        "results": _safe_get("results", {}),
        "history": _safe_get("history", []),
        "current_file": _safe_get("current_file"),
        "current_sheet": _safe_get("current_sheet"),
    }

    return project


def restore_project(project):
    """
    Restore all project information into Streamlit session state.
    """

    # -------------------------------------------------
    # DATASETS
    # -------------------------------------------------
    for key in [
        "dataset",
        "raw_dataset",
        "preprocessed_dataset",
        "preprocessed_X",
    ]:
        val = project.get(key)
        if val is not None:
            st.session_state[key] = val

    # -------------------------------------------------
    # PREPROCESSING INFO
    # -------------------------------------------------
    st.session_state["preprocessing_info"] = project.get(
        "preprocessing_info", {}
    )

    # -------------------------------------------------
    # PREPROCESSING SETTINGS (widgets)
    # -------------------------------------------------
    prep_set = project.get("preprocessing_settings", {})
    st.session_state["preprocessing_variables"] = prep_set.get(
        "variables", []
    )
    st.session_state["missing_method"] = prep_set.get(
        "missing_method", "None"
    )
    st.session_state["transformation_method"] = prep_set.get(
        "transformation_method", "None"
    )
    st.session_state["scaling_method"] = prep_set.get(
        "scaling_method", "None"
    )
    st.session_state["apply_snv"] = prep_set.get("apply_snv", False)
    st.session_state["apply_savgol"] = prep_set.get(
        "apply_savgol", False
    )
    if prep_set.get("savgol_window") is not None:
        st.session_state["savgol_window"] = prep_set["savgol_window"]
    if prep_set.get("savgol_polyorder") is not None:
        st.session_state["savgol_polyorder"] = prep_set[
            "savgol_polyorder"
        ]

    # -------------------------------------------------
    # DATA SETUP
    # -------------------------------------------------
    setup = project.get("data_setup", {})
    st.session_state["sample_id"] = setup.get("sample_id")
    st.session_state["X_variables"] = setup.get("X_variables", [])
    st.session_state["y_variable"] = setup.get("y_variable")
    st.session_state["group_variable"] = setup.get("group_variable")

    # -------------------------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------------------------
    split = project.get("split", {})
    for key in SPLIT_KEYS:
        val = split.get(key)
        if val is not None:
            st.session_state[key] = val
        elif key in st.session_state:
            # Clear stale keys if not present in the project
            del st.session_state[key]

    # -------------------------------------------------
    # DISCRIMINANT ANALYSIS
    # -------------------------------------------------
    da = project.get("discriminant_analysis", {})
    for key in DA_KEYS:
        val = da.get(key)
        if val is not None:
            st.session_state[key] = val
        elif key in st.session_state:
            del st.session_state[key]

    # -------------------------------------------------
    # GENERIC / FUTURE SECTIONS
    # -------------------------------------------------
    for section in [
        "exploratory_analysis",
        "statistics",
        "chemometrics",
        "machine_learning",
        "results",
    ]:
        st.session_state[section] = project.get(section, {})

    if project.get("history") is not None:
        st.session_state["history"] = project["history"]
    if project.get("current_file") is not None:
        st.session_state["current_file"] = project["current_file"]
    if project.get("current_sheet") is not None:
        st.session_state["current_sheet"] = project["current_sheet"]


def _status_badge(ok, label_ok, label_missing):
    if ok:
        st.success(label_ok)
    else:
        st.caption(label_missing)


# =====================================================
# CURRENT PROJECT STATUS
# =====================================================

st.divider()
st.header("Current Project")

# --- Dataset ---
if "dataset" in st.session_state and isinstance(
    st.session_state["dataset"], pd.DataFrame
):
    df = st.session_state["dataset"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", df.shape[0])
    c2.metric("Variables", df.shape[1])
    c3.metric(
        "Predictors (X)",
        len(_safe_get("X_variables", [])),
    )
    c4.metric(
        "Target (y)",
        _safe_get("y_variable") or "—",
    )

    st.success("Dataset loaded.")
else:
    st.warning("No dataset is currently loaded.")

# --- Preprocessed ---
if "preprocessed_dataset" in st.session_state and isinstance(
    st.session_state["preprocessed_dataset"], pd.DataFrame
):
    pre = st.session_state["preprocessed_dataset"]
    st.info(
        f"Preprocessed dataset: "
        f"{pre.shape[0]} samples × {pre.shape[1]} variables"
    )

# --- Split ---
if (
    "train_dataset" in st.session_state
    and "test_dataset" in st.session_state
):
    n_tr = len(st.session_state["train_dataset"])
    n_te = len(st.session_state["test_dataset"])
    info = _safe_get("split_info", {})
    stratified = info.get("stratified", False)
    st.info(
        f"Train/Test split: {n_tr} train + {n_te} test"
        + (" (stratified)" if stratified else "")
    )

# --- DA model ---
if "da_model" in st.session_state:
    mtype = _safe_get("da_model_type", "?")
    st.info(f"Discriminant model fitted: **{mtype}**")


# =====================================================
# PROJECT CONFIGURATION SUMMARY
# =====================================================

st.divider()
st.header("Project configuration")

col1, col2 = st.columns(2)

with col1:
    st.write("**Sample ID:**", _safe_get("sample_id"))
    st.write("**Target variable (y):**", _safe_get("y_variable"))

with col2:
    st.write("**Grouping variable:**", _safe_get("group_variable"))
    st.write(
        "**Number of predictors:**",
        len(_safe_get("X_variables", [])),
    )


# =====================================================
# PREPROCESSING STATUS
# =====================================================

prep = _safe_get("preprocessing_info", {})

if prep:
    st.divider()
    st.subheader("Preprocessing")

    if "variables" in prep:
        st.write(
            "**Variables processed:**", len(prep["variables"])
        )

    prep_summary = {
        "Missing values": prep.get("missing", "None"),
        "Transformation": prep.get("transformation", "None"),
        "Scaling": prep.get("scaling", "None"),
        "SNV": "Yes" if prep.get("snv", False) else "No",
        "Savitzky-Golay": (
            "Yes" if prep.get("savgol", False) else "No"
        ),
    }

    st.dataframe(
        pd.DataFrame(
            list(prep_summary.items()),
            columns=["Step", "Method"],
        ),
        hide_index=True,
        use_container_width=True,
    )


# =====================================================
# SPLIT STATUS
# =====================================================

split_info = _safe_get("split_info", {})

if split_info:
    st.divider()
    st.subheader("Train / Test Split")

    rows = [{"Item": k, "Value": str(v)} for k, v in split_info.items()]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )


# =====================================================
# DA STATUS
# =====================================================

if "da_model" in st.session_state:
    st.divider()
    st.subheader("Discriminant Analysis")

    da_rows = [
        ("Model", _safe_get("da_model_type")),
        ("X variables", len(_safe_get("da_X_vars") or [])),
        ("Target", _safe_get("da_y_var")),
        ("Components", _safe_get("da_n_components")),
        ("Data source", _safe_get("da_data_source")),
        (
            "Train results",
            "Yes" if "da_train_results" in st.session_state else "No",
        ),
        (
            "Test results",
            "Yes" if "da_test_results" in st.session_state else "No",
        ),
    ]
    st.dataframe(
        pd.DataFrame(da_rows, columns=["Item", "Value"]),
        hide_index=True,
        use_container_width=True,
    )


# =====================================================
# SAVE PROJECT
# =====================================================

st.divider()
st.header("Save Project")

st.info(
    """
The project file (`.foodai`) contains:

- Original & preprocessed datasets  
- Data setup (Sample ID, X, y, group)  
- Preprocessing configuration  
- Train / Test split (if available)  
- LDA / QDA model and results (if fitted)  
"""
)

project_name = st.text_input(
    "Project name",
    value="Food_AI_Project",
)

project_name = project_name.strip().replace(" ", "_")
if not project_name:
    project_name = "Food_AI_Project"

project_data = build_project()

project_bytes = io.BytesIO()
try:
    pickle.dump(
        project_data,
        project_bytes,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    project_bytes.seek(0)

    st.download_button(
        label="⬇️ Download Project",
        data=project_bytes.getvalue(),
        file_name=f"{project_name}.foodai",
        mime="application/octet-stream",
        use_container_width=True,
        key="download_foodai_project",
    )
except Exception as e:
    st.error(f"Unable to create project file: {e}")


# =====================================================
# LOAD PROJECT
# =====================================================

st.divider()
st.header("Load Project")

uploaded_project = st.file_uploader(
    "Upload Food AI Project",
    type=["foodai", "pkl"],
    key="project_uploader",
)

if uploaded_project is not None:
    try:
        loaded_project = pickle.load(uploaded_project)

        if not isinstance(loaded_project, dict):
            st.error("Invalid project file.")
            st.stop()

        if "metadata" not in loaded_project:
            st.error(
                "Invalid Food AI project file: "
                "metadata section not found."
            )
            st.stop()

        metadata = loaded_project.get("metadata", {})

        st.success("Project file successfully loaded.")

        col1, col2 = st.columns(2)
        with col1:
            st.write(
                "**Project version:**",
                metadata.get("project_version", "Unknown"),
            )
        with col2:
            st.write(
                "**Saved at:**",
                metadata.get("saved_at", "Unknown"),
            )

        # Dataset preview
        loaded_df = loaded_project.get("dataset")
        if isinstance(loaded_df, pd.DataFrame):
            st.write(
                "**Dataset:**",
                f"{loaded_df.shape[0]} samples × "
                f"{loaded_df.shape[1]} variables",
            )
            st.dataframe(
                loaded_df.head(),
                use_container_width=True,
            )
        else:
            st.warning("No valid dataset found in this project.")

        # Preprocessed
        loaded_pre = loaded_project.get("preprocessed_dataset")
        if isinstance(loaded_pre, pd.DataFrame):
            st.write(
                "**Preprocessed dataset:**",
                f"{loaded_pre.shape[0]} × {loaded_pre.shape[1]}",
            )

        # Split
        split_sec = loaded_project.get("split", {})
        if split_sec.get("train_dataset") is not None:
            n_tr = len(split_sec["train_dataset"])
            n_te = (
                len(split_sec["test_dataset"])
                if split_sec.get("test_dataset") is not None
                else 0
            )
            st.write(
                "**Train / Test split:**",
                f"{n_tr} train + {n_te} test",
            )

        # DA
        da_sec = loaded_project.get("discriminant_analysis", {})
        if da_sec.get("da_model") is not None:
            st.write(
                "**Discriminant model:**",
                da_sec.get("da_model_type", "fitted"),
            )

        # Restore button
        if st.button(
            "Restore Project",
            type="primary",
            use_container_width=True,
            key="restore_project_button",
        ):
            restore_project(loaded_project)
            st.success("Project successfully restored!")
            st.rerun()

    except Exception as e:
        st.error(
            f"Unable to load project: {type(e).__name__}: {e}"
        )


# =====================================================
# CURRENT X VARIABLES
# =====================================================

if "X_variables" in st.session_state:
    st.divider()
    st.subheader("Current Predictor Variables")

    X_variables = _safe_get("X_variables", [])

    if X_variables:
        st.dataframe(
            pd.DataFrame(X_variables, columns=["Variable"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No predictor variables selected.")
