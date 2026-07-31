import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import savgol_filter
import io


# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Data Split",
    page_icon="✂️",
    layout="wide"
)
st.title("Train / Test Split")


# =====================================================
# HELPER: PREPROCESSING PIPELINE
# =====================================================
def apply_preprocessing(
    X: pd.DataFrame,
    missing_method: str,
    transformation: str,
    transformation_shift: float,
    scaling_method: str,
    apply_snv: bool,
    apply_savgol: bool,
    savgol_window,
    savgol_polyorder,
    fit_params: dict | None = None,
):
    """
    Apply the same preprocessing steps used in the Preprocessing page.
    If fit_params is None → fit on this data (training set) and return
                             both the processed matrix and the parameters.
    If fit_params is given → apply the already-fitted parameters (test set).
    """
    X_work = X.copy()
    params = {} if fit_params is None else fit_params.copy()

    # -------------------------------------------------
    # 1. Missing values
    # -------------------------------------------------
    if missing_method == "Drop rows":
        X_work = X_work.dropna()
    elif missing_method == "Mean":
        if fit_params is None:
            fill_values = X_work.mean(numeric_only=True)
            params["fill_values"] = fill_values
        else:
            fill_values = params["fill_values"]
        X_work = X_work.fillna(fill_values)
    elif missing_method == "Median":
        if fit_params is None:
            fill_values = X_work.median(numeric_only=True)
            params["fill_values"] = fill_values
        else:
            fill_values = params["fill_values"]
        X_work = X_work.fillna(fill_values)
    # "None" / "Keep as is" → do nothing

    # -------------------------------------------------
    # 2. Transformation
    # -------------------------------------------------
    shift = transformation_shift if fit_params is None else params.get(
        "transformation_shift", 0
    )
    if fit_params is None:
        params["transformation_shift"] = shift

    if transformation == "Log10":
        if fit_params is None:
            min_val = X_work.min().min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                params["transformation_shift"] = shift
        X_work = np.log10(X_work + shift)
    elif transformation == "Natural log":
        if fit_params is None:
            min_val = X_work.min().min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                params["transformation_shift"] = shift
        X_work = np.log(X_work + shift)
    elif transformation == "Square root":
        if fit_params is None:
            min_val = X_work.min().min()
            if min_val < 0:
                shift = abs(min_val)
                params["transformation_shift"] = shift
        X_work = np.sqrt(X_work + shift)

    # -------------------------------------------------
    # 3. Scaling
    # -------------------------------------------------
    if scaling_method == "Mean Centering":
        if fit_params is None:
            means = X_work.mean()
            params["means"] = means
        else:
            means = params["means"]
        X_work = X_work - means
    elif scaling_method == "Autoscaling":
        if fit_params is None:
            scaler = StandardScaler()
            X_work = pd.DataFrame(
                scaler.fit_transform(X_work),
                columns=X_work.columns,
                index=X_work.index,
            )
            params["scaler"] = scaler
        else:
            scaler = params["scaler"]
            X_work = pd.DataFrame(
                scaler.transform(X_work),
                columns=X_work.columns,
                index=X_work.index,
            )
    elif scaling_method == "Pareto":
        if fit_params is None:
            means = X_work.mean()
            stds = X_work.std(ddof=1).replace(0, 1)
            params["means"] = means
            params["stds"] = stds
        else:
            means = params["means"]
            stds = params["stds"]
        X_work = (X_work - means) / np.sqrt(stds)
    elif scaling_method == "MinMax":
        if fit_params is None:
            scaler = MinMaxScaler()
            X_work = pd.DataFrame(
                scaler.fit_transform(X_work),
                columns=X_work.columns,
                index=X_work.index,
            )
            params["scaler"] = scaler
        else:
            scaler = params["scaler"]
            X_work = pd.DataFrame(
                scaler.transform(X_work),
                columns=X_work.columns,
                index=X_work.index,
            )

    # -------------------------------------------------
    # 4. SNV
    # -------------------------------------------------
    if apply_snv:
        def snv(data):
            row_mean = np.mean(data, axis=1, keepdims=True)
            row_std = np.std(data, axis=1, keepdims=True, ddof=1)
            row_std[row_std == 0] = 1
            return (data - row_mean) / row_std

        X_work = pd.DataFrame(
            snv(X_work.values),
            columns=X_work.columns,
            index=X_work.index,
        )

    # -------------------------------------------------
    # 5. Savitzky-Golay
    # -------------------------------------------------
    if apply_savgol and savgol_window is not None and savgol_polyorder is not None:
        X_work = pd.DataFrame(
            savgol_filter(
                X_work.values,
                window_length=savgol_window,
                polyorder=savgol_polyorder,
                axis=1,
            ),
            columns=X_work.columns,
            index=X_work.index,
        )

    X_work = X_work.astype(np.float64)
    if fit_params is None:
        return X_work, params
    return X_work


def build_full_dataset(original_df, processed_X, selected_columns, keep_index):
    """Rebuild a full dataframe keeping non-X columns and the processed X."""
    result = original_df.loc[keep_index].copy()
    # Force selected X columns to float64 BEFORE assignment.
    # Otherwise pandas raises TypeError when original dtype is int64
    # and processed values are floats (scaling / SNV / etc.).
    for col in selected_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float64)
    common = result.index.intersection(processed_X.index)
    result.loc[common, selected_columns] = (
        processed_X.loc[common, selected_columns].to_numpy()
    )
    return result


def safe_read_excel(file, sheet_name=0):
    """Robust Excel reader that reduces unwanted None/NaN conversions."""
    try:
        df = pd.read_excel(
            file,
            sheet_name=sheet_name,
            engine="openpyxl",
            keep_default_na=False,   # do not treat string "None", "NA" etc. as NaN
            na_values=[""],          # only truly empty cells become NaN
        )
    except Exception:
        # fallback
        df = pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")
    return df


def safe_read_csv(file):
    """Robust CSV reader."""
    try:
        df = pd.read_csv(
            file,
            keep_default_na=False,
            na_values=[""],
        )
    except Exception:
        df = pd.read_csv(file)
    return df


# =====================================================
# PREPROCESSING UI (reusable)
# =====================================================
def preprocessing_ui(prefix: str, default_info: dict | None = None, selected_X: list | None = None):
    """
    Renders a complete preprocessing configuration UI.
    Returns a dict with the chosen parameters (or None if user chooses to skip).
    """
    st.subheader("Preprocessing options")

    apply_prep = st.checkbox(
        "Apply preprocessing (fit on train → transform test)",
        value=True if default_info is not None else False,
        key=f"{prefix}_apply_prep",
    )

    if not apply_prep:
        st.info("No preprocessing will be applied. Raw data will be used.")
        return None

    # Pre-fill from previous page if available
    di = default_info or {}

    col_a, col_b = st.columns(2)

    with col_a:
        missing_opts = ["None", "Drop rows", "Mean", "Median"]
        default_missing = di.get("missing", "None")
        if default_missing not in missing_opts:
            default_missing = "None"
        missing_method = st.selectbox(
            "Missing values",
            missing_opts,
            index=missing_opts.index(default_missing),
            key=f"{prefix}_missing",
        )

        transf_opts = ["None", "Log10", "Natural log", "Square root"]
        default_transf = di.get("transformation", "None")
        if default_transf not in transf_opts:
            default_transf = "None"
        transformation = st.selectbox(
            "Transformation",
            transf_opts,
            index=transf_opts.index(default_transf),
            key=f"{prefix}_transf",
        )

        transformation_shift = st.number_input(
            "Transformation shift (auto-adjusted if needed)",
            value=float(di.get("transformation_shift", 0.0)),
            step=0.1,
            key=f"{prefix}_shift",
        )

    with col_b:
        scaling_opts = ["None", "Mean Centering", "Autoscaling", "Pareto", "MinMax"]
        default_scaling = di.get("scaling", "Autoscaling")
        if default_scaling not in scaling_opts:
            default_scaling = "Autoscaling"
        scaling_method = st.selectbox(
            "Scaling",
            scaling_opts,
            index=scaling_opts.index(default_scaling),
            key=f"{prefix}_scaling",
        )

        apply_snv = st.checkbox(
            "SNV (Standard Normal Variate)",
            value=bool(di.get("snv", False)),
            key=f"{prefix}_snv",
        )

        apply_savgol = st.checkbox(
            "Savitzky-Golay filter",
            value=bool(di.get("savgol", False)),
            key=f"{prefix}_savgol",
        )

    savgol_window = None
    savgol_polyorder = None
    if apply_savgol:
        c1, c2 = st.columns(2)
        with c1:
            savgol_window = st.number_input(
                "SavGol window length (odd)",
                min_value=3,
                max_value=101,
                value=int(di.get("savgol_window") or 11),
                step=2,
                key=f"{prefix}_sg_window",
            )
        with c2:
            savgol_polyorder = st.number_input(
                "SavGol poly order",
                min_value=1,
                max_value=5,
                value=int(di.get("savgol_polyorder") or 2),
                key=f"{prefix}_sg_poly",
            )
        if savgol_window % 2 == 0:
            st.warning("Window length should be odd. It will be adjusted automatically.")
            savgol_window += 1

    info = {
        "variables": selected_X or di.get("variables", []),
        "missing": missing_method,
        "transformation": transformation,
        "transformation_shift": transformation_shift,
        "scaling": scaling_method,
        "snv": apply_snv,
        "savgol": apply_savgol,
        "savgol_window": savgol_window,
        "savgol_polyorder": savgol_polyorder,
    }
    return info


# =====================================================
# MODE SELECTION
# =====================================================
st.divider()
mode = st.radio(
    "Split mode",
    [
        "Split from current dataset",
        "Import separate Train & Test files",
    ],
    horizontal=True,
)


# =====================================================
# MODE 1 – SPLIT FROM CURRENT DATASET
# =====================================================
if mode == "Split from current dataset":
    if "dataset" not in st.session_state:
        st.warning("Please load a dataset first (Data Import).")
        st.stop()

    source_options = ["Raw dataset"]
    if "preprocessed_dataset" in st.session_state:
        source_options.append("Preprocessed dataset")

    source_choice = st.radio(
        "Source dataset",
        source_options,
        key="split_source_choice",
    )

    if source_choice == "Preprocessed dataset":
        df = st.session_state["preprocessed_dataset"].copy()
        st.info(
            "Using the already-preprocessed dataset. "
            "You can still choose additional preprocessing below if desired."
        )
    else:
        df = st.session_state["dataset"].copy()

    st.success(
        f"Dataset loaded: {df.shape[0]} samples × {df.shape[1]} variables"
    )

    # -------------------------------------------------
    # Variable detection
    # -------------------------------------------------
    numeric_variables = df.select_dtypes(include=np.number).columns.tolist()
    categorical_variables = df.select_dtypes(exclude=np.number).columns.tolist()

    discrete_numeric = []
    for col in numeric_variables:
        if df[col].nunique(dropna=True) <= 20:
            discrete_numeric.append(col)

    strat_candidates = sorted(list(set(categorical_variables + discrete_numeric)))

    # Sample ID
    sample_id = st.session_state.get("sample_id")
    if sample_id is not None and sample_id not in df.columns:
        sample_id = None

    # -------------------------------------------------
    # X variables
    # -------------------------------------------------
    st.divider()
    st.header("Predictor variables (X)")
    default_X = st.session_state.get("X_variables", numeric_variables)
    default_X = [x for x in default_X if x in numeric_variables]

    selected_X = st.multiselect(
        "Select predictor variables",
        numeric_variables,
        default=default_X if default_X else numeric_variables,
        key="split_X",
    )
    if len(selected_X) < 1:
        st.warning("Select at least one predictor variable.")
        st.stop()

    # -------------------------------------------------
    # Target / stratification variable
    # -------------------------------------------------
    st.divider()
    st.header("Target / Stratification variable")
    st.info(
        """
        Stratification requires a **categorical** or **discrete numerical**
        variable (few unique values). Continuous numeric targets are allowed
        for regression but cannot be used for stratified splitting.
        """
    )

    y_options = ["None"] + [c for c in strat_candidates if c not in selected_X]
    continuous_numeric = [
        c for c in numeric_variables
        if c not in selected_X and c not in discrete_numeric and c != sample_id
    ]
    y_options += continuous_numeric

    default_y_idx = 0
    saved_y = st.session_state.get("y_variable")
    if saved_y is not None and saved_y in y_options:
        default_y_idx = y_options.index(saved_y)

    y_variable = st.selectbox(
        "Target variable (y)",
        y_options,
        index=default_y_idx,
        key="split_y",
    )
    if y_variable == "None":
        y_variable = None

    can_stratify = y_variable is not None and y_variable in strat_candidates

    # -------------------------------------------------
    # SPLIT METHOD
    # -------------------------------------------------
    st.divider()
    st.header("Split method")

    split_method = st.radio(
        "How do you want to create the test set?",
        [
            "Random / Stratified",
            "Manual selection (by sample ID or index)",
            "By index range",
            "By Sample ID range",
        ],
        key="split_method",
    )

    test_idx = None
    train_idx = None
    use_stratify = False
    test_size = 0.2
    random_state = 42

    if split_method == "Random / Stratified":
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider(
                "Test set proportion",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="test_size",
            )
        with col2:
            random_state = st.number_input(
                "Random seed",
                min_value=0,
                max_value=99999,
                value=42,
                step=1,
                key="random_state",
            )
        with col3:
            use_stratify = st.checkbox(
                "Stratified split",
                value=can_stratify,
                disabled=not can_stratify,
                key="use_stratify",
            )
            if not can_stratify and y_variable is not None:
                st.caption(
                    "Stratification disabled: target is continuous or has too many unique values."
                )
            elif y_variable is None:
                st.caption("No target selected → random (non-stratified) split.")

    elif split_method == "Manual selection (by sample ID or index)":
        st.info(
            "Select the samples that should go into the **test set**. "
            "All remaining samples will form the training set."
        )
        if sample_id is not None and sample_id in df.columns:
            id_values = df[sample_id].astype(str).tolist()
            selected_test_ids = st.multiselect(
                f"Select Test samples by `{sample_id}`",
                options=id_values,
                key="manual_test_ids",
            )
            if selected_test_ids:
                mask = df[sample_id].astype(str).isin(selected_test_ids)
                test_idx = df.index[mask].tolist()
                train_idx = df.index[~mask].tolist()
        else:
            st.warning(
                "No Sample ID defined. Using row index instead. "
                "You can define a Sample ID in the Data Import page."
            )
            all_indices = df.index.tolist()
            selected_test_indices = st.multiselect(
                "Select Test samples by row index",
                options=all_indices,
                key="manual_test_indices",
            )
            if selected_test_indices:
                test_idx = selected_test_indices
                train_idx = [i for i in all_indices if i not in test_idx]

    elif split_method == "By index range":
        st.info("Define one or more inclusive index ranges for the test set.")
        n_ranges = st.number_input("Number of ranges", min_value=1, max_value=10, value=1, key="n_idx_ranges")
        ranges = []
        for i in range(int(n_ranges)):
            c1, c2 = st.columns(2)
            with c1:
                start = st.number_input(f"Range {i+1} – start index", value=0, key=f"idx_start_{i}")
            with c2:
                end = st.number_input(f"Range {i+1} – end index", value=min(10, len(df)-1), key=f"idx_end_{i}")
            ranges.append((int(start), int(end)))

        test_mask = pd.Series(False, index=df.index)
        for start, end in ranges:
            test_mask.loc[(df.index >= start) & (df.index <= end)] = True
        test_idx = df.index[test_mask].tolist()
        train_idx = df.index[~test_mask].tolist()
        st.write(f"Test samples selected: **{len(test_idx)}**")

    elif split_method == "By Sample ID range":
        if sample_id is None or sample_id not in df.columns:
            st.error("Sample ID is not defined or not present in the dataset.")
            st.stop()

        st.info(
            f"Define inclusive ranges on the column `{sample_id}`. "
            "Works best when Sample ID is numeric or sortable strings."
        )
        # Try to treat as numeric if possible
        id_series = df[sample_id]
        try:
            id_numeric = pd.to_numeric(id_series, errors="coerce")
            is_numeric_id = id_numeric.notna().all()
        except Exception:
            is_numeric_id = False

        n_ranges = st.number_input("Number of ID ranges", min_value=1, max_value=10, value=1, key="n_id_ranges")
        ranges = []
        for i in range(int(n_ranges)):
            c1, c2 = st.columns(2)
            with c1:
                start = st.text_input(f"Range {i+1} – start ID", value="", key=f"id_start_{i}")
            with c2:
                end = st.text_input(f"Range {i+1} – end ID", value="", key=f"id_end_{i}")
            ranges.append((start.strip(), end.strip()))

        test_mask = pd.Series(False, index=df.index)
        for start, end in ranges:
            if not start and not end:
                continue
            if is_numeric_id:
                try:
                    s = float(start) if start else -np.inf
                    e = float(end) if end else np.inf
                    test_mask |= (id_numeric >= s) & (id_numeric <= e)
                except ValueError:
                    st.warning(f"Could not convert range ({start}, {end}) to numbers.")
            else:
                # string comparison
                s = start if start else ""
                e = end if end else "zzzzzzzzzz"
                test_mask |= (id_series.astype(str) >= s) & (id_series.astype(str) <= e)

        test_idx = df.index[test_mask].tolist()
        train_idx = df.index[~test_mask].tolist()
        st.write(f"Test samples selected: **{len(test_idx)}**")

    # -------------------------------------------------
    # Preprocessing (user choice)
    # -------------------------------------------------
    st.divider()
    preprocessing_info = preprocessing_ui(
        prefix="internal",
        default_info=st.session_state.get("preprocessing_info"),
        selected_X=selected_X,
    )

    # -------------------------------------------------
    # Perform split
    # -------------------------------------------------
    st.divider()
    if st.button("Perform train/test split", type="primary", key="do_split"):
        work_df = df.copy()

        # Drop rows with missing target
        if y_variable is not None:
            work_df = work_df.dropna(subset=[y_variable])

        if split_method == "Random / Stratified":
            stratify_labels = None
            if use_stratify and y_variable is not None:
                stratify_labels = work_df[y_variable]
                class_counts = stratify_labels.value_counts()
                min_count = class_counts.min()
                if min_count < 2:
                    st.error(
                        f"Cannot perform stratified split: class "
                        f"'{class_counts.idxmin()}' has only {min_count} sample(s). "
                        "Each class needs at least 2 samples."
                    )
                    st.stop()

            indices = work_df.index.tolist()
            try:
                train_idx, test_idx = train_test_split(
                    indices,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=stratify_labels,
                )
            except ValueError as e:
                st.error(f"Split failed: {e}")
                st.stop()
        else:
            # Manual / range methods already computed train_idx / test_idx
            if test_idx is None or len(test_idx) == 0:
                st.error("No test samples selected. Please select at least one sample for the test set.")
                st.stop()
            if train_idx is None or len(train_idx) == 0:
                st.error("Training set would be empty. Adjust your selection.")
                st.stop()
            # Ensure indices still exist after possible dropna on y
            train_idx = [i for i in train_idx if i in work_df.index]
            test_idx = [i for i in test_idx if i in work_df.index]

        train_raw = work_df.loc[train_idx].copy()
        test_raw = work_df.loc[test_idx].copy()

        # Apply preprocessing if requested
        if preprocessing_info is not None:
            prep_vars = [v for v in selected_X if v in train_raw.columns]
            X_train_raw = train_raw[prep_vars]
            X_test_raw = test_raw[prep_vars]

            X_train_proc, fit_params = apply_preprocessing(
                X_train_raw,
                missing_method=preprocessing_info.get("missing", "None"),
                transformation=preprocessing_info.get("transformation", "None"),
                transformation_shift=preprocessing_info.get("transformation_shift", 0),
                scaling_method=preprocessing_info.get("scaling", "None"),
                apply_snv=preprocessing_info.get("snv", False),
                apply_savgol=preprocessing_info.get("savgol", False),
                savgol_window=preprocessing_info.get("savgol_window"),
                savgol_polyorder=preprocessing_info.get("savgol_polyorder"),
                fit_params=None,
            )
            train_proc_index = X_train_proc.index

            X_test_proc = apply_preprocessing(
                X_test_raw,
                missing_method=preprocessing_info.get("missing", "None"),
                transformation=preprocessing_info.get("transformation", "None"),
                transformation_shift=preprocessing_info.get("transformation_shift", 0),
                scaling_method=preprocessing_info.get("scaling", "None"),
                apply_snv=preprocessing_info.get("snv", False),
                apply_savgol=preprocessing_info.get("savgol", False),
                savgol_window=preprocessing_info.get("savgol_window"),
                savgol_polyorder=preprocessing_info.get("savgol_polyorder"),
                fit_params=fit_params,
            )

            train_df = build_full_dataset(train_raw, X_train_proc, prep_vars, train_proc_index)
            test_df = build_full_dataset(test_raw, X_test_proc, prep_vars, X_test_proc.index)

            st.session_state["split_preprocessing_params"] = fit_params
            st.session_state["split_preprocessing_info"] = preprocessing_info
        else:
            train_df = train_raw
            test_df = test_raw
            st.session_state.pop("split_preprocessing_params", None)
            st.session_state.pop("split_preprocessing_info", None)

        # Save to session
        st.session_state["train_dataset"] = train_df
        st.session_state["test_dataset"] = test_df
        st.session_state["train_indices"] = list(train_df.index)
        st.session_state["test_indices"] = list(test_df.index)
        st.session_state["split_source"] = "internal_split"
        st.session_state["X_variables"] = selected_X
        if y_variable is not None:
            st.session_state["y_variable"] = y_variable

        st.session_state["split_info"] = {
            "mode": "internal_split",
            "source": source_choice,
            "split_method": split_method,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "test_size": test_size if split_method == "Random / Stratified" else None,
            "random_state": int(random_state) if split_method == "Random / Stratified" else None,
            "stratified": bool(use_stratify),
            "y_variable": y_variable,
            "X_variables": selected_X,
            "preprocessing_applied": preprocessing_info is not None,
        }

        st.success(
            f"✅ Split completed! Train: {len(train_df)} samples | Test: {len(test_df)} samples"
        )
        st.rerun()


# =====================================================
# MODE 2 – IMPORT SEPARATE TRAIN & TEST FILES
# =====================================================
else:
    st.info(
        "Upload two separate Excel/CSV files. "
        "You will select the worksheet, X variables and y variable "
        "for each file. Preprocessing (if chosen) is fit on train and applied to test."
    )

    col_train, col_test = st.columns(2)

    with col_train:
        st.subheader("📘 Training set file")
        train_file = st.file_uploader(
            "Upload training file",
            type=["xlsx", "xls", "csv"],
            key="train_upload",
        )
        train_df_raw = None
        if train_file is not None:
            if train_file.name.lower().endswith((".xlsx", ".xls")):
                xl = pd.ExcelFile(train_file, engine="openpyxl")
                train_sheet = st.selectbox("Training sheet", xl.sheet_names, key="train_sheet")
                train_df_raw = safe_read_excel(train_file, sheet_name=train_sheet)
            else:
                train_df_raw = safe_read_csv(train_file)
            st.write(f"Loaded: {train_df_raw.shape[0]} × {train_df_raw.shape[1]}")
            st.dataframe(train_df_raw.head(3), use_container_width=True)

    with col_test:
        st.subheader("📙 Test set file")
        test_file = st.file_uploader(
            "Upload test file",
            type=["xlsx", "xls", "csv"],
            key="test_upload",
        )
        test_df_raw = None
        if test_file is not None:
            if test_file.name.lower().endswith((".xlsx", ".xls")):
                xl = pd.ExcelFile(test_file, engine="openpyxl")
                test_sheet = st.selectbox("Test sheet", xl.sheet_names, key="test_sheet")
                test_df_raw = safe_read_excel(test_file, sheet_name=test_sheet)
            else:
                test_df_raw = safe_read_csv(test_file)
            st.write(f"Loaded: {test_df_raw.shape[0]} × {test_df_raw.shape[1]}")
            st.dataframe(test_df_raw.head(3), use_container_width=True)

    if train_df_raw is None or test_df_raw is None:
        st.warning("Please upload both training and test files.")
        st.stop()

    # -------------------------------------------------
    # Common columns & variable selection
    # -------------------------------------------------
    st.divider()
    st.header("Variable selection")

    common_cols = sorted(list(set(train_df_raw.columns) & set(test_df_raw.columns)))
    if len(common_cols) == 0:
        st.error("No columns in common between the two files.")
        st.stop()

    train_numeric = train_df_raw.select_dtypes(include=np.number).columns.tolist()
    test_numeric = test_df_raw.select_dtypes(include=np.number).columns.tolist()
    common_numeric = sorted(list(set(train_numeric) & set(test_numeric)))

    if len(common_numeric) == 0:
        # Fallback: try to convert common columns that look numeric
        st.warning(
            "No common numeric columns detected automatically. "
            "Trying to coerce possible numeric columns..."
        )
        candidate = []
        for c in common_cols:
            try:
                pd.to_numeric(train_df_raw[c].dropna().head(20), errors="raise")
                candidate.append(c)
            except Exception:
                pass
        common_numeric = candidate

    if len(common_numeric) == 0:
        st.error("No common numeric columns found even after coercion attempt.")
        st.stop()

    sample_id_options = ["None"] + common_cols
    sample_id_imp = st.selectbox("Sample ID (optional)", sample_id_options, key="imp_sample_id")
    if sample_id_imp == "None":
        sample_id_imp = None

    default_X_imp = [
        c for c in st.session_state.get("X_variables", common_numeric)
        if c in common_numeric and c != sample_id_imp
    ]
    if not default_X_imp:
        default_X_imp = [c for c in common_numeric if c != sample_id_imp]

    selected_X_imp = st.multiselect(
        "Predictor variables (X)",
        [c for c in common_numeric if c != sample_id_imp],
        default=default_X_imp,
        key="imp_X",
    )
    if len(selected_X_imp) < 1:
        st.warning("Select at least one predictor.")
        st.stop()

    remaining = [c for c in common_cols if c not in selected_X_imp and c != sample_id_imp]
    y_options_imp = ["None"] + remaining
    default_y_imp = 0
    saved_y = st.session_state.get("y_variable")
    if saved_y is not None and saved_y in y_options_imp:
        default_y_imp = y_options_imp.index(saved_y)

    y_imp = st.selectbox("Target variable (y)", y_options_imp, index=default_y_imp, key="imp_y")
    if y_imp == "None":
        y_imp = None

    # -------------------------------------------------
    # Preprocessing (user choice)
    # -------------------------------------------------
    st.divider()
    preprocessing_info = preprocessing_ui(
        prefix="import",
        default_info=st.session_state.get("preprocessing_info"),
        selected_X=selected_X_imp,
    )

    # -------------------------------------------------
    # Confirm import
    # -------------------------------------------------
    st.divider()
    if st.button("Import & apply preprocessing", type="primary", key="do_import_split"):
        train_raw = train_df_raw.copy()
        test_raw = test_df_raw.copy()

        if preprocessing_info is not None:
            prep_vars = [v for v in selected_X_imp if v in train_raw.columns and v in test_raw.columns]
            X_train_raw = train_raw[prep_vars].apply(pd.to_numeric, errors="coerce")
            X_test_raw = test_raw[prep_vars].apply(pd.to_numeric, errors="coerce")

            X_train_proc, fit_params = apply_preprocessing(
                X_train_raw,
                missing_method=preprocessing_info.get("missing", "None"),
                transformation=preprocessing_info.get("transformation", "None"),
                transformation_shift=preprocessing_info.get("transformation_shift", 0),
                scaling_method=preprocessing_info.get("scaling", "None"),
                apply_snv=preprocessing_info.get("snv", False),
                apply_savgol=preprocessing_info.get("savgol", False),
                savgol_window=preprocessing_info.get("savgol_window"),
                savgol_polyorder=preprocessing_info.get("savgol_polyorder"),
                fit_params=None,
            )
            X_test_proc = apply_preprocessing(
                X_test_raw,
                missing_method=preprocessing_info.get("missing", "None"),
                transformation=preprocessing_info.get("transformation", "None"),
                transformation_shift=preprocessing_info.get("transformation_shift", 0),
                scaling_method=preprocessing_info.get("scaling", "None"),
                apply_snv=preprocessing_info.get("snv", False),
                apply_savgol=preprocessing_info.get("savgol", False),
                savgol_window=preprocessing_info.get("savgol_window"),
                savgol_polyorder=preprocessing_info.get("savgol_polyorder"),
                fit_params=fit_params,
            )

            train_df = build_full_dataset(train_raw, X_train_proc, prep_vars, X_train_proc.index)
            test_df = build_full_dataset(test_raw, X_test_proc, prep_vars, X_test_proc.index)

            st.session_state["split_preprocessing_params"] = fit_params
            st.session_state["split_preprocessing_info"] = preprocessing_info
        else:
            train_df = train_raw
            test_df = test_raw
            st.session_state.pop("split_preprocessing_params", None)
            st.session_state.pop("split_preprocessing_info", None)

        st.session_state["train_dataset"] = train_df
        st.session_state["test_dataset"] = test_df
        st.session_state["train_indices"] = list(train_df.index)
        st.session_state["test_indices"] = list(test_df.index)
        st.session_state["split_source"] = "external_files"
        st.session_state["X_variables"] = selected_X_imp
        st.session_state["sample_id"] = sample_id_imp
        if y_imp is not None:
            st.session_state["y_variable"] = y_imp

        st.session_state["split_info"] = {
            "mode": "external_files",
            "n_train": len(train_df),
            "n_test": len(test_df),
            "y_variable": y_imp,
            "X_variables": selected_X_imp,
            "sample_id": sample_id_imp,
            "preprocessing_applied": preprocessing_info is not None,
        }

        st.success(
            f"✅ Train & Test imported! Train: {len(train_df)} | Test: {len(test_df)}"
        )
        st.rerun()


# =====================================================
# RESULTS SECTION (common to both modes)
# =====================================================
if "train_dataset" in st.session_state and "test_dataset" in st.session_state:
    st.divider()
    st.header("Current split summary")

    train_df = st.session_state["train_dataset"]
    test_df = st.session_state["test_dataset"]
    info = st.session_state.get("split_info", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train samples", len(train_df))
    c2.metric("Test samples", len(test_df))
    c3.metric(
        "X variables",
        len(info.get("X_variables", st.session_state.get("X_variables", []))),
    )
    c4.metric("Stratified", "Yes" if info.get("stratified") else "No")

    # Class distribution
    y_var = info.get("y_variable") or st.session_state.get("y_variable")
    if y_var is not None and y_var in train_df.columns:
        st.subheader(f"Class distribution – `{y_var}`")
        train_counts = train_df[y_var].value_counts(dropna=False).rename("Train")
        test_counts = test_df[y_var].value_counts(dropna=False).rename("Test")
        dist = pd.concat([train_counts, test_counts], axis=1).fillna(0)
        dist["Train %"] = (dist["Train"] / dist["Train"].sum() * 100).round(1)
        dist["Test %"] = (dist["Test"] / dist["Test"].sum() * 100).round(1)
        st.dataframe(dist, use_container_width=True)

    # Preview
    tab_train, tab_test = st.tabs(["Train preview", "Test preview"])
    with tab_train:
        st.dataframe(train_df.head(10), use_container_width=True)
    with tab_test:
        st.dataframe(test_df.head(10), use_container_width=True)

    # Download
    st.divider()
    st.header("⬇️ Download split")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        train_df.to_excel(writer, sheet_name="Train", index=True)
        test_df.to_excel(writer, sheet_name="Test", index=True)

        summary_rows = [{"Item": k, "Value": str(v)} for k, v in info.items()]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Split_Info", index=False)

        prep_info = st.session_state.get("split_preprocessing_info")
        if prep_info is not None:
            prep_rows = [{"Step": k, "Value": str(v)} for k, v in prep_info.items()]
            pd.DataFrame(prep_rows).to_excel(writer, sheet_name="Preprocessing_Info", index=False)

    st.download_button(
        "⬇️ Download Train/Test (Excel)",
        data=buffer.getvalue(),
        file_name="train_test_split.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        key="download_split",
    )

    # Clear
    st.divider()
    if st.button("🗑 Clear current split", key="clear_split"):
        for k in [
            "train_dataset", "test_dataset", "train_indices", "test_indices",
            "split_info", "split_source",
            "split_preprocessing_params", "split_preprocessing_info",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Split cleared.")
        st.rerun()
else:
    st.divider()
    st.info(
        "No train/test split available yet. "
        "Perform a split or import external files above."
    )
