import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import savgol_filter
import io


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Preprocessing",
    page_icon="⚙️",
    layout="wide"
)

st.title("Data Preprocessing")


# =====================================================
# CHECK DATASET
# =====================================================

if "dataset" not in st.session_state:

    st.warning(
        "Please load a dataset first."
    )

    st.stop()


df = st.session_state[
    "dataset"
].copy()


# =====================================================
# SELECT NUMERIC VARIABLES
# =====================================================

numeric_variables = (
    df
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)


if len(numeric_variables) == 0:

    st.error(
        "No numeric variables were found in the dataset."
    )

    st.stop()


# =====================================================
# DEFAULT VARIABLES
# =====================================================

if (
    "X_variables" in st.session_state
    and st.session_state["X_variables"]
):

    default_vars = [

        x

        for x in st.session_state[
            "X_variables"
        ]

        if x in numeric_variables

    ]

else:

    default_vars = numeric_variables


selected_vars = st.multiselect(

    "Variables to preprocess",

    options=numeric_variables,

    default=default_vars,

    key="preprocessing_variables"

)


if len(selected_vars) == 0:

    st.warning(
        "Select at least one variable."
    )

    st.stop()


# =====================================================
# ORIGINAL DATA
# =====================================================

X_original = df[
    selected_vars
].copy()


# =====================================================
# 1. MISSING VALUES
# =====================================================

st.divider()

st.header(
    "1. Missing values"
)


missing_method = st.selectbox(

    "Handling strategy",

    [

        "None",

        "Drop rows",

        "Mean",

        "Median"

    ],

    key="missing_method"

)


X = X_original.copy()


if missing_method == "Drop rows":

    X = X.dropna()


elif missing_method == "Mean":

    X = X.fillna(
        X.mean()
    )


elif missing_method == "Median":

    X = X.fillna(
        X.median()
    )


# Check remaining missing values

if X.isna().sum().sum() > 0:

    st.warning(

        "Missing values are still present. "
        "Some preprocessing methods may fail."

    )


# =====================================================
# 2. DATA TRANSFORMATION
# =====================================================

st.divider()

st.header(
    "2. Data transformation"
)


transformation = st.selectbox(

    "Transformation",

    [

        "None",

        "Log10",

        "Natural log",

        "Square root"

    ],

    key="transformation_method"

)


X_transf = X.copy()


transformation_shift = 0


# -----------------------------------------------------
# LOG10
# -----------------------------------------------------

if transformation == "Log10":

    min_val = X_transf.min().min()


    if min_val <= 0:

        transformation_shift = (
            abs(min_val) + 1
        )


    X_transf = np.log10(

        X_transf
        + transformation_shift

    )


    st.info(

        f"Log10 transformation applied. "
        f"Shift = {transformation_shift:.4f}"

    )


# -----------------------------------------------------
# NATURAL LOG
# -----------------------------------------------------

elif transformation == "Natural log":

    min_val = X_transf.min().min()


    if min_val <= 0:

        transformation_shift = (
            abs(min_val) + 1
        )


    X_transf = np.log(

        X_transf
        + transformation_shift

    )


    st.info(

        f"Natural log transformation applied. "
        f"Shift = {transformation_shift:.4f}"

    )


# -----------------------------------------------------
# SQUARE ROOT
# -----------------------------------------------------

elif transformation == "Square root":

    min_val = X_transf.min().min()


    if min_val < 0:

        transformation_shift = (
            abs(min_val)
        )


    X_transf = np.sqrt(

        X_transf
        + transformation_shift

    )


    if transformation_shift > 0:

        st.info(

            f"Square root transformation applied. "
            f"Shift = {transformation_shift:.4f}"

        )


# =====================================================
# 3. SCALING
# =====================================================

st.divider()

st.header(
    "3. Scaling"
)


scaling_method = st.selectbox(

    "Scaling method",

    [

        "None",

        "Mean Centering",

        "Autoscaling",

        "Pareto",

        "MinMax"

    ],

    key="scaling_method"

)


X_scaled = X_transf.copy()


# -----------------------------------------------------
# MEAN CENTERING
# -----------------------------------------------------

if scaling_method == "Mean Centering":

    X_scaled = (

        X_transf
        - X_transf.mean()

    )


# -----------------------------------------------------
# AUTOSCALING
# -----------------------------------------------------

elif scaling_method == "Autoscaling":

    scaler = StandardScaler()


    X_scaled = pd.DataFrame(

        scaler.fit_transform(
            X_transf
        ),

        columns=X_transf.columns,

        index=X_transf.index

    )


# -----------------------------------------------------
# PARETO
# -----------------------------------------------------

elif scaling_method == "Pareto":

    mean = X_transf.mean()

    std = X_transf.std(
        ddof=1
    )


    # Avoid division by zero

    std = std.replace(
        0,
        1
    )


    X_scaled = (

        X_transf
        - mean

    ) / np.sqrt(std)


# -----------------------------------------------------
# MINMAX
# -----------------------------------------------------

elif scaling_method == "MinMax":

    scaler = MinMaxScaler()


    X_scaled = pd.DataFrame(

        scaler.fit_transform(
            X_transf
        ),

        columns=X_transf.columns,

        index=X_transf.index

    )


# =====================================================
# 4. SNV
# =====================================================

st.divider()

st.header(
    "4. SNV (Standard Normal Variate)"
)


apply_snv = st.checkbox(

    "Apply SNV",

    key="apply_snv"

)


def snv(input_data):

    row_mean = np.mean(

        input_data,

        axis=1,

        keepdims=True

    )


    row_std = np.std(

        input_data,

        axis=1,

        keepdims=True,

        ddof=1

    )


    # Avoid division by zero

    row_std[
        row_std == 0
    ] = 1


    return (

        input_data
        - row_mean

    ) / row_std


if apply_snv:

    X_scaled = pd.DataFrame(

        snv(
            X_scaled.values
        ),

        columns=X_scaled.columns,

        index=X_scaled.index

    )


# =====================================================
# 5. SAVITZKY-GOLAY
# =====================================================

st.divider()

st.header(
    "5. Savitzky-Golay smoothing"
)


apply_savgol = st.checkbox(

    "Apply Savitzky-Golay",

    key="apply_savgol"

)


window = None

polyorder = None


if apply_savgol:

    n_variables = X_scaled.shape[1]


    # Maximum odd window

    max_window = min(

        31,

        n_variables
        if n_variables % 2 == 1

        else n_variables - 1

    )


    if max_window < 3:

        st.error(

            "Savitzky-Golay smoothing requires "
            "at least 3 numeric variables."

        )

        st.stop()


    window = st.slider(

        "Window length",

        min_value=3,

        max_value=max_window,

        value=min(
            7,
            max_window
        ),

        step=2,

        key="savgol_window"

    )


    max_polyorder = min(

        5,

        window - 1

    )


    polyorder = st.slider(

        "Polynomial order",

        min_value=1,

        max_value=max_polyorder,

        value=min(
            2,
            max_polyorder
        ),

        key="savgol_polyorder"

    )


    X_scaled = pd.DataFrame(

        savgol_filter(

            X_scaled.values,

            window_length=window,

            polyorder=polyorder,

            axis=1

        ),

        columns=X_scaled.columns,

        index=X_scaled.index

    )


# =====================================================
# ENSURE NUMERIC FLOAT DATA
# =====================================================

X_scaled = X_scaled.astype(
    np.float64
)


# =====================================================
# REMOVE INF / NAN FOR VISUALIZATION
# =====================================================

X_plot = (

    X_scaled

    .replace(

        [

            np.inf,

            -np.inf

        ],

        np.nan

    )

    .dropna()

)


# =====================================================
# 6. PREVIEW
# =====================================================

st.divider()

st.header(
    "6. Preview"
)


col1, col2 = st.columns(
    2
)


with col1:

    st.subheader(
        "Original"
    )


    st.dataframe(

        X_original.head(),

        use_container_width=True

    )


with col2:

    st.subheader(
        "Preprocessed"
    )


    st.dataframe(

        X_scaled.head(),

        use_container_width=True

    )


# =====================================================
# 7. PARALLEL COORDINATES
# =====================================================

st.divider()

st.header(
    "7. Parallel Coordinates"
)


if X_plot.empty:

    st.warning(

        "The parallel coordinates plot cannot be generated "
        "because there are no valid rows after preprocessing."

    )


else:

    tab1, tab2 = st.tabs(

        [

            "Original",

            "Preprocessed"

        ]

    )


    # -------------------------------------------------
    # ORIGINAL
    # -------------------------------------------------

    with tab1:

        X_original_plot = (

            X_original

            .replace(

                [

                    np.inf,

                    -np.inf

                ],

                np.nan

            )

            .dropna()

        )


        if X_original_plot.empty:

            st.warning(

                "No valid data available for the original plot."

            )


        else:

            fig_original = px.parallel_coordinates(

                X_original_plot
                .reset_index(drop=True)

            )


            st.plotly_chart(

                fig_original,

                use_container_width=True

            )


    # -------------------------------------------------
    # PREPROCESSED
    # -------------------------------------------------

    with tab2:

        fig_preprocessed = px.parallel_coordinates(

            X_plot
            .reset_index(drop=True)

        )


        st.plotly_chart(

            fig_preprocessed,

            use_container_width=True

        )


# =====================================================
# 8. PREPROCESSING SUMMARY
# =====================================================

st.divider()

st.header(
    "8. Preprocessing summary"
)


summary = pd.DataFrame({

    "Step": [

        "Missing values",

        "Transformation",

        "Scaling",

        "SNV",

        "Savitzky-Golay"

    ],

    "Method": [

        missing_method,

        transformation,

        scaling_method,

        "Yes"
        if apply_snv
        else "No",

        "Yes"
        if apply_savgol
        else "No"

    ]

})


st.dataframe(

    summary,

    hide_index=True,

    use_container_width=True

)


# =====================================================
# FUNCTION TO CREATE FULL PREPROCESSED DATASET
# =====================================================

def build_preprocessed_dataset(

    original_df,

    processed_X,

    selected_columns,

    missing_method

):

    # Copy original dataset

    result_df = original_df.copy()


    # IMPORTANT:
    # Convert selected columns to float
    # before assigning processed values.
    # This prevents int64 -> float assignment errors.

    result_df[
        selected_columns
    ] = (

        result_df[
            selected_columns
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )

        .astype(
            np.float64
        )

    )


    # Find common indexes

    common_index = (

        result_df.index

        .intersection(
            processed_X.index
        )

    )


    # Assign processed values

    result_df.loc[

        common_index,

        selected_columns

    ] = (

        processed_X

        .loc[

            common_index,

            selected_columns

        ]

        .to_numpy()

    )


    # If rows were dropped,
    # remove them from the final dataset

    if missing_method == "Drop rows":

        result_df = (

            result_df

            .loc[
                processed_X.index
            ]

            .copy()

        )


    return result_df


# =====================================================
# CREATE PREPROCESSED DATASET
# =====================================================

preprocessed_df = build_preprocessed_dataset(

    original_df=df,

    processed_X=X_scaled,

    selected_columns=selected_vars,

    missing_method=missing_method

)


# =====================================================
# 9. SAVE PREPROCESSED DATASET
# =====================================================
st.divider()
st.header(
    "9. Save preprocessing - Mandatory step before moving to the next pages"
)

if st.button(
    "💾 Save preprocessed dataset",
    type="primary",
    key="save_preprocessed_dataset"
):
    # Full preprocessed dataset
    st.session_state["preprocessed_dataset"] = preprocessed_df.copy()

    # Only the scaled X matrix
    st.session_state["preprocessed_X"] = X_scaled.copy()

    # IMPORTANT: make downstream pages use only the preprocessed variables
    st.session_state["X_variables"] = list(selected_vars)

    # Preprocessing metadata
    st.session_state["preprocessing_info"] = {
        "variables": list(selected_vars),
        "missing": missing_method,
        "transformation": transformation,
        "transformation_shift": transformation_shift,
        "scaling": scaling_method,
        "snv": apply_snv,
        "savgol": apply_savgol,
        "savgol_window": window,
        "savgol_polyorder": polyorder
    }

    # Clear any previous split (indices may no longer match)
    for k in [
        "train_dataset", "test_dataset",
        "train_indices", "test_indices",
        "split_info", "split_source",
        "da_model", "da_scaler", "da_train_results", "da_test_results"
    ]:
        if k in st.session_state:
            del st.session_state[k]

    st.success(
        f"✅ Preprocessed dataset saved successfully! "
        f"({len(selected_vars)} variables, scaling = {scaling_method})"
    )

# =====================================================
# 10. DOWNLOAD PREPROCESSED DATA
# =====================================================

st.divider()

st.header(
    "10. Download preprocessed data"
)


# Create Excel buffer

buffer = io.BytesIO()


with pd.ExcelWriter(

    buffer,

    engine="openpyxl"

) as writer:


    # Sheet 1:
    # Only preprocessed variables

    X_scaled.to_excel(

        writer,

        sheet_name="Preprocessed_X",

        index=True

    )


    # Sheet 2:
    # Complete dataset

    preprocessed_df.to_excel(

        writer,

        sheet_name="Full_Dataset",

        index=True

    )


    # Sheet 3:
    # Preprocessing summary

    summary.to_excel(

        writer,

        sheet_name="Preprocessing_Summary",

        index=False

    )


st.download_button(

    label="⬇️ Download preprocessed data (Excel)",

    data=buffer.getvalue(),

    file_name="preprocessed_data.xlsx",

    mime=(

        "application/vnd.openxmlformats-officedocument."

        "spreadsheetml.sheet"

    ),

    type="primary",

    key="download_preprocessed_excel"

)
