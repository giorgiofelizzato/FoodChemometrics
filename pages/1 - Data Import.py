import streamlit as st
import pandas as pd
import numpy as np
import io


# ====================================================
# CONFIGURATION
# ====================================================

st.set_page_config(
    page_title="Data Import",
    page_icon="📂",
    layout="wide"
)


st.title("Data Import & Dataset Management")


# ====================================================
# SESSION STATE INITIALIZATION
# ====================================================

if "raw_dataset" not in st.session_state:
    st.session_state.raw_dataset = None

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "history" not in st.session_state:
    st.session_state.history = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "current_sheet" not in st.session_state:
    st.session_state.current_sheet = None



# ====================================================
# FILE UPLOAD
# ====================================================

uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["xlsx", "xls", "csv"]
)



if uploaded_file is not None:


    # ------------------------------------------------
    # READ FILE
    # ------------------------------------------------

    if uploaded_file.name.endswith(
        (".xlsx", ".xls")
    ):

        excel_file = pd.ExcelFile(
            uploaded_file
        )


        sheet_names = excel_file.sheet_names


        selected_sheet = st.selectbox(
            "Select Excel worksheet",
            sheet_names
        )


        df_loaded = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )


    else:

        selected_sheet = "CSV"

        df_loaded = pd.read_csv(
            uploaded_file
        )



    # ------------------------------------------------
    # CHECK IF NEW FILE OR NEW SHEET
    # ------------------------------------------------

    new_file = (
        st.session_state.current_file
        != uploaded_file.name
    )


    new_sheet = (
        st.session_state.current_sheet
        != selected_sheet
    )


    if new_file or new_sheet:


        # Save RAW dataset

        st.session_state.raw_dataset = (
            df_loaded.copy()
        )


        # Create working dataset

        st.session_state.dataset = (
            df_loaded.copy()
        )


        # Reset history

        st.session_state.history = []


        st.session_state.history.append(
            f"Loaded file: {uploaded_file.name}"
        )


        st.session_state.history.append(
            f"Selected sheet: {selected_sheet}"
        )


        # Update memory

        st.session_state.current_file = (
            uploaded_file.name
        )

        st.session_state.current_sheet = (
            selected_sheet
        )



# ====================================================
# CHECK DATASET
# ====================================================

if st.session_state.dataset is None:

    st.info(
        "Please upload a dataset."
    )

    st.stop()



# Current working dataset

df = st.session_state.dataset



# ====================================================
# DATASET STATUS
# ====================================================

st.divider()

st.header("Dataset versions")


col1, col2 = st.columns(2)


with col1:

    st.subheader("🟦 Raw Dataset")

    st.write(
        "Original data"
    )

    st.write(
        f"Rows: {st.session_state.raw_dataset.shape[0]}"
    )

    st.write(
        f"Columns: {st.session_state.raw_dataset.shape[1]}"
    )



with col2:

    st.subheader("🟩 Modified Dataset")

    st.write(
        "Current working data"
    )

    st.write(
        f"Rows: {df.shape[0]}"
    )

    st.write(
        f"Columns: {df.shape[1]}"
    )



# ====================================================
# INFORMATION
# ====================================================

st.divider()

st.header("Dataset Information")


c1,c2,c3,c4 = st.columns(4)


c1.metric(
    "Rows",
    df.shape[0]
)


c2.metric(
    "Columns",
    df.shape[1]
)


c3.metric(
    "Missing values",
    int(df.isna().sum().sum())
)


c4.metric(
    "Duplicate rows",
    int(df.duplicated().sum())
)



c1,c2,c3,c4 = st.columns(4)


c1.metric(
    "Numeric variables",
    df.select_dtypes(
        include=np.number
    ).shape[1]
)


c2.metric(
    "Categorical variables",
    df.select_dtypes(
        exclude=np.number
    ).shape[1]
)


memory = (
    df.memory_usage(deep=True)
    .sum()/1024
)


c3.metric(
    "Memory",
    f"{memory:.2f} KB"
)


c4.metric(
    "Missing (%)",
    f"{df.isna().mean().mean()*100:.2f}%"
)



# ====================================================
# VARIABLE INFORMATION
# ====================================================

st.divider()

st.header("Variable Information")


variable_info = pd.DataFrame({

    "Variable": df.columns,

    "Type": df.dtypes.astype(str),

    "Missing":
        df.isna().sum().values,

    "Missing (%)":
        (
            df.isna()
            .mean()
            .values*100
        ).round(2),

    "Unique":
        df.nunique().values

})


st.dataframe(
    variable_info,
    use_container_width=True
)



# ====================================================
# DATA PREVIEW
# ====================================================

st.divider()

st.header("Data Preview")


st.dataframe(
    df,
    height=400,
    use_container_width=True
)



# ====================================================
# MISSING VALUES
# ====================================================

st.divider()

st.header("Missing Values")


missing = pd.DataFrame({

    "Variable": df.columns,

    "Missing values":
        df.isna().sum().values,

    "Missing (%)":
        (
            df.isna()
            .mean()
            .values*100
        ).round(2)

})


missing = missing[
    missing["Missing values"] > 0
]


if len(missing)==0:

    st.success(
        "No missing values"
    )

else:

    st.dataframe(
        missing,
        use_container_width=True
    )



# ====================================================
# REMOVE COLUMNS
# ====================================================

st.divider()

st.header("Remove Variables")


columns_remove = st.multiselect(
    "Select columns to remove",
    df.columns
)


if st.button(
    "Remove selected columns"
):

    if len(columns_remove)>0:


        df = df.drop(
            columns=columns_remove
        )


        st.session_state.dataset = df


        st.session_state.history.append(
            "Removed columns: "
            +
            ", ".join(columns_remove)
        )


        st.success(
            "Columns removed"
        )

        st.rerun()



# ====================================================
# REMOVE ROWS
# ====================================================

st.divider()

st.header("Remove Samples")


rows_remove = st.multiselect(
    "Select rows to remove",
    df.index.tolist()
)


if st.button(
    "Remove selected rows"
):

    if len(rows_remove)>0:


        df = (
            df.drop(rows_remove)
            .reset_index(drop=True)
        )


        st.session_state.dataset = df


        st.session_state.history.append(
            f"Removed rows: {rows_remove}"
        )


        st.success(
            "Rows removed"
        )


        st.rerun()



# ====================================================
# RESET DATA
# ====================================================

st.divider()

if st.button(
    "Restore RAW dataset"
):


    st.session_state.dataset = (
        st.session_state.raw_dataset.copy()
    )


    st.session_state.history.append(
        "Restored RAW dataset"
    )


    st.success(
        "Dataset restored"
    )


    st.rerun()



# ====================================================
# HISTORY
# ====================================================

st.divider()

st.header("Modification History")


for i, item in enumerate(
    st.session_state.history,
    start=1
):

    st.write(
        f"{i}. {item}"
    )



# ====================================================
# DOWNLOAD
# ====================================================

st.divider()

st.header("⬇Export")


buffer = io.BytesIO()


with pd.ExcelWriter(
    buffer,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        index=False,
        sheet_name="Modified"
    )


st.download_button(

    "Download modified dataset",

    data=buffer.getvalue(),

    file_name="modified_dataset.xlsx",

    mime=(
        "application/vnd.openxmlformats-"
        "officedocument.spreadsheetml.sheet"
    )

)