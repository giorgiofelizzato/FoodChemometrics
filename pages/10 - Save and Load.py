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
    layout="wide"
)

st.title("💾 Project Management")

st.markdown(
    """
Save your complete Food AI project into a single file
and reload it later to continue your analysis.
"""
)


# =====================================================
# PROJECT VERSION
# =====================================================

PROJECT_VERSION = "1.1"


# =====================================================
# HELPER FUNCTIONS
# =====================================================

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

            "app": "Food AI App"

        },


        # =================================================
        # ORIGINAL DATASET
        # =================================================

        "dataset":
            st.session_state.get(
                "dataset",
                None
            ),


        # =================================================
        # PREPROCESSED DATA
        # =================================================

        "preprocessed_dataset":
            st.session_state.get(
                "preprocessed_dataset",
                None
            ),

        "preprocessed_X":
            st.session_state.get(
                "preprocessed_X",
                None
            ),


        # =================================================
        # PREPROCESSING INFORMATION
        # =================================================

        "preprocessing_info":
            st.session_state.get(
                "preprocessing_info",
                {}
            ),


        # =================================================
        # PREPROCESSING SETTINGS
        # =================================================

        "preprocessing_settings": {

            "variables":
                st.session_state.get(
                    "preprocessing_variables",
                    []
                ),

            "missing_method":
                st.session_state.get(
                    "missing_method",
                    "None"
                ),

            "transformation_method":
                st.session_state.get(
                    "transformation_method",
                    "None"
                ),

            "scaling_method":
                st.session_state.get(
                    "scaling_method",
                    "None"
                ),

            "apply_snv":
                st.session_state.get(
                    "apply_snv",
                    False
                ),

            "apply_savgol":
                st.session_state.get(
                    "apply_savgol",
                    False
                ),

            "savgol_window":
                st.session_state.get(
                    "savgol_window",
                    None
                ),

            "savgol_polyorder":
                st.session_state.get(
                    "savgol_polyorder",
                    None
                )

        },


        # =================================================
        # DATA SETUP
        # =================================================

        "data_setup": {

            "sample_id":
                st.session_state.get(
                    "sample_id",
                    None
                ),

            "X_variables":
                st.session_state.get(
                    "X_variables",
                    []
                ),

            "y_variable":
                st.session_state.get(
                    "y_variable",
                    None
                ),

            "group_variable":
                st.session_state.get(
                    "group_variable",
                    None
                )

        },


        # =================================================
        # FUTURE SECTIONS
        # =================================================

        "exploratory_analysis":
            st.session_state.get(
                "exploratory_analysis",
                {}
            ),

        "statistics":
            st.session_state.get(
                "statistics",
                {}
            ),

        "chemometrics":
            st.session_state.get(
                "chemometrics",
                {}
            ),

        "machine_learning":
            st.session_state.get(
                "machine_learning",
                {}
            ),

        "results":
            st.session_state.get(
                "results",
                {}
            )

    }

    return project


# =====================================================
# RESTORE PROJECT
# =====================================================

def restore_project(project):

    """
    Restore all project information
    into Streamlit session state.
    """


    # =================================================
    # ORIGINAL DATASET
    # =================================================

    if project.get("dataset") is not None:

        st.session_state["dataset"] = (
            project["dataset"]
        )


    # =================================================
    # PREPROCESSED DATA
    # =================================================

    if project.get("preprocessed_dataset") is not None:

        st.session_state["preprocessed_dataset"] = (
            project["preprocessed_dataset"]
        )


    if project.get("preprocessed_X") is not None:

        st.session_state["preprocessed_X"] = (
            project["preprocessed_X"]
        )


    # =================================================
    # PREPROCESSING INFORMATION
    # =================================================

    st.session_state["preprocessing_info"] = (

        project.get(
            "preprocessing_info",
            {}
        )

    )


    # =================================================
    # PREPROCESSING SETTINGS
    # =================================================

    preprocessing_settings = (

        project.get(
            "preprocessing_settings",
            {}
        )

    )


    st.session_state["preprocessing_variables"] = (

        preprocessing_settings.get(
            "variables",
            []
        )

    )


    st.session_state["missing_method"] = (

        preprocessing_settings.get(
            "missing_method",
            "None"
        )

    )


    st.session_state["transformation_method"] = (

        preprocessing_settings.get(
            "transformation_method",
            "None"
        )

    )


    st.session_state["scaling_method"] = (

        preprocessing_settings.get(
            "scaling_method",
            "None"
        )

    )


    st.session_state["apply_snv"] = (

        preprocessing_settings.get(
            "apply_snv",
            False
        )

    )


    st.session_state["apply_savgol"] = (

        preprocessing_settings.get(
            "apply_savgol",
            False
        )

    )


    if preprocessing_settings.get(
        "savgol_window"
    ) is not None:

        st.session_state["savgol_window"] = (

            preprocessing_settings[
                "savgol_window"
            ]

        )


    if preprocessing_settings.get(
        "savgol_polyorder"
    ) is not None:

        st.session_state["savgol_polyorder"] = (

            preprocessing_settings[
                "savgol_polyorder"
            ]

        )


    # =================================================
    # DATA SETUP
    # =================================================

    data_setup = (

        project.get(
            "data_setup",
            {}
        )

    )


    st.session_state["sample_id"] = (

        data_setup.get(
            "sample_id",
            None
        )

    )


    st.session_state["X_variables"] = (

        data_setup.get(
            "X_variables",
            []
        )

    )


    st.session_state["y_variable"] = (

        data_setup.get(
            "y_variable",
            None
        )

    )


    st.session_state["group_variable"] = (

        data_setup.get(
            "group_variable",
            None
        )

    )


    # =================================================
    # FUTURE SECTIONS
    # =================================================

    future_sections = [

        "exploratory_analysis",

        "statistics",

        "chemometrics",

        "machine_learning",

        "results"

    ]


    for section in future_sections:

        st.session_state[section] = (

            project.get(
                section,
                {}
            )

        )


# =====================================================
# CURRENT PROJECT STATUS
# =====================================================

st.divider()

st.header("📊 Current Project")


if "dataset" in st.session_state:

    df = st.session_state["dataset"]


    if isinstance(df, pd.DataFrame):

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Samples",
                df.shape[0]
            )


        with col2:

            st.metric(
                "Variables",
                df.shape[1]
            )


        with col3:

            st.metric(
                "Predictors (X)",
                len(
                    st.session_state.get(
                        "X_variables",
                        []
                    )
                )
            )


        st.success(
            "A dataset is currently loaded."
        )


    else:

        st.error(
            "The current dataset is not a valid pandas DataFrame."
        )


    # =================================================
    # PREPROCESSED DATASET STATUS
    # =================================================

    if "preprocessed_dataset" in st.session_state:

        pre_df = (

            st.session_state[
                "preprocessed_dataset"
            ]

        )


        if isinstance(
            pre_df,
            pd.DataFrame
        ):

            st.info(

                f"Preprocessed dataset available: "
                f"{pre_df.shape[0]} samples × "
                f"{pre_df.shape[1]} variables"

            )


else:

    st.warning(
        "No dataset is currently loaded."
    )


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

st.divider()

st.header("Project configuration")


col1, col2 = st.columns(2)


with col1:

    st.write(
        "**Sample ID:**",
        st.session_state.get(
            "sample_id",
            None
        )
    )


    st.write(
        "**Target variable (y):**",
        st.session_state.get(
            "y_variable",
            None
        )
    )


with col2:

    st.write(
        "**Grouping variable:**",
        st.session_state.get(
            "group_variable",
            None
        )
    )


    st.write(
        "**Number of predictors:**",
        len(
            st.session_state.get(
                "X_variables",
                []
            )
        )
    )


# =====================================================
# PREPROCESSING STATUS
# =====================================================

prep = st.session_state.get(
    "preprocessing_info",
    {}
)


if prep:

    st.divider()

    st.subheader("⚙️ Preprocessing")


    # ---------------------------------------------
    # VARIABLES
    # ---------------------------------------------

    if "variables" in prep:

        st.write(
            "**Variables processed:**",
            len(
                prep["variables"]
            )
        )


    # ---------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------

    prep_summary = {

        "Missing values":
            prep.get(
                "missing",
                "None"
            ),

        "Transformation":
            prep.get(
                "transformation",
                "None"
            ),

        "Scaling":
            prep.get(
                "scaling",
                "None"
            ),

        "SNV":
            "Yes"
            if prep.get(
                "snv",
                False
            )
            else "No",

        "Savitzky-Golay":
            "Yes"
            if prep.get(
                "savgol",
                False
            )
            else "No"

    }


    prep_table = pd.DataFrame(

        list(
            prep_summary.items()
        ),

        columns=[
            "Step",
            "Method"
        ]

    )


    st.dataframe(

        prep_table,

        hide_index=True,

        use_container_width=True

    )


# =====================================================
# SAVE PROJECT
# =====================================================

st.divider()

st.header("💾 Save Project")


st.info(
    """
The project file contains the dataset, preprocessing,
data setup, preprocessing settings and the current
configuration of your analysis.
"""
)


project_name = st.text_input(

    "Project name",

    value="Food_AI_Project"

)


project_name = (

    project_name

    .strip()

    .replace(
        " ",
        "_"
    )

)


if not project_name:

    project_name = "Food_AI_Project"


# =====================================================
# BUILD PROJECT
# =====================================================

project_data = build_project()


# =====================================================
# SERIALIZE PROJECT
# =====================================================

project_bytes = io.BytesIO()


try:

    pickle.dump(

        project_data,

        project_bytes,

        protocol=pickle.HIGHEST_PROTOCOL

    )


    project_bytes.seek(0)


    st.download_button(

        label="⬇️ Download Project",

        data=project_bytes.getvalue(),

        file_name=f"{project_name}.foodai",

        mime="application/octet-stream",

        use_container_width=True,

        key="download_foodai_project"

    )


except Exception as e:

    st.error(

        f"Unable to create project file: {e}"

    )


# =====================================================
# LOAD PROJECT
# =====================================================

st.divider()

st.header("📂 Load Project")


uploaded_project = st.file_uploader(

    "Upload Food AI Project",

    type=[
        "foodai",
        "pkl"
    ],

    key="project_uploader"

)


if uploaded_project is not None:

    try:

        loaded_project = pickle.load(

            uploaded_project

        )


        # ---------------------------------------------
        # VALIDATE PROJECT
        # ---------------------------------------------

        if not isinstance(
            loaded_project,
            dict
        ):

            st.error(
                "Invalid project file."
            )

            st.stop()


        if "metadata" not in loaded_project:

            st.error(

                "Invalid Food AI project file: "
                "metadata section not found."

            )

            st.stop()


        metadata = (

            loaded_project.get(
                "metadata",
                {}
            )

        )


        st.success(
            "Project file successfully loaded."
        )


        # ---------------------------------------------
        # PROJECT INFORMATION
        # ---------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.write(

                "**Project version:**",

                metadata.get(
                    "project_version",
                    "Unknown"
                )

            )


        with col2:

            st.write(

                "**Saved at:**",

                metadata.get(
                    "saved_at",
                    "Unknown"
                )

            )


        # ---------------------------------------------
        # DATASET
        # ---------------------------------------------

        loaded_df = (

            loaded_project.get(
                "dataset",
                None
            )

        )


        if isinstance(
            loaded_df,
            pd.DataFrame
        ):

            st.write(

                "**Dataset:**",

                f"{loaded_df.shape[0]} samples × "
                f"{loaded_df.shape[1]} variables"

            )


            st.dataframe(

                loaded_df.head(),

                use_container_width=True

            )


        else:

            st.warning(
                "No valid dataset found in this project."
            )


        # ---------------------------------------------
        # PREPROCESSED DATASET
        # ---------------------------------------------

        loaded_pre = (

            loaded_project.get(
                "preprocessed_dataset",
                None
            )

        )


        if isinstance(
            loaded_pre,
            pd.DataFrame
        ):

            st.write(

                "**Preprocessed dataset:**",

                f"{loaded_pre.shape[0]} samples × "
                f"{loaded_pre.shape[1]} variables"

            )


        # ---------------------------------------------
        # RESTORE
        # ---------------------------------------------

        if st.button(

            "🔄 Restore Project",

            type="primary",

            use_container_width=True,

            key="restore_project_button"

        ):

            restore_project(

                loaded_project

            )


            st.success(

                "Project successfully restored!"

            )


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

    st.subheader(
        "🔢 Current Predictor Variables"
    )


    X_variables = (

        st.session_state.get(
            "X_variables",
            []
        )

    )


    if X_variables:

        st.dataframe(

            pd.DataFrame(

                X_variables,

                columns=[
                    "Variable"
                ]

            ),

            hide_index=True,

            use_container_width=True

        )


    else:

        st.info(
            "No predictor variables selected."
        )