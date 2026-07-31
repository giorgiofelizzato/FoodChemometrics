import streamlit as st
import pandas as pd
import numpy as np


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Data Setup",
    page_icon="⚙️",
    layout="wide"
)


st.title("Data Setup")


# =====================================================
# LOAD DATASET
# =====================================================

if "dataset" not in st.session_state:

    st.warning(
        "Please load a dataset first."
    )

    st.stop()



df = st.session_state["dataset"]


st.success(
    f"Dataset: {df.shape[0]} samples × {df.shape[1]} variables"
)



# =====================================================
# VARIABLE DETECTION
# =====================================================

numeric_variables = (

    df.select_dtypes(include=np.number)
    .columns
    .tolist()

)


categorical_variables = (

    df.select_dtypes(exclude=np.number)
    .columns
    .tolist()

)



all_variables = df.columns.tolist()



# =====================================================
# SAMPLE ID
# =====================================================

st.divider()

st.header("Sample identification")


st.info(
"""
The Sample ID identifies each sample.

It is not used as predictor (X) or target (y)
"""
)



# automatic suggestion

sample_default = 0


possible_id_names = [

    "Sample_ID",
    "Sample ID",
    "Sample",
    "ID",
    "Code",
    "Unnamed: 0"

]


for i, col in enumerate(all_variables):

    if col in possible_id_names:

        sample_default = i

        break



sample_id = st.selectbox(

    "Select Sample ID variable",

    ["None"] + all_variables,

    index=sample_default + 1

)



if sample_id == "None":

    sample_id = None



# =====================================================
# X VARIABLES
# =====================================================

st.divider()

st.header("Predictor variables (X)")


st.info(
"""
Predictor variables are the input variables used for
machine learning models.

Examples:

- volatile compounds
- chemical markers
- sensory descriptors

They are the variables used to explain or predict
the target variable.
"""
)



# remove sample id from possible X

available_X = [

    x for x in numeric_variables

    if x != sample_id

]



selected_X = st.multiselect(

    "Select variables used as predictors",

    available_X,

    default=available_X

)



if selected_X:


    st.dataframe(

        pd.DataFrame(

            {
                "Selected predictors":
                selected_X
            }

        ),

        hide_index=True,

        use_container_width=True

    )



# =====================================================
# TARGET VARIABLE
# =====================================================

st.divider()

st.header("Target variable (y)")


st.info(
"""
The target variable is the variable that the model
tries to predict.

Categorical target:
→ Classification

Examples:
- Origin
- Variety
- Treatment


Numerical target:
→ Regression

Examples:
- intensity
- concentration
- sensory score
"""
)



target_options = (

    ["None"]

    +

    [
        x for x in categorical_variables
        if x != sample_id
    ]

    +

    [
        x for x in numeric_variables
        if x != sample_id
    ]

)



target = st.selectbox(

    "Select target variable",

    target_options

)



# =====================================================
# GROUPING VARIABLE
# =====================================================

st.divider()

st.header("Grouping variable")

st.info(
"""
The grouping variable is used only for visualization.

It can be categorical (text) or discrete numerical
(e.g. 1, 2, 3 representing classes).

Continuous numerical variables are excluded automatically.
"""
)

# -----------------------------------------------------
# Detect discrete numerical variables
# -----------------------------------------------------

discrete_numeric_variables = []

for col in numeric_variables:

    if col == sample_id:
        continue

    n_unique = df[col].nunique(dropna=True)

    # keep only variables with few unique values
    if n_unique <= 20:
        discrete_numeric_variables.append(col)

# -----------------------------------------------------
# Combine categorical + discrete numerical variables
# -----------------------------------------------------

group_candidates = sorted(
    list(
        set(categorical_variables + discrete_numeric_variables)
    )
)

group_options = ["None"] + [
    x for x in group_candidates if x != sample_id
]

# Default selection if previously saved
default_group_index = 0

if (
    "group_variable" in st.session_state
    and st.session_state["group_variable"] in group_options
):
    default_group_index = group_options.index(
        st.session_state["group_variable"]
    )

group = st.selectbox(
    "Select grouping variable",
    group_options,
    index=default_group_index
)

# =====================================================
# SAVE CONFIGURATION
# =====================================================

st.divider()


if st.button(
    "Save configuration for the next pages"
):


    st.session_state["sample_id"] = sample_id


    st.session_state["X_variables"] = selected_X


    st.session_state["y_variable"] = (

        None
        if target == "None"
        else target

    )


    st.session_state["group_variable"] = (

        None
        if group == "None"
        else group

    )


    st.success(
        "Configuration saved!"
    )



# =====================================================
# SUMMARY
# =====================================================

if "X_variables" in st.session_state:


    st.divider()

    st.header("Current configuration")



    col1, col2 = st.columns(2)



    with col1:


        st.subheader("🆔 Sample ID")

        st.write(

            st.session_state.get(
                "sample_id",
                None
            )

        )


        st.subheader("🎯 Target variable")

        st.write(

            st.session_state.get(
                "y_variable",
                None
            )

        )



    with col2:


        st.subheader("🏷 Grouping variable")

        st.write(

            st.session_state.get(
                "group_variable",
                None
            )

        )


        st.subheader("🔢 Number of predictors")

        st.write(

            len(
                st.session_state["X_variables"]
            )

        )



    st.subheader("Predictor variables (X)")


    st.dataframe(

        pd.DataFrame(

            st.session_state["X_variables"],

            columns=["Variable"]

        ),

        hide_index=True,

        use_container_width=True

    )