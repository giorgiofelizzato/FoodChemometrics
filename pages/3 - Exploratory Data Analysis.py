import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy.stats import pearsonr, spearmanr, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Plots",
    page_icon="📊",
    layout="wide"
)


st.title("Exploratory Data Analysis")



# =====================================================
# LOAD DATA
# =====================================================

if "dataset" not in st.session_state:

    st.warning(
        "Please load dataset in Data Import first."
    )

    st.stop()


df = st.session_state["dataset"].copy()



# =====================================================
# SAMPLE ID
# =====================================================

sample_id = None


if (
    "sample_id" in st.session_state
    and st.session_state["sample_id"] is not None
    and st.session_state["sample_id"] in df.columns
):

    sample_id = st.session_state["sample_id"]



# =====================================================
# VARIABLE TYPES
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



# =====================================================
# VARIABLE SELECTION
# =====================================================

st.divider()

st.header("Variables selection")


st.info(
"""
Select numerical variables used for exploratory analysis.

Usually:
- chemical compounds
- sensory descriptors
- analytical responses
"""
)



default_variables = numeric_variables


if (
    "X_variables" in st.session_state
    and st.session_state["X_variables"]
):

    default_variables = [

        x for x in st.session_state["X_variables"]

        if x in numeric_variables

    ]



plot_variables = st.multiselect(

    "Variables for plots",

    numeric_variables,

    default=default_variables

)



if len(plot_variables) < 2:

    st.warning(
        "Select at least two variables."
    )

    st.stop()



# =====================================================
# GROUP VARIABLE
# =====================================================

st.divider()

st.header("Grouping variable")

st.info(
"""
The grouping variable can be:

- categorical (text labels)
- discrete numerical (e.g. 1, 2, 3 representing classes)

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

    # Keep only variables with few unique values
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
default_group = 0

if (
    "group_variable" in st.session_state
    and st.session_state["group_variable"] in group_options
):

    default_group = group_options.index(
        st.session_state["group_variable"]
    )

group_variable = st.selectbox(

    "Variable used for groups/classes",

    group_options,

    index=default_group

)

if group_variable == "None":

    group_variable = None
# =====================================================
# CORRELATION
# =====================================================

st.divider()

st.header("Correlation analysis")



st.info(
    """
Correlation analysis evaluates the relationship between numerical variables.

- **Pearson**: linear correlation
- **Spearman**: monotonic correlation

The **p-value matrix** indicates whether each correlation is statistically significant.
Common threshold: **p < 0.05**.
"""
)



method = st.selectbox(

    "Method",

    [
        "Pearson",
        "Spearman"
    ]

)



# -----------------------------------------------------
# Correlation matrix
# -----------------------------------------------------

corr = df[plot_variables].corr(
    method=method.lower()
)



# -----------------------------------------------------
# P-value matrix
# -----------------------------------------------------

p_matrix = pd.DataFrame(

    np.ones((len(plot_variables), len(plot_variables))),

    columns=plot_variables,

    index=plot_variables

)



for i in range(len(plot_variables)):

    for j in range(i + 1, len(plot_variables)):


        x = df[plot_variables[i]]
        y = df[plot_variables[j]]


        valid = pd.concat([x, y], axis=1).dropna()


        if len(valid) > 2:


            if method == "Pearson":

                r, p = pearsonr(
                    valid.iloc[:, 0],
                    valid.iloc[:, 1]
                )

            else:

                r, p = spearmanr(
                    valid.iloc[:, 0],
                    valid.iloc[:, 1]
                )


            p_matrix.iloc[i, j] = p
            p_matrix.iloc[j, i] = p



# Set diagonal to zero without using np.fill_diagonal
for i in range(len(plot_variables)):

    p_matrix.iloc[i, i] = 0



# -----------------------------------------------------
# Significant correlations
# -----------------------------------------------------

sig_mask = p_matrix < 0.05


sig_corr = corr.where(sig_mask)



# -----------------------------------------------------
# Tabs
# -----------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Correlation table",
    "Correlation heatmap",
    "P-value matrix",
    "Significant correlations"
])



# -----------------------------------------------------
# Correlation table
# -----------------------------------------------------

with tab1:


    st.dataframe(
        corr.round(3),
        use_container_width=True
    )



# -----------------------------------------------------
# Correlation heatmap
# -----------------------------------------------------

with tab2:


    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu",

        zmin=-1,

        zmax=1,

        aspect="auto",

        title=f"{method} correlation matrix"

    )


    fig.update_layout(height=700)


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# -----------------------------------------------------
# P-value table
# -----------------------------------------------------

with tab3:


    st.dataframe(
        p_matrix.round(4),
        use_container_width=True
    )


    st.caption(
        "Values < 0.05 are usually considered statistically significant."
    )



# -----------------------------------------------------
# Significant correlations heatmap
# -----------------------------------------------------

with tab4:


    fig_sig = px.imshow(

        sig_corr,

        text_auto=".2f",

        color_continuous_scale="RdBu",

        zmin=-1,

        zmax=1,

        aspect="auto",

        title="Significant correlations (p < 0.05)"

    )


    fig_sig.update_layout(height=700)


    st.plotly_chart(
        fig_sig,
        use_container_width=True
    )



# -----------------------------------------------------
# Significant pairs table
# -----------------------------------------------------

st.subheader("Significant pairs")


pairs = []


for i in range(len(plot_variables)):

    for j in range(i + 1, len(plot_variables)):


        p = p_matrix.iloc[i, j]


        if p < 0.05:


            pairs.append({

                "Variable 1": plot_variables[i],

                "Variable 2": plot_variables[j],

                "Correlation": round(corr.iloc[i, j], 3),

                "p-value": round(p, 5)

            })



if len(pairs) > 0:


    pairs_df = pd.DataFrame(pairs)


    pairs_df = pairs_df.sort_values(
        by="p-value"
    )


    st.dataframe(
        pairs_df,
        use_container_width=True
    )

else:


    st.info(
        "No significant correlations found at p < 0.05."
    )

    
# =====================================================
# DISTRIBUTION
# =====================================================

st.divider()

st.header("Variable distribution")



variable = st.selectbox(

    "Select variable",

    plot_variables,

    key="distribution"

)



distribution_type = st.selectbox(

    "Plot type",

    [
        "Histogram",
        "Boxplot",
        "Violin"
    ]

)



hover_information = (

    [sample_id]

    if sample_id is not None

    else None

)



if distribution_type == "Histogram":


    fig = px.histogram(

        df,

        x=variable,

        nbins=30,

        hover_data=hover_information

    )



elif distribution_type == "Boxplot":


    fig = px.box(

        df,

        y=variable,

        points="all",

        hover_data=hover_information

    )



else:


    fig = px.violin(

        df,

        y=variable,

        box=True,

        points="all",

        hover_data=hover_information

    )



st.plotly_chart(

    fig,

    use_container_width=True

)



# =====================================================
# CLASS COMPARISON
# =====================================================

st.divider()

st.header("Compound comparison by class")



if group_variable is not None:


    available_classes = (

        df[group_variable]

        .dropna()

        .unique()

        .tolist()

    )



    selected_classes = st.multiselect(

        "Select classes to display",

        available_classes,

        default=available_classes

    )



    compound = st.selectbox(

        "Select compound",

        plot_variables,

        key="compound_box"

    )



    plot_df = df[
        df[group_variable].isin(selected_classes)
    ]



    fig = px.box(

        plot_df,

        x=group_variable,

        y=compound,

        color=group_variable,

        points="all",

        hover_data=hover_information,

        title=f"{compound} by {group_variable}"

    )



    fig.update_layout(

        xaxis_title=group_variable,

        yaxis_title=compound

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )


else:


    st.info(
        "Select a grouping variable."
    )

# =====================================================
# ANOVA
# =====================================================

st.divider()

st.header("One-way ANOVA")


if group_variable is not None:

    st.info(
        "One-way ANOVA tests whether the mean of the selected "
        "variable differs significantly among the selected groups."
    )


    # -------------------------------------------------
    # Variable selection
    # -------------------------------------------------

    anova_variable = st.selectbox(

        "Select variable for ANOVA",

        plot_variables,

        key="anova_variable"

    )


    # -------------------------------------------------
    # Class selection
    # -------------------------------------------------

    available_classes = (

        df[group_variable]

        .dropna()

        .unique()

        .tolist()

    )


    anova_classes = st.multiselect(

        "Select classes",

        available_classes,

        default=available_classes,

        key="anova_classes"

    )


    # -------------------------------------------------
    # Prepare data
    # -------------------------------------------------

    anova_columns = [

        group_variable,

        anova_variable

    ]


    # Add Sample ID for hover information

    if sample_id is not None:

        anova_columns.append(
            sample_id
        )


    anova_df = df[
        anova_columns
    ].dropna(
        subset=[
            group_variable,
            anova_variable
        ]
    )


    anova_df = anova_df[
        anova_df[group_variable].isin(
            anova_classes
        )
    ]


    # -------------------------------------------------
    # Perform ANOVA
    # -------------------------------------------------

    groups = [

        group[anova_variable].values

        for _, group in anova_df.groupby(
            group_variable
        )

    ]


    group_names = (

        anova_df[group_variable]

        .unique()

        .tolist()

    )


    if len(groups) >= 2:

        f_stat, p_value = f_oneway(
            *groups
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(

                "F-statistic",

                f"{f_stat:.3f}"

            )


        with col2:

            st.metric(

                "p-value",

                f"{p_value:.4e}"

            )


        if p_value < 0.05:

            st.success(

                "The difference among groups is statistically significant "
                "(p < 0.05)."

            )

        else:

            st.info(

                "No statistically significant difference was detected "
                "among groups (p ≥ 0.05)."

            )


        # -------------------------------------------------
        # Boxplot
        # -------------------------------------------------

        st.subheader(

            f"{anova_variable} distribution by {group_variable}"

        )


        fig = px.box(

            anova_df,

            x=group_variable,

            y=anova_variable,

            color=group_variable,

            points="all",

            hover_data=[sample_id]
            if sample_id is not None
            else None,

            title=(

                f"{anova_variable} by "
                f"{group_variable} "
                f"(ANOVA p = {p_value:.3e})"

            )

        )


        fig.update_layout(

            xaxis_title=group_variable,

            yaxis_title=anova_variable,

            showlegend=False

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


        # -------------------------------------------------
        # Tukey HSD
        # -------------------------------------------------

        st.subheader(

            "Post-hoc Tukey HSD"

        )


        if p_value < 0.05 and len(group_names) >= 2:

            tukey = pairwise_tukeyhsd(

                endog=anova_df[anova_variable],

                groups=anova_df[group_variable],

                alpha=0.05

            )


            tukey_df = pd.DataFrame(

                data=tukey._results_table.data[1:],

                columns=tukey._results_table.data[0]

            )


            st.dataframe(

                tukey_df,

                use_container_width=True

            )


    else:

        st.warning(

            "Select at least two groups with available data."

        )


else:

    st.info(

        "Select a grouping variable above to perform ANOVA."

    )

# =====================================================
# SCATTER
# =====================================================

st.divider()

st.header("Scatter plot")



col1, col2 = st.columns(2)



with col1:

    x_variable = st.selectbox(

        "X axis",

        plot_variables,

        key="scatter_x"

    )



with col2:

    y_variable = st.selectbox(

        "Y axis",

        plot_variables,

        key="scatter_y"

    )



marker_size = st.slider(

    "Marker size",

    5,

    50,

    15

)



scatter_columns = [

    x_variable,

    y_variable

]


if group_variable is not None:

    scatter_columns.append(
        group_variable
    )


if sample_id is not None:

    scatter_columns.append(
        sample_id
    )



scatter_df = df[
    scatter_columns
].dropna()



fig = px.scatter(

    scatter_df,

    x=x_variable,

    y=y_variable,

    color=group_variable,

    hover_name=sample_id,

    title=f"{y_variable} vs {x_variable}"

)



fig.update_traces(

    marker=dict(

        size=marker_size

    )

)



st.plotly_chart(

    fig,

    use_container_width=True

)



# =====================================================
# SUMMARY
# =====================================================

st.divider()

st.header("Current settings")



st.write(
    "**Sample ID:**",
    sample_id
)


st.write(
    "**Variables:**"
)

st.write(
    plot_variables
)


st.write(
    "**Grouping:**",
    group_variable
)