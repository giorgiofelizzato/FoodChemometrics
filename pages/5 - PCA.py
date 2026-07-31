import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy import stats
import io


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PCA",
    page_icon="📈",
    layout="wide"
)


st.title("📈 Principal Component Analysis (PCA)")


# =====================================================
# DATASET SELECTION
# =====================================================

if "dataset" not in st.session_state:

    st.warning(
        "Please load a dataset first."
    )

    st.stop()


dataset_options = ["Raw dataset"]


if "preprocessed_dataset" in st.session_state:

    dataset_options.append(
        "Preprocessed dataset"
    )


dataset_choice = st.radio(
    "Dataset",
    dataset_options
)


if dataset_choice == "Preprocessed dataset":

    df = st.session_state[
        "preprocessed_dataset"
    ].copy()

else:

    df = st.session_state[
        "dataset"
    ].copy()


# =====================================================
# SAMPLE ID
# =====================================================

sample_id = None


if (
    "sample_id" in st.session_state
    and st.session_state["sample_id"] in df.columns
):

    sample_id = st.session_state["sample_id"]


# =====================================================
# VARIABLE SELECTION
# =====================================================

numeric_variables = (
    df.select_dtypes(include=np.number)
    .columns
    .tolist()
)


default_vars = numeric_variables


if (
    "X_variables" in st.session_state
    and st.session_state["X_variables"]
):

    default_vars = [

        x for x in st.session_state["X_variables"]

        if x in numeric_variables

    ]


selected_vars = st.multiselect(

    "Variables for PCA",

    numeric_variables,

    default=default_vars

)


if len(selected_vars) < 2:

    st.warning(
        "Select at least two variables."
    )

    st.stop()


X = df[selected_vars].dropna()


# =====================================================
# PCA PARAMETERS
# =====================================================

st.divider()

st.header("PCA settings")


max_pc = min(X.shape[0], X.shape[1])


n_components = st.slider(

    "Number of principal components",

    min_value=2,

    max_value=max_pc,

    value=min(5, max_pc)

)


# =====================================================
# PCA MODEL
# =====================================================

pca = PCA(
    n_components=n_components
)


scores_array = pca.fit_transform(X)


pc_names = [

    f"PC{i+1}"

    for i in range(n_components)

]


scores = pd.DataFrame(

    scores_array,

    columns=pc_names,

    index=X.index

)


loadings = pd.DataFrame(

    pca.components_.T,

    columns=pc_names,

    index=X.columns

)


# =====================================================
# ADD METADATA
# =====================================================

for col in df.columns:

    if col not in selected_vars:

        scores[col] = df.loc[
            scores.index,
            col
        ]


# =====================================================
# EXPLAINED VARIANCE
# =====================================================

st.divider()

st.header("Explained variance")


ev = (
    pca.explained_variance_ratio_
    * 100
)


cev = np.cumsum(ev)


ev_table = pd.DataFrame({

    "PC": pc_names,

    "Explained variance (%)":
        np.round(
            ev,
            2
        ),

    "Cumulative variance (%)":
        np.round(
            cev,
            2
        )

})


st.dataframe(

    ev_table,

    hide_index=True,

    use_container_width=True

)


col1, col2 = st.columns(2)


with col1:

    fig_ev = px.bar(

        ev_table,

        x="PC",

        y="Explained variance (%)",

        title="Scree plot"

    )


    st.plotly_chart(

        fig_ev,

        use_container_width=True

    )


with col2:

    fig_cev = px.line(

        ev_table,

        x="PC",

        y="Cumulative variance (%)",

        markers=True,

        title="Cumulative explained variance"

    )


    fig_cev.update_yaxes(

        range=[
            0,
            100
        ]

    )


    st.plotly_chart(

        fig_cev,

        use_container_width=True

    )


# =====================================================
# SCORE PLOT 2D
# =====================================================

st.divider()

st.header("Scores plot (2D)")


col1, col2 = st.columns(2)


with col1:

    x_pc = st.selectbox(

        "X axis",

        pc_names,

        index=0

    )


with col2:

    y_pc = st.selectbox(

        "Y axis",

        pc_names,

        index=1

    )


# =====================================================
# AXIS LABELS WITH EXPLAINED VARIANCE
# =====================================================

x_pc_index = pc_names.index(
    x_pc
)


y_pc_index = pc_names.index(
    y_pc
)


x_pc_variance = ev[
    x_pc_index
]


y_pc_variance = ev[
    y_pc_index
]


x_axis_label = (

    f"{x_pc} "
    f"({x_pc_variance:.2f}% explained variance)"

)


y_axis_label = (

    f"{y_pc} "
    f"({y_pc_variance:.2f}% explained variance)"

)


# =====================================================
# COLOR VARIABLE
# =====================================================

group_var = None


if (
    "group_variable" in st.session_state
    and st.session_state["group_variable"]
    in scores.columns
):

    group_var = st.session_state[
        "group_variable"
    ]


color_options = [

    "None"

] + scores.columns.tolist()


color_var = st.selectbox(

    "Color by",

    color_options

)


if color_var == "None":

    color_var = None


# =====================================================
# SIZE VARIABLE
# =====================================================

size_options = [

    "None"

] + scores.columns.tolist()


size_var = st.selectbox(

    "Size by",

    size_options

)


if size_var == "None":

    size_var = None


marker_size = st.slider(

    "Default marker size",

    5,

    25,

    10

)


# =====================================================
# 2D SCORE PLOT
# =====================================================

fig_scores = px.scatter(

    scores,

    x=x_pc,

    y=y_pc,

    color=color_var,

    size=size_var,

    hover_name=sample_id,

    title=(
        f"{x_pc} vs {y_pc}"
    )

)


if size_var is None:

    fig_scores.update_traces(

        marker=dict(

            size=marker_size

        )

    )


# =====================================================
# UPDATE AXES WITH EXPLAINED VARIANCE
# =====================================================

fig_scores.update_xaxes(

    title=x_axis_label,

    zeroline=True,

    zerolinecolor="black"

)


fig_scores.update_yaxes(

    title=y_axis_label,

    zeroline=True,

    zerolinecolor="black"

)


st.plotly_chart(

    fig_scores,

    use_container_width=True

)


# =====================================================
# SCORE PLOT 3D
# =====================================================

st.divider()

st.header("Scores plot (3D)")


col1, col2, col3 = st.columns(3)


with col1:

    pc_x3 = st.selectbox(

        "X",

        pc_names,

        index=0,

        key="x3"

    )


with col2:

    pc_y3 = st.selectbox(

        "Y",

        pc_names,

        index=1,

        key="y3"

    )


with col3:

    pc_z3 = st.selectbox(

        "Z",

        pc_names,

        index=(
            2
            if n_components > 2
            else 1
        ),

        key="z3"

    )


# =====================================================
# 3D AXIS VARIANCE
# =====================================================

pc_x3_index = pc_names.index(
    pc_x3
)


pc_y3_index = pc_names.index(
    pc_y3
)


pc_z3_index = pc_names.index(
    pc_z3
)


pc_x3_variance = ev[
    pc_x3_index
]


pc_y3_variance = ev[
    pc_y3_index
]


pc_z3_variance = ev[
    pc_z3_index
]


x3_axis_label = (

    f"{pc_x3} "
    f"({pc_x3_variance:.2f}% explained variance)"

)


y3_axis_label = (

    f"{pc_y3} "
    f"({pc_y3_variance:.2f}% explained variance)"

)


z3_axis_label = (

    f"{pc_z3} "
    f"({pc_z3_variance:.2f}% explained variance)"

)


# =====================================================
# 3D MARKER SIZE
# =====================================================

size_candidates = [

    c

    for c in scores.columns

    if pd.api.types.is_numeric_dtype(
        scores[c]
    )

]


size_options_3d = [

    "Fixed size"

] + size_candidates


size_var_3d = st.selectbox(

    "Marker size (3D)",

    size_options_3d,

    key="size3d"

)


fixed_size_3d = st.slider(

    "Fixed marker size",

    3,

    20,

    8,

    key="fixed3d"

)


# =====================================================
# 3D SCORE PLOT
# =====================================================

if size_var_3d == "Fixed size":

    fig3d = px.scatter_3d(

        scores,

        x=pc_x3,

        y=pc_y3,

        z=pc_z3,

        color=color_var,

        hover_name=sample_id,

        title=(
            f"3D Scores: "
            f"{pc_x3}, "
            f"{pc_y3}, "
            f"{pc_z3}"
        )

    )


    fig3d.update_traces(

        marker=dict(

            size=fixed_size_3d

        )

    )


else:

    fig3d = px.scatter_3d(

        scores,

        x=pc_x3,

        y=pc_y3,

        z=pc_z3,

        color=color_var,

        size=size_var_3d,

        hover_name=sample_id,

        title=(
            f"3D Scores: "
            f"{pc_x3}, "
            f"{pc_y3}, "
            f"{pc_z3}"
        )

    )


# =====================================================
# 3D AXES WITH EXPLAINED VARIANCE
# =====================================================

fig3d.update_layout(

    height=700,

    scene=dict(

        xaxis=dict(

            title=x3_axis_label

        ),

        yaxis=dict(

            title=y3_axis_label

        ),

        zaxis=dict(

            title=z3_axis_label

        )

    )

)


st.plotly_chart(

    fig3d,

    use_container_width=True

)


# =====================================================
# LOADINGS PLOT
# =====================================================

st.divider()

st.header("Loadings plot")


col1, col2 = st.columns(2)


with col1:

    load_x = st.selectbox(

        "Loading X",

        pc_names,

        index=0,

        key="loadx"

    )


with col2:

    load_y = st.selectbox(

        "Loading Y",

        pc_names,

        index=1,

        key="loady"

    )


load_plot = loadings.copy()


load_plot["Variable"] = load_plot.index


fig_load = px.scatter(

    load_plot,

    x=load_x,

    y=load_y,

    text="Variable",

    title=(
        f"Loadings: "
        f"{load_x} vs {load_y}"
    )

)


fig_load.update_traces(

    textposition="top center"

)


fig_load.update_xaxes(

    zeroline=True,

    zerolinecolor="black"

)


fig_load.update_yaxes(

    zeroline=True,

    zerolinecolor="black"

)


st.plotly_chart(

    fig_load,

    use_container_width=True

)


# =====================================================
# LOADINGS BARPLOT
# =====================================================

st.divider()

st.header("Variable contributions")


loading_pc = st.selectbox(

    "Principal component",

    pc_names

)


bar_df = loadings[

    [loading_pc]

].copy()


bar_df["Variable"] = bar_df.index


bar_df = bar_df.sort_values(

    by=loading_pc,

    key=np.abs,

    ascending=False

)


fig_bar = px.bar(

    bar_df,

    x="Variable",

    y=loading_pc,

    title=(
        f"Contributions to "
        f"{loading_pc}"
    )

)


fig_bar.update_layout(

    xaxis_tickangle=90

)


st.plotly_chart(

    fig_bar,

    use_container_width=True

)


# =====================================================
# BIPLOT
# =====================================================

st.divider()

st.header("Biplot")


# =====================================================
# BIPLOT AXES
# =====================================================

col1, col2 = st.columns(2)


with col1:

    biplot_x = st.selectbox(

        "Biplot X axis",

        pc_names,

        index=0,

        key="biplot_x"

    )


with col2:

    biplot_y = st.selectbox(

        "Biplot Y axis",

        pc_names,

        index=1,

        key="biplot_y"

    )


# =====================================================
# BIPLOT AXIS VARIANCE
# =====================================================

biplot_x_index = pc_names.index(

    biplot_x

)


biplot_y_index = pc_names.index(

    biplot_y

)


biplot_x_variance = ev[

    biplot_x_index

]


biplot_y_variance = ev[

    biplot_y_index

]


biplot_x_axis_label = (

    f"{biplot_x} "
    f"({biplot_x_variance:.2f}% explained variance)"

)


biplot_y_axis_label = (

    f"{biplot_y} "
    f"({biplot_y_variance:.2f}% explained variance)"

)


# =====================================================
# LOADING SCALE
# =====================================================

biplot_user_scale = st.slider(

    "Biplot loading scale",

    min_value=0.5,

    max_value=10.0,

    value=2.0,

    step=0.5,

    key="biplot_scale"

)


# =====================================================
# BIPLOT COLOR
# =====================================================

biplot_color_options = [

    "None"

] + scores.columns.tolist()


biplot_color_var = st.selectbox(

    "Color samples by",

    biplot_color_options,

    index=(

        biplot_color_options.index(

            color_var

        )

        if (

            color_var is not None

            and color_var
            in biplot_color_options

        )

        else 0

    ),

    key="biplot_color"

)


if biplot_color_var == "None":

    biplot_color_var = None


# =====================================================
# CREATE BIPLOT
# =====================================================

fig_bi = go.Figure()


# =====================================================
# HOVER COLUMNS
# =====================================================

hover_columns = []


if sample_id is not None:

    hover_columns.append(

        sample_id

    )


if biplot_color_var is not None:

    if biplot_color_var not in hover_columns:

        hover_columns.append(

            biplot_color_var

        )


if hover_columns:

    hover_text = (

        scores[hover_columns]

        .astype(str)

        .agg(

            "<br>".join,

            axis=1

        )

    )

else:

    hover_text = (

        scores.index.astype(str)

    )


# =====================================================
# SAMPLE POINTS - BIPLOT
# =====================================================

if biplot_color_var is None:

    # ---------------------------------------------
    # NO COLORING
    # ---------------------------------------------

    fig_bi.add_trace(

        go.Scatter(

            x=scores[biplot_x],

            y=scores[biplot_y],

            mode="markers",

            text=hover_text,

            hovertemplate=(

                "%{text}<br>"

                + biplot_x

                + ": %{x:.3f}<br>"

                + biplot_y

                + ": %{y:.3f}"

                + "<extra></extra>"

            ),

            name="Samples",

            marker=dict(

                size=marker_size

            )

        )

    )


else:

    color_values = scores[

        biplot_color_var

    ]


    # =================================================
    # CATEGORICAL VARIABLE
    # =================================================

    if not pd.api.types.is_numeric_dtype(

        color_values

    ):

        categories = (

            color_values

            .astype(str)

            .dropna()

            .unique()

        )


        for category in categories:

            mask = (

                color_values

                .astype(str)

                == category

            )


            category_hover = (

                scores.loc[

                    mask,

                    hover_columns

                ]

                .astype(str)

                .agg(

                    "<br>".join,

                    axis=1

                )

                if hover_columns

                else scores.index[mask]

                .astype(str)

            )


            fig_bi.add_trace(

                go.Scatter(

                    x=scores.loc[

                        mask,

                        biplot_x

                    ],

                    y=scores.loc[

                        mask,

                        biplot_y

                    ],

                    mode="markers",

                    name=category,

                    text=category_hover,

                    hovertemplate=(

                        "%{text}<br>"

                        + biplot_x

                        + ": %{x:.3f}<br>"

                        + biplot_y

                        + ": %{y:.3f}"

                        + "<extra></extra>"

                    ),

                    marker=dict(

                        size=marker_size

                    )

                )

            )


    # =================================================
    # CONTINUOUS NUMERICAL VARIABLE
    # =================================================

    else:

        fig_bi.add_trace(

            go.Scatter(

                x=scores[biplot_x],

                y=scores[biplot_y],

                mode="markers",

                text=hover_text,

                hovertemplate=(

                    "%{text}<br>"

                    + biplot_x

                    + ": %{x:.3f}<br>"

                    + biplot_y

                    + ": %{y:.3f}"

                    + "<extra></extra>"

                ),

                name=biplot_color_var,

                marker=dict(

                    size=marker_size,

                    color=color_values,

                    colorscale="Viridis",

                    showscale=True,

                    colorbar=dict(

                        title=biplot_color_var

                    )

                )

            )

        )


# =====================================================
# AUTOMATIC LOADING SCALING
# =====================================================

score_x_max = np.max(

    np.abs(

        scores[biplot_x]

    )

)


score_y_max = np.max(

    np.abs(

        scores[biplot_y]

    )

)


loading_x_max = np.max(

    np.abs(

        loadings[biplot_x]

    )

)


loading_y_max = np.max(

    np.abs(

        loadings[biplot_y]

    )

)


# Avoid division by zero

if loading_x_max == 0:

    loading_x_max = 1


if loading_y_max == 0:

    loading_y_max = 1


scale_x = (

    score_x_max

    / loading_x_max

)


scale_y = (

    score_y_max

    / loading_y_max

)


scale_factor = (

    min(

        scale_x,

        scale_y

    )

    * 0.7

    * biplot_user_scale

)


# =====================================================
# LOADING VECTORS
# =====================================================

for var in loadings.index:

    x_end = (

        loadings.loc[

            var,

            biplot_x

        ]

        * scale_factor

    )


    y_end = (

        loadings.loc[

            var,

            biplot_y

        ]

        * scale_factor

    )


    # ---------------------------------------------
    # Arrow line
    # ---------------------------------------------

    fig_bi.add_trace(

        go.Scatter(

            x=[

                0,

                x_end

            ],

            y=[

                0,

                y_end

            ],

            mode="lines",

            line=dict(

                width=2

            ),

            showlegend=False,

            hoverinfo="skip"

        )

    )


    # ---------------------------------------------
    # Variable label
    # ---------------------------------------------

    fig_bi.add_annotation(

        x=x_end,

        y=y_end,

        text=var,

        showarrow=False,

        xanchor="center",

        yanchor="bottom"

    )


# =====================================================
# BIPLOT AXES
# =====================================================

fig_bi.update_xaxes(

    title=biplot_x_axis_label,

    zeroline=True,

    zerolinecolor="black",

    zerolinewidth=1

)


fig_bi.update_yaxes(

    title=biplot_y_axis_label,

    zeroline=True,

    zerolinecolor="black",

    zerolinewidth=1

)


# =====================================================
# BIPLOT LAYOUT
# =====================================================

fig_bi.update_layout(

    title=(

        f"Biplot: "

        f"{biplot_x} vs "

        f"{biplot_y}"

    ),

    height=750,

    template="plotly_white",

    legend=dict(

        title=(

            biplot_color_var

            if biplot_color_var is not None

            and not pd.api.types.is_numeric_dtype(

                scores[biplot_color_var]

            )

            else "Classes"

        )

    )

)


st.plotly_chart(

    fig_bi,

    use_container_width=True

)


# =====================================================
# OUTLIER DETECTION
# =====================================================

st.divider()

st.header(

    "Outlier detection "
    "(Hotelling T² & Q residuals)"

)


T = scores[

    pc_names

].values


P = loadings[

    pc_names

].values


X_reconstructed = np.dot(

    T,

    P.T

)


Err = (

    X.values

    - X_reconstructed

)


Q = np.sum(

    Err**2,

    axis=1

)


Tsq = np.sum(

    (

        T

        / np.std(

            T,

            axis=0

        )

    )**2,

    axis=1

)


conf = 0.95


Tsq_conf = np.percentile(

    Tsq,

    conf * 100

)


Q_conf = np.percentile(

    Q,

    conf * 100

)


outlier_df = pd.DataFrame({

    "Sample":

        X.index.astype(str),

    "T2":

        Tsq,

    "Qres":

        Q,

    "Outlier":

        (

            Tsq > Tsq_conf

        )

        |

        (

            Q > Q_conf

        )

})


if group_var is not None:

    outlier_df[

        group_var

    ] = df.loc[

        X.index,

        group_var

    ].values


fig_out = px.scatter(

    outlier_df,

    x="T2",

    y="Qres",

    color=(

        group_var

        if group_var in outlier_df.columns

        else None

    ),

    hover_name="Sample",

    title=(

        "Hotelling T² vs Q residuals"

    )

)


fig_out.add_vline(

    x=Tsq_conf,

    line_dash="dash",

    line_color="red"

)


fig_out.add_hline(

    y=Q_conf,

    line_dash="dash",

    line_color="red"

)


st.plotly_chart(

    fig_out,

    use_container_width=True

)


st.subheader(

    "Detected outliers"

)


st.dataframe(

    outlier_df[

        outlier_df["Outlier"]

    ],

    use_container_width=True

)


# =====================================================
# DOWNLOAD RESULTS
# =====================================================

st.divider()

st.header(

    "Download PCA results"

)


buffer = io.BytesIO()


with pd.ExcelWriter(

    buffer,

    engine="openpyxl"

) as writer:

    scores.to_excel(

        writer,

        sheet_name="Scores"

    )


    loadings.to_excel(

        writer,

        sheet_name="Loadings"

    )


    ev_table.to_excel(

        writer,

        sheet_name="Explained_variance",

        index=False

    )


    outlier_df.to_excel(

        writer,

        sheet_name="Outliers",

        index=False

    )


st.download_button(

    "⬇️ Download PCA results",

    data=buffer.getvalue(),

    file_name="PCA_results.xlsx",

    mime=(

        "application/vnd.openxmlformats-officedocument."

        "spreadsheetml.sheet"

    )

)
