import streamlit as st


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Food Chemometrics Studio",
    page_icon="🧪",
    layout="wide"
)

# -----------------------------------------------------
# BACKGROUND GRADIENT
# -----------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f3ef 0%, #e8ddd1 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TITLE
# =====================================================

st.title("Food Chemometrics Studio")

st.write(
    "Welcome!"
)


st.markdown(
    """
    Integrated platform for food data analysis - Food Chemistry Group DSTF

- Data import
- Experimental setup
- Preprocessing
- Exploratory analysis
- PCA
- Machine learning
    """
)