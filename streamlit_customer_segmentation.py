import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")
sns.set(style="whitegrid")

st.title("📊 Customer Segmentation Dashboard")

# Sidebar
menu = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "EDA", "Clustering", "PCA Visualization", "Downloads"]
)

# -------------------------------
# 1) UPLOAD DATA
# -------------------------------
if menu == "Upload Data":
    st.header("Upload Dataset")
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

    if uploaded:
        if uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)

        st.session_state["df"] = df
        st.success("Dataset Uploaded Successfully!")

        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# 2) EDA
# -------------------------------
if menu == "EDA":

    if "df" not in st.session_state:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state["df"]

    tab1, tab2, tab3 = st.tabs(["Overview", "Univariate", "Bivariate"])

    # OVERVIEW TAB
    with tab1:
        st.subheader("Basic Info")
        st.write("Shape:", df.shape)
        st.write(df.head())

        st.write("Missing Values")
        st.write(df.isnull().sum())

        st.write("Data Types")
        st.write(df.dtypes)

    # UNIVARIATE TAB
    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        col = st.selectbox("Select numeric column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[col], ax=ax2)
        st.pyplot(fig2)

    # BIVARIATE TAB
    with tab3:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:
            x = st.selectbox("X-axis", num_cols)
            y = st.selectbox("Y-axis", num_cols)

            fig3, ax3 = plt.subplots()
            sns.scatterplot(x=df[x], y=df[y], ax=ax3)
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(9,6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
            st.pyplot(fig4)

# -------------------------------
# 3) CLUSTERING
# -------------------------------
if menu == "Clustering":

    if "df" not in st.session_state:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state["df"]

    st.header("Run K-Means Clustering")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = st.multiselect("Select features", num_cols, default=num_cols)

    if st.button("Cluster"):

        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute optimal K
        sse, sil = [], []
        K_range = range(2, 11)

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X_scaled)
            sse.append(km.inertia_)
            sil.append(silhouette_score(X_scaled, labels))

        # Elbow
        st.subheader("Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(list(K_range), sse, marker='o')
        st.pyplot(fig)

        # Silhouette
        st.subheader("Silhouette Scores")
        fig2, ax2 = plt.subplots()
        ax2.plot(list(K_range), sil, marker='o')
        st.pyplot(fig2)

        best_k = int(K_range[np.argmax(sil)])
        st.success(f"Optimal K = {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=42)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        st.session_state["clustered_df"] = df
        st.session_state["X_scaled"] = X_scaled

        st.subheader("Cluster Counts")
        st.write(df["cluster"].value_counts())

        st.dataframe(df.head())

# -------------------------------
# 4) PCA VISUALIZATION
# -------------------------------
if menu == "PCA Visualization":

    if "clustered_df" not in st.session_state:
        st.warning("Run clustering first")
        st.stop()

    df = st.session_state["clustered_df"]
    X_scaled = st.session_state["X_scaled"]

    st.header("PCA 2D Cluster Visualization")

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(comps, columns=["PC1","PC2"])
    pca_df["cluster"] = df["cluster"]

    fig, ax = plt.subplots(figsize=(9,6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="tab10")
    st.pyplot(fig)

# -------------------------------
# 5) DOWNLOAD OUTPUTS
# -------------------------------
if menu == "Downloads":

    if "clustered_df" not in st.session_state:
        st.warning("Run clustering first")
        st.stop()

    df = st.session_state["clustered_df"]
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button("Download Clustered Dataset", csv, "customer_clusters.csv", "text/csv")
