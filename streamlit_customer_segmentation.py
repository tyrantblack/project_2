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
sns.set(style="whitegrid")

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation (K-Means) Dashboard")


# -------------------------------------------------
# Sidebar Menu
# -------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "EDA", "Clustering", "PCA Visualization", "Downloads"]
)


# -------------------------------------------------
# 1️⃣ UPLOAD DATA
# -------------------------------------------------
if menu == "Upload Data":

    st.header("📥 Upload Dataset")
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

    if uploaded:
        if uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)

        st.session_state["df"] = df

        st.success("Dataset uploaded successfully! 🎉")
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")


# -------------------------------------------------
# 2️⃣ EDA SECTION
# -------------------------------------------------
if menu == "EDA":

    if "df" not in st.session_state:
        st.warning("⚠ Please upload data first.")
        st.stop()

    df = st.session_state["df"]

    tab1, tab2, tab3 = st.tabs(["Overview", "Univariate", "Bivariate"])

    # -------- 2.1 Overview --------
    with tab1:
        st.subheader("📌 Data Overview")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Missing Values")
            st.write(df.isnull().sum())

        with col2:
            st.write("### Duplicate Rows")
            st.write(df.duplicated().sum())

        st.write("### Data Types")
        st.write(df.dtypes)

    # -------- 2.2 Univariate --------
    with tab2:
        st.subheader("📊 Univariate Analysis")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            col = st.selectbox("Select numeric column", num_cols)

            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[col], ax=ax2)
            st.pyplot(fig2)
        else:
            st.info("No numeric columns found.")

    # -------- 2.3 Bivariate --------
    with tab3:
        st.subheader("🔗 Bivariate Analysis")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:

            x = st.selectbox("X-axis", num_cols)
            y = st.selectbox("Y-axis", num_cols)

            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x], y=df[y], ax=ax)
            st.pyplot(fig)

            # ------- PROFESSIONAL CORRELATION HEATMAP -------
            st.write("### Correlation Heatmap (Professional)")

            try:
                corr = df[num_cols].corr()

                fig = plt.figure(figsize=(14, 10))

                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    annot_kws={"size": 6},
                    cbar_kws={"shrink": 0.8},
                    linewidths=0.3,
                    linecolor="white"
                )

                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.title("Correlation Matrix (Professional)", fontsize=16, pad=15)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating heatmap: {e}")

        else:
            st.info("Not enough numeric columns for correlation.")


# -------------------------------------------------
# 3️⃣ CLUSTERING (STREAMLIT CLOUD SAFE)
# -------------------------------------------------
if menu == "Clustering":

    if "df" not in st.session_state:
        st.warning("⚠ Please upload data first.")
        st.stop()

    df = st.session_state["df"].copy()

    st.header("🧮 K-Means Clustering")

    df = df.apply(pd.to_numeric, errors="ignore")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        st.error("Not enough numeric columns for clustering.")
        st.stop()

    features = st.multiselect("Select features for clustering:", num_cols, default=num_cols)

    df[features] = df[features].fillna(df[features].median())

    if st.button("Run Clustering"):

        scaler = StandardScaler()
        X = df[features]
        X_scaled = scaler.fit_transform(X)

        K_range = range(2, 11)
        sse, sil = [], []

        for k in K_range:
            try:
                km = KMeans(n_clusters=k, random_state=42)
                labels = km.fit_predict(X_scaled)
                sse.append(km.inertia_)
                sil.append(silhouette_score(X_scaled, labels))
            except:
                sse.append(None)
                sil.append(None)

        st.subheader("📉 Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(list(K_range), sse, marker="o")
        st.pyplot(fig)

        st.subheader("📈 Silhouette Scores")
        fig2, ax2 = plt.subplots()
        ax2.plot(list(K_range), sil, marker="o")
        st.pyplot(fig2)

        best_k = int(K_range[np.argmax(sil)])
        st.success(f"Optimal number of clusters = **{best_k}**")

        km = KMeans(n_clusters=best_k, random_state=42)
        df["cluster"] = km.fit_predict(X_scaled)

        st.session_state["clustered_df"] = df
        st.session_state["X_scaled"] = X_scaled

        st.subheader("📊 Cluster Counts")
        st.write(df["cluster"].value_counts())

        st.dataframe(df.head(), use_container_width=True)


# -------------------------------------------------
# 4️⃣ PCA VISUALIZATION
# -------------------------------------------------
if menu == "PCA Visualization":

    if "clustered_df" not in st.session_state:
        st.warning("⚠ Please run clustering first.")
        st.stop()

    df = st.session_state["clustered_df"]
    X_scaled = st.session_state["X_scaled"]

    st.header("🎨 PCA Visualization (2D)")

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
    pca_df["cluster"] = df["cluster"]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=100)
    st.pyplot(fig)


# -------------------------------------------------
# 5️⃣ DOWNLOAD OUTPUTS
# -------------------------------------------------
if menu == "Downloads":

    if "clustered_df" not in st.session_state:
        st.warning("⚠ Please run clustering first.")
        st.stop()

    df = st.session_state["clustered_df"]

    st.header("📥 Download Outputs")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Clustered Dataset (CSV)",
        csv,
        "customer_clusters.csv",
        "text/csv"
    )

    st.success("CSV ready for download! 🎯")
