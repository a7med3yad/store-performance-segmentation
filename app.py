import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import dendrogram, linkage

st.title("Store Data Clustering and Analysis App")

uploaded_file = st.file_uploader("Upload your Stores Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Initial Data")
    st.dataframe(df.head())

    if "Store No." in df.columns:
        df = df.drop("Store No.", axis=1)

    n_df = df.select_dtypes(include=np.number)

    st.subheader("Boxplot Before Outlier Removal")
    df_melted = n_df.melt(var_name='Feature', value_name='Value')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Feature', y='Value', data=df_melted, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    def remove_outliers(feature):
        Q1 = feature.quantile(0.25)
        Q3 = feature.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return feature.where((feature >= lower) & (feature <= upper))

    for col in n_df.columns:
        n_df[col] = remove_outliers(n_df[col])

    n_df.fillna(n_df.median(), inplace=True)

    st.subheader("Boxplot After Outlier Removal")
    df_melted = n_df.melt(var_name='Feature', value_name='Value')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Feature', y='Value', data=df_melted, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(n_df), columns=n_df.columns)

    if 'Mng-Sex (Num)' in X_scaled.columns:
        X_scaled.drop('Mng-Sex (Num)', axis=1, inplace=True)

    st.subheader("PCA Explained Variance")
    pca = PCA()
    pca.fit(X_scaled)
    exp_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(exp_var)+1), exp_var, marker='o')
    ax.axhline(y=0.95, color='r', linestyle='--')
    ax.set_title("Cumulative Explained Variance by PCA")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Explained Variance")
    st.pyplot(fig)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

    best_score, best_k = -1, -1
    for k in range(2, 11):
        model = KMedoids(n_clusters=k, random_state=42).fit(X_pca)
        score = silhouette_score(X_pca, model.labels_)
        if score > best_score:
            best_score, best_k = score, k

    model = KMedoids(n_clusters=best_k, random_state=42).fit(X_pca)
    X_pca_df['Cluster'] = model.labels_

    st.subheader("K-Medoids Clustering (PCA View)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=X_pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

    vif_df = X_scaled.copy()
    removed_cols = []
    threshold = 10
    while True:
        vif = pd.DataFrame()
        vif["feature"] = vif_df.columns
        vif["VIF"] = [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])]
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            col_to_remove = vif.loc[vif["VIF"].idxmax(), "feature"]
            removed_cols.append(col_to_remove)
            vif_df.drop(columns=[col_to_remove], inplace=True)
        else:
            break

    st.subheader("Final Features After VIF Selection")
    st.write(vif_df.columns.tolist())

    st.subheader("Final Cluster Visual (with KDE)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=X_pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=50, alpha=0.7)
    for label in X_pca_df['Cluster'].unique():
        cluster_data = X_pca_df[X_pca_df['Cluster'] == label]
        sns.kdeplot(data=cluster_data, x="PC1", y="PC2", ax=ax, levels=[0.05], linewidths=1.2)
    st.pyplot(fig)
