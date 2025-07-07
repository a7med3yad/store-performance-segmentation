# Store Clustering and Analysis

This project applies advanced data mining techniques to cluster retail stores based on their characteristics and performance metrics. The workflow includes data preprocessing, outlier handling, dimensionality reduction (PCA), feature selection, and clustering using K-Medoids and Agglomerative Clustering. The results are visualized and interpreted to identify key differentiators among store clusters.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Workflow](#workflow)
- [Key Features](#key-features)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Visualization](#visualization)
- [License](#license)

## Project Overview

The goal is to segment stores into meaningful groups using unsupervised learning, enabling better business insights and targeted strategies. The analysis leverages both raw and transformed (PCA, VIF-selected) features to compare clustering performance.

## Data

- **Input:** `StoresData.xlsx`  
  Contains store-level data such as sales, wages, staff count, advertising, and car spaces.
- **Notebooks:**  
  - [`thebestever.ipynb`](thebestever.ipynb): Main analysis and clustering workflow.
  - [`practical.ipynb`](practical.ipynb): Additional clustering and data exploration.

## Workflow

1. **Data Preprocessing:**  
   - Load data, remove irrelevant columns, handle duplicates and missing values.
   - Outlier detection and removal using IQR.
   - Feature scaling with StandardScaler.

2. **Dimensionality Reduction:**  
   - Principal Component Analysis (PCA) to reduce dimensionality and visualize explained variance.

3. **Feature Selection:**  
   - Correlation analysis and Variance Inflation Factor (VIF) to remove multicollinear features.

4. **Clustering:**  
   - K-Medoids and Agglomerative Clustering on scaled, PCA-transformed, and VIF-selected data.
   - Silhouette score used to determine optimal number of clusters.

5. **Visualization:**  
   - Scatter plots, box plots, and KDE contours to interpret clusters.
   - Feature distributions within best-performing clusters.

## Key Features

- Robust data cleaning and preprocessing pipeline.
- Outlier detection and imputation.
- Dimensionality reduction with PCA.
- Feature selection using VIF.
- Multiple clustering algorithms with silhouette-based model selection.
- Comprehensive visualizations for cluster interpretation.

## Results

- Clustering performance is evaluated using silhouette scores.
- PCA-transformed data yielded the best clustering results.
- Cluster characteristics are visualized to identify high-performing store segments.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scikit-learn-extra
- statsmodels
- jupyter

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Usage

1. Place `StoresData.xlsx` in the project directory.
2. Open [`thebestever.ipynb`](thebestever.ipynb) in Jupyter Notebook or VS Code.
3. Run the notebook cells sequentially to reproduce the analysis and visualizations.

## Visualization

The notebook generates:
- Elbow and silhouette plots for cluster selection.
- PCA explained variance plots.
- Cluster scatter plots with KDE contours.
- Box plots and histograms for feature distributions by cluster.

## License

This project is licensed under the MIT License.
