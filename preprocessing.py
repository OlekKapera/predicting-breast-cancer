import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt

def dataset_info(df):
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())

    # Display basic statistics including min, max, mean
    print("\nBasic Statistics:")
    print(df.describe())

    # Count missing values in each column
    print("\nMissing Values Count:")
    print(df.isnull().sum())


def remove_outliers_lof(df):
    # Create a copy of the dataframe for LOF
    df_lof = df.copy()
    if 'Status' in df_lof.columns:
        df_lof = df_lof.drop('Status', axis=1)

    # Initialize and fit LOF
    lof = LocalOutlierFactor(n_neighbors=20)
    outlier_labels = lof.fit_predict(df_lof)

    # Inliers labeled as 1
    df_clean = df[outlier_labels == 1]
    df_outliers = df[outlier_labels == -1]

    # Print information about removed outliers
    print("\nNumber of outliers removed using LOF:", len(df) - len(df_clean))
    print("Shape of dataset after removing outliers:", df_clean.shape)
    return df_clean, df_outliers


def handle_missing_values(df):
    # Fill missing values with the mean of the column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


def normalize_data(df):
    # Normalize the data using Z-score standardization
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


def visualize_reduced_dimensions(inliers, outliers):
    # Prepare data
    _inliers = inliers.copy()
    _outliers = outliers.copy()
    if 'Status' in _inliers.columns:
        y1 = _inliers.pop('Status')
    if 'Status' in _outliers.columns:
        y2 = _outliers.pop('Status')

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    _inliers_reduced = pca.fit_transform(_inliers)
    _outliers_reduced = pca.fit_transform(_outliers)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(_inliers_reduced[:, 0], _inliers_reduced[:, 1], c='blue')
    scatter2 = plt.scatter(_outliers_reduced[:, 0], _outliers_reduced[:, 1], c='red')

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Visualization of Breast Cancer Dataset')
    plt.legend((scatter1, scatter2), ('Inliers', 'Outliers'))
    plt.show()

    # Print explained variance ratio
    print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", sum(pca.explained_variance_ratio_))


# Load the breast cancer dataset
df = pd.read_csv('Breast_Cancer_dataset.csv')

dataset_info(df)
handle_missing_values(df)
dataset_info(df)

# Encode categorical columns using OrdinalEncoder
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.difference(['Status'])
if len(categorical_cols) > 0:
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

normalize_data(df)

df_clean, df_outliers = remove_outliers_lof(df)

visualize_reduced_dimensions(df_clean, df_outliers)

df.to_csv('preprocessed_data.csv', index=False)
