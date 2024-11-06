import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import LabelEncoder
import sklearn.feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


def feature_ranking(df):
    # Create feature matrix X by handling numeric and categorical columns separately
    X = df.copy()
    
    # Remove target column from features
    if 'Status' in X.columns:
        X = X.drop('Status', axis=1)

    # Perform feature ranking using chi-squared test
    ranking = sklearn.feature_selection.mutual_info_classif(X, df['Status'])
    
    # Create dictionary mapping feature names to their ranking scores
    feature_scores = dict(zip(X.columns, ranking))
    
    # Sort feature scores by value in descending order
    sorted_feature_scores = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
    return sorted_feature_scores
    

def feature_selection(df):
    # Prepare data
    X = df.copy()
    if 'Status' in X.columns:
        y = X.pop('Status')

    # Initialize logistic regression model
    model = DecisionTreeClassifier()

    # Initialize sequential feature selector
    sfs = sklearn.feature_selection.SequentialFeatureSelector(
        model,
        n_features_to_select='auto',
        direction='backward',
        scoring='accuracy',
    )
    
    # Fit the selector
    sfs.fit(X, y)

    # Get selected feature names
    selected_features = X.columns[sfs.get_support()].tolist()

    return selected_features



df = pd.read_csv('preprocessed_data.csv')
ranking = feature_ranking(df)
print(ranking)


# Drop features with ranking < 0.01
df_selected = df.copy()
low_ranked_features = [feature for feature, rank in ranking.items() if rank < 0.01]
df_selected = df_selected.drop(columns=low_ranked_features)

print("\nFeatures dropped due to low ranking:")
print(low_ranked_features)
print("\nRemaining features:", list(df_selected.columns))


selected_features = feature_selection(df)
print("\nSelected features:", selected_features)

# Create final dataset with only selected features (and Status column)
final_df = df[selected_features + ['Status']]

# Save the final dataset to CSV
final_df.to_csv('selected_features_data.csv', index=False)
print("\nFinal dataset saved to 'selected_features_data.csv' with", len(selected_features), "features")

# backward
# ['Race', 'Marital Status', 'T Stage ', '6th Stage', 'differentiate', 'Grade', 'Progesterone Status', 'Survival Months']

# forward
# ['Race', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'Survival Months']