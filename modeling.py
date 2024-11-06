from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from knn import KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier


# Load the selected features dataset
data = pd.read_csv('selected_features_data.csv')
X = data.drop('Status', axis=1)
y = data['Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


### KNN
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test, predictions))

### Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy}")


### C4.5 Decision Tree
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"C4.5 Decision Tree Accuracy: {dt_accuracy}")

### Random Forest
rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

### Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb.fit(X_train, y_train)
gb_predictions = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f"Gradient Boosting Accuracy: {gb_accuracy}")

### Neural Network
nn = MLPClassifier(
    hidden_layer_sizes=(8),  # One hidden layers with 8 neurons
    activation='relu',
    solver='adam',
    max_iter=1000,
)

nn.fit(X_train, y_train)
nn_predictions = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
print(f"Neural Network Accuracy: {nn_accuracy}")


### Hyperparameter Tuning

# Random Forest parameter grid
rf_param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': [3, 5, 10, 15, 20, None]
}

# Gradient Boosting parameter grid
gb_param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2, 0.3]
}

# Perform Grid Search for Random Forest
rf_grid = GridSearchCV(
    RandomForestClassifier(criterion='entropy'),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

# Perform Grid Search for Gradient Boosting
gb_grid = GridSearchCV(
    GradientBoostingClassifier(),
    gb_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
gb_grid.fit(X_train, y_train)

# Create results table for Random Forest
rf_results = pd.DataFrame(rf_grid.cv_results_)
rf_results = rf_results.sort_values('mean_test_score', ascending=False)
print("\nRandom Forest Grid Search Results:")
print("Best parameters:", rf_grid.best_params_)
print("Best accuracy: {:.4f}".format(rf_grid.best_score_))
print("\nTop 5 parameter combinations:")
print(rf_results[['params', 'mean_test_score']].head())

# Create results table for Gradient Boosting
gb_results = pd.DataFrame(gb_grid.cv_results_)
gb_results = gb_results.sort_values('mean_test_score', ascending=False)
print("\nGradient Boosting Grid Search Results:")
print("Best parameters:", gb_grid.best_params_)
print("Best accuracy: {:.4f}".format(gb_grid.best_score_))
print("\nTop 5 parameter combinations:")
print(gb_results[['params', 'mean_test_score']].head())

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Random Forest heatmap
rf_pivot = rf_results.pivot_table(
    index='param_max_depth',
    columns='param_n_estimators',
    values='mean_test_score'
)
sns.heatmap(rf_pivot, annot=True, fmt='.3f', ax=ax1, cmap='YlOrRd')
ax1.set_title('Random Forest Accuracy\nby max_depth and n_estimators')

# Gradient Boosting heatmap
gb_pivot = gb_results.pivot_table(
    index='param_learning_rate',
    columns='param_n_estimators',
    values='mean_test_score'
)
sns.heatmap(gb_pivot, annot=True, fmt='.3f', ax=ax2, cmap='YlOrRd')
ax2.set_title('Gradient Boosting Accuracy\nby learning_rate and n_estimators')

plt.tight_layout()
plt.show()

## Feature Importance Random Forest
# Train Random Forest with best parameters
best_rf = RandomForestClassifier(n_estimators=1000, max_depth=5, criterion='entropy')
best_rf.fit(X_train, y_train)

# Get feature importance scores
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
})

# Sort features by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

### Feature Importance Gradient Boosting
# Train Gradient Boosting with best parameters
best_gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2)
best_gb.fit(X_train, y_train)

# Get feature importance scores
gb_feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_gb.feature_importances_
})

# Sort features by importance
gb_feature_importance = gb_feature_importance.sort_values('importance', ascending=False)

print("\nGradient Boosting Feature Importance:")
print(gb_feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=gb_feature_importance)
plt.title('Gradient Boosting Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
