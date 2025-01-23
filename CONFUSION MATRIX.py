import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load your dataset
data = pd.read_excel(r"C:\Users\dell\Pictures\ce671\Data.xlsx",sheet_name='2022')

# Step 2: Prepare training and validation sets with common feature names ('Longitude' and 'Latitude')
# Training data for each class
X_train_forest = data[['Forest_Train_Long', 'Forest_Train_Lat']].copy()
X_train_openland = data[['OpenLand_Train_Long', 'OpenLand_Train_Lat']].copy()
X_train_buildings = data[['Buildings_Train_Long', 'Buildings_Train_Lat']].copy()
X_train_roads = data[['Roads_Train_Long', 'Roads_Train_Lat']].copy()

# Rename columns to have consistent feature names
X_train_forest.columns = ['Longitude', 'Latitude']
X_train_openland.columns = ['Longitude', 'Latitude']
X_train_buildings.columns = ['Longitude', 'Latitude']
X_train_roads.columns = ['Longitude', 'Latitude']

# Combine all training data and create corresponding labels
X_train = pd.concat([X_train_forest, X_train_openland, X_train_buildings, X_train_roads], axis=0)
y_train = ['Forest'] * len(X_train_forest) + ['Open Land'] * len(X_train_openland) + \
          ['Buildings'] * len(X_train_buildings) + ['Roads'] * len(X_train_roads)

# Validation data for each class
X_test_forest = data[['Forest_Vali_Long', 'Forest_Vali_Lat']].copy()
X_test_openland = data[['OpenLand_Vali_Long', 'OpenLand_Vali_Lat']].copy()
X_test_buildings = data[['Buildings_Vali_Long', 'Buildings_Vali_Lat']].copy()
X_test_roads = data[['Roads_Vali_Long', 'Roads_Vali_Lat']].copy()

# Rename columns to have consistent feature names
X_test_forest.columns = ['Longitude', 'Latitude']
X_test_openland.columns = ['Longitude', 'Latitude']
X_test_buildings.columns = ['Longitude', 'Latitude']
X_test_roads.columns = ['Longitude', 'Latitude']

# Combine all validation data and create corresponding labels
X_test = pd.concat([X_test_forest, X_test_openland, X_test_buildings, X_test_roads], axis=0)
y_test = ['Forest'] * len(X_test_forest) + ['Open Land'] * len(X_test_openland) + \
         ['Buildings'] * len(X_test_buildings) + ['Roads'] * len(X_test_roads)

# Step 3: Handle missing values in the training and test data using imputation
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Step 4: Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 5: Make predictions on the validation set
y_pred = rf.predict(X_test)

# Step 6: Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 7: Compute overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)

# Step 8: Compute producer's and user's accuracy from confusion matrix
producer_accuracy = cm.diagonal() / cm.sum(axis=1)  # Recall per class (row sum)
user_accuracy = cm.diagonal() / cm.sum(axis=0)  # Precision per class (column sum)

# Step 9: Compute kappa coefficient
kappa = cohen_kappa_score(y_test, y_pred)

# Step 10: Plot the confusion matrix with accuracy metrics
fig, ax = plt.subplots(figsize=(10, 8))

# Create a new matrix with additional rows/columns for producer's and user's accuracy
extended_cm = np.vstack([cm, user_accuracy])  # Add user accuracy row
extended_cm = np.column_stack([extended_cm, np.append(producer_accuracy, overall_accuracy)])  # Add producer accuracy column

# Annotate the confusion matrix
sns.heatmap(extended_cm, annot=True, fmt='.2f', cmap="Blues", cbar=False,
            xticklabels=['Forest', 'Open Land', 'Buildings', 'Roads', 'Producer\'s Accuracy'],
            yticklabels=['Forest', 'Open Land', 'Buildings', 'Roads', 'User\'s Accuracy'])
plt.title(f'Confusion Matrix with \nOverall Accuracy: {overall_accuracy:.2f}, Kappa: {kappa:.2f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 11: Display classification report for precision, recall, and F1-score
print(classification_report(y_test, y_pred))
