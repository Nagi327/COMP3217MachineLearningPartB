# Imports necessary libraries/modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import torch

# Load the training data from a CSV file
data = pd.read_csv('./data/TrainingDataMulti.csv', header=None)

# Extract features and labels from the data
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

# Convert the data to PyTorch tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Print the training and testing tensors
print(train_features)
print(train_labels)
print(test_features)
print(test_labels)

# Count the number of samples for each label in the training and testing sets
test_label_counts = torch.unique(test_labels, return_counts=True)
train_label_counts = torch.unique(train_labels, return_counts=True)

print("Test Label Counts:")
for label, count in zip(test_label_counts[0], test_label_counts[1]):
    print(f"Label {label}: {count}")

print("\nTrain Label Counts:")
for label, count in zip(train_label_counts[0], train_label_counts[1]):
    print(f"Label {label}: {count}")

# Create an instance of the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=271, random_state=42)

# Find the best n_estimators for rfc
cross = []
for i in range(0, 300, 10):
    rfc = RandomForestClassifier(n_estimators=i+1, random_state=0)
    cross_score = cross_val_score(rfc, train_features, train_labels, cv=5).mean()
    cross.append(cross_score)
plt.plot(range(1, 301, 10), cross)
plt.xlabel('n_estimators')
plt.ylabel('acc')
plt.show()
print("The best n_estimators")
print((cross.index(max(cross))*10)+1,max(cross))

# Train the model
model.fit(train_features, train_labels)

# Predict the labels for training features
predicted_train_labels = model.predict(train_features)

# Evaluate the model on the testing data
predicted_labels = model.predict(test_features)

# Compute classification metrics
training_accuracy = accuracy_score(train_labels, predicted_train_labels)
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average='weighted')
recall = recall_score(test_labels, predicted_labels, average='weighted')
f1 = f1_score(test_labels, predicted_labels, average='weighted')

# Print the classification metrics
print("Classification Metrics:")
print("Training Accuracy:", training_accuracy)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Compute confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1", "2"])
disp.plot(cmap=plt.cm.Reds, colorbar=True)
plt.title("Confusion Matrix")
plt.show()

# Read the testing data from a CSV file
data = pd.read_csv('./data/TestingDataMulti.csv', header=None)
test_features = data.values

# Perform inference on the testing data
predicted_labels = model.predict(test_features)

# Show all predicted labels
print(predicted_labels)

# Create a DataFrame with predicted labels
result_df = pd.DataFrame({'Predicted Label': predicted_labels})

# Save the predicted labels to a CSV file
result_df.to_csv('TestingResultsMulti.csv', index=False)

