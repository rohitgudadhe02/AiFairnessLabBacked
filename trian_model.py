import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
data = pd.read_csv('loan_approval_dataset.csv')

# NEW: Strip any hidden spaces from the column names!
data.columns = data.columns.str.strip()

# 2. Drop the ID column
if 'loan_id' in data.columns:
    data = data.drop('loan_id', axis=1)

# 3. Fill missing values
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# NEW FIX: Explicitly include 'object' and 'string' to silence the warning
text_cols = data.select_dtypes(include=['object', 'string']).columns
for col in text_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# 4. Convert all text columns into numbers
label_encoders = {}
for col in text_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 5. Separate the data into Features (X) and Target (y)
# Now 'loan_status' will perfectly match!
X = data.drop('loan_status', axis=1)
y = data['loan_status']

print("--- Data Processing Complete ---")
print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\nHere is what our data looks like now (all numbers!):")
print(X.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 6. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining the AI Model... Please wait...")

# 7. Create and Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Give the model its "final exam" using the Testing set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Training Complete ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 9. Save the trained model and the encoders to files
joblib.dump(model, 'loan_approval_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model and Encoders saved successfully! You can find the .pkl files in your project folder.")

