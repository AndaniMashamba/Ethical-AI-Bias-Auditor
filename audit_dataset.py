import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import accuracy_score

print("Loading dataset (this might take a few seconds)...")
# Fetching the Adult Census dataset (predicting if income > 50K)
data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
df = data.frame.dropna()

# --- NEW CODE TO SEE THE DATA ---
print("\n--- FIRST 5 ROWS OF THE DATASET ---")
print(df.head())

# Save a copy to your folder so you can look at it later:
df.to_csv("adult_census_data.csv", index=False)
print("\n✅ Dataset saved as 'adult_census_data.csv' in your folder!\n")
# --------------------------------


# 1. Define Target and Sensitive Feature
# Target: Does the person make more than 50K? (1 = Yes, 0 = No)
y = (df['class'] == '>50K').astype(int)

# Sensitive feature we want to audit for bias (e.g., sex)
sensitive_feature = df['sex']

# 2. Prepare Data for Model
# Drop the target and sensitive feature from the training data, then encode categories
X = pd.get_dummies(df.drop(columns=['class', 'sex']))

# Split into training and testing sets
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42
)

# 3. Train a basic Machine Learning Model
print("Training the AI model...")
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# 4. The Audit: Evaluate Accuracy and Fairness
acc = accuracy_score(y_test, y_pred)
print(f"\n--- AUDIT REPORT ---")
print(f"Model Accuracy: {acc:.2%}")

# Demographic Parity Difference calculates the difference in selection rates between groups
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)

print(f"Demographic Parity Difference: {dpd:.4f}")

if dpd > 0.1:
    print("⚠️ WARNING: Significant bias detected! The model favors one demographic group over another.")
    print("Recommendation: Apply fairness mitigation algorithms (like Fairlearn's ExponentiatedGradient) before deployment.")
else:
    print("✅ Fairness metrics are within acceptable ranges.")