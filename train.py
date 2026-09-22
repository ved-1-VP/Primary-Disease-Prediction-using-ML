import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_df = pd.read_csv("Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the pkl files
joblib.dump(model, "disease_prediction_model.pkl")
joblib.dump(list(X_train.columns), "feature_columns.pkl")
print("Successfully generated disease_prediction_model.pkl and feature_columns.pkl")