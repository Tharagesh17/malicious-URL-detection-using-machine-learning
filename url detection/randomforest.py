import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Handle categorical variables
def encode_categorical_features(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Model Training and Evaluation
def train_and_evaluate_models(df):
    # Check if 'type' column exists
    if 'type' not in df.columns:
        print("Error: 'type' column not found in the dataset.")
        return

    # Split data into features and target
    X = df.drop(columns=['type'])  
    y = df['type']
    
    # Check for NaN values and handle them
    if df.isnull().values.any():
        print("Missing values detected. Filling NaN values with the mean of numeric columns.")
        numeric_cols = df.select_dtypes(include=[int, float]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode categorical features
    df = encode_categorical_features(df)

    # Ensure all columns in X are numeric
    non_numeric_cols = X.select_dtypes(exclude=[int, float]).columns
    if len(non_numeric_cols) > 0:
        print(f"Non-numeric columns found: {non_numeric_cols.tolist()}. Dropping these columns.")
        X = X.drop(columns=non_numeric_cols)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the Random Forest model
    model = RandomForestClassifier()
    
    # Train and evaluate the model
    model.fit(X_train, y_train)
    
    # Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Model: Random Forest")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 50)

# Main execution
file_path = r"C:\Users\Asus\Downloads\processed_dataset.csv"   # Path to processed data
df = load_data(file_path)  # Load the processed data
train_and_evaluate_models(df)  # Train and evaluate the Random Forest model