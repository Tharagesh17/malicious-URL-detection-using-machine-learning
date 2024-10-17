import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(file_path, sample_size=None):
    df = pd.read_csv(file_path)
    
    # Take a random sample if sample_size is specified
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)  # Set random_state for reproducibility
    
    return df

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['benign', 'defacement', 'malware', 'phishing'],
                yticklabels=['benign', 'defacement', 'malware', 'phishing'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Model Training and Evaluation
def train_and_evaluate_models(df):
    # Split data into features and target
    X = df.drop(columns=['type'])  # Ensure 'type' column is present in your processed dataset
    y = df['type']
    
    # Check for NaN values and handle them
    if df.isnull().values.any():
        print("Missing values detected. Filling NaN values with the mean of numeric columns.")
        numeric_cols = df.select_dtypes(include=[int, float]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


    # Convert categorical features to numeric if necessary
    for col in X.select_dtypes(include=['object']).columns:
        if col != 'type':
            label_encoder = LabelEncoder()
            X[col] = label_encoder.fit_transform(X[col])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Cross-Validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f"Model: {name}")
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
            
            # Plot the confusion matrix for the current model
            plot_confusion_matrix(y_test, y_pred, name)
        
        except Exception as e:
            print(f"An error occurred while evaluating {name}: {e}")

# Main execution
file_path = r"C:\Users\Tharagesh\Desktop\data science certificates\processed_dataset.csv"  # Path to processed data
sample_size = 10000  # Set the desired sample size here
df = load_data(file_path, sample_size)  # Load the processed data
train_and_evaluate_models(df)  # Train and evaluate models

