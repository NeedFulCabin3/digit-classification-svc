import sys
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def main():
    try:
        print("[*] Loading digits dataset...")
        digits = load_digits()
        X, y = digits.data, digits.target
        
        print(f"    - Total samples loaded: {X.shape[0]}")
        print(f"    - Image feature dimensions: {X.shape[1]} pixels (8x8)")

        # Scale pixel values from [0, 16] to [0, 1] using MinMaxScaler
        print("[*] Scaling pixel values with MinMaxScaler...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training (80%) and testing (20%) sets
        print("[*] Splitting dataset into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Initialize and train the Support Vector Machine classifier
        print("[*] Training Support Vector Machine (SVC) model...")
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        print("[*] Evaluating model on test data...")
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[+] Model Accuracy: {accuracy * 100:.2f}%\n")
        
        print("Detailed Classification Report:")
        print("-" * 50)
        print(classification_report(y_test, y_pred))

        # Demonstrate a quick prediction on a few test samples
        print("[*] Sample Predictions (Actual vs Predicted):")
        for i in range(5):
            print(f"    Sample {i+1} -> Actual: {y_test[i]} | Predicted: {y_pred[i]}")

    except Exception as e:
        print(f"[!] An error occurred during execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()