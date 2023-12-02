import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import os

def main():
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Classifier and Random State Demo")

    # Add command line arguments
    parser.add_argument("--clf_name", type=str, choices=["svm", "tree"], help="Classifier name (svm or tree)")
    parser.add_argument("--random_state", type=int, help="Random seed/state value for dataset splitting")

    # Additional SVM hyperparameters
    parser.add_argument("--gamma", type=float, default=1.0, help="SVM hyperparameter gamma")
    parser.add_argument("--C", type=float, default=1.0, help="SVM hyperparameter C")

    # Additional Decision Tree hyperparameters
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the Decision Tree")

    # Parse the command line arguments
    args = parser.parse_args()

    # Load digits dataset (just for demonstration)
    digits = load_digits()
    X, y = digits.data, digits.target

    # Split the dataset with the specified random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

    if args.clf_name == "svm":
        # Example for SVM
        print("Using SVM classifier")

        # Initialize SVM classifier with a radial basis function (RBF) kernel
        svm_classifier = SVC(kernel='rbf', gamma=args.gamma, C=args.C, random_state=args.random_state)

        # Train the SVM classifier
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = svm_classifier.predict(X_test)

        # Print and save accuracy and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"test accuracy: {accuracy}")
        print(f"test macro-f1: {macro_f1}")

        # Save the model
        model_filename = f"./models/svm_gamma={args.gamma}_C={args.C}_{args.random_state}.joblib"
        dump(svm_classifier, model_filename)
        print(f"model saved at {model_filename}")

    elif args.clf_name == "tree":
        # Example for Decision Tree
        print("Using Decision Tree classifier")

        # Set random_state for both dataset splitting and the Decision Tree classifier
        decision_tree_classifier = DecisionTreeClassifier(random_state=args.random_state, splitter="best", max_depth=args.max_depth)

        # Train the Decision Tree classifier
        decision_tree_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = decision_tree_classifier.predict(X_test)

        # Print and save accuracy and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"test accuracy: {accuracy}")
        print(f"test macro-f1: {macro_f1}")

        # Save the model
        model_filename = f"./models/tree_max_depth={args.max_depth}_{args.random_state}.joblib"
        dump(decision_tree_classifier, model_filename)
        print(f"model saved at {model_filename}")

    # Save results to a file
    results_filename = f"results/{args.clf_name}_{args.random_state}.txt"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, 'w') as results_file:
        results_file.write(f"test accuracy: {accuracy}\n")
        results_file.write(f"test macro-f1: {macro_f1}\n")
        results_file.write(f"model saved at {model_filename}\n")

if __name__ == "__main__":
    main()
