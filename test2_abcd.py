import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Classifier and Random State Demo")

    # Add command line arguments
    parser.add_argument("--clf_name", type=str, choices=["svm", "tree"], help="Classifier name (svm or tree)")
    parser.add_argument("--random_state", type=int, help="Random seed/state value for dataset splitting")

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
        svm_classifier = SVC(kernel='rbf', random_state=args.random_state)

        # Train the SVM classifier
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = svm_classifier.predict(X_test)

        # Print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

    elif args.clf_name == "tree":
        # Example for Decision Tree
        print("Using Decision Tree classifier")

        # Set random_state for both dataset splitting and the Decision Tree classifier
        # decision_tree_classifier = DecisionTreeClassifier(random_state=args.random_state)
        
        # By setting splitter="best", the best split is always chosen, making the Decision Tree results fully reproducible
        decision_tree_classifier = DecisionTreeClassifier(random_state=args.random_state, splitter="best")


        # Train the Decision Tree classifier
        decision_tree_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = decision_tree_classifier.predict(X_test)

        # Print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
