import numpy as np
from sklearn.model_selection import train_test_split
import unittest

def create_splits(X, y, test_size, random_state):
    # Split the dataset with the provided random seed
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Further split the temporary set into dev and test
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

class TestDatasetSplitting(unittest.TestCase):
    def test_same_random_seed(self):
        # Generate a simple dataset for demonstration
        X = np.random.rand(100, 10)
        y = np.random.randint(2, size=100)

        # Set a random seed
        seed = 42

        # Create splits with the random seed
        splits_1 = create_splits(X, y, test_size=0.4, random_state=seed)

        # Re-run the splitting with the same seed
        splits_2 = create_splits(X, y, test_size=0.4, random_state=seed)

        # Print or use the splits as needed
        print("\nResults with the same random seed:")
        print("Train set shape:", splits_1[0].shape)
        print("Dev set shape:", splits_1[1].shape)
        print("Test set shape:", splits_1[2].shape)

        # Assert that the splits are exactly the same
        self.assertTrue(np.array_equal(splits_1[0], splits_2[0]), "Train sets are not equal")
        self.assertTrue(np.array_equal(splits_1[1], splits_2[1]), "Dev sets are not equal")
        self.assertTrue(np.array_equal(splits_1[2], splits_2[2]), "Test sets are not equal")
        self.assertTrue(np.array_equal(splits_1[3], splits_2[3]), "Train labels are not equal")
        self.assertTrue(np.array_equal(splits_1[4], splits_2[4]), "Dev labels are not equal")
        self.assertTrue(np.array_equal(splits_1[5], splits_2[5]), "Test labels are not equal")

    def test_different_random_seed(self):
        # Generate a simple dataset for demonstration
        X = np.random.rand(100, 10)
        y = np.random.randint(2, size=100)

        # Set different random seeds
        seed_1 = 42
        seed_2 = 43

        # Create splits with the first seed
        splits_1 = create_splits(X, y, test_size=0.4, random_state=seed_1)

        # Create splits with the second seed
        splits_2 = create_splits(X, y, test_size=0.4, random_state=seed_2)

        # Print or use the splits as needed
        print("\nResults with different random seeds:")
        print("Train set shape with seed_1:", splits_1[0].shape)
        print("Dev set shape with seed_1:", splits_1[1].shape)
        print("Test set shape with seed_1:", splits_1[2].shape)
        print("Train set shape with seed_2:", splits_2[0].shape)
        print("Dev set shape with seed_2:", splits_2[1].shape)
        print("Test set shape with seed_2:", splits_2[2].shape)

        # Assert that the splits are not exactly the same
        self.assertFalse(np.array_equal(splits_1[0], splits_2[0]), "Train sets are equal")
        self.assertFalse(np.array_equal(splits_1[1], splits_2[1]), "Dev sets are equal")
        self.assertFalse(np.array_equal(splits_1[2], splits_2[2]), "Test sets are equal")
        self.assertFalse(np.array_equal(splits_1[3], splits_2[3]), "Train labels are equal")
        self.assertFalse(np.array_equal(splits_1[4], splits_2[4]), "Dev labels are equal")
        self.assertFalse(np.array_equal(splits_1[5], splits_2[5]), "Test labels are equal")

if __name__ == '__main__':
    unittest.main()
