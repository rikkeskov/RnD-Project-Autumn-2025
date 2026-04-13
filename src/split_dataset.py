import pandas as pd # type: ignore

def split_dataset(input_file_path: str, train_percentage: float = 0.8) -> None:
    """
    Divides a dataset into training and test sets and saves them as separate CSV files.

    Parameters:
    - input_file_path: Path to the input CSV file.
    - output_dir: Directory where the output CSV files will be saved.
    - train_percentage: Percentage of the data to be used for training (default is 80%).
    """

    # Load the CSV file
    df = pd.read_csv(input_file_path) # type: ignore

    # Calculate the split index
    split_idx = int(len(df) * train_percentage)

    # Split the dataset
    train_df = df.iloc[:split_idx, 1:2].reset_index(drop=True)
    test_df = df.iloc[split_idx:, 1:2].reset_index(drop=True)

    train_df.insert(0, 'index', range(len(train_df)))
    test_df.insert(0, 'index', range(len(test_df)))

    # Find file name only
    filename = input_file_path.split(".")[0]

    # Define output file paths
    train_file_path = filename + '_TRAIN.csv'
    test_file_path = filename + '_TEST.csv'

    # Save the training and test sets to CSV files
    train_df.to_csv(train_file_path, index=False, header=False)
    test_df.to_csv(test_file_path, index=False, header=False)

    print(f"Training set saved to {train_file_path}")
    print(f"Test set saved to {test_file_path}")

# Example usage
input_file_path = 'data/BIDMC32_dim_V.csv'
train_percentage = 0.8  # 80% for training, 20% for testing

split_dataset(input_file_path, train_percentage)