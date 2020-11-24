import joblib
import os

from sklearn.model_selection import train_test_split

def generate_datasets(features, labels):

    datasets = []
    
    train_test_splits = train_test_split(features, labels, test_size=0.1, random_state=42)
    train_val_features = train_test_splits[0]
    test_features = train_test_splits[1]
    train_val_labels = train_test_splits[2]
    test_labels = train_test_splits[3]

    train_val_splits = train_test_split(train_val_features, train_val_labels, test_size=0.1, random_state=42)
    train_features = train_val_splits[0]
    val_features = train_val_splits[1]
    train_labels = train_val_splits[2]
    val_labels = train_val_splits[3]

    datasets.append(("mfcc_train.joblib", train_features))
    datasets.append(("labels_train.joblib", train_labels))
    datasets.append(("mfcc_val.joblib", val_features))
    datasets.append(("labels_val.joblib", val_labels))
    datasets.append(("mfcc_test.joblib", test_features))
    datasets.append(("labels_test.joblib", test_labels))

    return datasets




if __name__ == "__main__":
    
    data_dir = "../../data/features/sequential"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.normpath(os.path.join(script_dir, data_dir))

    mfcc_train_val = joblib.load(os.path.join(data_dir_path, "mfcc_train_val.joblib"))
    labels_train_val = joblib.load(os.path.join(data_dir_path, "labels_train_val.joblib"))

    print(mfcc_train_val.shape)
    print(labels_train_val.shape)
    datasets = generate_datasets(mfcc_train_val, labels_train_val)

    for dataset in datasets:
        output_file_path = os.path.join(data_dir_path, dataset[0])
        joblib.dump(dataset[1], output_file_path)

