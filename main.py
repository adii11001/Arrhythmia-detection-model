from torch.utils.data import DataLoader, random_split
import model
import preprocessing

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    segments, labels = preprocessing.loading_and_segmenting()
    full_dataset = preprocessing.ECGDataset(segments, labels)

    # 2. Split (70/15/15)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    # 3. Create Loaders (Note: only train is balanced)
    train_loader = preprocessing.balanced_loader(train_ds, batch_size=64)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 4. Initialize and Train
    ECGModel = model.ECGClassifier(num_classes=5)
    trained_model = model.train_model(ECGModel, train_loader, val_loader, 5)
