import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGClassifier, self).__init__()

        # Block 1: Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)

        # Block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        # Block 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)

        # Global Average Pooling replaces flatten monster
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers
        # With window_size 256 and three poolings ( 64 -> 32)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: [Batch, 1, 256]
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 256 -> 128
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 128 -> 64
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 64 -> 32


        x = self.global_pool(x)  # (B, 64, 1)
        x = x.squeeze(-1)  # (B, 64)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, epochs=50, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, correct = 0, 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        # Logging Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_ecg_model.pth')
            early_stop_counter = 0
            print("  --> Model saved!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the best weights before returning
    model.load_state_dict(torch.load('best_ecg_model.pth', weights_only=True))
    return model
