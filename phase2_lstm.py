import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# 1. Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# 2. Load processed data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data = torch.load(
    os.path.join(BASE_DIR, "phase1_data.pt"),
    map_location="cpu",
    weights_only=True
)

train_dataset = TensorDataset(*data["train_data"])
val_dataset = TensorDataset(*data["validation_data"])

VOCAB_SIZE = data["vocab_size"]

# 3. DataLoaders
BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# 4. LSTM Model
class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 100, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        output, (hidden, cell) = self.lstm(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden), dim=1
        )

        combined = self.dropout(combined)

        return self.fc(combined).squeeze(1)


# 5. Initialize model
model = SentimentLSTM(VOCAB_SIZE).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 8

# 6. Training
history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_accuracy = 0

print("\nStarting LSTM Model Training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for reviews, lengths, labels in train_loader:

        reviews = reviews.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        outputs = model(reviews, lengths)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = (torch.sigmoid(outputs) >= 0.5).long()

        correct += (predictions == labels.long()).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total

    # Validation
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for reviews, lengths, labels in val_loader:

            reviews = reviews.to(device)
            labels = labels.float().to(device)

            outputs = model(reviews, lengths)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = (torch.sigmoid(outputs) >= 0.5).long()

            val_correct += (
                predictions == labels.long()
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            os.path.join(BASE_DIR, "best_lstm_model.pth")
        )


# 7. Save training history
torch.save(
    history,
    os.path.join(BASE_DIR, "training_history.pt")
)

# 8. Accuracy Graph
plt.figure(figsize=(8, 5))

plt.plot(history["train_accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.title("LSTM Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(
    os.path.join(BASE_DIR, "accuracy_graph.png")
)

plt.close()

# 9. Loss Graph
plt.figure(figsize=(8, 5))

plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.title("LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    os.path.join(BASE_DIR, "loss_graph.png")
)

plt.close()

# 10. Final results
print("\nPhase 2 Completed Successfully!")

print(
    "Best Validation Accuracy:",
    f"{best_accuracy * 100:.2f}%"
)

print("\nGenerated Files:")
print("1. best_lstm_model.pth")
print("2. accuracy_graph.png")
print("3. loss_graph.png")
print("4. training_history.pt")