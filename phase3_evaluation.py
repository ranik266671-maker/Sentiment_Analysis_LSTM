import os
import re
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# 1. Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Load processed data
data = torch.load(
    os.path.join(BASE_DIR, "phase1_data.pt"),
    map_location="cpu",
    weights_only=True
)

vocab = data["vocab"]
VOCAB_SIZE = data["vocab_size"]
MAX_LENGTH = data["max_length"]

test_dataset = TensorDataset(*data["test_data"])

test_loader = DataLoader(
    test_dataset,
    batch_size=64
)

# 3. Define LSTM model
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


# 4. Load trained model
model = SentimentLSTM(VOCAB_SIZE).to(device)

model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "best_lstm_model.pth"),
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Trained LSTM model loaded successfully!")

# 5. Evaluate on test dataset
correct = 0
total = 0
test_loss = 0

criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():

    for reviews, lengths, labels in test_loader:

        reviews = reviews.to(device)
        labels = labels.float().to(device)

        outputs = model(reviews, lengths)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).long()

        correct += (
            predictions == labels.long()
        ).sum().item()

        total += labels.size(0)

test_accuracy = correct / total * 100
average_loss = test_loss / len(test_loader)

print("\nTEST RESULTS")
print("-------------------------")
print("Test Accuracy:", f"{test_accuracy:.2f}%")
print("Test Loss:", f"{average_loss:.4f}")


# 6. Tokenization for custom reviews
def tokenize(text):

    text = text.lower()

    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


# 7. Convert custom review into numbers
def encode_review(text):

    words = tokenize(text)

    sequence = [
        vocab.get(word, vocab["<UNK>"])
        for word in words
    ]

    sequence = sequence[:MAX_LENGTH]

    if len(sequence) == 0:
        sequence = [vocab["<UNK>"]]

    length = len(sequence)

    sequence += [vocab["<PAD>"]] * (
        MAX_LENGTH - len(sequence)
    )

    return sequence, length


# 8. Custom review prediction
def predict_sentiment(review):

    sequence, length = encode_review(review)

    review_tensor = torch.tensor(
        [sequence], dtype=torch.long
    ).to(device)

    length_tensor = torch.tensor([length])

    with torch.no_grad():

        output = model(review_tensor, length_tensor)

        probability = torch.sigmoid(output).item()

    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return sentiment, probability


# 9. Test five custom reviews
custom_reviews = [
    "This movie was absolutely amazing. The story was interesting from beginning to end. The actors performed brilliantly, the direction was excellent, and the background music made every scene enjoyable. I would definitely recommend this movie to everyone.",

    "I was very disappointed with this movie. The story was boring and predictable. The acting was poor, the dialogues were weak, and the direction was not impressive. I wasted my time watching it and would not recommend it.",

    "At first, I thought this movie would be terrible because the trailer was not impressive. However, the story became interesting after the first few scenes. The actors performed naturally, the music was good, and the ending was satisfying. Overall, I really enjoyed it.",

    "The movie had some good moments, but most of the story was disappointing. The acting was average, the screenplay was slow, and several scenes felt unnecessary. Although the cinematography was beautiful, it could not save the weak storyline. I would not watch it again.",

    "I did not expect much from this movie, but it surprised me. The characters were well developed, the story was engaging, and the performances were excellent. Although a few scenes were slow, the overall experience was enjoyable. It was definitely worth watching."
]

print("\nCUSTOM REVIEW PREDICTIONS")
print("-------------------------")

results = []

for i, review in enumerate(custom_reviews, 1):

    sentiment, probability = predict_sentiment(review)

    result = (
        f"\nReview {i}: {review}\n"
        f"Prediction: {sentiment}\n"
        f"Positive probability: {probability * 100:.2f}%"
    )

    print(result)

    results.append(result)


# 10. Save results
result_path = os.path.join(
    BASE_DIR, "phase3_results.txt"
)

with open(result_path, "w", encoding="utf-8") as file:

    file.write("IMDB SENTIMENT ANALYSIS RESULTS\n")
    file.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    file.write(f"Test Loss: {average_loss:.4f}\n")

    for result in results:
        file.write(result + "\n")

print("\nPhase 3 Completed Successfully!")
print("Results saved in phase3_results.txt")