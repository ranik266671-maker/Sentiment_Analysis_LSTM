import os
import re
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

# 1. Folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_PATH = os.path.join(
    DATA_DIR, "train-00000-of-00001.parquet"
)

TEST_PATH = os.path.join(
    DATA_DIR, "test-00000-of-00001.parquet"
)

# 2. Check dataset files
if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        "Dataset files not found. Check your dataset folder."
    )

# 3. Load IMDB dataset
print("Loading IMDB dataset...")

train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

print("Training data:", len(train_df))
print("Testing data:", len(test_df))

# 4. Split training data into train and validation
train_df = train_df.sample(
    frac=1, random_state=42
).reset_index(drop=True)

split_index = int(len(train_df) * 0.8)

training_df = train_df.iloc[:split_index]
validation_df = train_df.iloc[split_index:]

print("Training samples:", len(training_df))
print("Validation samples:", len(validation_df))

# 5. Tokenization
def tokenize(text):
    text = text.lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return words

# 6. Create vocabulary using training data
word_counts = Counter()

for review in training_df["text"]:
    word_counts.update(tokenize(review))

MAX_VOCAB_SIZE = 20000

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in word_counts.most_common(MAX_VOCAB_SIZE - 2):
    vocab[word] = len(vocab)

print("Vocabulary size:", len(vocab))

# 7. Convert reviews into numbers
MAX_LENGTH = 400

def encode_review(text):
    words = tokenize(text)

    sequence = [
        vocab.get(word, vocab["<UNK>"])
        for word in words
    ]

    # Truncate long reviews
    sequence = sequence[:MAX_LENGTH]

    # Handle empty reviews
    if len(sequence) == 0:
        sequence = [vocab["<UNK>"]]

    actual_length = len(sequence)

    # Padding short reviews
    sequence += [vocab["<PAD>"]] * (
        MAX_LENGTH - len(sequence)
    )

    return sequence, actual_length

# 8. Prepare tensors
def prepare_data(dataframe):
    sequences = []
    lengths = []
    labels = []

    for _, row in dataframe.iterrows():
        sequence, length = encode_review(row["text"])

        sequences.append(sequence)
        lengths.append(length)
        labels.append(int(row["label"]))

    X = torch.tensor(sequences, dtype=torch.long)
    L = torch.tensor(lengths, dtype=torch.long)
    Y = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(X, L, Y)

train_dataset = prepare_data(training_df)
validation_dataset = prepare_data(validation_df)
test_dataset = prepare_data(test_df)

# 9. Create DataLoaders
BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

# 10. Save processed data
save_path = os.path.join(BASE_DIR, "phase1_data.pt")

torch.save({
    "train_data": train_dataset.tensors,
    "validation_data": validation_dataset.tensors,
    "test_data": test_dataset.tensors,
    "vocab": vocab,
    "vocab_size": len(vocab),
    "max_length": MAX_LENGTH
}, save_path)

# 11. Display results
print("\nPhase 1 Completed Successfully!")

print("Vocabulary size:", len(vocab))
print("Maximum review length:", MAX_LENGTH)

print("Train batches:", len(train_loader))
print("Validation batches:", len(validation_loader))
print("Test batches:", len(test_loader))

# Display one sample
X, L, Y = next(iter(train_loader))

print("\nSample review shape:", X.shape)
print("Sample review length:", L[0].item())
print("Sample sentiment label:", Y[0].item())

print("\nProcessed data saved as phase1_data.pt")