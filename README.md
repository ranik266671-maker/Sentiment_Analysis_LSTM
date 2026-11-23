# Sentiment Analysis Using LSTM

## Project Overview

This project is a Deep Learning-based Sentiment Analysis system that classifies movie reviews as **Positive** or **Negative** using a Long Short-Term Memory (LSTM) neural network.

The model is trained on the IMDB Movie Reviews dataset containing 50,000 labeled movie reviews. It uses Natural Language Processing (NLP) techniques to process text and learn sentiment patterns.

## Objectives

- Understand and implement LSTM networks.
- Perform text preprocessing and tokenization.
- Convert text reviews into numerical sequences.
- Train a deep learning model for sentiment classification.
- Evaluate model performance using accuracy and loss.
- Predict sentiment for new and longer movie reviews.

## Dataset

**Dataset:** IMDB Movie Reviews Dataset

- Total reviews: 50,000
- Training reviews: 25,000
- Testing reviews: 25,000
- Positive reviews: 25,000
- Negative reviews: 25,000

Dataset source: Hugging Face – Stanford IMDB Dataset.

The dataset is downloaded in Parquet format.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Natural Language Processing (NLP)
- Deep Learning
- LSTM (Long Short-Term Memory)

## Project Structure

```text
Sentiment_Analysis_LSTM/
│
├── dataset/
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
│
├── phase1_preprocessing.py
├── phase1_data.pt
│
├── phase2_lstm.py
├── best_lstm_model.pth
├── accuracy_graph.png
├── loss_graph.png
├── training_history.pt
│
├── phase3_evaluation.py
├── phase3_results.txt
│
└── README.md
```

## Project Implementation

### Phase 1: Data Acquisition and Preprocessing

- Loaded the IMDB dataset using Pandas.
- Divided the training data into training and validation sets.
- Applied text tokenization.
- Created a vocabulary of 20,000 words.
- Converted text into numerical sequences.
- Applied padding and truncation.
- Set maximum review length to 400 tokens.
- Created PyTorch tensors and DataLoaders.

### Phase 2: LSTM Model Training

Developed an LSTM-based neural network using PyTorch.

**Model Architecture:**

- Embedding Layer: 100 dimensions
- Bidirectional LSTM: 64 hidden units per direction
- Dropout: 0.3
- Fully Connected Layer: 128 input features
- Output Layer: 1 neuron
- Activation: Sigmoid during prediction

**Training Configuration:**

| Parameter | Value |
|---|---|
| Vocabulary Size | 20,000 |
| Maximum Sequence Length | 400 |
| Embedding Dimension | 100 |
| Hidden Units | 64 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Epochs | 8 |
| Optimizer | Adam |
| Loss Function | BCEWithLogitsLoss |

The best-performing model is saved as `best_lstm_model.pth`.

### Phase 3: Model Evaluation and Prediction

- Loaded the trained LSTM model.
- Evaluated the model using the test dataset.
- Calculated test accuracy and test loss.
- Implemented custom review prediction.
- Tested positive and negative movie reviews.
- Tested longer reviews to examine model performance.

## Model Performance

Latest training results:

| Metric | Result |
|---|---:|
| Training Accuracy | 90.15% |
| Validation Accuracy | 84.50% |
| Training Loss | 0.2449 |
| Validation Loss | 0.3780 |

The final test accuracy is calculated separately during Phase 3 evaluation.

## Generated Outputs

The project generates the following files:

- `best_lstm_model.pth` – Trained LSTM model.
- `accuracy_graph.png` – Training and validation accuracy graph.
- `loss_graph.png` – Training and validation loss graph.
- `training_history.pt` – Training history.
- `phase1_data.pt` – Preprocessed dataset.
- `phase3_results.txt` – Test evaluation and custom prediction results.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd Sentiment_Analysis_LSTM
```

### 3. Install Required Libraries

```bash
pip install torch pandas pyarrow matplotlib
```

### 4. Download the Dataset

Download the Stanford IMDB dataset from Hugging Face.

Place the training and testing Parquet files inside the `dataset` folder.

### 5. Run Phase 1

```bash
python phase1_preprocessing.py
```

### 6. Run Phase 2

```bash
python phase2_lstm.py
```

### 7. Run Phase 3

```bash
python phase3_evaluation.py
```

## Applications

- Movie review sentiment classification.
- Customer feedback analysis.
- Product review analysis.
- Social media opinion analysis.
- Text classification systems.

## Future Improvements

- Improve prediction accuracy.
- Experiment with different LSTM architectures.
- Implement attention mechanisms.
- Test longer and more complex reviews.
- Develop a user-friendly web interface.

## Conclusion

This project demonstrates how Deep Learning and Natural Language Processing can be used to perform sentiment analysis.

By implementing a Bidirectional LSTM model, the system learns sentiment patterns from movie reviews and predicts whether a review expresses positive or negative sentiment.

The project provides practical experience in text preprocessing, neural network training, model evaluation, and real-world text classification.

---

**Project Name:** Sentiment Analysis Using LSTM  
**Domain:** Artificial Intelligence and Machine Learning  
**Framework:** PyTorch  
**Language:** Python
