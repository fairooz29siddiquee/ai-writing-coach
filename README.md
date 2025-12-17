# AI Writing Coach

## Project Overview
The AI Writing Coach is a Natural Language Processing (NLP) system that analyzes user-written text and provides structured, persona-based feedback. The system is designed to simulate an AI writing coach by combining classical NLP techniques, machine learning classifiers, and rule-based feedback generation.

The goal of this project is to demonstrate how interpretable NLP models can be integrated into an AI-style interaction loop without relying on external APIs.

---

## System Pipeline
User Text → NLP Model → Interpretable Output → AI-Style Feedback

1. The user provides a text sample.
2. The system preprocesses the text using classical NLP techniques.
3. A trained classifier predicts the writing style.
4. The prediction and confidence scores are used to generate coaching feedback.

---

## Dataset
This project uses the **Sentiment140 dataset**, which contains labeled tweets for sentiment analysis.

- Source: Kaggle  
- File used: `training.1600000.processed.noemoticon.csv`
- Labels:
  - `0` → Negative
  - `4` → Positive (mapped to 1)

For computational efficiency, a random subset of 5,000 samples is used during training.

---

## Classical NLP Pipeline
The classical NLP pipeline includes:

- Tokenization (NLTK)
- Stopword removal
- Text normalization
- TF-IDF vectorization (unigrams + bigrams)
- Feature extraction for interpretability

---

## Machine Learning Models
The following models are trained and evaluated:

- Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression

Each model is evaluated using:
- Precision
- Recall
- F1-score
- Confusion Matrix (saved as PNG files)

---

## Neural Baseline
A neural baseline is implemented using an LSTM model with TensorFlow/Keras:

- Token embedding layer
- Bidirectional LSTM layers
- Dense output layer
- Training and validation accuracy/loss visualization

If TensorFlow is not available, the system skips this component.

---

## AI Writing Coach (No External API)
Instead of using an external LLM API, this project implements a **simulated AI coach**.

The system:
- Uses classifier predictions and confidence scores
- Applies rule-based reasoning
- Generates persona-specific writing feedback

This approach ensures transparency, reproducibility, and compliance with environments where API usage is restricted.

---

## Example Output
The AI Writing Coach outputs:
- Detected writing style
- Confidence score
- Actionable writing suggestions based on model predictions

---

## How to Run the Project

### 1. Install dependencies
pip install numpy pandas scikit-learn nltk matplotlib seaborn tensorflow

### 2. Download NLTK resources:
These are automatically downloaded when the script runs:

punkt

stopwords

averaged_perceptron_tagger

### 3.Place dataset

Download the dataset from Kaggle and place in the project root directory:

training.1600000.processed.noemoticon.csv

### 4.Run the program
python src/writing_coach.py

### Output Files

After running, the following files are generated:

confusion_matrix_nb.png                                                                                                                                                      
Ethical Considerations

The model may reflect biases present in the dataset.

Predictions are probabilistic and should not be treated as absolute judgments.

No personal data is stored or transmitted externally.

Limitations and Future Work

Extend style classification beyond binary labels

Add more nuanced feedback categories

Integrate a real LLM API when permitted

Improve UI with a web or desktop interface

confusion_matrix_svm.png

confusion_matrix_lr.png

lstm_training_history.png (if applicable)

