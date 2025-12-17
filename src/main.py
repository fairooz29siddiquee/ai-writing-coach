"""
AI Writing Coach - Complete Training Pipeline
Implements classical NLP + neural models + evaluation

Requirements:
- scikit-learn
- numpy
- pandas
- nltk
- tensorflow/pytorch
- matplotlib
- seaborn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """Handles all text preprocessing tasks"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text.lower())

    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [t for t in tokens if t not in self.stop_words and t.isalnum()]

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Tokenization
        tokens = self.tokenize(text)
        # Stopword removal
        tokens = self.remove_stopwords(tokens)
        # Rejoin for vectorization
        return ' '.join(tokens)

    def extract_features(self, text):
        """Extract classical NLP features"""
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)

        features = {
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in tokens]),
            'avg_sentence_length': len(tokens) / max(len(sentences), 1),
            'vocabulary_richness': len(set(tokens)) / len(tokens) if tokens else 0,
            'stopword_ratio': sum(1 for t in tokens if t.lower() in self.stop_words) / len(tokens) if tokens else 0
        }
        return features


class StyleClassifier:
    """
    Multi-class classifier for writing style detection
    Implements: Naive Bayes, SVM, Logistic Regression
    """

    def __init__(self, model_type='nb'):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

        if model_type == 'nb':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif model_type == 'lr':
            self.model = LogisticRegression(max_iter=1000)
        else:
            raise ValueError("Model type must be 'nb', 'svm', or 'lr'")

        self.model_type = model_type

    def train(self, X_train, y_train):
        """Train the classifier"""
        # Preprocess texts
        X_train_processed = [self.preprocessor.preprocess(text) for text in X_train]

        # Vectorize
        X_train_vectors = self.vectorizer.fit_transform(X_train_processed)

        # Train
        self.model.fit(X_train_vectors, y_train)

        return self

    def predict(self, X_test):
        """Make predictions"""
        X_test_processed = [self.preprocessor.preprocess(text) for text in X_test]
        X_test_vectors = self.vectorizer.transform(X_test_processed)
        return self.model.predict(X_test_vectors)

    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        X_test_processed = [self.preprocessor.preprocess(text) for text in X_test]
        X_test_vectors = self.vectorizer.transform(X_test_processed)
        return self.model.predict_proba(X_test_vectors)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }

    def plot_confusion_matrix(self, X_test, y_test, labels):
        """Plot confusion matrix"""
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {self.model_type.upper()} Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.model_type}.png')
        plt.close()


# Example Neural Network Implementation (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences


    class LSTMClassifier:
        """
        LSTM-based neural classifier for writing style
        """

        def __init__(self, vocab_size=5000, embedding_dim=128, max_length=200):
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.max_length = max_length
            self.tokenizer = Tokenizer(num_words=vocab_size)
            self.model = None

        def build_model(self, num_classes):
            """Build LSTM architecture"""
            model = Sequential([
                Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.5),
                Bidirectional(LSTM(32)),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            self.model = model
            return model

        def prepare_data(self, texts):
            """Tokenize and pad sequences"""
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
            return padded

        def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
            """Train the LSTM model"""
            # Fit tokenizer
            self.tokenizer.fit_on_texts(X_train)

            # Prepare sequences
            X_train_seq = self.prepare_data(X_train)
            X_val_seq = self.prepare_data(X_val)

            # Train
            history = self.model.fit(
                X_train_seq, y_train,
                validation_data=(X_val_seq, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            return history

        def predict(self, X_test):
            """Make predictions"""
            X_test_seq = self.prepare_data(X_test)
            predictions = self.model.predict(X_test_seq)
            return np.argmax(predictions, axis=1)

        def evaluate(self, X_test, y_test):
            """Evaluate model"""
            y_pred = self.predict(X_test)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )

            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

        def plot_training_history(self, history):
            """Plot training metrics"""
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot accuracy
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()

            # Plot loss
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Val Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()

            plt.tight_layout()
            plt.savefig('lstm_training_history.png')
            plt.close()

except ImportError:
    print("TensorFlow not available. LSTM classifier will not be available.")
    LSTMClassifier = None


# Example usage and training pipeline

def ai_writing_coach(text, classifier):
    """
    Simulated AI Writing Coach (NO API)
    Uses classifier outputs + rules to respond
    """
    prediction = classifier.predict([text])[0]
    probs = classifier.predict_proba([text])[0]
    confidence = max(probs)

    feedback = f"\n📝 Writing Coach Feedback\n"
    feedback += f"Detected style: {prediction} (confidence: {confidence:.2f})\n\n"

    if prediction == "academic":
        feedback += (
            "Your writing follows an academic tone. "
            "Strengths include formal structure and objective language.\n"
            "Suggestions:\n"
            "- Consider adding clearer topic sentences\n"
            "- Reduce sentence length for readability\n"
        )
    elif prediction == "casual":
        feedback += (
            "Your writing has a casual, conversational tone.\n"
            "Suggestions:\n"
            "- Watch for slang in formal contexts\n"
            "- Add clearer transitions between ideas\n"
        )
    else:
        feedback += "Try refining clarity and consistency in tone."

    return feedback


def main():
    print("Loading dataset...")

    df = pd.read_csv(
        "training.1600000.processed.noemoticon.csv",
        encoding="latin-1",
        header=None
    )

    df.columns = ["target", "id", "date", "flag", "user", "text"]

    df["target"] = df["target"].map({0: 0, 4: 1})


    df = df.sample(5000, random_state=42)

    texts = df["text"].tolist()
    labels = df["target"].tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Split again for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    unique_labels = sorted(list(set(labels)))

    print("\n" + "=" * 50)
    print("TRAINING CLASSICAL MODELS")
    print("=" * 50)

    # Train Naive Bayes
    print("\n1. Training Naive Bayes...")
    nb_clf = StyleClassifier(model_type='nb')
    nb_clf.train(X_train, y_train)
    nb_results = nb_clf.evaluate(X_test, y_test)


    print("\n" + "=" * 50)
    print("AI WRITING COACH DEMO")
    print("=" * 50)

    sample_text = (
        "This paper explores the cool impacts of sustainable stuff on the future of fashion and environmental responsibility."
    )

    coach_feedback = ai_writing_coach(sample_text, nb_clf)
    print(coach_feedback)

    print(
        f"NB - Precision: {nb_results['precision']:.3f}, Recall: {nb_results['recall']:.3f}, F1: {nb_results['f1_score']:.3f}")

    # Train SVM
    print("\n2. Training SVM...")
    svm_clf = StyleClassifier(model_type='svm')
    svm_clf.train(X_train, y_train)
    svm_results = svm_clf.evaluate(X_test, y_test)
    svm_clf.plot_confusion_matrix(X_test, y_test, unique_labels)

    print(
        f"SVM - Precision: {svm_results['precision']:.3f}, Recall: {svm_results['recall']:.3f}, F1: {svm_results['f1_score']:.3f}")

    # Train Logistic Regression
    print("\n3. Training Logistic Regression...")
    lr_clf = StyleClassifier(model_type='lr')
    lr_clf.train(X_train, y_train)
    lr_results = lr_clf.evaluate(X_test, y_test)
    print(
        f"LR - Precision: {lr_results['precision']:.3f}, Recall: {lr_results['recall']:.3f}, F1: {lr_results['f1_score']:.3f}")

    # Plot confusion matrices
    unique_labels = sorted(set(labels))
    nb_clf.plot_confusion_matrix(X_test, y_test, unique_labels)

    if LSTMClassifier is not None:
        print("\n" + "=" * 50)
        print("TRAINING NEURAL MODEL (LSTM)")
        print("=" * 50)

        # Encode labels
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_train_encoded = np.array([label_to_idx[l] for l in y_train])
        y_val_encoded = np.array([label_to_idx[l] for l in y_val])
        y_test_encoded = np.array([label_to_idx[l] for l in y_test])

        # Train LSTM
        lstm_clf = LSTMClassifier()
        lstm_clf.build_model(num_classes=len(unique_labels))
        print(lstm_clf.model.summary())

        history = lstm_clf.train(
            X_train, y_train_encoded,
            X_val, y_val_encoded,
            epochs=10,
            batch_size=32
        )

        lstm_results = lstm_clf.evaluate(X_test, y_test_encoded)
        print(
            f"\nLSTM - Precision: {lstm_results['precision']:.3f}, Recall: {lstm_results['recall']:.3f}, F1: {lstm_results['f1_score']:.3f}")

        lstm_clf.plot_training_history(history)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print("\nResults saved:")
    print("- confusion_matrix_*.png")
    print("- lstm_training_history.png (if applicable)")

    return {
        'nb': nb_clf,
        'svm': svm_clf,
        'lr': lr_clf,
        'lstm': lstm_clf if LSTMClassifier else None
    }


if __name__ == "__main__":
    models = main()
