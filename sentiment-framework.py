import argparse
import os
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
import logging
import pandas as pd
from sklearn.base import BaseEstimator
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_nltk_resource():
    nltk.download('stopwords')
    nltk.download('punkt')


class DataLoader:
    """Data loading class: responsible for reading and organizing data from processed CSV files"""

    def __init__(self, data_dir: str):
        """
        Initialize data loader
        Args:
            data_dir: path to aclImdb directory
        """
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / 'csv'
        self.train_file = self.csv_dir / 'train.csv'
        self.test_file = self.csv_dir / 'test.csv'
        self.train_preprocessed_file = self.csv_dir / 'train_preprocessed.csv'
        self.test_preprocessed_file = self.csv_dir / 'test_preprocessed.csv'

        # Verify if original CSV directory exists
        if not self.train_file.exists() or not self.test_file.exists():
            raise FileNotFoundError(
                "CSV files not found. Please run process_dataset.py first."
            )

    def load_data(self, preprocessor: 'TextPreprocessor' = None) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load training and test data
        Create preprocessed files if they don't exist
        Args:
            preprocessor: TextPreprocessor instance
        Returns:
            training texts, training labels, test texts, test labels
        """
        # Create preprocessed files if they don't exist
        if not self.train_preprocessed_file.exists() or not self.test_preprocessed_file.exists():
            if preprocessor is None:
                raise ValueError("Preprocessed files don't exist, TextPreprocessor instance required for preprocessing")

            logger.info("Preprocessed files don't exist, starting preprocessing...")
            # Load original data
            train_df = pd.read_csv(self.train_file)
            test_df = pd.read_csv(self.test_file)

            # Preprocess text
            logger.info("Preprocessing training set...")
            train_processed = preprocessor.preprocess(train_df['review'].tolist())
            logger.info("Preprocessing test set...")
            test_processed = preprocessor.preprocess(test_df['review'].tolist())

            # Create and save preprocessed data
            self.csv_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({
                'review': train_df['review'],
                'processed_review': train_processed,
                'sentiment': train_df['sentiment']
            }).to_csv(self.train_preprocessed_file, index=False)

            pd.DataFrame({
                'review': test_df['review'],
                'processed_review': test_processed,
                'sentiment': test_df['sentiment']
            }).to_csv(self.test_preprocessed_file, index=False)

            logger.info(f"Preprocessed data saved to: {self.csv_dir}")

        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        train_df = pd.read_csv(self.train_preprocessed_file)
        test_df = pd.read_csv(self.test_preprocessed_file)

        # Extract data and labels
        train_texts = train_df['processed_review'].tolist()
        train_labels = [1 if label == 'positive' else 0 for label in train_df['sentiment']]
        test_texts = test_df['processed_review'].tolist()
        test_labels = [1 if label == 'positive' else 0 for label in test_df['sentiment']]

        logger.info(f"Successfully loaded data: {len(train_texts)} training samples, {len(test_texts)} test samples")
        return train_texts, train_labels, test_texts, test_labels


class TextPreprocessor:
    """Text preprocessing class: implements all text cleaning and preprocessing steps"""

    def __init__(self):
        """Initialize preprocessor, set up required tools"""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.pattern = re.compile(r'[^a-zA-Z\s]')  # Keep only letters and spaces

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess single text
        Args:
            text: original text
        Returns:
            processed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = self.pattern.sub('', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and perform stemming
        processed_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2  # Keep only words longer than 2 characters
        ]

        return ' '.join(processed_tokens)

    def preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess list of texts
        Args:
            texts: list of original texts
        Returns:
            list of processed texts
        """
        return [self.preprocess_text(text) for text in texts]


class WordEmbedding:
    """Word embedding class: converts text to vector representation using Skip-gram model"""

    def __init__(self, embedding_size: int = 100, window: int = 5, min_count: int = 5):
        """
        Initialize word embedding model
        Args:
            embedding_size: word vector dimension
            window: context window size
            min_count: minimum word frequency
        """
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.vocab = None

    def _tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        """Convert list of texts to list of word lists"""
        return [text.split() for text in texts]

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Train word embedding model and transform texts
        Args:
            texts: list of preprocessed texts
        Returns:
            vector representation of texts
        """
        # Convert texts to word lists
        tokenized_texts = self._tokenize_texts(texts)

        # Train Skip-gram model
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            sg=1  # Use Skip-gram model
        )

        # Build vocabulary
        self.vocab = self.model.wv.key_to_index

        # Calculate document vectors (using average of word vectors)
        return self._texts_to_vectors(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using trained model
        Args:
            texts: list of preprocessed texts
        Returns:
            vector representation of texts
        """
        if self.model is None:
            raise ValueError("Model not trained, please call fit_transform first")
        return self._texts_to_vectors(texts)

    def _texts_to_vectors(self, texts: List[str]) -> np.ndarray:
        """Convert texts to vector representation"""
        vectors = np.zeros((len(texts), self.embedding_size))

        for i, text in enumerate(texts):
            words = text.split()
            word_vectors = []

            for word in words:
                if word in self.vocab:
                    word_vectors.append(self.model.wv[word])

            if word_vectors:
                # Use average of word vectors as document vector
                vectors[i] = np.mean(word_vectors, axis=0)

        return vectors


class SentimentClassifier:
    """Sentiment classifier: implements model training and prediction"""

    def __init__(self, model: BaseEstimator):
        """
        Initialize classifier
        Args:
            model: sklearn classifier (e.g., SVM, Naive Bayes)
        """
        self.model = model
        self.best_params = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train model, use grid search to find best parameters
        Args:
            X: feature matrix
            y: label vector
            **kwargs: other training parameters
        Returns:
            training result information
        """
        # Define parameter grid (adjust based on classifier)
        if isinstance(self.model, LinearSVC):
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            }
        else:
            # Can add parameter grids for other classifiers
            param_grid = {}

        # Use grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )

        # Train model
        grid_search.fit(X, y)

        # Save best parameters
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict new samples
        Args:
            X: feature matrix
        Returns:
            prediction results
        """
        return self.model.predict(X)


class ModelEvaluator:
    """Model evaluator: calculate various evaluation metrics"""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            dictionary containing various evaluation metrics
        """
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Calculate specific metrics from confusion matrix
        tn, fp, fn, tp = conf_matrix.ravel()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }


def main():
    parser = argparse.ArgumentParser(description='Process IMDB dataset into CSV format')
    parser.add_argument('--base_dir', default='./aclImdb', help='Path to aclImdb directory')
    parser.add_argument('--embedding_size', type=int, default=100, help='Word embedding dimension size')
    args = parser.parse_args()

    DATA_DIR = args.base_dir
    EMBEDDING_SIZE = args.embedding_size

    try:
        # 1. Initialize data loader and preprocessor
        logger.info("Initializing...")
        data_loader = DataLoader(DATA_DIR)
        preprocessor = TextPreprocessor()

        # 2. Load data (load directly from preprocessed files if they exist, otherwise preprocess)
        logger.info("Starting data loading...")
        train_texts, train_labels, test_texts, test_labels = data_loader.load_data(preprocessor)

        # 3. Word embedding
        logger.info("Starting word embedding conversion...")
        embedder = WordEmbedding(embedding_size=EMBEDDING_SIZE)
        X_train = embedder.fit_transform(train_texts)
        X_test = embedder.transform(test_texts)

        # 4. Model training
        logger.info("Starting model training...")
        classifier = SentimentClassifier(LinearSVC())
        train_results = classifier.train(X_train, train_labels)

        # 5. Model evaluation
        logger.info("Starting model evaluation...")
        y_pred = classifier.predict(X_test)
        evaluation_results = ModelEvaluator.evaluate(test_labels, y_pred)

        # 6. Output results
        logger.info("Evaluation results:")
        for metric, value in evaluation_results.items():
            if isinstance(value, np.ndarray):
                logger.info(f"{metric}:\n{value}")
            else:
                logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        raise


if __name__ == "__main__":
    download_nltk_resource()
    main()
