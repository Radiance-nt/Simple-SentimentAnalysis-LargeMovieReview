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
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_performance_comparison(results: List[Dict[str, Any]], save_path: str = None):
    """
    Create a bar plot comparing performance metrics across different models
    Args:
        results: list of evaluation results for different models
        save_path: path to save the plot (optional)
    """
    # Extract metrics for comparison
    performance_data = []
    for result in results:
        performance_data.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1 Score': result['f1_score']
        })

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(performance_data)

    # Set up the plot style
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(df['Model']))
    width = 0.2

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, df[metric], width, label=metric)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')

    # Customize plot
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['Model'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_confusion_matrices(results: List[Dict[str, Any]], save_path: str = None):
    """
    Create confusion matrix plots for all models
    Args:
        results: list of evaluation results for different models
        save_path: path to save the plot (optional)
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    plt.style.use('seaborn')

    for ax, result in zip(axes, results):
        conf_matrix = result['confusion_matrix']
        model_name = result['model_name']

        # Create confusion matrix heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )

        ax.set_title(f'Confusion Matrix - {model_name}\nAccuracy: {result["accuracy"]:.3f}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


class SentimentClassifier:
    """Sentiment classifier: implements model training and prediction"""

    def __init__(self, model: BaseEstimator, model_name: str):
        """
        Initialize classifier
        Args:
            model: sklearn classifier (e.g., SVM, Naive Bayes)
            model_name: name of the model for identification
        """
        self.model = model
        self.model_name = model_name
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
        # Define parameter grid based on classifier type
        if isinstance(self.model, LinearSVC):
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            }
        elif isinstance(self.model, LogisticRegression):
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            }
        else:
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
            'model_name': self.model_name,
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
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance
        Args:
            y_true: true labels
            y_pred: predicted labels
            model_name: name of the model being evaluated
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
            'model_name': model_name,
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
    parser.add_argument('--output_dir', default='./outputs', help='Directory to save outputs')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    DATA_DIR = args.base_dir
    EMBEDDING_SIZE = args.embedding_size

    try:
        # 1. Initialize data loader and preprocessor
        logger.info("Initializing...")
        data_loader = DataLoader(DATA_DIR)
        preprocessor = TextPreprocessor()

        # 2. Load data
        logger.info("Starting data loading...")
        train_texts, train_labels, test_texts, test_labels = data_loader.load_data(preprocessor)

        # 3. Word embedding
        logger.info("Starting word embedding conversion...")
        embedder = WordEmbedding(embedding_size=EMBEDDING_SIZE)
        X_train = embedder.fit_transform(train_texts)
        X_test = embedder.transform(test_texts)

        # 4. Initialize different classifiers
        classifiers = [
            SentimentClassifier(LinearSVC(), "SVM"),
            SentimentClassifier(LogisticRegression(), "Logistic Regression"),
        ]

        # 5. Train and evaluate all classifiers
        evaluation_results = []

        for classifier in classifiers:
            logger.info(f"Training {classifier.model_name}...")
            train_results = classifier.train(X_train, train_labels)

            logger.info(f"Evaluating {classifier.model_name}...")
            y_pred = classifier.predict(X_test)
            eval_result = ModelEvaluator.evaluate(test_labels, y_pred, classifier.model_name)
            evaluation_results.append(eval_result)

            # Log individual model results
            logger.info(f"\nResults for {classifier.model_name}:")
            for metric, value in eval_result.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"{metric}:\n{value}")
                elif isinstance(value, str):
                    logger.info(f"{metric}: {value}")
                else:
                    logger.info(f"{metric}: {value:.4f}")

        # 6. Create and save visualizations
        logger.info("Creating visualizations...")

        # Plot and save confusion matrices
        confusion_matrices_path = output_dir / 'confusion_matrices.png'
        plot_confusion_matrices(evaluation_results, str(confusion_matrices_path))

        # Plot and save performance comparison
        performance_comparison_path = output_dir / 'performance_comparison.png'
        plot_performance_comparison(evaluation_results, str(performance_comparison_path))

        logger.info(f"Visualizations saved in: {output_dir}")

    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        raise


if __name__ == "__main__":
    download_nltk_resource()
    main()
