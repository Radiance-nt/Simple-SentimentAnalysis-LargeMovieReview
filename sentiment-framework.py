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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from get_embeddings import get_fasttext_embeddings, get_word2vec_embeddings, train_word2vec, train_fasttext, \
    get_bert_embeddings
import matplotlib.pyplot as plt
import seaborn as sns
import time

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


def generate_embeddings(
        train_texts: List[str],
        test_texts: List[str],
        embedding_size: int = 100
) -> Dict[str, Dict[str, Any]]:
    """
    Generate embeddings for both training and test sets using different techniques
    Args:
        train_texts: list of training texts
        test_texts: list of test texts
        embedding_size: dimension of embeddings (default: 100)
    Returns:
        dictionary with embedding results for each method
    """
    embeddings_dict = {}

    # Word2Vec embeddings
    logger.info("\nGenerating Word2Vec embeddings...")
    # Train model on training data
    start_time = time.time()
    word2vec_model = train_word2vec(train_texts, embedding_size)
    word2vec_train = get_word2vec_embeddings(train_texts, word2vec_model, embedding_size)
    train_time = time.time() - start_time

    # Use trained model for test data
    start_time = time.time()
    word2vec_test = get_word2vec_embeddings(test_texts, word2vec_model, embedding_size)
    inference_time = time.time() - start_time

    embeddings_dict['Word2Vec'] = {
        'train': word2vec_train,
        'test': word2vec_test,
        'train_time': train_time,
        'inference_time': inference_time
    }

    # FastText embeddings
    logger.info("\nGenerating FastText embeddings...")
    # Train model on training data
    start_time = time.time()
    fasttext_model = train_fasttext(train_texts, embedding_size)
    fasttext_train = get_fasttext_embeddings(train_texts, fasttext_model, embedding_size)
    train_time = time.time() - start_time

    # Use trained model for test data
    start_time = time.time()
    fasttext_test = get_fasttext_embeddings(test_texts, fasttext_model, embedding_size)
    inference_time = time.time() - start_time

    embeddings_dict['FastText'] = {
        'train': fasttext_train,
        'test': fasttext_test,
        'train_time': train_time,
        'inference_time': inference_time
    }

    # BERT embeddings (using pre-trained model)
    # logger.info("\nGenerating BERT embeddings...")
    # start_time = time.time()
    # bert_train = get_bert_embeddings(train_texts, embedding_size)
    # train_time = time.time() - start_time
    #
    # start_time = time.time()
    # bert_test = get_bert_embeddings(test_texts, embedding_size)
    # inference_time = time.time() - start_time
    #
    # embeddings_dict['BERT'] = {
    #     'train': bert_train,
    #     'test': bert_test,
    #     'train_time': train_time,
    #     'inference_time': inference_time
    # }

    return embeddings_dict


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


def main():
    """Main function incorporating both sentiment analysis and embedding comparison"""
    parser = argparse.ArgumentParser(description='Process IMDB dataset with multiple embedding techniques')
    parser.add_argument('--base_dir', default='./aclImdb', help='Path to aclImdb directory')
    parser.add_argument('--embedding_size', type=int, default=100, help='Word embedding dimension size')
    parser.add_argument('--output_dir', default='./outputs', help='Directory to save outputs')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize components
    logger.info("Initializing components...")
    data_loader = DataLoader(args.base_dir)
    preprocessor = TextPreprocessor()

    # 2. Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_texts, train_labels, test_texts, test_labels = data_loader.load_data(preprocessor)

    # 3. Generate embeddings using different techniques
    logger.info("Generating embeddings using different techniques...")
    embeddings_dict = generate_embeddings(
        train_texts,
        test_texts,
        embedding_size=args.embedding_size
    )
    # 4. Initialize classifiers
    classifiers = [
        SentimentClassifier(LinearSVC(), "SVM"),
        SentimentClassifier(LogisticRegression(), "Logistic Regression"),
    ]

    # 5. Evaluate each embedding technique with each classifier
    logger.info("Evaluating embedding techniques with different classifiers...")
    all_evaluation_results = []  # For confusion matrix visualization

    for embed_name, embed_data in embeddings_dict.items():
        for classifier in classifiers:
            model_name = f"{embed_name}-{classifier.model_name}"
            logger.info(f"Training {model_name}...")

            # Train classifier
            start_time = time.time()
            train_results = classifier.train(embed_data['train'], train_labels)
            training_time = time.time() - start_time

            # Make predictions
            start_time = time.time()
            y_pred = classifier.predict(embed_data['test'])
            inference_time = time.time() - start_time

            # Evaluate results
            eval_result = ModelEvaluator.evaluate(test_labels, y_pred, model_name)
            eval_result['training_time'] = training_time
            eval_result['inference_time'] = inference_time
            eval_result['embedding_name'] = embed_name

            all_evaluation_results.append(eval_result)

            # Log individual model results
            logger.info(f"Results for {model_name}:")
            for metric, value in eval_result.items():
                if isinstance(value, (np.ndarray, str)):
                    logger.info(f"{metric}:\n{value}")
                else:
                    logger.info(f"{metric}: {value:.4f}")

    # 6. Create performance comparisons and visualizations
    logger.info("\nCreating performance visualizations...")

    # Confusion matrices
    confusion_matrices_path = output_dir / 'confusion_matrices.png'
    plot_confusion_matrices(all_evaluation_results, str(confusion_matrices_path))

    # Performance comparison
    performance_comparison_path = output_dir / 'performance_comparison.png'
    plot_performance_comparison(all_evaluation_results, str(performance_comparison_path))

    # 8. Save performance summary
    results_df = pd.DataFrame([{
        'Model': result['model_name'],
        'Embedding': result['embedding_name'],
        'Accuracy': f"{result['accuracy']:.4f}",
        'Precision': f"{result['precision']:.4f}",
        'Recall': f"{result['recall']:.4f}",
        'F1': f"{result['f1_score']:.4f}",
        'Train_Time': f"{result['training_time']:.4f}",
        'Infer_Time': f"{result['inference_time']:.4f}"
    } for result in all_evaluation_results])

    # Save results
    results_path = output_dir / 'model_comparison_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Log best model
    best_model = results_df.loc[results_df['Accuracy'].astype(float).idxmax()]
    logger.info(f"Best model (accuracy): {best_model['Model']} ({best_model['Accuracy']})")
    logger.info(f"\nAll outputs saved in: {output_dir}")


if __name__ == "__main__":
    download_nltk_resource()
    main()
