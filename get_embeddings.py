import torch
from transformers import AutoTokenizer, AutoModel
from gensim.models import FastText, Word2Vec
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def initialize_bert():
    """Initialize BERT model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model


def get_bert_embeddings(texts: List[str], embedding_size: int = 100) -> np.ndarray:
    """
    Generate BERT embeddings for texts using GPU if available
    Args:
        texts: list of texts to embed
        embedding_size: not used for BERT but kept for consistent interface
    Returns:
        numpy array of embeddings
    """
    tokenizer, model = initialize_bert()

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        # Move model to GPU if available
        model = model.to(device)
        embeddings = []

        # Process in batches
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize and prepare input
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            # Move input tensors to GPU if available
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**encoded)
                # Move to CPU and convert to numpy
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

            # Log progress for long sequences
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {i + batch_size} texts out of {len(texts)}")

        # Stack all embeddings
        return np.vstack(embeddings)

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("GPU out of memory. Trying with smaller batch size...")
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Recursively try with smaller batch size
            return get_bert_embeddings_with_batch_size(texts, batch_size // 2, tokenizer, model, device)
        else:
            raise e
    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")
        raise


def get_bert_embeddings_with_batch_size(
        texts: List[str],
        batch_size: int,
        tokenizer,
        model,
        device
) -> np.ndarray:
    """
    Helper function to retry embedding generation with smaller batch size
    """
    if batch_size < 1:
        raise ValueError("Batch size cannot be smaller than 1")

    logger.info(f"Retrying with batch size: {batch_size}")
    embeddings = []

    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {i + batch_size} texts out of {len(texts)}")

        return np.vstack(embeddings)
    except RuntimeError as e:
        if "out of memory" in str(e) and batch_size > 1:
            logger.error("Still out of memory. Trying with even smaller batch size...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return get_bert_embeddings_with_batch_size(texts, batch_size // 2, tokenizer, model, device)
        else:
            raise e


def train_word2vec(texts: List[str], embedding_size: int = 100) -> Word2Vec:
    """
    Train Word2Vec model on texts
    Args:
        texts: list of texts to train on
        embedding_size: size of word vectors
    Returns:
        trained Word2Vec model
    """
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=embedding_size,
        window=5,
        min_count=5,
        sg=1  # Use Skip-gram model
    )
    return model


def get_word2vec_embeddings(texts: List[str], word2vec_model: Word2Vec = None, embedding_size: int = 100) -> np.ndarray:
    """
    Generate Word2Vec embeddings for texts
    Args:
        texts: list of texts to embed
        word2vec_model: pre-trained Word2Vec model (optional)
        embedding_size: size of word vectors
    Returns:
        numpy array of embeddings
    """
    if word2vec_model is None:
        word2vec_model = train_word2vec(texts, embedding_size)

    # Convert texts to vectors
    vocab = word2vec_model.wv.key_to_index
    vectors = np.zeros((len(texts), embedding_size))

    for i, text in enumerate(texts):
        words = text.split()
        word_vectors = []

        for word in words:
            if word in vocab:
                word_vectors.append(word2vec_model.wv[word])

        if word_vectors:
            vectors[i] = np.mean(word_vectors, axis=0)

    return vectors


def train_fasttext(texts: List[str], embedding_size: int = 100) -> FastText:
    """
    Train FastText model on texts
    Args:
        texts: list of texts to train on
        embedding_size: size of word vectors
    Returns:
        trained FastText model
    """
    tokenized_texts = [text.split() for text in texts]
    model = FastText(
        tokenized_texts,
        vector_size=embedding_size,
        window=5,
        min_count=5
    )
    return model


def get_fasttext_embeddings(texts: List[str], fasttext_model: FastText = None, embedding_size: int = 100) -> np.ndarray:
    """
    Generate FastText embeddings for texts
    Args:
        texts: list of texts to embed
        fasttext_model: pre-trained FastText model (optional)
        embedding_size: size of word vectors
    Returns:
        numpy array of embeddings
    """
    if fasttext_model is None:
        fasttext_model = train_fasttext(texts, embedding_size)

    embeddings = np.zeros((len(texts), embedding_size))
    for i, text in enumerate(texts):
        words = text.split()
        word_vectors = [fasttext_model.wv[word] for word in words if word in fasttext_model.wv]
        if word_vectors:
            embeddings[i] = np.mean(word_vectors, axis=0)

    return embeddings
