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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载类：负责从处理好的CSV文件读取和组织数据"""

    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        Args:
            data_dir: aclImdb目录的路径
        """
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / 'csv'
        self.train_file = self.csv_dir / 'train.csv'
        self.test_file = self.csv_dir / 'test.csv'

        # 验证文件是否存在
        if not self.train_file.exists() or not self.test_file.exists():
            raise FileNotFoundError(
                "CSV files not found. Please run process_dataset.py first."
            )

    def load_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        加载训练集和测试集数据
        Returns:
            训练文本, 训练标签, 测试文本, 测试标签
        """
        logger.info("Loading training data...")
        train_df = pd.read_csv(self.train_file)

        logger.info("Loading test data...")
        test_df = pd.read_csv(self.test_file)

        # 转换标签为数值
        label_map = {'positive': 1, 'negative': 0}

        # 提取数据和标签
        train_texts = train_df['review'].tolist()
        train_labels = [label_map[label] for label in train_df['sentiment']]

        test_texts = test_df['review'].tolist()
        test_labels = [label_map[label] for label in test_df['sentiment']]

        logger.info(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")

        # 验证数据完整性
        self._validate_data(train_texts, train_labels, test_texts, test_labels)

        return train_texts, train_labels, test_texts, test_labels

    def _validate_data(
            self,
            train_texts: List[str],
            train_labels: List[int],
            test_texts: List[str],
            test_labels: List[int]
    ) -> None:
        """验证数据集的完整性和平衡性"""
        # 检查数量是否匹配
        assert len(train_texts) == len(train_labels), "训练集文本和标签数量不匹配"
        assert len(test_texts) == len(test_labels), "测试集文本和标签数量不匹配"

        # 检查标签是否为0和1
        assert set(train_labels) == {0, 1}, "训练集标签不是二元的"
        assert set(test_labels) == {0, 1}, "测试集标签不是二元的"

        # 检查数据集是否平衡
        train_pos = sum(train_labels)
        train_neg = len(train_labels) - train_pos
        test_pos = sum(test_labels)
        test_neg = len(test_labels) - test_pos

        logger.info("数据集统计：")
        logger.info(f"训练集：正面评价 {train_pos}，负面评价 {train_neg}")
        logger.info(f"测试集：正面评价 {test_pos}，负面评价 {test_neg}")

    def get_data_info(self) -> dict:
        """获取数据集的基本信息"""
        train_df = pd.read_csv(self.train_file)
        test_df = pd.read_csv(self.test_file)

        return {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_positive': (train_df['sentiment'] == 'positive').sum(),
            'train_negative': (train_df['sentiment'] == 'negative').sum(),
            'test_positive': (test_df['sentiment'] == 'positive').sum(),
            'test_negative': (test_df['sentiment'] == 'negative').sum(),
        }


class TextPreprocessor:
    """文本预处理类：实现所有文本清理和预处理步骤"""

    def __init__(self):
        """初始化预处理器，设置所需的工具"""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.pattern = re.compile(r'[^a-zA-Z\s]')  # 只保留字母和空格

    def preprocess_text(self, text: str) -> str:
        """
        预处理单个文本
        Args:
            text: 原始文本
        Returns:
            处理后的文本
        """
        # 转换小写
        text = text.lower()

        # 去除标点符号和特殊字符
        text = self.pattern.sub('', text)

        # 分词
        tokens = word_tokenize(text)

        # 去除停用词和词干提取
        processed_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2  # 只保留长度大于2的词
        ]

        return ' '.join(processed_tokens)

    def preprocess(self, texts: List[str]) -> List[str]:
        """
        对文本列表进行预处理
        Args:
            texts: 原始文本列表
        Returns:
            处理后的文本列表
        """
        return [self.preprocess_text(text) for text in texts]


class WordEmbedding:
    """词嵌入类：使用Skip-gram模型将文本转换为向量表示"""

    def __init__(self, embedding_size: int = 100, window: int = 5, min_count: int = 5):
        """
        初始化词嵌入模型
        Args:
            embedding_size: 词向量维度
            window: 上下文窗口大小
            min_count: 最小词频
        """
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.vocab = None

    def _tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        """将文本列表转换为词列表的列表"""
        return [text.split() for text in texts]

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        训练词嵌入模型并转换文本
        Args:
            texts: 预处理后的文本列表
        Returns:
            文本的向量表示
        """
        # 将文本转换为词列表
        tokenized_texts = self._tokenize_texts(texts)

        # 训练Skip-gram模型
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            sg=1  # 使用Skip-gram模型
        )

        # 构建词汇表
        self.vocab = self.model.wv.key_to_index

        # 计算文档向量（使用词向量的平均值）
        return self._texts_to_vectors(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        使用训练好的模型转换新文本
        Args:
            texts: 预处理后的文本列表
        Returns:
            文本的向量表示
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit_transform")
        return self._texts_to_vectors(texts)

    def _texts_to_vectors(self, texts: List[str]) -> np.ndarray:
        """将文本转换为向量表示"""
        vectors = np.zeros((len(texts), self.embedding_size))

        for i, text in enumerate(texts):
            words = text.split()
            word_vectors = []

            for word in words:
                if word in self.vocab:
                    word_vectors.append(self.model.wv[word])

            if word_vectors:
                # 使用词向量的平均值作为文档向量
                vectors[i] = np.mean(word_vectors, axis=0)

        return vectors


class SentimentClassifier:
    """情感分类器：实现模型训练和预测"""

    def __init__(self, model: BaseEstimator):
        """
        初始化分类器
        Args:
            model: sklearn分类器（如SVM、朴素贝叶斯等）
        """
        self.model = model
        self.best_params = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        训练模型，使用网格搜索找到最佳参数
        Args:
            X: 特征矩阵
            y: 标签向量
            **kwargs: 其他训练参数
        Returns:
            训练结果信息
        """
        # 定义参数网格（根据具体分类器调整）
        if isinstance(self.model, LinearSVC):
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            }
        else:
            # 可以添加其他分类器的参数网格
            param_grid = {}

        # 使用网格搜索
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )

        # 训练模型
        grid_search.fit(X, y)

        # 保存最佳参数
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本
        Args:
            X: 特征矩阵
        Returns:
            预测结果
        """
        return self.model.predict(X)


class ModelEvaluator:
    """模型评估器：计算各种评估指标"""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        评估模型性能
        Args:
            y_true: 真实标签
            y_pred: 预测标签
        Returns:
            包含各种评估指标的字典
        """
        # 计算基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        conf_matrix = confusion_matrix(y_true, y_pred)

        # 计算混淆矩阵中的具体指标
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
    """主函数：组织整个工作流程"""

    # 1. 设置参数
    DATA_DIR = "path/to/aclImdb"
    EMBEDDING_SIZE = 100

    try:
        # 2. 加载数据
        logger.info("开始加载数据...")
        data_loader = DataLoader(DATA_DIR)
        train_texts, train_labels, test_texts, test_labels = data_loader.load_data()

        # 3. 文本预处理
        logger.info("开始预处理文本...")
        preprocessor = TextPreprocessor()
        train_texts_processed = preprocessor.preprocess(train_texts)
        test_texts_processed = preprocessor.preprocess(test_texts)

        # 4. 词嵌入
        logger.info("开始词嵌入转换...")
        embedder = WordEmbedding(embedding_size=EMBEDDING_SIZE)
        X_train = embedder.fit_transform(train_texts_processed)
        X_test = embedder.transform(test_texts_processed)

        # 5. 模型训练
        logger.info("开始训练模型...")
        from sklearn.svm import LinearSVC
        classifier = SentimentClassifier(LinearSVC())
        train_results = classifier.train(X_train, train_labels)

        # 6. 模型评估
        logger.info("开始评估模型...")
        y_pred = classifier.predict(X_test)
        evaluation_results = ModelEvaluator.evaluate(test_labels, y_pred)

        # 7. 输出结果
        logger.info("评估结果：")
        for metric, value in evaluation_results.items():
            if isinstance(value, np.ndarray):
                logger.info(f"{metric}:\n{value}")
            else:
                logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"程序执行出错：{str(e)}")
        raise


if __name__ == "__main__":
    main()
