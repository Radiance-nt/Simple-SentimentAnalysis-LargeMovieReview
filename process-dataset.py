import os
import re
import string
import pandas as pd
from typing import Tuple
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IMDBProcessor:
    """IMDB数据集处理类"""
    
    def __init__(self, base_dir: str):
        """
        初始化处理器
        Args:
            base_dir: aclImdb目录的路径
        """
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / 'train'
        self.test_dir = self.base_dir / 'test'
        self.csv_dir = self.base_dir / 'csv'
        self.csv_dir.mkdir(exist_ok=True)
        
    @staticmethod
    def remove_non_ascii(text: str) -> str:
        """移除非ASCII字符"""
        return re.sub(r'[^\x00-\x7F]', ' ', text)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        # 移除非ASCII字符
        text = IMDBProcessor.remove_non_ascii(text)
        # 转换小写
        text = text.lower()
        # 移除标点符号
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text
    
    def process_files(self, input_dir: Path, sentiment: str, output_file: Path) -> None:
        """
        处理指定目录下的所有评论文件
        Args:
            input_dir: 输入文件目录
            sentiment: 情感标签
            output_file: 输出CSV文件路径
        """
        files = list(input_dir.glob('*.txt'))
        total_files = len(files)
        
        # 创建CSV文件并写入头部
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("review,sentiment\n")
            
            # 处理每个文件
            for i, file_path in enumerate(files, 1):
                try:
                    # 读取并清理文本
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as review_file:
                        text = review_file.read()
                    
                    cleaned_text = self.clean_text(text)
                    # 将文本中的逗号替换为空格，以防止CSV格式错误
                    cleaned_text = cleaned_text.replace(',', ' ')
                    
                    # 写入CSV
                    f.write(f'"{cleaned_text}",{sentiment}\n')
                    
                    if i % 1000 == 0:
                        logger.info(f'Processed {i}/{total_files} files in {input_dir.name}')
                        
                except Exception as e:
                    logger.error(f'Error processing {file_path}: {str(e)}')
    
    def process_dataset(self) -> Tuple[Path, Path]:
        """
        处理整个数据集
        Returns:
            训练集和测试集CSV文件的路径
        """
        # 创建输出目录
        train_csv = self.csv_dir / 'train.csv'
        test_csv = self.csv_dir / 'test.csv'
        
        # 处理训练集
        logger.info("Processing training set...")
        self.process_files(self.train_dir / 'pos', 'positive', self.csv_dir / 'train_pos.csv')
        self.process_files(self.train_dir / 'neg', 'negative', self.csv_dir / 'train_neg.csv')
        
        # 处理测试集
        logger.info("Processing test set...")
        self.process_files(self.test_dir / 'pos', 'positive', self.csv_dir / 'test_pos.csv')
        self.process_files(self.test_dir / 'neg', 'negative', self.csv_dir / 'test_neg.csv')
        
        # 合并训练集文件
        logger.info("Merging training files...")
        train_pos = pd.read_csv(self.csv_dir / 'train_pos.csv')
        train_neg = pd.read_csv(self.csv_dir / 'train_neg.csv')
        pd.concat([train_pos, train_neg], ignore_index=True).to_csv(train_csv, index=False)
        
        # 合并测试集文件
        logger.info("Merging test files...")
        test_pos = pd.read_csv(self.csv_dir / 'test_pos.csv')
        test_neg = pd.read_csv(self.csv_dir / 'test_neg.csv')
        pd.concat([test_pos, test_neg], ignore_index=True).to_csv(test_csv, index=False)
        
        # 删除中间文件
        for file in ['train_pos.csv', 'train_neg.csv', 'test_pos.csv', 'test_neg.csv']:
            (self.csv_dir / file).unlink()
        
        return train_csv, test_csv

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process IMDB dataset into CSV format')
    parser.add_argument('base_dir', help='Path to aclImdb directory')
    args = parser.parse_args()
    
    processor = IMDBProcessor(args.base_dir)
    train_csv, test_csv = processor.process_dataset()
    
    logger.info(f"Processing complete. Files saved as:")
    logger.info(f"Training set: {train_csv}")
    logger.info(f"Test set: {test_csv}")

if __name__ == "__main__":
    main()
