import os
import argparse
import re
import string
import pandas as pd
from typing import Tuple
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IMDBProcessor:

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / 'train'
        self.test_dir = self.base_dir / 'test'
        self.csv_dir = self.base_dir / 'csv'
        self.csv_dir.mkdir(exist_ok=True)

    @staticmethod
    def remove_non_ascii(text: str) -> str:
        return re.sub(r'[^\x00-\x7F]', ' ', text)

    @staticmethod
    def clean_text(text: str) -> str:
        text = IMDBProcessor.remove_non_ascii(text)
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def process_files(self, input_dir: Path, sentiment: str, output_file: Path) -> None:

        files = list(input_dir.glob('*.txt'))
        total_files = len(files)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("review,sentiment\n")

            for i, file_path in enumerate(files, 1):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as review_file:
                        text = review_file.read()

                    cleaned_text = self.clean_text(text)
                    cleaned_text = cleaned_text.replace(',', ' ')

                    f.write(f'"{cleaned_text}",{sentiment}\n')

                    if i % 1000 == 0:
                        logger.info(f'Processed {i}/{total_files} files in {input_dir.name}')

                except Exception as e:
                    logger.error(f'Error processing {file_path}: {str(e)}')

    def process_dataset(self) -> Tuple[Path, Path]:
        train_csv = self.csv_dir / 'train.csv'
        test_csv = self.csv_dir / 'test.csv'

        logger.info("Processing training set...")
        self.process_files(self.train_dir / 'pos', 'positive', self.csv_dir / 'train_pos.csv')
        self.process_files(self.train_dir / 'neg', 'negative', self.csv_dir / 'train_neg.csv')

        logger.info("Processing test set...")
        self.process_files(self.test_dir / 'pos', 'positive', self.csv_dir / 'test_pos.csv')
        self.process_files(self.test_dir / 'neg', 'negative', self.csv_dir / 'test_neg.csv')

        logger.info("Merging training files...")
        train_pos = pd.read_csv(self.csv_dir / 'train_pos.csv')
        train_neg = pd.read_csv(self.csv_dir / 'train_neg.csv')
        pd.concat([train_pos, train_neg], ignore_index=True).to_csv(train_csv, index=False)

        logger.info("Merging test files...")
        test_pos = pd.read_csv(self.csv_dir / 'test_pos.csv')
        test_neg = pd.read_csv(self.csv_dir / 'test_neg.csv')
        pd.concat([test_pos, test_neg], ignore_index=True).to_csv(test_csv, index=False)

        for file in ['train_pos.csv', 'train_neg.csv', 'test_pos.csv', 'test_neg.csv']:
            (self.csv_dir / file).unlink()

        return train_csv, test_csv


def main():
    parser = argparse.ArgumentParser(description='Process IMDB dataset into CSV format')
    parser.add_argument('--base_dir', default='./aclImdb', help='Path to aclImdb directory')
    args = parser.parse_args()

    processor = IMDBProcessor(args.base_dir)
    train_csv, test_csv = processor.process_dataset()

    logger.info(f"Processing complete. Files saved as:")
    logger.info(f"Training set: {train_csv}")
    logger.info(f"Test set: {test_csv}")


if __name__ == "__main__":
    main()
