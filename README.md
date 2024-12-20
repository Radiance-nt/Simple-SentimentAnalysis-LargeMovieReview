# Sentiment Analysis on Large Movie Review Dataset

This project demonstrates a simple machine learning model designed to analyze user sentiment based on their comments and reviews. The workflow involves the following key steps:

1. **Text Data Preprocessing**  
2. **Classifier Training and Testing**  
3. **Model Performance Evaluation**  

### Result  
The results of the analysis are summarized in the image below:  

<p align="center">  
  <img src="https://github.com/Radiance-nt/Simple-SentimentAnalysis-LargeMovieReview/blob/os/result.jpg?raw=true" alt="Model Performance Result">  
</p>  

---

## Dataset: Large Movie Review Dataset  

The dataset used in this project can be downloaded from the link below. Place the downloaded dataset in the root directory of the repository.  

[Download Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)  

---

## Project Structure  

1. **Preprocessing the Dataset:**  
   Run `process-dataset.py` to generate a folder named `acllmbd/csv` containing two CSV files: `train.csv` and `test.csv`. Each CSV file has two columns: `review` and `sentiment`.  

2. **(Optional) Basic Sentiment Analysis:**  
   Run `sentiment-simple.py` for a basic sentiment analysis implementation. This script works independently of the other files.  

3.  **(Optional) Advanced Sentiment Analysis:**  
   Run `sentiment-framework.py`, which imports functionality from `get_embeddings.py`. This script offers a more comprehensive sentiment analysis approach.  

4.  **(Optional) Colab Integration:**  
   Upload the `colab_sentiment.ipynb` notebook to Google Colab. Then, upload the `acllmbd/csv` folder to Colab for running the analysis.  

