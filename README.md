## Sentiment Analysis Assignment

This project analyzes English text reviews with `pandas` and `TextBlob`.

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### What the script does

1. Loads a CSV dataset of reviews.
2. Cleans the review text by lowercasing, removing punctuation, and optionally removing stop words.
3. Uses `TextBlob` to calculate a sentiment polarity score for each review.
4. Adds a label column with `Positive`, `Negative`, or `Neutral`.
5. Prints final statistics and saves the processed CSV.

### Dataset source

The dataset used in this project is `movie.csv`, downloaded from the following Kaggle dataset:

`IMDb Movie Ratings and Sentiment Analysis`

https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis

The current script configuration analyzes the first 500 reviews from that file. 
