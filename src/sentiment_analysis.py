import re
import string
from pathlib import Path
import pandas as pd
from stop_words import get_stop_words
from textblob import TextBlob

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "movie.csv"
OUTPUT_CSV = PROJECT_ROOT / "output" / "processed_movie_reviews_500.csv"
MAX_ROWS = 500
REMOVE_STOPWORDS = False

STOP_WORDS = set(get_stop_words("english"))


def find_text_column(df: pd.DataFrame) -> str:

    if "text" in df.columns:
        return "text"

    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            return column

    available = ", ".join(df.columns)
    raise KeyError(f"No text review column was found. Available columns: {available}")


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        tokens = [word for word in text.split() if word not in STOP_WORDS]
        text = " ".join(tokens)

    return text


def sentiment_label(score: float) -> str:
    if score > 0.10:
        return "Positive"
    if score < -0.10:
        return "Negative"
    return "Neutral"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    print(f"Loading dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    text_column = find_text_column(df)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()

    df["cleaned_review"] = df[text_column].apply(
        lambda text: clean_text(text, remove_stopwords=REMOVE_STOPWORDS)
    )
    df["sentiment_score"] = df["cleaned_review"].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )
    df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    label_counts = df["sentiment_label"].value_counts().sort_index()
    label_percentages = (
        df["sentiment_label"].value_counts(normalize=True).sort_index() * 100
    ).round(2)

    print("\nSentiment Statistics")
    print("-" * 50)
    print(f"Text column used: {text_column}")
    print(f"Total reviews: {len(df)}")
    print(f"Average sentiment score: {df['sentiment_score'].mean():.4f}")

    print("\nLabel counts:")
    print(label_counts.to_string())

    print("\nLabel percentages:")
    print(label_percentages.to_string())

    print(f"\nProcessed file saved to: {OUTPUT_CSV}")

    print(f"Total reviews: {len(df)}")
    print(f"Average sentiment score: {df['sentiment_score'].mean():.4f}")

if __name__ == "__main__":
    main()
