import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from textblob import TextBlob
import spacy
import argparse

# Load spaCy's English model
nlp = spacy.load("en_core_web_md")

def preprocess_data(input_file):
    """
    Load and split the data into training and validation sets.
    Add redacted word length and context-based features.
    """
    print("Step 1: Preprocessing data...")
    with open(input_file, "r", encoding="utf-8") as f:
        valid_rows = []
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) == 3:  # Expecting 3 fields
                valid_rows.append(fields)
    # Convert to DataFrame
    data = pd.DataFrame(valid_rows, columns=["split", "name", "context"])
    data["id"] = range(1, len(data) + 1)  # Add numerical ID column

    # Add redacted word length
    data["redacted_length"] = data["context"].apply(lambda x: x.count("█"))

    # Add previous and next words
    data["prev_word"] = data["context"].apply(lambda x: extract_previous_word(x))
    data["next_word"] = data["context"].apply(lambda x: extract_next_word(x))

    # Add number of words in context
    data["num_words"] = data["context"].apply(lambda x: len(x.split()))

    print("Step 1 Complete: Data preprocessed.")
    return data[data["split"] == "training"], data[data["split"] == "validation"]

def extract_previous_word(text):
    """Extract the word before the redaction block."""
    parts = text.split("█")
    return parts[0].split()[-1] if len(parts[0].split()) > 0 else ""

def extract_next_word(text):
    """Extract the word after the redaction block."""
    parts = text.split("█")
    return parts[1].split()[0] if len(parts) > 1 and len(parts[1].split()) > 0 else ""

def add_sentiment_features(data):
    """
    Add sentiment polarity as a feature to the dataset.
    """
    print("Step 2: Adding sentiment features...")
    data["sentiment"] = data["context"].apply(lambda x: TextBlob(x).sentiment.polarity)
    print("Step 2 Complete: Sentiment features added.")
    return data

def vectorize_data(train_data, val_data):
    """
    Convert context (text) into numerical features using TfidfVectorizer
    and append additional features.
    """
    print("Step 3: Vectorizing data...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(train_data["context"])
    X_val_tfidf = vectorizer.transform(val_data["context"])

    # Combine TF-IDF with additional features
    additional_features_train = train_data[["redacted_length", "sentiment", "num_words"]].values
    additional_features_val = val_data[["redacted_length", "sentiment", "num_words"]].values

    X_train = np.hstack((X_train_tfidf.toarray(), additional_features_train))
    X_val = np.hstack((X_val_tfidf.toarray(), additional_features_val))
    print("Step 3 Complete: Data vectorized.")
    return X_train, X_val, vectorizer

def train_and_predict(X_train, y_train, X_val):
    """
    Train a RandomForest model and predict redacted names.
    """
    print("Step 4: Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Step 4 Complete: Model trained. Predicting on validation set...")
    y_pred = model.predict(X_val)
    print("Step 4 Complete: Predictions generated.")
    return model, y_pred

def predict_on_test(test_file, model, vectorizer, output_file, train_names):
    """
    Generate predictions for test data and save to submission file.
    """
    print("Step 6: Running model on test data...")
    test_data = pd.read_csv(test_file, sep="\t", names=["id", "context"], encoding="utf-8")

    # Add additional features to test data
    test_data["redacted_length"] = test_data["context"].apply(lambda x: x.count("█"))
    test_data["prev_word"] = test_data["context"].apply(lambda x: extract_previous_word(x))
    test_data["next_word"] = test_data["context"].apply(lambda x: extract_next_word(x))
    test_data["num_words"] = test_data["context"].apply(lambda x: len(x.split()))
    test_data["sentiment"] = test_data["context"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Vectorize test data
    X_test_tfidf = vectorizer.transform(test_data["context"])
    additional_features_test = test_data[["redacted_length", "sentiment", "num_words"]].values
    X_test = np.hstack((X_test_tfidf.toarray(), additional_features_test))

    # Predict redacted names
    test_data["name"] = model.predict(X_test)

    print(f"Saving predictions to {output_file}...")
    test_data[["id", "name"]].to_csv(output_file, sep="\t", index=False)
    print(f"Step 6 Complete: Submission file saved to {output_file}.")

def evaluate(val_data, y_true, y_pred):
    """
    Evaluate model predictions on validation set.
    """
    print("Step 5: Evaluating predictions...")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Unredactor Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to the input TSV file (unredactor.tsv)")
    parser.add_argument("--test", type=str, required=True, help="Path to the test TSV file (test.tsv)")
    parser.add_argument("--output", type=str, default="submission.tsv", help="Path to save the submission TSV file")
    args = parser.parse_args()

    input_file = args.data
    test_file = args.test
    submission_file = args.output

    print("Starting Unredactor pipeline...")
    train_data, val_data = preprocess_data(input_file)

    train_data = add_sentiment_features(train_data)
    val_data = add_sentiment_features(val_data)

    X_train, X_val, vectorizer = vectorize_data(train_data, val_data)

    y_train = train_data["name"]
    y_true = val_data["name"]

    # Extract train names
    train_names = train_data["name"]

    model, y_pred = train_and_predict(X_train, y_train, X_val)

    evaluate(val_data, y_true, y_pred)

    predict_on_test(test_file, model, vectorizer, submission_file, train_names)

    print("Unredactor pipeline completed successfully.")

if __name__ == "__main__":
    main()
