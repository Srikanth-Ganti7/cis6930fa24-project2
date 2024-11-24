
# cis6930fa24 -- Project2

## Author

**Name**: Balasai Srikanth Ganti

**UFID** : 5251-6075

## Assignment Description 
This project implements the Unredactor, a tool designed to predict redacted names from given contexts in text data. It uses machine learning techniques, including feature extraction and classification, to achieve this goal. The core pipeline involves preprocessing data, extracting features, training a model, and making predictions.


## Pipeline Overview

The pipeline implemented for the Unredactor project is designed to be modular, scalable, and efficient, focusing on extracting meaningful features from textual data to predict redacted names accurately. The chosen approach addresses the challenges of working with unstructured text and combines feature engineering with machine learning for robust results. Below is an explanation of the pipeline’s components and the reasoning behind them:

### Preprocessing:
The pipeline starts by reading the input data and splitting it into training and validation datasets. During this step, key features like the redacted block length and the words immediately before and after the redaction are extracted. This ensures the data is prepared in a structured format for the machine learning model.

#### This Step is Necessary:

- Extracting features such as redacted block length and surrounding words provides the model with essential clues about the redaction context.
- A well-defined split between training and validation sets ensures reliable evaluation and avoids data leakage.

### Feature Engineering:

Key features are engineered to capture the contextual meaning of the text:

- TF-IDF (Term Frequency-Inverse Document Frequency) represents the importance of words in the text.
- Additional features such as the length of the redacted block, number of words in the context, and sentiment polarity are included.

#### Why These Features Were Chosen:

- TF-IDF: It highlights the significance of words in the context while ignoring common, less meaningful words (e.g., articles, prepositions).
- Sentiment Polarity: Sentiment, calculated using TextBlob, provides insights into the emotional tone of the context, which can relate to specific names.
- Redacted Length: The length of the redacted block serves as a direct indicator of the missing name’s character count.

### Vectorization
The pipeline uses a hybrid approach by combining TF-IDF features with the additional engineered features. This comprehensive feature set is fed into the model to train and make predictions.

#### Reason for This Approach:

- TF-IDF captures the textual patterns, while additional features provide structural and contextual information. Together, they create a richer representation of the data.
- Combining these features ensures that the model learns from both statistical patterns in the text and manually crafted indicators.

### Model Training:
A RandomForestClassifier is used to train the model. Random Forest is a robust ensemble-based algorithm that can handle various types of features and prevents overfitting.

#### Why Random Forest Was Chosen:

- It effectively handles the diverse feature set comprising of categorical data.
- Its ensemble nature reduces overfitting, making it suitable for tasks involving high-dimensional data, such as TF-IDF vectors.

### Testing and Predictions:
The predict_on_test() function applies the trained model to the test dataset to generate predictions. The results are saved in the required submission.tsv format, ensuring compatibility with the evaluation system.

#### Purpose of This Step:

- This function maintains consistency in feature engineering and vectorization between training and testing phases.
- It ensures the output adheres to the specified format, enabling smooth submission and evaluation.

### Evaluation:
The pipeline evaluates model performance using metrics such as accuracy, precision, recall, and F1-score. These metrics are computed on the validation set to assess how well the model generalizes.

#### Why These Metrics Were Selected:

- Precision and recall are critical for identifying names accurately, as incorrect predictions could lead to significant errors in sensitive contexts.
- F1-score provides a balanced assessment of the model’s precision and recall, offering a comprehensive view of its effectiveness.

### Modular Design
The pipeline is organized into individual functions for preprocessing, feature extraction, vectorization, training, prediction, and evaluation. Each function performs a specific task, making the codebase easy to debug, test, and extend.

#### Benefits of Modularity:

Each component can be independently verified, reducing the risk of cascading errors.
The design allows flexibility to incorporate future improvements, such as adding new features or testing alternative machine learning models.

### Scalability
The use of TF-IDF and Random Forest makes the pipeline scalable for large datasets, such as the IMDB reviews corpus. The additional features are lightweight to compute, ensuring computational efficiency.

#### Why Scalability Matters:

The IMDB dataset contains thousands of entries, requiring a pipeline that can handle significant data volumes without compromising performance.
The balance between computational efficiency and predictive accuracy ensures the pipeline remains practical for larger datasets.

---

This pipeline is carefully crafted to integrate statistical feature extraction with machine learning, providing a robust and reliable approach to predicting redacted names. Each component has been thoughtfully chosen to enhance the pipeline’s ability to generalize and deliver accurate results across a variety of contexts.

## How to install

To set up the environment, use pipenv to install the dependencies.

```bash
pipenv install 
```

## How to run
Run the program using the following commands:

### Run the pipeline:
```bash
pipenv run python unredactor2.py --data data/unredactor.tsv --test test.tsv --output submission.tsv
```

Check the results in submission.tsv.


## Documentation of Features

### Features Used:

#### Redacted Length:
Number of █ characters in the redacted section.
Helps approximate the length of the missing name.
#### Previous Word:
The word immediately preceding the redacted block.
Provides contextual clues about the missing name.
#### Next Word:
The word immediately following the redacted block.
Adds context for identifying the missing name.
#### Number of Words in Context:
Total word count in the context.
Measures the richness of the context.
#### Sentiment Polarity:
Polarity score of the context text using TextBlob.
Indicates the emotional tone of the text.
#### TF-IDF Features:
- Numerical representation of the context text using TfidfVectorizer.
- Captures n-grams (1 to 3 words) and their importance in the text.


### Evaluation Metrics
The model's performance is evaluated using the following metrics:

#### Accuracy: 
Proportion of correctly predicted names out of all predictions.
#### Precision: 
Proportion of true positive predictions among all positive predictions.
#### Recall: 
Proportion of true positives among all actual positives.
#### F1-Score: 
Harmonic mean of precision and recall.
These metrics are calculated using the validation set:

```bash
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="micro")
recall = recall_score(y_true, y_pred, average="micro")
f1 = f1_score(y_true, y_pred, average="micro")

```


## Output Example

The final output submission.tsv is generated by the predict_on_test() function and includes:

- id: The identifier of the redacted entry from the test data.
- name: The predicted unredacted name.

Example:

| id | name              |
|----|-------------------|
| 1  | Ashton Kutcher    |
| 2  | Lawrence Fishburne |

---

## Functions Used

### 1. `preprocess_data(input_file)`
**Description**:  
Loads and splits the data into training and validation sets. Adds features like redacted word length, previous/next words, and the number of words in the context.

**Parameters**:
- `input_file`: Path to the input TSV file.

**Returns**:
- Training and validation dataframes.

---

### 2. `extract_previous_word(text)`
**Description**:  
Extracts the word immediately preceding the redacted block in the context.

**Parameters**:
- `text`: A string containing the context with redacted blocks.

**Returns**:
- The word preceding the redaction block.

---

### 3. `extract_next_word(text)`
**Description**:  
Extracts the word immediately following the redacted block in the context.

**Parameters**:
- `text`: A string containing the context with redacted blocks.

**Returns**:
- The word following the redaction block.

---

### 4. `add_sentiment_features(data)`
**Description**:  
Adds a sentiment polarity score as a feature for each context using TextBlob.

**Parameters**:
- `data`: A dataframe containing the context.

**Returns**:
- A dataframe with an additional `sentiment` column.

---

### 5. `vectorize_data(train_data, val_data)`
**Description**:  
Converts the context text into numerical features using `TfidfVectorizer` and combines these with additional features (redacted length, sentiment, and word count).

**Parameters**:
- `train_data`: Dataframe containing the training set.
- `val_data`: Dataframe containing the validation set.

**Returns**:
- TF-IDF vectorized training and validation data along with the vectorizer.

---

### 6. `train_and_predict(X_train, y_train, X_val)`
**Description**:  
Trains a Random Forest classifier and predicts redacted names for the validation set.

**Parameters**:
- `X_train`: Training feature matrix.
- `y_train`: Training labels (redacted names).
- `X_val`: Validation feature matrix.

**Returns**:
- The trained model and predictions for the validation set.

---

### 7. `predict_on_test(test_file, model, vectorizer, output_file, train_names)`
**Description**:  
Predicts redacted names from test data and saves the predictions in a submission file.

**Parameters**:
- `test_file`: Path to the test data file.
- `model`: Trained Random Forest model.
- `vectorizer`: Trained TF-IDF vectorizer.
- `output_file`: Path to save the submission file.
- `train_names`: List of names from the training data.

**Output**:
- A TSV file containing the predicted names for the test data.

---

### 8. `evaluate(val_data, y_true, y_pred)`
**Description**:  
Calculates and prints evaluation metrics (accuracy, precision, recall, and F1-score) for the validation set predictions.

**Parameters**:
- `val_data`: Dataframe containing the validation set.
- `y_true`: True labels for the validation set.
- `y_pred`: Predicted labels for the validation set.

---

### 9. `main()`
**Description**:  
The main function to run the entire pipeline:
1. Load and preprocess data.
2. Extract features.
3. Train and evaluate the model.
4. Predict and save results for the test set.


## Libraries Used

Below is a list of the Python libraries used in the code along with their purpose:

### Core Libraries
- **pandas**: For loading, preprocessing, and manipulating tabular data.
- **numpy**: For numerical operations and feature matrix manipulations.
- **argparse**: To handle command-line arguments.

### Machine Learning Libraries
- **scikit-learn**:
  - `TfidfVectorizer`: To convert context text into numerical features.
  - `RandomForestClassifier`: To train a model for predicting redacted names.
  - `metrics` (e.g., `accuracy_score`, `precision_score`, `recall_score`, `f1_score`): For evaluating model performance.

### Natural Language Processing Libraries
- **TextBlob**: To calculate sentiment polarity of the context text.
- **spaCy**: For NLP tasks such as tokenization (loaded with the `en_core_web_sm` English model).

---

### Installation
To install the required libraries, use the following command:
```bash
pipenv install pandas numpy scikit-learn textblob spacy
```
Additionally, ensure that the en_core_web_md model for spaCy is downloaded:

```bash
pipenv run python -m spacy download en_core_web_sm

```


## Bugs and Assumptions

### Potential Bugs
1. **Input Format Issues**:
  - The `preprocess_data()` function expects every line in the input file to have exactly three tab-separated fields (`split`, `name`, `context`). If the file format is inconsistent, it might skip those lines or behave unexpectedly.

2. **Redaction Block Limitations**:
  - The `extract_previous_word()` and `extract_next_word()` functions only account for the first occurrence of the redaction block (`█`). If multiple blocks exist, the extracted context may be inaccurate.

3. **Sentiment Analysis Constraints**:
  - Sentiment polarity from `TextBlob` might not fully capture nuanced or domain-specific sentiment in the context.

4. **Memory Overhead with TF-IDF**:
  - Using up to 50,000 features in the `TfidfVectorizer` can lead to excessive memory usage, especially with large datasets, potentially causing performance issues.

5. **Generalization on Unknown Names**:
  - The `RandomForestClassifier` may perform poorly when predicting names not seen during training, leading to reduced accuracy on new data.

6. **Evaluation Metric Assumption**:
  - The `average="micro"` argument in evaluation metrics assumes a flat label distribution, which may not work well with imbalanced data.

7. **Column Overlaps in Test Data**:
  - If the test dataset already includes a `name` column, appending predictions might overwrite existing data or cause misalignment.

---

### Assumptions
1. **Input File Structure**:
  - The input file (`unredactor.tsv`) is well-formed, with exactly three tab-separated fields (`split`, `name`, `context`) per line.
  - The test file (`test.tsv`) contains two tab-separated columns: `id` and `context`.

2. **Single Redaction Per Context**:
  - Each context contains a single redaction block represented by a contiguous set of `█` characters.

3. **Context Length**:
  - The provided context is sufficiently long to allow meaningful feature extraction without requiring truncation or padding.

4. **Feature Independence**:
  - Features such as `redacted_length`, `sentiment`, and `num_words` are assumed to independently contribute to the model's predictions without significant interaction effects.

5. **SpaCy Model Accuracy**:
  - The `en_core_web_md` model from spaCy is assumed to provide accurate tokenization and linguistic features for the dataset.

6. **Validation Data Representativeness**:
  - The validation dataset from `unredactor.tsv` is assumed to be a reliable representation of the test data and generalizes well to unseen cases.

7. **Unique Name Correspondence**:
  - Each redaction corresponds to a single unique name, which is expected to appear in the training dataset.

8. **Dependency Availability**:
  - The required Python libraries (`pandas`, `numpy`, `scikit-learn`, `TextBlob`, `spaCy`) are installed and compatible with the runtime environment.





