
## Make Data Count — Dataset Reference Classification

Dataset:[https://www.kaggle.com/competitions/make-data-count-finding-data-references/data]

### Problem Statement

The challenge is to **detect and classify mentions of scientific datasets** within research articles. For each article and its mentioned dataset, we need to predict whether the mention is:

- `Primary`: The dataset was used in the actual study.
- `Secondary`: The dataset was simply referenced or cited.

We’re given XML and PDF versions of scientific papers and a training label file mapping dataset mentions to article IDs and citation types.

---

### Solution Overview

This solution performs the following steps:

### 1. Data Extraction from XML/PDF
- XMLs are parsed using `lxml` to extract `title`, `abstract`, and body `sections`.
- PDFs (optional fallback) are parsed using `pdfplumber` if the XML is unavailable or incomplete.

```python
{
  "section": "Abstract",
  "text": "RNA splicing is a key mechanism..."
}
```

### 2. Matching Datasets with Article Text
Each article is scanned for dataset IDs. If a match is found, the text around it is extracted and saved with the associated label from `train_labels.csv`.

### 3. Text Classification
A simple pipeline is built with:
- `TfidfVectorizer` (word + bigrams)
- `LogisticRegression` with balanced class weights

```python
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000, ngram_range=(1,2)),
    LogisticRegression(max_iter=1000, class_weight="balanced")
)
```

This model learns to distinguish `Primary` vs `Secondary` mentions based on surrounding text.

### 4. Evaluation
The model is trained on labeled data and evaluated using:
- Precision, recall, F1-score
- Accuracy and support per class

### 5. Prediction on Test Set
Each test article is processed to:
- Predict the type (`Primary` or `Secondary`)
- Extract dataset IDs using regex and custom normalizers
- Ensure one prediction per article (fallback if needed)

---

### File Structure

```
├── train/
│   ├── XML/
│   └── PDF/
├── test/
│   └── XML/
├── train_labels.csv
├── extract_text.py
├── train_model.py
├── predict.py
└── submission.csv
```

---

### Dependencies

- `lxml`
- `pdfplumber`
- `scikit-learn`
- `pandas`
- `tqdm`
- `joblib` (for parallel processing)

Install them via:

```bash
pip install lxml pdfplumber scikit-learn pandas tqdm joblib
```

---

### Key Techniques

- Robust XML and PDF parsing with fallbacks
- Dataset ID normalization and regex cleaning
- Pipeline-based ML model with TF-IDF + Logistic Regression
- Parallelized feature extraction using `joblib` for speed

---

### Notes

- PDF parsing is used **only if XML fails**.
- Duplicate predictions are avoided using a `seen` set.
- Fallbacks ensure at least one row is predicted per article.
