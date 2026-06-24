# Smart Text Mining for Review Retrieval, Classification, and Clustering

---

## Overview

This project builds a smart information retrieval and text analysis system that processes customer reviews from two domains: **beauty salons** (domain-related) and **gyms** (out-of-domain). It applies NLP techniques to clean and analyze text, retrieve similar documents using TF-IDF and cosine similarity, classify emails as spam/ham, and cluster documents using K-Means.

---

## Requirements

```bash
pip install nltk spacy scikit-learn pandas matplotlib seaborn ipywidgets
python -m spacy download en_core_web_sm
```

---

## How to Run

1. Open the notebook in **Google Colab**
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Place the `.txt` review files under `All folders/` in your Drive
4. Upload `spam.csv` to `/content/`
5. Run cells in order from top to bottom

---

## Project Parts

### Part 1 — Text Normalization & TF-IDF

A dataset of 30 `.txt` documents was created — 15 beauty salon reviews (domain-related) and 15 gym reviews (out-of-domain).

**Cleaning steps applied to all documents:**
1. **Tokenization & Lowercasing** — split text into words, convert to lowercase
2. **Stopword Removal** — remove common words like "is", "the", "and" using NLTK
3. **Non-Alphabetic Filtering** — remove numbers and punctuation
4. **Lemmatization** — reduce words to base form using spaCy

All steps are wrapped in a single `normalize(text)` function.

**TF-IDF** is then computed manually using `math.log10`, producing a weighted term-document matrix displayed as a DataFrame.

**Libraries:** `nltk`, `spacy`, `glob`, `pandas`, `math`

---

### Part 2 — Information Retrieval

Retrieves the most relevant documents for a given query using cosine similarity.

**Process for each query:**
1. Normalize the query using the same `normalize()` function
2. Build a TF-IDF matrix covering all documents + the query
3. Compute cosine similarity between the query vector and each document vector
4. Rank and display the top 10 most similar documents
5. Calculate **Precision** and **Recall** against a predefined set of relevant documents

**Queries used:**
- `"bad customer service at the salon"`
- `"unpleasant salon experience"`
- `"overpriced and not professional"`
- `"salon appointment was delayed and staff unfriendly"`
- `"gorgeous quality hair treatment and nail cleaning"`
- `"the good haircut service and results"`
- `"waited too long and didn't like the haircut"`
- `"the salon service was professional and friendly"`

**Sample output:**
```
Precision: 0.60
Recall: 0.43
```

**Libraries:** `sklearn`, `numpy`

> **Bug fix applied:** Moved all TF-IDF and similarity logic inside the loop so each query is evaluated independently.

---

### Part 3 — Document Classification

Classifies emails as **spam** or **ham** using two supervised learning models trained on `spam.csv`.

**Steps:**
1. Load and clean data — remove 415 duplicate rows, check for missing values
2. Convert text to TF-IDF vectors (with English stopwords removed)
3. Split: 80% training / 20% testing
4. Train two models:
   - **Decision Tree Classifier**
   - **Multinomial Naive Bayes**
5. Evaluate with Accuracy, Classification Report, and Confusion Matrix

**Results:**

| Model | Accuracy |
|-------|----------|
| Decision Tree | 97% |
| Naive Bayes | 97% |

**Libraries:** `sklearn`, `seaborn`, `matplotlib`, `pandas`

---

### Part 4 — Document Clustering

Groups documents into two clusters (spam / ham) using unsupervised learning on `spam.csv`.

**Steps:**
1. Load `spam.csv` and extract the `Message` column
2. Vectorize text using TF-IDF
3. Apply **K-Means** with `k=2`
4. Reduce to 2D using **PCA** for visualization
5. Plot a scatter plot where each color represents a cluster

**Evaluation (when true labels are available):**
- Clustering Accuracy: ~85–87%
- Silhouette Score: 0.53

**Libraries:** `sklearn`, `matplotlib`, `pandas`

---

### Bonus — Interactive Widget

A GUI built with `ipywidgets` that demonstrates the three core tasks on a small set of sample documents.

| Button | Function |
|--------|----------|
| Retrieve Similar Documents | Shows top 3 most similar documents to the input text |
| Classify | Predicts Positive or Negative sentiment using Naive Bayes |
| Group | Clusters the sample documents into 2 groups using K-Means |

**Libraries:** `ipywidgets`, `sklearn`

---

## Bugs Fixed

| Location | Issue | Fix |
|----------|-------|-----|
| Part 2 | Loop defined `query_normalized` but all TF-IDF/similarity logic ran outside the loop — only the last query was evaluated | Moved all logic inside the loop so each query is processed independently |
| Last cell | `IndentationError` in nested `if/else` + `similarities` variable used outside its scope | Removed the broken cell; the widget cell above already covers this functionality |

---

## References

- NLTK: https://www.nltk.org/
- spaCy: https://spacy.io/
- scikit-learn: https://scikit-learn.org/
- Matplotlib: https://matplotlib.org/
- Google Colab: https://colab.research.google.com/
- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- Precision & Recall: https://en.wikipedia.org/wiki/Precision_and_recall
