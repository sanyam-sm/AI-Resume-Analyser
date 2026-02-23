# 🤖 AI Resume Analyzer

A production-grade AI-powered resume analyzer built with **Flask**, **scikit-learn**, and a stunning dark-theme UI. Uses ML models trained on thousands of real resumes (via Kaggle) to classify your resume, score it, and deliver smart recommendations.

---

## ✨ Features

| Feature | Details |
|---|---|
| **ML Classification** | Trained 7+ models (LR, SVM, RF, GBM, NB, XGBoost, LightGBM) — best one auto-selected |
| **Resume Scoring** | 9-section scoring system (Contact, Skills, Projects, Certs, etc.) |
| **Top-N Predictions** | Confidence-ranked predictions across all job categories |
| **Skill Analysis** | Detects 40+ skills, recommends category-specific additions |
| **Course Recommendations** | Curated courses for detected job domain |
| **Beautiful UI** | Dark theme, animated charts, drag & drop upload |
| **Experience Detection** | Auto-detects Fresher / Junior / Mid-Level / Senior |

---

## 📁 Project Structure

```
ai-resume-analyzer/
├── notebooks/
│   └── train_model.ipynb        ← Jupyter: train ML models
├── models/                       ← Auto-created by notebook
│   ├── resume_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── model_meta.json
├── static/
│   └── js/
│       └── main.js               ← Frontend logic
├── templates/
│   └── index.html                ← Main HTML (dark theme)
├── uploads/                      ← Temp uploads (auto-cleaned)
├── utils/
│   └── parser.py                 ← PDF parsing + NLP utilities
├── app.py                        ← Flask backend
└── requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train the Models (Jupyter Notebook)

```bash
cd notebooks
jupyter notebook train_model.ipynb
```

> **What it does:**
> 1. Loads the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (2,484 resumes, 25 categories)
> 2. Preprocesses text (lemmatization, stop-word removal)
> 3. Builds TF-IDF features (15,000 n-grams)
> 4. Trains 7–9 models and compares them
> 5. Saves best model + vectorizer + label encoder as `.pkl` files

> **Kaggle Authentication:** You need a Kaggle account and `~/.kaggle/kaggle.json` API key. [See instructions](https://www.kaggle.com/docs/api).

### Step 3 — Run the Web App

```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🧠 ML Pipeline

```
Resume PDF
    ↓
Text Extraction (pdfplumber)
    ↓
Preprocessing (lowercase, lemmatize, remove stopwords, strip URLs/emails/phones)
    ↓
TF-IDF Vectorization (15k features, unigrams + bigrams)
    ↓
Multi-Model Training & Comparison:
    - Logistic Regression
    - Linear SVC
    - Random Forest
    - Gradient Boosting
    - Naive Bayes
    - K-Nearest Neighbors
    - XGBoost (if installed)
    - LightGBM (if installed)
    ↓
5-Fold Cross-Validation
    ↓
Best Model Selected by F1-Weighted Score
    ↓
Saved as resume_model.pkl
```

---

## 📊 Resume Scoring System

| Section | Points |
|---|---|
| Contact Info | 10 |
| Summary / Objective | 8 |
| Education | 15 |
| Experience | 20 |
| Skills | 15 |
| Projects | 12 |
| Certifications | 10 |
| Achievements | 5 |
| Hobbies / Interests | 5 |
| **Total** | **100** |

---

## 🎨 UI Components

- **Resume Strength Ring** — animated doughnut chart (Chart.js)
- **Category Confidence Bars** — animated progress bars for top-5 predictions
- **Score Breakdown** — per-section scoring with visual fill bars
- **Skill Tags** — current skills vs. recommended skills
- **Course Cards** — direct Udemy/Coursera enrollment links
- **PDF Preview** — embedded PDF viewer (base64)
- **Model Banner** — shows which ML model was used and its accuracy

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Main UI |
| `POST /api/analyze` | POST | Upload PDF, returns JSON analysis |
| `GET /api/model-info` | GET | Returns model name, accuracy, classes |
| `GET /api/demo` | GET | Returns demo analysis (no file needed) |

### Example Response (`/api/analyze`)

```json
{
  "status": "success",
  "extracted": {
    "name": "Alex Johnson",
    "email": "alex@email.com",
    "phone": "+1-555-0192",
    "pages": 2,
    "word_count": 680
  },
  "prediction": {
    "category": "Data Science",
    "experience_level": "Mid-Level",
    "top_predictions": [
      { "label": "Data Science",   "confidence": 82.4 },
      { "label": "Python Developer", "confidence": 9.8 }
    ]
  },
  "skills": {
    "current": ["Python", "Machine Learning", "SQL"],
    "recommended": ["Deep Learning", "NLP", "Big Data"]
  },
  "score": { "total": 71, "max": 100, "breakdown": { ... } },
  "courses": [
    { "name": "Machine Learning A-Z", "url": "https://..." }
  ]
}
```

---

## 📦 Dataset

- **Source:** [Kaggle — Resume Dataset by Sneha Anbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Size:** 2,484 resumes across 25 job categories
- **Categories include:** Data Science, Python Developer, Java Developer, React Developer, Network Security, HR, Business Analyst, and more

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask 3.x |
| ML | scikit-learn, XGBoost, LightGBM |
| NLP | NLTK (lemmatization, stopwords) |
| PDF | pdfplumber |
| Frontend | Vanilla JS + Chart.js |
| Fonts | Syne (headings) + DM Sans (body) |
| Training | Jupyter Notebook |
