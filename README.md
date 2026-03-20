# AI Resume Analyzer

<div align="center">

**Production-grade AI-powered resume analysis platform powered by Machine Learning, BERT NER, and Intelligent Skill Matching**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-orange.svg)
![NER](https://img.shields.io/badge/NER-BERT-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [API](#-api-documentation) • [Model Performance](#-model-performance)

</div>

---

## 📋 Overview

AI Resume Analyzer is a comprehensive resume analysis platform that combines **XGBoost classification**, **BERT-based Named Entity Recognition**, and **intelligent job matching** to provide actionable insights for job seekers. Built on a dataset of 14,000+ resumes across 52 job categories, it delivers professional-grade analysis with 84% accuracy.

### What It Does

- **Classifies resumes** into 52+ job categories with confidence scores
- **Extracts structured data** (name, email, phone, skills, experience, education) using BERT NER
- **Scores resumes** on a 100-point scale across 9 critical sections
- **Matches candidates** to 18 predefined job roles with skill gap analysis
- **Recommends projects** using Gemini 2.0 Flash AI based on detected skills
- **Suggests courses** from YouTube, Coursera, and Udemy for skill development

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Model ML Classification** | Trains 6+ models (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, Naive Bayes) and auto-selects the best performer |
| **BERT Named Entity Recognition** | Extracts entities using `yashpwr/resume-ner-bert-v2` with regex fallback |
| **52 Job Categories** | Data Science, Python Developer, DevOps Engineer, Java Developer, React Developer, Blockchain, and more |
| **Resume Scoring (0-100)** | 9-section heuristic scoring: Contact Info, Summary, Education, Experience, Skills, Projects, Certifications, Achievements, Hobbies |
| **Job Role Matching** | Matches against 18 job roles with skill gap analysis (Core: 70%, Preferred: 30%) |
| **AI-Powered Project Ideas** | Gemini 2.0 Flash generates personalized project suggestions based on skills |
| **Smart Course Recommendations** | YouTube Data API v3 + curated database of 30+ courses |
| **Experience Level Detection** | Automatically classifies as Fresher, Junior, Mid-Level, or Senior |
| **Beautiful Dark UI** | Drag-and-drop upload, animated Chart.js visualizations, embedded PDF preview |

### Extracted Information

- **Personal Details:** Name, Email, Phone Number
- **Skills:** 286 known skills with confidence percentages
- **Experience:** Companies worked at, job titles, years of experience
- **Education:** Degrees, institutions, certifications
- **Locations:** Geographic information from work history
- **Resume Metrics:** Page count, word count, text density

---

## 🎯 Demo

### Input
Upload a PDF resume via drag-and-drop or file picker.

### Output
```json
{
  "status": "success",
  "extracted": {
    "name": "Sarah Johnson",
    "email": "sarah.j@email.com",
    "phone": "+1-555-123-4567",
    "skills": ["Python", "Machine Learning", "Docker", "AWS", "SQL"],
    "experience_years": "3-5 years",
    "companies": ["TechCorp Inc.", "DataSolutions"],
    "designations": ["ML Engineer", "Data Analyst"],
    "education": ["B.S. Computer Science"]
  },
  "prediction": {
    "category": "Data Science",
    "confidence": 87.3,
    "experience_level": "Mid-Level",
    "top_predictions": [
      {"label": "Data Science", "confidence": 87.3},
      {"label": "Python Developer", "confidence": 8.2},
      {"label": "Machine Learning Engineer", "confidence": 3.1}
    ]
  },
  "score": {
    "total": 78,
    "max": 100,
    "breakdown": {
      "contact_info": 10,
      "summary": 6,
      "education": 13,
      "experience": 18,
      "skills": 13,
      "projects": 10,
      "certifications": 3,
      "achievements": 3,
      "hobbies": 2
    }
  },
  "job_matches": [
    {
      "role": "Data Scientist",
      "match_percentage": 78,
      "core_skills_match": 85,
      "preferred_skills_match": 60,
      "missing_skills": {
        "core": ["Deep Learning", "NLP"],
        "preferred": ["Big Data", "Spark"]
      }
    }
  ]
}
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Kaggle API credentials (for model training)
- pip or conda package manager

### 1. Clone the Repository

```bash
git clone https://github.com/sanyam-sm/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>View requirements.txt</summary>

```
flask>=3.0.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.3.0
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.3.0
nltk>=3.8.0
transformers>=4.36.0
torch>=2.1.0
accelerate>=0.25.0
pdfplumber>=0.11.0
kagglehub[pandas-datasets]>=0.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
google-generativeai
```
</details>

### 3. Download NLTK Data

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# Google Gemini API (for AI project suggestions)
GEMINI_API_KEY=your_gemini_api_key_here

# YouTube Data API v3 (for course recommendations)
YOUTUBE_API_KEY=your_youtube_api_key_here
```

**Get API Keys:**
- **Gemini API:** [Google AI Studio](https://aistudio.google.com/app/apikey)
- **YouTube API:** [Google Cloud Console](https://console.cloud.google.com/apis/credentials)

> **Note:** Application works without API keys but falls back to curated databases instead of live recommendations.

### 5. Train the Model (First Time Setup)

#### Option A: Train from Scratch

```bash
jupyter notebook notebooks/train_model_v2.ipynb
```

Run all cells to:
1. Download Kaggle dataset (14,308 resumes)
2. Train 6+ ML models with 5-fold cross-validation
3. Export best model (resume_model_v2.pkl, tfidf_vectorizer_v2.pkl, label_encoder_v2.pkl)



#### Option B: Use Pre-trained Models

If models already exist in `notebooks/models/`, skip training and proceed to step 6.

### 6. Run the Application

```bash
python app.py
```

Open your browser to **http://localhost:5000**

---

## 📊 Model Performance

### Version Comparison

| Metric | v1.1 (LightGBM) | v2.0 (XGBoost) ⭐ |
|--------|----------------|------------------|
| **Accuracy** | 82.7% | **84.0%** |
| **F1-Weighted** | 82.1% | **83.8%** |
| **F1-Macro** | 77.6% | **83.2%** |
| **Categories** | 24 | **52** |
| **Dataset Size** | 2,484 samples | **14,308 samples** |
| **TF-IDF Features** | 15,000 | 10,000 |
| **Training Samples** | 1,987 | 11,446 |
| **Test Samples** | 497 | 2,862 |

**v2.0 Status:** Currently active in production

### Model Training Pipeline

```
Raw Resume Dataset (Kaggle)
         ↓
Text Preprocessing
  • Lowercase normalization
  • Lemmatization (NLTK WordNet)
  • Stopword removal
  • URL/Email/Phone stripping
         ↓
TF-IDF Vectorization
  • 10,000 features (v2.0) or 15,000 (v1.1)
  • Unigrams + Bigrams
  • Max DF: 0.8, Min DF: 2
         ↓
Model Training & Selection
  • Logistic Regression
  • Linear SVC
  • Random Forest
  • Gradient Boosting
  • K-Nearest Neighbors
  • Naive Bayes
  • XGBoost
  • LightGBM
         ↓
5-Fold Cross-Validation
         ↓
Best Model Selection (F1-Weighted)
         ↓
Serialization (joblib)
  • resume_model_v2.pkl
  • tfidf_vectorizer_v2.pkl
  • label_encoder_v2.pkl
  • model_meta_v2.json
```

---

## 🏗️ Architecture

### Application Workflow

```
┌─────────────────┐
│  Upload Resume  │
│     (PDF)       │
└────────┬────────┘
         ↓
┌─────────────────────────────────────┐
│  Text Extraction (pdfplumber)      │
│  Fallback: pdfminer3                │
└────────┬────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  BERT NER Extraction                     │
│  Model: yashpwr/resume-ner-bert-v2       │
│  Extracts: Name, Email, Phone, Skills,   │
│  Companies, Education, Locations, YOE    │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  Skill Detection (Hybrid)                │
│  • NER entities (SKILLS tag)             │
│  • Keyword matching (286 skills)         │
│  • Confidence scoring                    │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  Text Preprocessing                      │
│  • Lemmatization                         │
│  • Stopword removal                      │
│  • Lowercasing                           │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  TF-IDF Vectorization                    │
│  Transform to 10k-dimensional vector     │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  ML Classification (XGBoost v2.0)        │
│  Predict job category + confidence       │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  Post-Processing Pipeline                │
│  • Experience level detection (regex)    │
│  • Job role matching (18 roles)          │
│  • Skill gap analysis                    │
│  • Resume scoring (9 sections)           │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  Recommendation Engine                   │
│  • Gemini 2.0 Flash (project ideas)      │
│  • YouTube API (courses)                 │
│  • Skill-based ranking                   │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  JSON Response + UI Visualization        │
│  Chart.js animations, skill tags, bars   │
└──────────────────────────────────────────┘
```

### Project Structure

```
AI-Resume-Analyzer\
│
├── app.py                          # Flask backend 
│   ├── Routes: /, /api/analyze, /api/model-info, /api/demo
│   ├── Model loading (v2.0 XGBoost)
│   ├── Job matching engine
│   ├── Gemini AI integration
│   └── YouTube API integration
│
├── utils/
│   ├── __init__.py
│   └── parser.py                  # NLP utilities 
│       ├── PDF text extraction (pdfplumber + pdfminer3)
│       ├── BERT NER extraction (Transformers)
│       ├── Text preprocessing (NLTK)
│       ├── Skill detection (286 skills)
│       └── Resume scoring logic
│
├── templates/
│   └── index.html                 # Dark-themed frontend 
│       ├── Drag-and-drop upload
│       ├── Animated UI components
│       └── Embedded PDF preview
│
├── static/
│   └── js/
│       └── main.js                # Frontend logic 
│           ├── File upload handling
│           ├── Chart.js visualizations
│           └── Dynamic results rendering
│
├── notebooks/
│   ├── train_model.ipynb          # v1.1 training (24 categories)
│   ├── train_model_v2.ipynb       # v2.0 training (52 categories) ⭐
│   └── models/
│       ├── resume_model_v2.pkl    # Active XGBoost model
│       ├── tfidf_vectorizer_v2.pkl
│       ├── label_encoder_v2.pkl
│       ├── model_meta_v2.json
│       ├── resume_model.pkl       # Legacy v1.1 (LightGBM)
│       ├── tfidf_vectorizer.pkl
│       ├── label_encoder.pkl
│       └── model_meta.json
│
├── uploads/                       # Temporary PDF storage (auto-cleaned)
├── requirements.txt               # Python dependencies
├── .env                          # API keys (Git-ignored)
└── README.md                     # This file
```

---

## 🔥 Key Features in Detail

### 1. Multi-Model Machine Learning Pipeline

The system trains and evaluates multiple classification algorithms:

- **Logistic Regression** (baseline)
- **Linear Support Vector Classifier**
- **Random Forest** (ensemble)
- **Gradient Boosting**
- **Naive Bayes**
- **K-Nearest Neighbors**
- **XGBoost** ⭐ (current best - v2.0)
- **LightGBM** (v1.1 winner)

**Selection Criteria:** F1-weighted score from 5-fold cross-validation

### 2. BERT-Based Named Entity Recognition

Uses Hugging Face's `yashpwr/resume-ner-bert-v2` transformer model to extract:

- **PERSON:** Full name
- **EMAIL:** Email addresses
- **PHONE:** Phone numbers
- **SKILLS:** Technical and soft skills
- **COMPANIES:** Organizations worked at
- **DESIGNATION:** Job titles
- **EDUCATION:** Degrees and institutions
- **LOCATION:** Cities, states, countries
- **YEARS OF EXPERIENCE:** Tenure information

**Hybrid Approach:** Combines NER predictions with regex patterns for robustness.

### 3. Intelligent Job Matching

Compares candidate skills against 18 job role profiles:

- Backend Developer
- Data Scientist
- DevOps Engineer
- Frontend Developer
- Full Stack Developer
- Mobile App Developer
- Data Analyst
- Cloud Architect
- Cybersecurity Analyst
- Machine Learning Engineer
- Product Manager
- QA Engineer
- Database Administrator
- Blockchain Developer
- AI Research Scientist
- ...and more

**Scoring Algorithm:**
- Core skills: 70% weight
- Preferred skills: 30% weight
- Outputs match percentage and skill gaps

### 4. Resume Scoring System

| Section | Weight | Criteria |
|---------|--------|----------|
| **Contact Info** | 10 pts | Email (5) + Phone (5) |
| **Summary/Objective** | 8 pts | Presence of career summary |
| **Education** | 15 pts | Degrees, institutions, certifications |
| **Experience** | 20 pts | Work history, companies, roles |
| **Skills** | 15 pts | Technical and soft skills |
| **Projects** | 12 pts | Portfolio projects |
| **Certifications** | 10 pts | Professional certifications |
| **Achievements** | 5 pts | Awards, recognitions |
| **Hobbies/Interests** | 5 pts | Personal interests |
| **Total** | **100 pts** | Overall resume strength |

### 5. AI-Powered Recommendations

**Project Ideas:**
- Gemini 2.0 Flash generates personalized project suggestions
- Fallback to curated database of 50+ project ideas
- Matched by skill relevance and difficulty

**Course Recommendations:**
- YouTube Data API v3 searches for trending courses
- Curated database of 30+ courses (freeCodeCamp, Udemy, Coursera)
- Skill-specific recommendations with direct enrollment links

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Flask 3.x |
| **ML Framework** | scikit-learn 1.4+, XGBoost 2.0+, LightGBM 4.3+ |
| **Deep Learning** | PyTorch 2.1+, Hugging Face Transformers 4.36+ |
| **NLP** | NLTK (lemmatization, stopwords) |
| **NER Model** | BERT (`yashpwr/resume-ner-bert-v2`) |
| **PDF Parsing** | pdfplumber 0.11+ (primary), pdfminer3 (fallback) |
| **Frontend** | Vanilla JavaScript, HTML5, CSS3 |
| **Visualization** | Chart.js 4.4.1 |
| **Typography** | Google Fonts (Syne, DM Sans) |
| **AI APIs** | Google Gemini 2.0 Flash, YouTube Data API v3 |
| **Training** | Jupyter Notebook, Matplotlib, Seaborn |

---

## 📦 Dataset Information

### Training Data

- **Source:** [Kaggle Resume Dataset by Sneha Anbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), [jillanisofttech/updated-resume-dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset), [ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas)
- **Size:** 14,308 resumes (v2.0) | 2,484 resumes (v1.1)
- **Categories:** 52 job roles (v2.0) | 24 categories (v1.1)
- **Format:** PDF text extracts with labeled categories

### Categories (v2.0)

```
Data Science | Python Developer | DevOps Engineer | Java Developer |
React Developer | Full Stack Developer | Testing | Blockchain |
Network Security Engineer | Database | Automation Testing | ETL Developer |
Android Developer | iOS Developer | Machine Learning Engineer |
Frontend Developer | Backend Developer | Data Analyst | Quality Assurance |
Business Analyst | Cloud Architect | UI/UX Designer | Product Manager |
Technical Writer | Cybersecurity Analyst | HR Manager | Sales Manager |
Marketing Manager | Financial Analyst | Operations Manager | and 22 more...
```

---

## 🔌 API Documentation

### `POST /api/analyze`

Upload and analyze a resume.

**Request:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "status": "success",
  "extracted": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1-555-999-8888",
    "skills": ["Python", "Flask", "React"],
    "experience_years": "2-3 years",
    "companies": ["Tech Startup Inc."],
    "designations": ["Software Engineer"],
    "education": ["B.Tech Computer Science"],
    "locations": ["San Francisco, CA"],
    "pages": 2,
    "word_count": 512
  },
  "prediction": {
    "category": "Python Developer",
    "confidence": 91.2,
    "experience_level": "Junior",
    "top_predictions": [...]
  },
  "skills": {
    "current": ["Python", "Flask", "React"],
    "all_skills": ["Python", "Flask", "React", "Git"],
    "recommended": ["Django", "PostgreSQL", "Docker"]
  },
  "score": {
    "total": 73,
    "max": 100,
    "percentage": 73.0,
    "breakdown": {...}
  },
  "job_matches": [...],
  "skill_gaps": [...],
  "projects": [...],
  "courses": [...],
  "model_info": {
    "name": "XGBoost v2.0",
    "accuracy": 84.0,
    "f1_weighted": 83.8,
    "trained_on": "14,308 samples"
  }
}
```

### `GET /api/model-info`

Get metadata about the loaded ML model.

**Response:**
```json
{
  "model_name": "XGBoost",
  "accuracy": 84.0,
  "f1_weighted": 83.8,
  "f1_macro": 83.2,
  "num_classes": 52,
  "num_samples": 14308,
  "features": 10000,
  "version": "2.0"
}
```

### `GET /api/demo`

Returns a demo analysis without file upload (for testing).

---

## 🎨 UI Features

### Dark Theme Design

- **Color Palette:** Deep purple/blue gradients with cyan accents
- **Fonts:** Syne (headings, 600 weight) + DM Sans (body, 400/500/600)
- **Animations:** Fade-up keyframes, animated progress bars, Chart.js transitions

### Components

1. **Drag-and-Drop Upload Zone** - SVG upload icon with hover effects
2. **Loading Animation** - Animated dots during analysis
3. **Resume Strength Chart** - Chart.js doughnut chart (0-100 score)
4. **Category Confidence Bars** - Top-5 predictions with animated fill
5. **Score Breakdown Grid** - 9-section scoring with visual indicators
6. **Skill Tags** - Color-coded chips with confidence percentages
7. **Job Match Cards** - Role matching with progress bars
8. **Course Cards** - Clickable course recommendations with thumbnails
9. **PDF Embeded Preview** - Base64-encoded PDF display
10. **Model Info Banner** - Shows active model and accuracy

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for project suggestions | Optional |
| `YOUTUBE_API_KEY` | YouTube Data API v3 key for course search | Optional |

### Application Settings (app.py)

```python
# Flask Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model Configuration
MODEL_VERSION = 'v2'  # 'v1' or 'v2'
MODEL_PATH = 'notebooks/models/resume_model_v2.pkl'
VECTORIZER_PATH = 'notebooks/models/tfidf_vectorizer_v2.pkl'
ENCODER_PATH = 'notebooks/models/label_encoder_v2.pkl'
```

### Skill Database (utils/parser.py)

286 known skills across categories:
- **Programming:** Python, Java, JavaScript, C++, Go, Rust, etc.
- **Frameworks:** React, Angular, Vue, Django, Flask, Spring, etc.
- **DevOps:** Docker, Kubernetes, Jenkins, AWS, Azure, GCP, etc.
- **Data Science:** Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch, etc.
- **Databases:** MySQL, PostgreSQL, MongoDB, Redis, etc.
- **Tools:** Git, JIRA, Postman, etc.

---

## 🚦 Usage

### Running the Application

```bash
# Development mode (default)
python app.py

# Access at http://localhost:5000
```

### Training New Models

```bash
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook train_model_v2.ipynb

# Follow notebook instructions:
# 1. Run data loading cells
# 2. Execute EDA cells (optional)
# 3. Run preprocessing pipeline
# 4. Train all models
# 5. Export best model
```

### Switching Model Versions

Edit `app.py` line ~20-30:

```python
# Use v2.0 (current)
model = joblib.load('notebooks/models/resume_model_v2.pkl')
vectorizer = joblib.load('notebooks/models/tfidf_vectorizer_v2.pkl')
encoder = joblib.load('notebooks/models/label_encoder_v2.pkl')

# Or use v1.1 (legacy)
model = joblib.load('notebooks/models/resume_model.pkl')
vectorizer = joblib.load('notebooks/models/tfidf_vectorizer.pkl')
encoder = joblib.load('notebooks/models/label_encoder.pkl')
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Model files not found**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'notebooks/models/resume_model_v2.pkl'
```
**Solution:** Run `train_model_v2.ipynb` notebook to generate models.

**2. NLTK data missing**
```bash
LookupError: Resource 'corpora/stopwords' not found
```
**Solution:**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**3. Kaggle API authentication failed**
```bash
OSError: Could not find kaggle.json
```
**Solution:** Place `kaggle.json` in `~/.kaggle/` directory. [Get API key](https://www.kaggle.com/settings/account).

**4. BERT model download slow**
```
Downloading yashpwr/resume-ner-bert-v2: 100%|████████████| 420M
```
**Solution:** First run downloads ~420MB model. Subsequent runs use cached version from `~/.cache/huggingface/`.

**5. PDF parsing errors**
```bash
Error extracting text from PDF: <error_message>
```
**Solution:** Application automatically falls back to pdfminer3. If both fail, PDF may be image-based (requires OCR).

**6. API key errors (Gemini/YouTube)**
```bash
google.api_core.exceptions.PermissionDenied: API key not valid
```
**Solution:** Verify `.env` file exists and contains valid keys. App continues with fallback databases if APIs fail.

---

## 🔒 Security Considerations

### Current .env Exposure

The `.env` file contains API keys and is **gitignored**, but placeholder keys are visible in project files.

**Recommendations:**
1. Never commit real API keys to version control
2. Use environment-specific .env files (.env.development, .env.production)
3. Rotate exposed API keys immediately
4. Consider using secret management tools (AWS Secrets Manager, Azure Key Vault)

### File Upload Security

- **Max file size:** 16MB
- **Allowed extensions:** `.pdf` only
- **Storage:** Temporary uploads folder (cleaned after processing)
- **Validation:** File extension and MIME type checking

**Recommendations:**
1. Add virus scanning for uploaded files
2. Implement rate limiting on `/api/analyze`
3. Add authentication for production deployment
4. Sanitize PDF metadata before processing

---

## 🚀 Deployment

### Local Development

```bash
python app.py
# Runs on http://localhost:5000
```

### Production Deployment (Docker)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "4", "app:app"]
```

Build and run:

```bash
docker build -t ai-resume-analyzer .
docker run -p 5000:5000 --env-file .env ai-resume-analyzer
```

### Production Recommendations

- **Web Server:** Use Gunicorn or uWSGI instead of Flask dev server
- **Reverse Proxy:** Nginx for SSL termination and static file serving
- **Caching:** Redis for model prediction caching
- **Queue System:** Celery for async analysis tasks
- **Monitoring:** Application performance monitoring (APM) tools
- **Logging:** Centralized logging (ELK stack, CloudWatch)

---

## 📈 Performance Metrics

### Model Performance (v2.0)

- **Inference Time:** ~1-3 seconds per resume
- **NER Extraction:** ~2-4 seconds (BERT on CPU)
- **Total Processing:** ~3-7 seconds end-to-end
- **Memory Usage:** ~1.2GB (BERT model loaded)
- **Model Size:**
  - XGBoost model: ~5MB
  - TF-IDF vectorizer: ~80MB
  - BERT NER model: ~420MB

### Scalability

- **Single-threaded:** ~8-12 resumes/minute
- **Multi-worker:** ~30-50 resumes/minute (4 workers)
- **Bottleneck:** BERT NER inference (CPU-bound)

**Optimization Tips:**
- Use GPU for BERT inference (3-5x speedup)
- Cache vectorized results for duplicate resumes
- Batch processing for multiple resumes
- Lazy-load BERT model only when needed

---

## 🧪 Testing

### Manual Testing

```bash
# Test demo endpoint (no file upload)
curl http://localhost:5000/api/demo

# Test file upload
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@/path/to/resume.pdf"

# Test model info
curl http://localhost:5000/api/model-info
```

### Model Validation

Check model metadata:

```bash
python -c "
import joblib
meta = joblib.load('notebooks/models/model_meta_v2.json')
print(f'Accuracy: {meta[\"accuracy\"]}%')
print(f'Classes: {meta[\"n_classes\"]}')
"
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Improvement

1. **Model Enhancements**
   - Fine-tune BERT on resume-specific data
   - Implement deep learning classifier (LSTM, Transformer)
   - Add multi-label classification (one resume, multiple roles)

2. **Features**
   - Batch resume processing
   - Resume comparison (A vs B)
   - ATS-friendliness score
   - Resume reformatting suggestions
   - Cover letter generator

3. **Infrastructure**
   - Add unit tests (pytest)
   - CI/CD pipeline (GitHub Actions)
   - API authentication (JWT, OAuth)
   - Rate limiting and caching
   - Database integration (PostgreSQL)

4. **UI/UX**
   - Mobile-responsive design
   - Resume editor with live scoring
   - Export analysis as PDF report
   - Dark/light mode toggle

### Development Workflow

```bash
# Fork and clone
git clone https://github.com/yourusername/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, commit
git add .
git commit -m "Add: your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

---

## 📝 Roadmap

- [ ] Add user authentication and resume history
- [ ] Implement batch processing API
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add support for DOCX/TXT formats
- [ ] Integrate with LinkedIn API for profile comparison
- [ ] Build Chrome extension for one-click analysis
- [ ] Add A/B testing for resume versions
- [ ] Implement resume anonymization for bias testing
- [ ] Create mobile app (React Native)
- [ ] Add internationalization (i18n) for non-English resumes

---

## 👥 Authors

This project was developed and maintained by:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/sanyam-sm/">
        <img src="https://github.com/sanyam-sm.png" width="100px;" alt="Sanyam"/>
        <br />
        <sub><b>Sanyam</b></sub>
      </a>
      <br />
      <a href="https://github.com/sanyam-sm/" title="GitHub Profile">💻</a>
    </td>
    <td align="center">
      <a href="https://github.com/Pankajdagar777">
        <img src="https://github.com/Pankajdagar777.png" width="100px;" alt="Pankaj"/>
        <br />
        <sub><b>Pankaj</b></sub>
      </a>
      <br />
      <a href="https://github.com/Pankajdagar777" title="GitHub Profile">💻</a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** [Kaggle Resume Dataset by Sneha Anbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), [jillanisofttech/updated-resume-dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset), [ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas)
- **NER Model:** [yashpwr/resume-ner-bert-v2](https://huggingface.co/yashpwr/resume-ner-bert-v2) on Hugging Face
- **AI APIs:** Google Gemini 2.0 Flash, YouTube Data API v3
- **Visualization:** Chart.js for beautiful data visualizations

---

## 📧 Contact

For questions, issues, or suggestions:
- **Open an issue** on [GitHub Issues](https://github.com/sanyam-sm/AI-Resume-Analyzer/issues)
- **Reach out to authors:**
  - [Sanyam](https://github.com/sanyam-sm/)
  - [Pankaj](https://github.com/Pankajdagar777)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star on GitHub!

---



