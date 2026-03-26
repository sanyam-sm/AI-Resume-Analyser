"""
app.py — AI Resume Analyzer Flask Backend

Extraction architecture:
  BERT NER (yashpwr/resume-ner-bert-v2)  — Primary DL extractor
    • name, email, skills, locations, years_of_experience → BERT
    • education_entries, experience_entries               → Gemini 2.5 Flash Lite (context-aware)
    • phone, email validation                             → Regex always
  Regex fallback                          — when BERT unavailable

Classification:
  XGBoost + TF-IDF (v2.0) — 52 categories, 84% accuracy, 11,446 training samples
  Keyword fallback when ML confidence < 50%
"""

import os
import re
import json
import base64
import uuid
import time
import atexit
import signal
import sys
import requests
from pathlib import Path

# ─── Load .env ────────────────────────────────────────────────────────────────
def _load_env(env_path='.env'):
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key, value = key.strip(), value.strip()
                if key and value and value != f'your_{key.lower()}_here':
                    os.environ.setdefault(key, value)
        print("  .env loaded ✓")
    except FileNotFoundError:
        pass

_load_env()

from flask import Flask, request, jsonify, render_template, send_from_directory, session, Response, stream_with_context
import joblib
import numpy as np

from utils.parser import (
    extract_text_from_pdf, clean_text, extract_email,
    extract_phone, extract_name, count_pages,
    score_resume, detect_experience_level,
    ResumeNERExtractor,
    compute_job_matches, compute_skill_gaps,
    get_project_ideas,
    extract_skills_by_keyword, merge_skills,
    compute_jd_match, generate_quick_win,
)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER']      = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.secret_key = os.environ.get('SECRET_KEY', 'resume-analyzer-dev-key')

ALLOWED         = {'pdf'}
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', '')
JSEARCH_API_KEY = os.environ.get('JSEARCH_API_KEY', '')
MODEL_DIR       = Path(r'notebooks\models')
UPLOAD_TTL_SECONDS = 3 * 60 * 60
_LAST_UPLOAD_CLEANUP_CHECK = 0.0
_SHUTDOWN_IN_PROGRESS = False


def _cleanup_expired_uploads(max_age_seconds: int = UPLOAD_TTL_SECONDS) -> int:
    now = time.time()
    removed = 0
    for pdf in app.config['UPLOAD_FOLDER'].glob('*.pdf'):
        try:
            if now - pdf.stat().st_mtime > max_age_seconds:
                pdf.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _cleanup_all_uploads() -> int:
    removed = 0
    for pdf in app.config['UPLOAD_FOLDER'].glob('*.pdf'):
        try:
            pdf.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def _graceful_shutdown_cleanup(*_args):
    global _SHUTDOWN_IN_PROGRESS
    if _SHUTDOWN_IN_PROGRESS:
        return
    _SHUTDOWN_IN_PROGRESS = True

    removed = _cleanup_all_uploads()
    if removed:
        print(f"  Graceful shutdown cleanup removed {removed} upload file(s)")
    print("  Shutting down...")
    sys.exit(0)


atexit.register(_graceful_shutdown_cleanup)
try:
    signal.signal(signal.SIGTERM, _graceful_shutdown_cleanup)
    signal.signal(signal.SIGINT, _graceful_shutdown_cleanup)
except Exception:
    pass


def load_models():
    components = {
        'ml_model': None, 'tfidf': None, 'label_encoder': None,
        'meta': {}, 'ner_extractor': None, 'mode': 'none',
    }
    try:
        components['ml_model']      = joblib.load(MODEL_DIR / 'resume_model_v2.pkl')
        components['tfidf']         = joblib.load(MODEL_DIR / 'tfidf_vectorizer_v2.pkl')
        components['label_encoder'] = joblib.load(MODEL_DIR / 'label_encoder_v2.pkl')
        with open(MODEL_DIR / 'model_meta_v2.json') as f:
            components['meta'] = json.load(f)
        components['mode'] = 'ml'
        print(f"  ML model loaded: {components['meta'].get('best_model_name', 'Unknown')}")
    except FileNotFoundError as e:
        print(f"  ML model not found: {e}")
    except Exception as e:
        print(f"  ML model failed: {e}")

    # ── BERT NER: Primary DL extractor ────────────────────────────────────────
    # Handles: name, email, skills, locations, years_of_experience
    # Gemini (inside BERT extractor) handles: education + experience entries
    # Regex fallback: when BERT unavailable
    try:
        components['ner_extractor'] = ResumeNERExtractor()
    except Exception as e:
        print(f"  BERT NER unavailable: {e}")
        print("  → Will use regex extraction fallback")

    return components


print("Loading models...")
models = load_models()
ner_status = 'BERT NER ✓' if models['ner_extractor'] else 'Regex fallback (BERT unavailable)'
print(f"Ready!  ML: {models['mode']} | Extraction: {ner_status}")
print(f"  YouTube API : {'configured ✓' if YOUTUBE_API_KEY else 'not set'}")
print(f"  Gemini API  : {'configured ✓ (edu/exp extraction)' if os.environ.get('GEMINI_API_KEY') else 'not set (regex fallback for edu/exp)'}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


def _build_job_search_query(category: str, top_categories: list = None, experience_level: str = None) -> str:
    # JSearch works best with short queries (3-6 words).
    # Strategy: merge ML top prediction + job match roles, deduplicate, pick best 2.
    top_categories = [str(c).strip() for c in (top_categories or []) if str(c).strip()]
    clean_category = (category or '').strip()

    # ML top prediction first, then job match roles, deduplicated
    seen, merged = set(), []
    for role in ([clean_category] + top_categories):
        if role and role.lower() not in seen:
            seen.add(role.lower())
            merged.append(role)
        if len(merged) == 2:
            break
    roles = merged or ['Software Engineer']
    role_part = ' OR '.join(roles)

    level = (experience_level or '').strip().lower()
    if 'fresher' in level or 'intern' in level:
        return f"Fresher {role_part}"
    elif level == 'junior':
        return f"Junior {role_part}"
    elif level == 'mid-level':
        return f"Mid {role_part}"
    elif level == 'senior':
        return f"Senior {role_part}"
    return role_part


def _normalize_job_source(publisher: str) -> str:
    publisher_l = (publisher or '').lower()
    if 'linkedin' in publisher_l:
        return 'LinkedIn'
    if 'indeed' in publisher_l:
        return 'Indeed'
    return ''


def _format_job_item(job: dict, source: str) -> dict:
    city = (job.get('job_city') or '').strip()
    country = (job.get('job_country') or '').strip()
    location = ', '.join([v for v in [city, country] if v]) or (job.get('job_location') or 'Location not specified')
    title = job.get('job_title') or 'Untitled role'
    return {
        'job_title': title,
        'company_name': job.get('employer_name') or 'Unknown company',
        'location': location,
        'apply_url': job.get('job_apply_link') or job.get('job_google_link') or '',
        'source_platform': source,
    }


# ─── Course Recommendations ───────────────────────────────────────────────────

_SKILL_COURSES = {
    'python'            : {'name': 'Python for Everybody (Coursera)',          'url': 'https://www.coursera.org/specializations/python'},
    'django'            : {'name': 'Django Full Course (freeCodeCamp)',         'url': 'https://www.youtube.com/watch?v=F5mRW0jo-U4'},
    'flask'             : {'name': 'Flask Tutorial (Tech With Tim)',            'url': 'https://www.youtube.com/watch?v=Z1RJmh_OqeA'},
    'machine learning'  : {'name': 'ML Specialization (Coursera - Andrew Ng)', 'url': 'https://www.coursera.org/specializations/machine-learning-introduction'},
    'deep learning'     : {'name': 'Deep Learning Specialization (Coursera)',  'url': 'https://www.coursera.org/specializations/deep-learning'},
    'pytorch'           : {'name': 'PyTorch Full Course (freeCodeCamp)',        'url': 'https://www.youtube.com/watch?v=Z_ikDlimN6A'},
    'tensorflow'        : {'name': 'TensorFlow 2.0 Full Course (YouTube)',     'url': 'https://www.youtube.com/watch?v=tPYj3fFJGjk'},
    'nlp'               : {'name': 'NLP with Python (Udemy)',                  'url': 'https://www.udemy.com/course/nlp-natural-language-processing-with-python/'},
    'data analysis'     : {'name': 'Data Analysis with Python (freeCodeCamp)','url': 'https://www.youtube.com/watch?v=r-uOLxNrNk8'},
    'statistics'        : {'name': 'Statistics for Data Science (YouTube)',    'url': 'https://www.youtube.com/watch?v=xxpc-HPKN28'},
    'big data'          : {'name': 'Big Data Specialization (Coursera)',       'url': 'https://www.coursera.org/specializations/big-data'},
    'hadoop'            : {'name': 'Hadoop Tutorial for Beginners (YouTube)',  'url': 'https://www.youtube.com/watch?v=1vbXmCrkT3Y'},
    'spark'             : {'name': 'Apache Spark Full Course (YouTube)',       'url': 'https://www.youtube.com/watch?v=S2MUhGA3lEw'},
    'sql'               : {'name': 'SQL Full Course (freeCodeCamp)',           'url': 'https://www.youtube.com/watch?v=HXV3zeQKqGY'},
    'mongodb'           : {'name': 'MongoDB Full Course (YouTube)',            'url': 'https://www.youtube.com/watch?v=ofme2o29ngU'},
    'postgresql'        : {'name': 'PostgreSQL Tutorial (YouTube)',            'url': 'https://www.youtube.com/watch?v=qw--VYLpxG4'},
    'aws'               : {'name': 'AWS Cloud Practitioner (freeCodeCamp)',    'url': 'https://www.youtube.com/watch?v=SOTamWNgDKc'},
    'docker'            : {'name': 'Docker Full Course (TechWorld YouTube)',   'url': 'https://www.youtube.com/watch?v=3c-iBn73dDE'},
    'kubernetes'        : {'name': 'Kubernetes Full Course (YouTube)',         'url': 'https://www.youtube.com/watch?v=X48VuDVv0do'},
    'ci/cd'             : {'name': 'CI/CD Pipeline Tutorial (YouTube)',        'url': 'https://www.youtube.com/watch?v=R8_veQiYBjI'},
    'mlflow'            : {'name': 'MLflow Tutorial (YouTube)',                'url': 'https://www.youtube.com/watch?v=AxYmj8ufKKY'},
    'react'             : {'name': 'React Full Course (freeCodeCamp)',         'url': 'https://www.youtube.com/watch?v=bMknfKXIFA8'},
    'javascript'        : {'name': 'JavaScript Full Course (freeCodeCamp)',    'url': 'https://www.youtube.com/watch?v=jS4aFq5-91M'},
    'node.js'           : {'name': 'Node.js Full Course (YouTube)',            'url': 'https://www.youtube.com/watch?v=Oe421EPjeBE'},
    'rest api'          : {'name': 'REST API Full Course (YouTube)',           'url': 'https://www.youtube.com/watch?v=-MTSQjw5DrM'},
    'tableau'           : {'name': 'Tableau Full Course (YouTube)',            'url': 'https://www.youtube.com/watch?v=TPMlZxRRaBQ'},
    'power bi'          : {'name': 'Power BI Full Course (YouTube)',           'url': 'https://www.youtube.com/watch?v=fnA454Gbkak'},
    'data visualization': {'name': 'Data Visualization with Python (YouTube)','url': 'https://www.youtube.com/watch?v=a9UrKTVEeZA'},
    'excel'             : {'name': 'Excel Full Course (freeCodeCamp)',         'url': 'https://www.youtube.com/watch?v=Vl0H-qTclOg'},
    'r'                 : {'name': 'R Programming Full Course (YouTube)',      'url': 'https://www.youtube.com/watch?v=_V8eKsto3Ug'},
    'java'              : {'name': 'Java Full Course (freeCodeCamp)',          'url': 'https://www.youtube.com/watch?v=grEKMHGYyns'},
    'spring boot'       : {'name': 'Spring Boot Full Course (YouTube)',        'url': 'https://www.youtube.com/watch?v=9SGDpanrc8U'},
    'kafka'             : {'name': 'Apache Kafka Full Course (YouTube)',       'url': 'https://www.youtube.com/watch?v=B5j3uNBH8X4'},
    'redis'             : {'name': 'Redis Full Course (YouTube)',              'url': 'https://www.youtube.com/watch?v=jgpVdJB2sKQ'},
    'terraform'         : {'name': 'Terraform Full Course (YouTube)',          'url': 'https://www.youtube.com/watch?v=SLB_c_ayRMo'},
    'git'               : {'name': 'Git & GitHub Full Course (freeCodeCamp)', 'url': 'https://www.youtube.com/watch?v=RGOj5yH7evk'},
    'transformers'      : {'name': 'Hugging Face Transformers Course (Free)',  'url': 'https://huggingface.co/learn/nlp-course/chapter1/1'},
    'hugging face'      : {'name': 'Hugging Face NLP Course (Free)',           'url': 'https://huggingface.co/learn/nlp-course/chapter1/1'},
}

_DOMAIN_COURSES = {
    'data science'    : {'name': 'IBM Data Science Certificate (Coursera)', 'url': 'https://www.coursera.org/professional-certificates/ibm-data-science'},
    'web designing'   : {'name': 'Web Design Full Course (freeCodeCamp)',   'url': 'https://www.youtube.com/watch?v=mU6anWqZJcc'},
    'java developer'  : {'name': 'Java Full Course (freeCodeCamp)',         'url': 'https://www.youtube.com/watch?v=grEKMHGYyns'},
    'python developer': {'name': 'Python Full Course (freeCodeCamp)',       'url': 'https://www.youtube.com/watch?v=rfscVS0vtbw'},
    'devops engineer' : {'name': 'DevOps Full Course (YouTube)',            'url': 'https://www.youtube.com/watch?v=j5Zsa_eOXeY'},
    'database'        : {'name': 'Database Design Course (freeCodeCamp)',   'url': 'https://www.youtube.com/watch?v=ztHopE5Wnpc'},
    'business analyst': {'name': 'Business Analysis Fundamentals (Udemy)', 'url': 'https://www.udemy.com/course/business-analysis-fundamentals/'},
    'testing'         : {'name': 'Software Testing Full Course (YouTube)', 'url': 'https://www.youtube.com/watch?v=sO8eGL6SFsA'},
    'engineering'     : {'name': 'System Design Full Course (YouTube)',    'url': 'https://www.youtube.com/watch?v=xpDnVSmNFX0'},
}


def get_courses_for_gaps(missing_skills, predicted_category, max_courses=6):
    if YOUTUBE_API_KEY:
        return _fetch_from_youtube(missing_skills, predicted_category, max_courses)
    return _curated_fallback(missing_skills, predicted_category, max_courses)


def _fetch_from_youtube(missing_skills, predicted_category, max_courses):
    courses, seen_ids = [], set()
    queries = [(s, f"{s} tutorial full course 2024") for s in missing_skills[:4]]
    if predicted_category:
        queries.append((predicted_category, f"{predicted_category} full course beginner to advanced"))
    for skill_name, query in queries:
        if len(courses) >= max_courses:
            break
        try:
            resp = requests.get(
                'https://www.googleapis.com/youtube/v3/search',
                params={'part': 'snippet', 'q': query, 'type': 'video',
                        'maxResults': 2, 'order': 'relevance', 'key': YOUTUBE_API_KEY},
                timeout=5
            )
            if resp.status_code != 200:
                break
            for item in resp.json().get('items', []):
                vid_id = item['id'].get('videoId')
                if not vid_id or vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)
                snippet = item['snippet']
                courses.append({
                    'name'     : snippet.get('title', query),
                    'url'      : f"https://www.youtube.com/watch?v={vid_id}",
                    'thumbnail': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
                    'channel'  : snippet.get('channelTitle', ''),
                    'reason'   : f"Fills gap: {skill_name}", 'source': 'YouTube',
                })
        except requests.exceptions.RequestException:
            break
    if len(courses) < max_courses:
        fallback  = _curated_fallback(missing_skills, predicted_category, max_courses - len(courses))
        existing  = {c['name'] for c in courses}
        for c in fallback:
            if c['name'] not in existing:
                courses.append(c)
    return courses[:max_courses]


def _curated_fallback(missing_skills, predicted_category, max_courses):
    courses, seen = [], set()
    for skill in missing_skills:
        key = skill.lower().strip()
        if key in _SKILL_COURSES and key not in seen:
            seen.add(key)
            c = _SKILL_COURSES[key].copy()
            c.update({'reason': f"Fills gap: {skill}", 'source': 'Curated', 'thumbnail': '', 'channel': ''})
            courses.append(c)
        if len(courses) >= max_courses:
            break
    if len(courses) < max_courses and predicted_category:
        domain_key = predicted_category.lower().strip()
        for key, course in _DOMAIN_COURSES.items():
            if key in domain_key or domain_key in key:
                if course['name'] not in seen:
                    c = course.copy()
                    c.update({'reason': f"Recommended for {predicted_category}", 'source': 'Curated', 'thumbnail': '', 'channel': ''})
                    courses.append(c)
                break
    return courses[:max_courses]


# ─── ML Prediction + Keyword Fallback ────────────────────────────────────────

_CATEGORY_KEYWORDS = {
    'Data Science'             : ['machine learning', 'deep learning', 'nlp', 'tensorflow',
                                  'pytorch', 'data science', 'neural network', 'pandas',
                                  'scikit', 'xgboost', 'lightgbm', 'pyspark', 'mlflow',
                                  'airflow', 'data pipeline', 'model training', 'jupyter',
                                  'numpy', 'matplotlib', 'data preprocessing',
                                  'feature engineering', 'ml model', 'generative ai',
                                  'classification', 'regression', 'power bi'],
    'Full Stack Developer'     : ['full stack', 'fullstack', 'react', 'node.js', 'express',
                                  'mongodb', 'rest api', 'angular', 'vue', 'next.js'],
    'Python Developer'         : ['python developer', 'django', 'flask', 'fastapi', 'celery'],
    'Java Developer'           : ['java developer', 'spring boot', 'spring mvc', 'hibernate', 'maven'],
    'React Developer'          : ['react developer', 'redux', 'typescript', 'webpack', 'frontend developer'],
    'DevOps Engineer'          : ['devops', 'kubernetes', 'ci/cd', 'terraform', 'ansible', 'jenkins', 'helm'],
    'Web Designing'            : ['web designer', 'ui designer', 'ux designer', 'figma', 'adobe xd', 'web design'],
    'Database'                 : ['dba', 'database administrator', 'oracle dba', 'sql server dba'],
    'Network Security Engineer': ['network security', 'cybersecurity', 'penetration testing', 'siem', 'ethical hacking'],
    'Testing'                  : ['test engineer', 'qa engineer', 'selenium', 'automation testing', 'quality assurance'],
    'Business Analyst'         : ['business analyst', 'requirements gathering', 'stakeholder management', 'user stories'],
    'DotNet Developer'         : ['asp.net', 'dotnet', '.net developer', 'c# developer', 'entity framework'],
    'SAP Developer'            : ['sap abap', 'sap hana', 'sap fiori', 'sap developer'],
    'Blockchain'               : ['blockchain developer', 'solidity', 'ethereum', 'smart contract', 'web3'],
    'ETL Developer'            : ['etl developer', 'data warehouse', 'informatica', 'talend', 'ssis'],
    'Hadoop'                   : ['hadoop developer', 'hive', 'hbase', 'mapreduce', 'hdfs'],
    'Information Technology'   : ['it support', 'system administrator', 'help desk', 'active directory', 'itil'],
    'HR'                       : ['human resources', 'recruitment', 'talent acquisition', 'hr manager', 'payroll'],
    'Sales'                    : ['sales manager', 'business development', 'sales executive', 'crm', 'lead generation'],
    'Finance'                  : ['financial analyst', 'investment banker', 'chartered accountant', 'cfa', 'balance sheet'],
    'Aviation'                 : ['pilot', 'aircraft', 'aviation', 'atpl', 'cpl', 'iata', 'icao', 'cockpit'],
    'Management'               : ['general manager', 'senior manager', 'vice president', 'coo', 'ceo', 'strategic planning'],
    'Architecture'             : ['revit', 'autocad architect', 'interior design', 'urban planning', 'bim'],
    'Civil Engineer'           : ['civil engineer', 'structural engineer', 'rcc', 'staad pro'],
    'Mechanical Engineer'      : ['mechanical engineer', 'solidworks', 'catia', 'ansys'],
    'PMO'                      : ['project manager', 'pmo', 'prince2', 'pmp', 'scrum master'],
    'BPO'                      : ['bpo', 'call center', 'customer service', 'voice process'],
}

CONFIDENCE_THRESHOLD = 50.0


def _keyword_predict(raw_text: str) -> str:
    """Score resume text against category keywords — low-confidence fallback."""
    text_lower = raw_text.lower()
    scores = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    return max(scores, key=scores.get) if scores else None


def _predict_domain(tfidf_vec, raw_text: str = ''):
    """
    ML classification with keyword fallback when confidence < threshold.
    Handles both XGBoost (predict_proba) and LinearSVC (decision_function).
    """
    ml_model = models['ml_model']
    le       = models['label_encoder']

    pred_id            = ml_model.predict(tfidf_vec)[0]
    predicted_category = le.classes_[pred_id]

    try:
        proba = ml_model.predict_proba(tfidf_vec)[0]
    except AttributeError:
        raw   = ml_model.decision_function(tfidf_vec)[0]
        e_x   = np.exp(raw - np.max(raw))
        proba = e_x / e_x.sum()

    top_ids = np.argsort(proba)[::-1][:5]
    top_predictions = [
        {'label': le.classes_[i], 'confidence': round(float(proba[i]) * 100, 1)}
        for i in top_ids
    ]

    top_confidence = top_predictions[0]['confidence']

    # ── Keyword override when model is uncertain ──────────────────────────────
    if top_confidence < CONFIDENCE_THRESHOLD and raw_text:
        keyword_cat = _keyword_predict(raw_text)
        if keyword_cat and keyword_cat != predicted_category:
            print(f"  ⚠ Low ML confidence ({top_confidence:.1f}%) → keyword override: {keyword_cat}")
            predicted_category = keyword_cat
            exists = next((p for p in top_predictions if p['label'] == keyword_cat), None)
            if exists:
                top_predictions = [exists] + [p for p in top_predictions if p['label'] != keyword_cat]
            else:
                top_predictions = [{'label': keyword_cat, 'confidence': top_confidence}] + top_predictions[:4]

    return predicted_category, top_predictions


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/model-info')
def api_model_info():
    meta = models['meta']
    if not meta:
        return jsonify({'status': 'error', 'message': 'Models not loaded. Run the training notebook first.'}), 503
    return jsonify({
        'status'       : 'ok',
        'model_name'   : meta.get('best_model_name', 'Unknown'),
        'model_mode'   : models['mode'],
        'accuracy'     : round(meta.get('accuracy',    0) * 100, 2),
        'f1_weighted'  : round(meta.get('f1_weighted', 0) * 100, 2),
        'f1_macro'     : round(meta.get('f1_macro',    0) * 100, 2),
        'ner_model'    : 'yashpwr/resume-ner-bert-v2' if models['ner_extractor'] else 'Regex fallback',
        'ner_available': models['ner_extractor'] is not None,
        'num_classes'  : meta.get('num_classes',    0),
        'classes'      : meta.get('classes',       []),
        'train_samples': meta.get('train_samples', 0),
        'test_samples' : meta.get('test_samples',  0),
        'cv_mean'      : round(meta.get('cv_mean', 0) * 100, 2),
        'cv_std'       : round(meta.get('cv_std',  0) * 100, 2),
        'all_results'  : meta.get('all_results',  {}),
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    global _LAST_UPLOAD_CLEANUP_CHECK
    now = time.time()
    if now - _LAST_UPLOAD_CLEANUP_CHECK > 300:
        _cleanup_expired_uploads()
        _LAST_UPLOAD_CLEANUP_CHECK = now

    if 'resume' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided.'}), 400
    file = request.files['resume']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Only PDF files are accepted.'}), 400
    if models['mode'] == 'none':
        return jsonify({'status': 'error', 'message': 'ML model not loaded. Run the training notebook first.'}), 503

    tmp_path = app.config['UPLOAD_FOLDER'] / f"{uuid.uuid4().hex}.pdf"

    try:
        file.save(tmp_path)
        raw_text = extract_text_from_pdf(str(tmp_path))
        if not raw_text.strip():
            return jsonify({'status': 'error', 'message': 'Could not extract text. Ensure the PDF is not a scanned image.'}), 422

        # Store resume text in session for JD matching
        session['resume_text'] = raw_text[:8000]

        pages      = count_pages(str(tmp_path))
        word_count = len(raw_text.split())

        # ── Entity Extraction ─────────────────────────────────────────────────
        # BERT NER handles: name, email, skills, locations, years_of_experience
        # Gemini (called inside BERT extractor): education + experience entries only
        # Regex: phone always + fallback when BERT unavailable
        if models['ner_extractor']:
            try:
                ner_results = models['ner_extractor'].extract(raw_text)
            except Exception as e:
                print(f"  BERT NER error: {e} — using regex fallback")
                ner_results = {
                    'name': extract_name(raw_text), 'email': extract_email(raw_text),
                    'phone': extract_phone(raw_text), 'skills': [],
                    'years_of_experience': '', 'locations': [],
                    'experience_entries': [], 'education_entries': [], 'all_entities': {},
                }
        else:
            print("  BERT NER unavailable — using regex fallback")
            ner_results = {
                'name': extract_name(raw_text), 'email': extract_email(raw_text),
                'phone': extract_phone(raw_text), 'skills': [],
                'years_of_experience': '', 'locations': [],
                'experience_entries': [], 'education_entries': [], 'all_entities': {},
            }

        name  = ner_results.get('name',  '') or extract_name(raw_text)
        email = ner_results.get('email', '') or extract_email(raw_text)
        phone = ner_results.get('phone', '') or extract_phone(raw_text)

        # ── Skills: NER/Gemini + keyword merge ────────────────────────────────
        ner_skill_items     = ner_results.get('skills', [])
        keyword_skill_items = extract_skills_by_keyword(raw_text)
        skill_items         = merge_skills(ner_skill_items, keyword_skill_items)

        # ── Domain Classification ─────────────────────────────────────────────
        cleaned                             = clean_text(raw_text)
        tfidf_vec                           = models['tfidf'].transform([cleaned])
        predicted_category, top_predictions = _predict_domain(tfidf_vec, raw_text)

        # ── Experience Level ──────────────────────────────────────────────────
        level   = detect_experience_level(raw_text, pages)
        ner_exp = ner_results.get('years_of_experience', '')
        if ner_exp:
            exp_nums = re.findall(r'(\d+)', ner_exp)
            if exp_nums:
                yrs   = max(int(y) for y in exp_nums)
                level = 'Senior' if yrs >= 5 else 'Mid-Level' if yrs >= 2 else 'Junior'

        # ── Job Matching + Skill Gaps ─────────────────────────────────────────
        job_matches = compute_job_matches(skill_items, top_n=5,
                                          predicted_category=predicted_category)
        skill_gaps  = compute_skill_gaps(skill_items,
                                         predicted_category=predicted_category)

        seen, unique_missing = set(), []
        for gap in skill_gaps:
            for s in gap.get('missing_core_skills', []) + gap.get('missing_skills', [])[:3]:
                if s not in seen:
                    unique_missing.append(s)
                    seen.add(s)

        # ── Projects + Courses ────────────────────────────────────────────────
        project_ideas = get_project_ideas(skill_items, max_projects=4,
                                          experience_level=level)
        courses       = get_courses_for_gaps(unique_missing, predicted_category, max_courses=6)

        # ── Resume Score ──────────────────────────────────────────────────────
        scoring = score_resume(raw_text)

        # ── PDF preview ───────────────────────────────────────────────────────
        with open(tmp_path, 'rb') as f:
            pdf_b64 = base64.b64encode(f.read()).decode('utf-8')

        resume_payload = {
            'name': name,
            'email': email,
            'phone': phone,
            'category': predicted_category,
            'experience_level': level,
            'skills': [s['text'] if isinstance(s, dict) else s for s in skill_items],
            'years_of_experience': ner_results.get('years_of_experience', ''),
            'experience_entries': ner_results.get('experience_entries', []),
            'education_entries': ner_results.get('education_entries', []),
            'uploaded_pdf_path': str(tmp_path),
            'uploaded_pdf_name': file.filename,
        }
        session['resume_apply_data'] = resume_payload
        session['uploaded_resume_path'] = str(tmp_path)
        session['job_matches'] = job_matches

        return jsonify({
            'status'   : 'success',
            'extracted': {
                'name': name, 'email': email, 'phone': phone,
                'pages': pages, 'word_count': word_count,
            },
            'ner_entities': {
                'experience_entries' : ner_results.get('experience_entries', []),
                'education_entries'  : ner_results.get('education_entries',  []),
                'years_of_experience': ner_results.get('years_of_experience', ''),
                'locations'          : [
                    e if isinstance(e, dict) else {'text': e}
                    for e in ner_results.get('locations', [])
                ],
            },
            'prediction': {
                'category'        : predicted_category,
                'experience_level': level,
                'top_predictions' : top_predictions,
                'model_used'      : models['mode'],
            },
            'skills': {
                'current'                : [s['text'] if isinstance(s, dict) else s for s in skill_items],
                'current_with_confidence': skill_items,
            },
            'job_matches'  : job_matches,
            'skill_gaps'   : skill_gaps,
            'project_ideas': project_ideas,
            'score'        : scoring,
            'courses'      : courses,
            'pdf_preview'  : pdf_b64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

    finally:
        pass


@app.route('/api/cleanup-resume', methods=['POST'])
def api_cleanup_resume():
    _cleanup_expired_uploads()
    resume_path = session.get('uploaded_resume_path')
    deleted = False
    if resume_path:
        p = Path(resume_path)
        if p.exists():
            p.unlink()
            deleted = True
    session.pop('uploaded_resume_path', None)
    session.pop('resume_apply_data', None)
    return jsonify({'status': 'success', 'deleted': deleted})


@app.route('/api/cleanup-uploads', methods=['POST'])
def api_cleanup_uploads():
    removed = _cleanup_all_uploads()
    session.pop('uploaded_resume_path', None)
    session.pop('resume_apply_data', None)
    return jsonify({'status': 'success', 'deleted_count': removed})


@app.route('/api/jobs')
def api_jobs():
    try:
        location = request.args.get('location', '').strip()
        country = request.args.get('country', 'us').strip().lower() or 'us'

        if not JSEARCH_API_KEY:
            return jsonify({
                'status': 'setup_required',
                'message': 'JSEARCH_API_KEY is missing. Add it to your .env file and restart the app.',
                'linkedin_jobs': [],
                'indeed_jobs': [],
            }), 200

        resume_data = session.get('resume_apply_data')
        if not resume_data:
            return jsonify({'status': 'error', 'message': 'No resume data in session. Analyze a resume first.'}), 400

        category = resume_data.get('category', 'Software Engineer')
        experience_level = resume_data.get('experience_level', '')
        job_matches_data = session.get('job_matches')
        top_categories = [m['role'] for m in (job_matches_data or [])[:3]]

        query = _build_job_search_query(
            category=category,
            top_categories=top_categories,
            experience_level=experience_level,
        )
        print(f"[JOB SEARCH] Query: {query}, Level: {experience_level}, Location: {location or 'Not specified'}, Country: {country}")

        endpoint = "https://jsearch.p.rapidapi.com/search"
        headers = {
            'x-rapidapi-key': JSEARCH_API_KEY,
            'x-rapidapi-host': 'jsearch.p.rapidapi.com',
        }

        collected_linkedin = []
        collected_indeed = []

        # Parse cities if there are multiple (separated by OR)
        cities = [c.strip() for c in location.split(' OR ')] if location else ['']
        print(f"[JOB SEARCH] Searching in cities: {cities}")

        for city in cities:
            if len(collected_linkedin) >= 30:  # Only LinkedIn, max 30 jobs
                break

            page = 1
            max_pages = 1  # Only 1 page per city

            while page <= max_pages and len(collected_linkedin) < 30:
                params = {
                    'query': query,
                    'page': page,
                    'num_pages': 1,
                    'date_posted': 'week',
                }

                # Add location (single city, not OR-separated)
                if city:
                    params['query'] = f"{query} in {city}"
                    print(f"[JOB SEARCH] Fetching page {page} for city: {city}")

                # Add country only if specified
                if country:
                    params['country'] = country

                # Retry logic for network issues
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        response = requests.get(endpoint, headers=headers, params=params, timeout=(5, 20))
                        break
                    except requests.exceptions.ReadTimeout:
                        if attempt < max_retries - 1:
                            print(f"[JOB SEARCH] Timeout on attempt {attempt + 1}, retrying...")
                            time.sleep(1)
                            continue
                        else:
                            print(f"[JOB SEARCH ERROR] All retry attempts exhausted (timeout)")
                            continue
                    except requests.exceptions.RequestException as e:
                        print(f"[JOB SEARCH ERROR] Network error: {str(e)}")
                        continue

                if response.status_code != 200:
                    print(f"[JOB SEARCH ERROR] JSearch API returned {response.status_code}: {response.text[:200]}")
                    page += 1
                    continue

                payload = response.json()
                jobs = payload.get('data', [])
                if not jobs:
                    print(f"[JOB SEARCH] No jobs found in {city} on page {page}")
                    break

                print(f"[JOB SEARCH] Found {len(jobs)} jobs in {city} on page {page}")

                for job in jobs:
                    if len(collected_linkedin) >= 50:
                        break
                    source = _normalize_job_source(job.get('job_publisher'))
                    formatted_job = _format_job_item(job, source or job.get('job_publisher', 'Other'))
                    collected_linkedin.append(formatted_job)

                page += 1

        # Sort: LinkedIn first, then by match percentage
        def sort_key(j):
            return 1 if j.get('source_platform', '').lower() == 'linkedin' else 0
        
        collected_linkedin.sort(key=sort_key, reverse=True)

        linkedin_count = sum(1 for j in collected_linkedin if j.get('source_platform', '').lower() == 'linkedin')
        print(f"[JOB SEARCH] Found {len(collected_linkedin)} total jobs ({linkedin_count} LinkedIn)")
        return jsonify({
            'status': 'success',
            'query': query,
            'jobs': collected_linkedin[:50],
            'linkedin_count': linkedin_count,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Failed to fetch jobs: {str(e)}'}), 502


@app.route('/api/demo')
def api_demo():
    demo_courses = get_courses_for_gaps(['PyTorch', 'Big Data', 'Kubernetes', 'AWS'], 'Data Science', max_courses=4)
    return jsonify({
        'status'   : 'success',
        'extracted': {'name': 'Alex Johnson', 'email': 'alex.johnson@email.com',
                      'phone': '+1-555-0192', 'pages': 2, 'word_count': 680},
        'ner_entities': {
            'experience_entries': [
                {'designation': 'Senior Data Scientist', 'company': 'Google',    'duration': 'Jan 2022 – Present',  'confidence': 0.97},
                {'designation': 'Data Analyst',           'company': 'Microsoft', 'duration': 'Jun 2019 – Dec 2021', 'confidence': 0.95},
            ],
            'education_entries': [
                {'degree': 'M.S. Computer Science', 'college': 'Stanford University', 'year': '2019', 'confidence': 0.98},
            ],
            'years_of_experience': '5 years',
            'locations': [{'text': 'San Francisco, CA', 'confidence': 0.96}],
        },
        'prediction': {
            'category': 'Data Science', 'experience_level': 'Senior',
            'top_predictions': [
                {'label': 'Data Science',     'confidence': 89.4},
                {'label': 'Python Developer', 'confidence':  5.8},
                {'label': 'Business Analyst', 'confidence':  2.3},
                {'label': 'Database',         'confidence':  1.5},
                {'label': 'Hadoop',           'confidence':  1.0},
            ],
            'model_used': models['mode'],
        },
        'skills': {
            'current': ['Python', 'Machine Learning', 'SQL', 'TensorFlow', 'Data Analysis',
                        'Deep Learning', 'NLP', 'Docker', 'Git', 'Statistics'],
            'current_with_confidence': [
                {'text': 'Python',           'confidence': 0.98},
                {'text': 'Machine Learning', 'confidence': 0.97},
                {'text': 'SQL',              'confidence': 0.96},
                {'text': 'TensorFlow',       'confidence': 0.95},
                {'text': 'Data Analysis',    'confidence': 0.94},
                {'text': 'Deep Learning',    'confidence': 0.93},
                {'text': 'NLP',              'confidence': 0.91},
                {'text': 'Docker',           'confidence': 0.89},
                {'text': 'Git',              'confidence': 0.88},
                {'text': 'Statistics',       'confidence': 0.87},
            ],
        },
        'job_matches': [
            {'role': 'Data Scientist', 'match_pct': 88.0,
             'matched_skills': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'Data Analysis', 'Deep Learning', 'TensorFlow', 'NLP'],
             'missing_core': [], 'missing_preferred': ['PyTorch', 'Big Data', 'Data Visualization', 'R'], 'total_required': 12},
            {'role': 'ML Engineer', 'match_pct': 82.0,
             'matched_skills': ['Python', 'Machine Learning', 'Docker', 'SQL', 'Git', 'Deep Learning', 'TensorFlow'],
             'missing_core': [], 'missing_preferred': ['Kubernetes', 'AWS', 'PyTorch', 'CI/CD', 'MLflow'], 'total_required': 12},
        ],
        'skill_gaps': [
            {'role': 'Data Scientist', 'missing_skills': ['PyTorch', 'Big Data', 'Data Visualization', 'R'],
             'missing_core_skills': [], 'message': "You have all core skills for Data Scientist! Consider adding 'PyTorch', 'Big Data' to stand out."},
        ],
        'project_ideas': [
            {'name': 'Heart Disease Prediction', 'description': 'Build a classification model using scikit-learn and deploy as a web app.',
             'matched_skills': ['Python', 'Machine Learning', 'SQL'], 'all_skills': ['Python', 'Machine Learning', 'SQL', 'Flask'], 'difficulty': 'Intermediate', 'relevance': 100.0},
            {'name': 'Sentiment Analysis Dashboard', 'description': 'Build an NLP pipeline analyzing social media sentiment on a live dashboard.',
             'matched_skills': ['Python', 'NLP'], 'all_skills': ['Python', 'NLP', 'Flask', 'Data Visualization'], 'difficulty': 'Intermediate', 'relevance': 67.0},
        ],
        'score': {
            'total': 71, 'max': 100,
            'breakdown': {
                'Contact Info'      : {'earned': 10, 'max': 10, 'present': True},
                'Summary/Objective' : {'earned':  0, 'max':  8, 'present': False},
                'Education'         : {'earned': 15, 'max': 15, 'present': True},
                'Experience'        : {'earned': 20, 'max': 20, 'present': True},
                'Skills'            : {'earned': 15, 'max': 15, 'present': True},
                'Projects'          : {'earned':  0, 'max': 12, 'present': False},
                'Certifications'    : {'earned': 10, 'max': 10, 'present': True},
                'Achievements'      : {'earned':  0, 'max':  5, 'present': False},
                'Hobbies/Interests' : {'earned':  0, 'max':  5, 'present': False},
            },
        },
        'courses': demo_courses,
    })


@app.route('/api/jd-match', methods=['POST'])
def api_jd_match():
    """Analyze resume against job description and return match analysis."""
    try:
        data = request.get_json()
        jd_text = data.get('jd_text', '') if data else ''
        resume_text = session.get('resume_text', '')

        if not jd_text:
            return jsonify({'status': 'error', 'message': 'Please paste a job description to analyze.'}), 400
        if not resume_text:
            return jsonify({'status': 'error', 'message': 'Please upload a resume first before analyzing JD match.'}), 400

        # Compute match analysis
        match_result = compute_jd_match(resume_text, jd_text)

        # Generate quick win tip
        quick_win = generate_quick_win(
            resume_text,
            match_result['genuine_gaps'],
            match_result['implied_matches'],
            jd_text,
            match_result['verdict']
        )

        return jsonify({
            'status': 'success',
            'verdict': match_result['verdict'],
            'hard_coverage': match_result['hard_coverage'],
            'breakdown': {
                'direct_matches': match_result['direct_matches'],
                'implied_matches': match_result['implied_matches'],
                'genuine_gaps': match_result['genuine_gaps'],
                'exp_aligned': match_result['exp_aligned'],
                'resume_level': match_result['resume_level'],
                'jd_level': match_result['jd_level'],
            },
            'quick_win': quick_win,
            'inflation_flags': match_result['inflation_flags'],
            'preferred_gaps': match_result['preferred_gaps'],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)