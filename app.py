"""
app.py — AI Resume Analyzer Flask Backend
ML domain classifier (TF-IDF) + NER (yashpwr/resume-ner-bert-v2)
Course recommendations via YouTube Data API v3 (free) with curated fallback.
"""

import os
import re
import json
import base64
import uuid
import requests
from pathlib import Path

# ─── Load .env file (so you never have to set keys manually in terminal) ───────
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
        pass  # No .env file — fall back to system environment variables

_load_env()

from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np

from utils.parser import (
    extract_text_from_pdf, clean_text, extract_email,
    extract_phone, extract_name, count_pages,
    score_resume, detect_experience_level,
    ResumeNERExtractor,
    compute_job_matches, compute_skill_gaps,
    get_project_ideas,
    extract_skills_by_keyword, merge_skills,   # ← NEW
)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

ALLOWED = {'pdf'}
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', '')
MODEL_DIR = Path(r'notebooks\models')


def load_models():
    components = {
        'ml_model': None, 'tfidf': None, 'label_encoder': None,
        'meta': {}, 'ner_extractor': None, 'mode': 'none',
    }
    try:
        components['ml_model']      = joblib.load(MODEL_DIR / 'resume_model.pkl')
        components['tfidf']         = joblib.load(MODEL_DIR / 'tfidf_vectorizer.pkl')
        components['label_encoder'] = joblib.load(MODEL_DIR / 'label_encoder.pkl')
        with open(MODEL_DIR / 'model_meta.json') as f:
            components['meta'] = json.load(f)
        components['mode'] = 'ml'
        print(f"  ML model loaded: {components['meta'].get('best_model_name', 'Unknown')}")
    except FileNotFoundError as e:
        print(f"  ML model not found: {e}\n  → Run the training notebook first.")
    except Exception as e:
        print(f"  ML model failed to load: {e}\n  → Likely a scikit-learn version mismatch.")

    try:
        components['ner_extractor'] = ResumeNERExtractor()
        print("  NER extractor loaded: yashpwr/resume-ner-bert-v2")
    except Exception as e:
        print(f"  NER extractor not available: {e}\n  → Falling back to regex extraction.")
    return components


print("Loading models...")
models = load_models()
print(f"Ready!  Mode: {models['mode']} | NER: {'yes' if models['ner_extractor'] else 'no (regex fallback)'}")
print(f"  YouTube API: {'configured ✓' if YOUTUBE_API_KEY else 'not set — using curated fallback'}")
print(f"  Gemini API:  {'configured ✓' if os.environ.get('GEMINI_API_KEY') else 'not set — using project idea fallback'}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


# ─── Course Recommendations ───────────────────────────────────────────────────

_SKILL_COURSES = {
    'python'            : {'name': 'Python for Everybody (Coursera)',           'url': 'https://www.coursera.org/specializations/python'},
    'django'            : {'name': 'Django Full Course (freeCodeCamp)',          'url': 'https://www.youtube.com/watch?v=F5mRW0jo-U4'},
    'flask'             : {'name': 'Flask Tutorial (Tech With Tim)',             'url': 'https://www.youtube.com/watch?v=Z1RJmh_OqeA'},
    'machine learning'  : {'name': 'ML Specialization (Coursera - Andrew Ng)',  'url': 'https://www.coursera.org/specializations/machine-learning-introduction'},
    'deep learning'     : {'name': 'Deep Learning Specialization (Coursera)',   'url': 'https://www.coursera.org/specializations/deep-learning'},
    'pytorch'           : {'name': 'PyTorch Full Course (freeCodeCamp)',         'url': 'https://www.youtube.com/watch?v=Z_ikDlimN6A'},
    'tensorflow'        : {'name': 'TensorFlow 2.0 Full Course (YouTube)',      'url': 'https://www.youtube.com/watch?v=tPYj3fFJGjk'},
    'nlp'               : {'name': 'NLP with Python (Udemy)',                   'url': 'https://www.udemy.com/course/nlp-natural-language-processing-with-python/'},
    'data analysis'     : {'name': 'Data Analysis with Python (freeCodeCamp)', 'url': 'https://www.youtube.com/watch?v=r-uOLxNrNk8'},
    'statistics'        : {'name': 'Statistics for Data Science (YouTube)',     'url': 'https://www.youtube.com/watch?v=xxpc-HPKN28'},
    'big data'          : {'name': 'Big Data Specialization (Coursera)',        'url': 'https://www.coursera.org/specializations/big-data'},
    'hadoop'            : {'name': 'Hadoop Tutorial for Beginners (YouTube)',   'url': 'https://www.youtube.com/watch?v=1vbXmCrkT3Y'},
    'spark'             : {'name': 'Apache Spark Full Course (YouTube)',        'url': 'https://www.youtube.com/watch?v=S2MUhGA3lEw'},
    'sql'               : {'name': 'SQL Full Course (freeCodeCamp)',            'url': 'https://www.youtube.com/watch?v=HXV3zeQKqGY'},
    'mongodb'           : {'name': 'MongoDB Full Course (YouTube)',             'url': 'https://www.youtube.com/watch?v=ofme2o29ngU'},
    'postgresql'        : {'name': 'PostgreSQL Tutorial (YouTube)',             'url': 'https://www.youtube.com/watch?v=qw--VYLpxG4'},
    'aws'               : {'name': 'AWS Cloud Practitioner (freeCodeCamp)',     'url': 'https://www.youtube.com/watch?v=SOTamWNgDKc'},
    'docker'            : {'name': 'Docker Full Course (TechWorld YouTube)',    'url': 'https://www.youtube.com/watch?v=3c-iBn73dDE'},
    'kubernetes'        : {'name': 'Kubernetes Full Course (YouTube)',          'url': 'https://www.youtube.com/watch?v=X48VuDVv0do'},
    'ci/cd'             : {'name': 'CI/CD Pipeline Tutorial (YouTube)',         'url': 'https://www.youtube.com/watch?v=R8_veQiYBjI'},
    'mlflow'            : {'name': 'MLflow Tutorial (YouTube)',                 'url': 'https://www.youtube.com/watch?v=AxYmj8ufKKY'},
    'react'             : {'name': 'React Full Course (freeCodeCamp)',          'url': 'https://www.youtube.com/watch?v=bMknfKXIFA8'},
    'javascript'        : {'name': 'JavaScript Full Course (freeCodeCamp)',     'url': 'https://www.youtube.com/watch?v=jS4aFq5-91M'},
    'node.js'           : {'name': 'Node.js Full Course (YouTube)',             'url': 'https://www.youtube.com/watch?v=Oe421EPjeBE'},
    'rest api'          : {'name': 'REST API Full Course (YouTube)',            'url': 'https://www.youtube.com/watch?v=-MTSQjw5DrM'},
    'tableau'           : {'name': 'Tableau Full Course (YouTube)',             'url': 'https://www.youtube.com/watch?v=TPMlZxRRaBQ'},
    'power bi'          : {'name': 'Power BI Full Course (YouTube)',            'url': 'https://www.youtube.com/watch?v=fnA454Gbkak'},
    'data visualization': {'name': 'Data Visualization with Python (YouTube)', 'url': 'https://www.youtube.com/watch?v=a9UrKTVEeZA'},
    'excel'             : {'name': 'Excel Full Course (freeCodeCamp)',          'url': 'https://www.youtube.com/watch?v=Vl0H-qTclOg'},
    'r'                 : {'name': 'R Programming Full Course (YouTube)',       'url': 'https://www.youtube.com/watch?v=_V8eKsto3Ug'},
    'java'              : {'name': 'Java Full Course (freeCodeCamp)',           'url': 'https://www.youtube.com/watch?v=grEKMHGYyns'},
    'spring boot'       : {'name': 'Spring Boot Full Course (YouTube)',         'url': 'https://www.youtube.com/watch?v=9SGDpanrc8U'},
    'kafka'             : {'name': 'Apache Kafka Full Course (YouTube)',        'url': 'https://www.youtube.com/watch?v=B5j3uNBH8X4'},
    'redis'             : {'name': 'Redis Full Course (YouTube)',               'url': 'https://www.youtube.com/watch?v=jgpVdJB2sKQ'},
    'terraform'         : {'name': 'Terraform Full Course (YouTube)',           'url': 'https://www.youtube.com/watch?v=SLB_c_ayRMo'},
    'git'               : {'name': 'Git & GitHub Full Course (freeCodeCamp)',  'url': 'https://www.youtube.com/watch?v=RGOj5yH7evk'},
    'transformers'      : {'name': 'Hugging Face Transformers Course (Free)',   'url': 'https://huggingface.co/learn/nlp-course/chapter1/1'},
    'hugging face'      : {'name': 'Hugging Face NLP Course (Free)',            'url': 'https://huggingface.co/learn/nlp-course/chapter1/1'},
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
                    'name': snippet.get('title', query),
                    'url': f"https://www.youtube.com/watch?v={vid_id}",
                    'thumbnail': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
                    'channel': snippet.get('channelTitle', ''),
                    'reason': f"Fills gap: {skill_name}", 'source': 'YouTube',
                })
        except requests.exceptions.RequestException:
            break
    if len(courses) < max_courses:
        fallback = _curated_fallback(missing_skills, predicted_category, max_courses - len(courses))
        existing = {c['name'] for c in courses}
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


# ─── ML Prediction Helper ─────────────────────────────────────────────────────

def _predict_domain(tfidf_vec):
    ml_model = models['ml_model']
    le = models['label_encoder']
    pred_id = ml_model.predict(tfidf_vec)[0]
    predicted_category = le.classes_[pred_id]
    try:
        proba = ml_model.predict_proba(tfidf_vec)[0]
    except AttributeError:
        raw = ml_model.decision_function(tfidf_vec)[0]
        e_x = np.exp(raw - np.max(raw))
        proba = e_x / e_x.sum()
    top_ids = np.argsort(proba)[::-1][:5]
    top_predictions = [
        {'label': le.classes_[i], 'confidence': round(float(proba[i]) * 100, 1)}
        for i in top_ids
    ]
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
        'status': 'ok',
        'model_name'   : meta.get('best_model_name', 'Unknown'),
        'model_mode'   : models['mode'],
        'accuracy'     : round(meta.get('accuracy', 0) * 100, 2),
        'f1_weighted'  : round(meta.get('f1_weighted', 0) * 100, 2),
        'f1_macro'     : round(meta.get('f1_macro', 0) * 100, 2),
        'ner_model'    : meta.get('ner_model', 'yashpwr/resume-ner-bert-v2'),
        'ner_available': models['ner_extractor'] is not None,
        'num_classes'  : meta.get('num_classes', 0),
        'classes'      : meta.get('classes', []),
        'train_samples': meta.get('train_samples', 0),
        'test_samples' : meta.get('test_samples', 0),
        'cv_mean'      : round(meta.get('cv_mean', 0) * 100, 2),
        'cv_std'       : round(meta.get('cv_std', 0) * 100, 2),
        'all_results'  : meta.get('all_results', {}),
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Main endpoint — accepts PDF resume, returns full structured analysis."""

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

        pages      = count_pages(str(tmp_path))
        word_count = len(raw_text.split())

        # ── NER Extraction ────────────────────────────────────────────────────
        ner_results = {}
        if models['ner_extractor']:
            try:
                ner_results = models['ner_extractor'].extract(raw_text[:2000])
            except Exception as e:
                print(f"NER extraction error: {e} — using regex fallback")

        name  = ner_results.get('name',  '') or extract_name(raw_text)
        email = ner_results.get('email', '') or extract_email(raw_text)
        phone = ner_results.get('phone', '') or extract_phone(raw_text)  # regex handles +91 formats

        # ── Skill Extraction: NER + keyword merge ─────────────────────────────
        # NER alone misses/garbles many skills. Keyword scan reliably catches
        # all explicitly listed technologies from the full resume text.
        ner_skill_items     = ner_results.get('skills', [])
        keyword_skill_items = extract_skills_by_keyword(raw_text)
        skill_items         = merge_skills(ner_skill_items, keyword_skill_items)

        # ── Domain Classification ─────────────────────────────────────────────
        cleaned                             = clean_text(raw_text)
        tfidf_vec                           = models['tfidf'].transform([cleaned])
        predicted_category, top_predictions = _predict_domain(tfidf_vec)

        # ── Experience Level ──────────────────────────────────────────────────
        level   = detect_experience_level(raw_text, pages)
        ner_exp = ner_results.get('years_of_experience', '')
        if ner_exp:
            exp_nums = re.findall(r'(\d+)', ner_exp)
            if exp_nums:
                yrs   = max(int(y) for y in exp_nums)
                level = 'Senior' if yrs >= 5 else 'Mid-Level' if yrs >= 2 else 'Junior'

        # ── Job Matching + Skill Gaps ─────────────────────────────────────────
        job_matches = compute_job_matches(skill_items, top_n=5)
        skill_gaps  = compute_skill_gaps(skill_items)

        seen, unique_missing = set(), []
        for gap in skill_gaps:
            for s in gap.get('missing_core_skills', []) + gap.get('missing_skills', [])[:3]:
                if s not in seen:
                    unique_missing.append(s)
                    seen.add(s)

        # ── Projects + Courses ────────────────────────────────────────────────
        project_ideas = get_project_ideas(skill_items, max_projects=4)
        courses       = get_courses_for_gaps(unique_missing, predicted_category, max_courses=6)

        # ── Resume Score ──────────────────────────────────────────────────────
        scoring = score_resume(raw_text)

        # ── PDF preview ───────────────────────────────────────────────────────
        with open(tmp_path, 'rb') as f:
            pdf_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'extracted': {
                'name': name, 'email': email, 'phone': phone,
                'pages': pages, 'word_count': word_count,
            },
            'ner_entities': {
                # Structured experience: [{designation, company, duration}]
                'experience_entries' : ner_results.get('experience_entries', []),
                # Structured education: [{degree, college, year}]
                'education_entries'  : ner_results.get('education_entries', []),
                'years_of_experience': ner_results.get('years_of_experience', ''),
                'locations'          : [e if isinstance(e, dict) else {'text': e} for e in ner_results.get('locations', [])],
            },
            'prediction': {
                'category': predicted_category, 'experience_level': level,
                'top_predictions': top_predictions, 'model_used': models['mode'],
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
        if tmp_path.exists():
            tmp_path.unlink()


@app.route('/api/demo')
def api_demo():
    demo_courses = get_courses_for_gaps(['PyTorch', 'Big Data', 'Kubernetes', 'AWS'], 'Data Science', max_courses=4)
    return jsonify({
        'status': 'success',
        'extracted': {'name': 'Alex Johnson', 'email': 'alex.johnson@email.com', 'phone': '+1-555-0192', 'pages': 2, 'word_count': 680},
        'ner_entities': {
            'companies': [{'text': 'Google', 'confidence': 0.97}, {'text': 'Microsoft', 'confidence': 0.95}],
            'designations': [{'text': 'Senior Data Scientist', 'confidence': 0.96}],
            'degrees': [{'text': 'M.S. Computer Science', 'confidence': 0.98}],
            'college_names': [{'text': 'Stanford University', 'confidence': 0.97}],
            'graduation_years': [{'text': '2019', 'confidence': 0.95}],
            'years_of_experience': '5 years',
            'locations': [{'text': 'San Francisco, CA', 'confidence': 0.96}],
        },
        'prediction': {
            'category': 'Data Science', 'experience_level': 'Senior',
            'top_predictions': [
                {'label': 'Data Science', 'confidence': 89.4},
                {'label': 'Python Developer', 'confidence': 5.8},
                {'label': 'Business Analyst', 'confidence': 2.3},
                {'label': 'Database', 'confidence': 1.5},
                {'label': 'Hadoop', 'confidence': 1.0},
            ],
            'model_used': models['mode'],
        },
        'skills': {
            'current': ['Python', 'Machine Learning', 'SQL', 'TensorFlow', 'Data Analysis', 'Deep Learning', 'NLP', 'Docker', 'Git', 'Statistics'],
            'current_with_confidence': [
                {'text': 'Python', 'confidence': 0.98}, {'text': 'Machine Learning', 'confidence': 0.97},
                {'text': 'SQL', 'confidence': 0.96}, {'text': 'TensorFlow', 'confidence': 0.95},
                {'text': 'Data Analysis', 'confidence': 0.94}, {'text': 'Deep Learning', 'confidence': 0.93},
                {'text': 'NLP', 'confidence': 0.91}, {'text': 'Docker', 'confidence': 0.89},
                {'text': 'Git', 'confidence': 0.88}, {'text': 'Statistics', 'confidence': 0.87},
            ],
        },
        'job_matches': [
            {'role': 'Data Scientist', 'match_pct': 88.0, 'matched_skills': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'Data Analysis', 'Deep Learning', 'TensorFlow', 'NLP'], 'missing_core': [], 'missing_preferred': ['PyTorch', 'Big Data', 'Data Visualization', 'R'], 'total_required': 12},
            {'role': 'ML Engineer', 'match_pct': 82.0, 'matched_skills': ['Python', 'Machine Learning', 'Docker', 'SQL', 'Git', 'Deep Learning', 'TensorFlow'], 'missing_core': [], 'missing_preferred': ['Kubernetes', 'AWS', 'PyTorch', 'CI/CD', 'MLflow'], 'total_required': 12},
        ],
        'skill_gaps': [
            {'role': 'Data Scientist', 'missing_skills': ['PyTorch', 'Big Data', 'Data Visualization', 'R'], 'missing_core_skills': [], 'message': "You have all core skills for Data Scientist! Consider adding 'PyTorch', 'Big Data' to stand out."},
        ],
        'project_ideas': [
            {'name': 'Heart Disease Prediction', 'description': 'Build a classification model to predict heart disease from patient data using scikit-learn and deploy as a web app.', 'matched_skills': ['Python', 'Machine Learning', 'SQL'], 'difficulty': 'Intermediate', 'relevance': 100.0},
            {'name': 'Sentiment Analysis Dashboard', 'description': 'Build an NLP pipeline that analyzes social media sentiment and displays results on a live dashboard.', 'matched_skills': ['Python', 'NLP'], 'difficulty': 'Intermediate', 'relevance': 67.0},
        ],
        'score': {
            'total': 71, 'max': 100,
            'breakdown': {
                'Contact Info': {'earned': 10, 'max': 10, 'present': True},
                'Summary/Objective': {'earned': 0, 'max': 8, 'present': False},
                'Education': {'earned': 15, 'max': 15, 'present': True},
                'Experience': {'earned': 20, 'max': 20, 'present': True},
                'Skills': {'earned': 15, 'max': 15, 'present': True},
                'Projects': {'earned': 0, 'max': 12, 'present': False},
                'Certifications': {'earned': 10, 'max': 10, 'present': True},
                'Achievements': {'earned': 0, 'max': 5, 'present': False},
                'Hobbies/Interests': {'earned': 0, 'max': 5, 'present': False},
            },
        },
        'courses': demo_courses,
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)