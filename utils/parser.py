"""
utils/parser.py
Resume text extraction, NER-based entity extraction, BERT domain classification,
job-role matching, skill-gap analysis, project ideas, and course recommendations.
"""
import re
import io
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import pdfplumber
    PDF_BACKEND = 'pdfplumber'
except ImportError:
    try:
        from pdfminer3.layout import LAParams
        from pdfminer3.pdfpage import PDFPage
        from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer3.converter import TextConverter
        PDF_BACKEND = 'pdfminer3'
    except ImportError:
        PDF_BACKEND = None

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


# =============================================================================
# PDF Extraction
# =============================================================================

def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF using best available backend."""
    text = ""

    if PDF_BACKEND == 'pdfplumber':
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

    elif PDF_BACKEND == 'pdfminer3':
        from pdfminer3.layout import LAParams
        from pdfminer3.pdfpage import PDFPage
        from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer3.converter import TextConverter
        resource_manager = PDFResourceManager()
        fake_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_handle, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, converter)
        with open(file_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                interpreter.process_page(page)
        text = fake_handle.getvalue()
        converter.close()
        fake_handle.close()

    return text


def count_pages(file_path: str) -> int:
    """Count pages in a PDF."""
    try:
        if PDF_BACKEND == 'pdfplumber':
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return len(pdf.pages)
        elif PDF_BACKEND == 'pdfminer3':
            from pdfminer3.pdfpage import PDFPage
            with open(file_path, 'rb') as fh:
                return sum(1 for _ in PDFPage.get_pages(fh, check_extractable=True))
    except Exception:
        pass
    return 1


# =============================================================================
# Text Cleaning (used for TF-IDF ML fallback)
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and preprocess resume text for ML inference."""
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# =============================================================================
# Regex Fallback Extractors
# =============================================================================

def extract_email(text: str) -> str:
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return emails[0] if emails else ""


def extract_phone(text: str) -> str:
    """Extract phone number — handles Indian (+91) and international formats."""
    patterns = [
        r'\+91[\s\-]?[6-9]\d{9}',           # +91 9876543210
        r'\+91[\s\-]?\d{5}[\s\-]?\d{5}',  # +91 87654 32109
        r'\b[6-9]\d{9}\b',                    # 10-digit Indian mobile
        r'\+?\d[\d\s\-\.\(\)]{9,14}\d', # generic international
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group().strip()
    return ""


def extract_name(text: str) -> str:
    """Heuristic: first non-empty line is often the name."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        first = lines[0]
        words = first.split()
        if 1 <= len(words) <= 5 and all(w[0].isupper() or w.istitle() for w in words):
            return first
    return ""


def extract_experience(text: str) -> list:
    """
    Extract work experience entries using line-by-line heuristics.
    Returns list of dicts: {company, designation, duration, confidence}
    """
    entries = []
    title_words = [
        'engineer', 'developer', 'analyst', 'manager', 'architect',
        'designer', 'scientist', 'lead', 'head', 'director', 'intern',
        'consultant', 'specialist', 'associate', 'officer', 'executive',
        'programmer', 'administrator', 'technician', 'coordinator',
    ]

    # Isolate the experience section if possible
    section_m = re.search(
        r'(?:WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|WORK\s+HISTORY)[^\n]*\n([\s\S]+?)'
        r'(?=\n(?:EDUCATION|PROJECTS|SKILLS|CERTIFICATIONS|TECHNICAL|$))',
        text, re.IGNORECASE
    )
    section = section_m.group(1) if section_m else text

    lines = [l.strip() for l in section.split('\n')]
    i = 0
    while i < len(lines):
        line = lines[i]
        # Candidate designation: short, no pipe, contains a job title word
        if (line and '|' not in line and len(line.split()) <= 8
                and any(w in line.lower() for w in title_words)):
            # Find next non-empty line (company/date info)
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines):
                info_line = lines[j]
                date_m = re.search(
                    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})'
                    r'[\w\s,–\-]*(?:Present|\d{4})?',
                    info_line, re.IGNORECASE
                )
                duration = date_m.group().strip() if date_m else ''
                if '|' in info_line:
                    company = info_line.split('|')[0].strip()
                elif ',' in info_line:
                    company = info_line.split(',')[0].strip()
                else:
                    company = info_line.strip()
                if company:
                    entries.append({
                        'designation': line,
                        'company'    : company,
                        'duration'   : duration,
                        'confidence' : 0.85,
                    })
                i = j + 1
                continue
        i += 1
    return entries


def extract_education(text: str) -> list:
    """
    Extract education entries: degree, college, year.
    Returns list of dicts: {degree, college, year}
    """
    entries = []

    section_match = re.search(
        r'(?:EDUCATION|ACADEMIC)[\s\S]{0,20}?\n([\s\S]+?)(?:\n(?:PROJECTS|SKILLS|CERTIFICATIONS|WORK|EXPERIENCE|TECHNICAL)|$)',
        text, re.IGNORECASE
    )
    section = section_match.group(1) if section_match else text

    degree_keywords = [
        r'B\.?Tech', r'B\.?E\.?', r'B\.?Sc', r'B\.?Com', r'B\.?A\.?',
        r'M\.?Tech', r'M\.?E\.?', r'M\.?Sc', r'M\.?S\.?', r'MBA',
        r'Ph\.?D', r'Bachelor', r'Master', r'Doctor', r'Diploma',
    ]
    degree_pat = r'(?:' + '|'.join(degree_keywords) + r')[^\n]{0,80}'

    for m in re.finditer(degree_pat, section, re.IGNORECASE):
        degree_line = m.group().strip()

        # Look for college name in next 1-2 lines after degree
        start = m.end()
        after = section[start:start+200]
        after_lines = [l.strip() for l in after.split('\n') if l.strip()]

        college = ''
        year    = ''

        for line in after_lines[:3]:
            # Year pattern
            yr_m = re.search(r'\b(19|20)\d{2}\b', line)
            if yr_m and not year:
                year = yr_m.group()
            # College: line with university/college/institute keywords
            if re.search(r'university|college|institute|school|iit|nit|bits', line, re.IGNORECASE):
                # Strip date info from college line
                college = re.sub(r'[\|,].*', '', line).strip()
            elif '|' in line and not college:
                college = line.split('|')[0].strip()

        # Also try year from the degree line itself
        if not year:
            yr_m = re.search(r'\b(19|20)\d{2}\b', degree_line)
            if yr_m:
                year = yr_m.group()

        if degree_line:
            entries.append({
                'degree'    : degree_line,
                'college'   : college,
                'year'      : year,
                'confidence': 0.85,
            })

    return entries


# =============================================================================
# Keyword-based Skill Extractor (supplements NER)
# =============================================================================

_KNOWN_SKILLS = [
    'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
    'Kotlin', 'Swift', 'Scala', 'R', 'Bash', 'SQL', 'PHP', 'Ruby',
    'Spring Boot', 'Spring MVC', 'Spring', 'Hibernate', 'JPA', 'Django',
    'Flask', 'FastAPI', 'Node.js', 'React', 'Angular', 'Vue', 'Next.js',
    'Express', 'GraphQL', 'REST API', 'gRPC', 'Microservices',
    'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Cassandra',
    'Oracle', 'SQLite', 'DynamoDB', 'Firestore',
    'Apache Kafka', 'Kafka', 'RabbitMQ', 'ActiveMQ', 'Celery',
    'Docker', 'Kubernetes', 'Terraform', 'Ansible', 'Jenkins',
    'AWS', 'GCP', 'Azure', 'EC2', 'S3', 'Lambda', 'RDS', 'SQS',
    'Git', 'GitHub', 'GitLab', 'Maven', 'Gradle', 'Linux',
    'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
    'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'Pandas', 'NumPy',
    'Spark', 'Hadoop', 'Airflow', 'MLflow', 'Hugging Face', 'Transformers',
    'CI/CD', 'DevOps', 'Agile', 'Scrum', 'JIRA',
    'JUnit', 'Mockito', 'Pytest', 'Selenium', 'Postman',
    'Figma', 'Adobe XD', 'Tableau', 'Power BI',
    'Android', 'iOS', 'Flutter', 'React Native',
    'Cybersecurity', 'Networking', 'Penetration Testing',
    'OAuth', 'JWT',
    'Data Analysis', 'Data Visualization', 'Statistics', 'Excel',
    'Big Data', 'ETL', 'Data Engineering',
    'Solidity', 'Ethereum', 'Smart Contracts',
    'Monitoring', 'Prometheus', 'Grafana',
]


def extract_skills_by_keyword(text: str) -> list:
    """Scan resume text for known skills using case-insensitive whole-word matching."""
    found = []
    seen  = set()
    text_lower = text.lower()
    for skill in _KNOWN_SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                found.append({'text': skill, 'confidence': 0.80})
    return found


def merge_skills(ner_skills: list, keyword_skills: list) -> list:
    """Merge NER + keyword skills, deduplicating by lowercase key. NER takes precedence."""
    merged = {}
    for s in ner_skills:
        key = s['text'].lower().strip()
        merged[key] = s
    for s in keyword_skills:
        key = s['text'].lower().strip()
        if key not in merged:
            merged[key] = s
    return sorted(merged.values(), key=lambda x: x['confidence'], reverse=True)


# =============================================================================
# Resume Scoring (section-based heuristic)
# =============================================================================

def score_resume(text: str) -> dict:
    """Score a resume across multiple dimensions (0-100)."""
    text_l = text.lower()
    sections = {
        'Contact Info':      10,
        'Summary/Objective':  8,
        'Education':         15,
        'Experience':        20,
        'Skills':            15,
        'Projects':          12,
        'Certifications':    10,
        'Achievements':       5,
        'Hobbies/Interests':  5,
    }
    checks = {
        'Contact Info':      any(k in text_l for k in ['email', 'phone', 'linkedin', 'github', 'contact']),
        'Summary/Objective': any(k in text_l for k in ['objective', 'summary', 'profile', 'about me']),
        'Education':         any(k in text_l for k in ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'b.tech', 'b.e']),
        'Experience':        any(k in text_l for k in ['experience', 'work experience', 'employment', 'internship']),
        'Skills':            any(k in text_l for k in ['skills', 'technical skills', 'core competencies']),
        'Projects':          any(k in text_l for k in ['project', 'projects', 'portfolio']),
        'Certifications':    any(k in text_l for k in ['certification', 'certifications', 'certificate', 'certified']),
        'Achievements':      any(k in text_l for k in ['achievement', 'achievements', 'award', 'honors', 'recognition']),
        'Hobbies/Interests': any(k in text_l for k in ['hobbies', 'interests', 'activities', 'volunteer']),
    }
    total = 0
    breakdown = {}
    for section, points in sections.items():
        got = points if checks[section] else 0
        breakdown[section] = {'earned': got, 'max': points, 'present': checks[section]}
        total += got

    return {'total': total, 'max': 100, 'breakdown': breakdown}


def detect_experience_level(text: str, pages: int) -> str:
    text_l = text.lower()
    exp_years = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience', text_l)
    if exp_years:
        yrs = max(int(y) for y in exp_years)
        if yrs >= 5:
            return 'Senior'
        elif yrs >= 2:
            return 'Mid-Level'
        else:
            return 'Junior'

    if any(k in text_l for k in ['senior', 'lead ', 'principal', 'architect', 'head of']):
        return 'Senior'
    if any(k in text_l for k in ['internship', 'intern ', 'fresher', 'entry level', 'trainee']):
        return 'Fresher/Intern'
    if any(k in text_l for k in ['experience', 'work experience', 'employment']):
        return 'Experienced'
    if pages <= 1:
        return 'Fresher/Intern'
    return 'Mid-Level'


# =============================================================================
# NER-Based Entity Extraction (yashpwr/resume-ner-bert-v2)
# =============================================================================

class ResumeNERExtractor:
    """Wraps yashpwr/resume-ner-bert-v2 for structured entity extraction."""

    MODEL_NAME = 'yashpwr/resume-ner-bert-v2'

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            'token-classification',
            model=self.MODEL_NAME,
            aggregation_strategy='simple',
            device=-1,  # CPU
        )
        self._loaded = True

    # Max word counts per entity type to filter out garbage long spans
    _MAX_WORDS = {
        'Name': 5, 'Email Address': 1, 'Phone': 1,
        'Skills': 6, 'Companies worked at': 8, 'Company': 8,
        'Designation': 8, 'Degree': 10, 'College Name': 10,
        'Graduation Year': 1, 'Years of Experience': 6, 'Location': 6,
    }

    def _clean_word(self, word: str) -> str:
        """Remove BERT subword ## prefix artifacts and clean whitespace."""
        word = re.sub(r'\s*##', '', word)
        word = re.sub(r'^\W+$', '', word)
        return word.strip()

    def extract(self, text: str) -> dict:
        """
        Process resume text through NER.
        Returns dict with: name, email, phone, skills, companies, designations,
        degrees, college_names, graduation_years, years_of_experience, locations.
        """
        chunks = self._chunk_text(text, max_chars=1800, overlap_chars=200)

        all_raw_entities = []
        for chunk in chunks:
            try:
                results = self.pipe(chunk)
                all_raw_entities.extend(results)
            except Exception:
                continue

        # Group by entity type — clean and filter
        grouped = {}
        for ent in all_raw_entities:
            group = ent['entity_group']
            word  = self._clean_word(ent['word'])
            score = float(ent['score'])

            if not word or len(word) < 2:
                continue

            # Filter out entities that are too long (sentences mistakenly tagged)
            max_words = self._MAX_WORDS.get(group, 10)
            if len(word.split()) > max_words:
                continue

            # Filter out low-confidence noise
            if score < 0.35:
                continue

            if group not in grouped:
                grouped[group] = []
            grouped[group].append({'text': word, 'confidence': round(score, 4)})

        # Deduplicate: keep highest confidence per normalized text
        for group in grouped:
            seen = {}
            for item in grouped[group]:
                key = item['text'].lower().strip()
                if key not in seen or item['confidence'] > seen[key]['confidence']:
                    seen[key] = item
            grouped[group] = sorted(seen.values(), key=lambda x: x['confidence'], reverse=True)

        # ── Always use regex for structured fields — NER is unreliable here ──
        # NER picks up garbage for companies, degrees, phone. Regex is more accurate.
        regex_exp = extract_experience(text)
        regex_edu = extract_education(text)

        # Phone: always use regex (NER returns fragments like "65")
        phone = extract_phone(text)

        # Experience entries: structured {designation, company, duration}
        # Always prefer regex results; only fall back to NER if regex finds nothing
        if regex_exp:
            experience_entries = regex_exp
        else:
            # Build from NER as last resort
            ner_co   = grouped.get('Companies worked at', grouped.get('Company', []))
            ner_desig = grouped.get('Designation', [])
            experience_entries = []
            for i in range(max(len(ner_co), len(ner_desig))):
                experience_entries.append({
                    'designation': ner_desig[i]['text'] if i < len(ner_desig) else '',
                    'company'    : ner_co[i]['text']    if i < len(ner_co)    else '',
                    'duration'   : '',
                    'confidence' : 0.5,
                })

        # Education entries: structured {degree, college, year}
        if regex_edu:
            education_entries = regex_edu
        else:
            ner_deg  = grouped.get('Degree', [])
            ner_col  = grouped.get('College Name', [])
            ner_yr   = grouped.get('Graduation Year', [])
            education_entries = []
            for i in range(max(len(ner_deg), len(ner_col))):
                education_entries.append({
                    'degree'    : ner_deg[i]['text'] if i < len(ner_deg) else '',
                    'college'   : ner_col[i]['text'] if i < len(ner_col) else '',
                    'year'      : ner_yr[i]['text']  if i < len(ner_yr)  else '',
                    'confidence': 0.5,
                })

        return {
            'name':               self._first(grouped.get('Name', [])),
            'email':              self._first(grouped.get('Email Address', [])) or extract_email(text),
            'phone':              phone,
            'skills':             grouped.get('Skills', []),
            'years_of_experience': self._first(grouped.get('Years of Experience', [])),
            'locations':          grouped.get('Location', []),
            'experience_entries': experience_entries,
            'education_entries':  education_entries,
            'all_entities':       grouped,
        }

    def _first(self, items):
        """Return the highest-confidence item's text, or empty string."""
        if not items:
            return ''
        return max(items, key=lambda x: x['confidence'])['text']

    def _chunk_text(self, text, max_chars=1800, overlap_chars=200):
        """Split text into overlapping chunks for the NER model."""
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap_chars
        return chunks


# =============================================================================
# BERT Domain Classifier (fine-tuned on Kaggle resume dataset)
# =============================================================================

class BERTDomainClassifier:
    """Loads the fine-tuned BERT domain classifier from local directory."""

    def __init__(self, model_dir: str, label_classes: list):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.classes = label_classes
        self._torch = torch

    def predict(self, text: str, top_k: int = 5) -> tuple:
        """
        Returns (predicted_category, top_k_predictions_list).
        Each item: {'label': str, 'confidence': float}
        """
        inputs = self.tokenizer(
            text, truncation=True, max_length=512,
            padding='max_length', return_tensors='pt'
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            probs = self._torch.softmax(outputs.logits, dim=1)[0]

        top_ids = self._torch.argsort(probs, descending=True)[:top_k]
        top_preds = [
            {'label': self.classes[idx.item()], 'confidence': round(probs[idx].item() * 100, 1)}
            for idx in top_ids
        ]
        predicted = self.classes[self._torch.argmax(probs).item()]
        return predicted, top_preds


# =============================================================================
# Job Role Requirements Database
# =============================================================================

JOB_ROLE_REQUIREMENTS = {
    'Backend Developer': {
        'core': ['Python', 'Java', 'SQL', 'REST API', 'Git'],
        'preferred': ['Docker', 'Kubernetes', 'AWS', 'Redis', 'PostgreSQL', 'CI/CD', 'Linux', 'Node.js'],
    },
    'Frontend Developer': {
        'core': ['JavaScript', 'React', 'HTML', 'CSS', 'Git'],
        'preferred': ['TypeScript', 'Vue', 'Angular', 'Webpack', 'Figma', 'REST API', 'Next.js'],
    },
    'Full Stack Developer': {
        'core': ['JavaScript', 'Python', 'React', 'SQL', 'Git', 'REST API'],
        'preferred': ['Docker', 'Kubernetes', 'AWS', 'TypeScript', 'Node.js', 'CI/CD', 'MongoDB'],
    },
    'Data Scientist': {
        'core': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'Data Analysis'],
        'preferred': ['Deep Learning', 'TensorFlow', 'PyTorch', 'NLP', 'Big Data', 'Data Visualization', 'R'],
    },
    'ML Engineer': {
        'core': ['Python', 'Machine Learning', 'Docker', 'SQL', 'Git'],
        'preferred': ['Deep Learning', 'Kubernetes', 'AWS', 'TensorFlow', 'PyTorch', 'CI/CD', 'MLflow'],
    },
    'Data Analyst': {
        'core': ['SQL', 'Excel', 'Data Analysis', 'Data Visualization', 'Statistics'],
        'preferred': ['Python', 'Tableau', 'Power BI', 'R', 'Machine Learning'],
    },
    'DevOps Engineer': {
        'core': ['Docker', 'Kubernetes', 'CI/CD', 'Linux', 'Git'],
        'preferred': ['AWS', 'Azure', 'GCP', 'Terraform', 'Ansible', 'Python', 'Monitoring'],
    },
    'Cloud Architect': {
        'core': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'],
        'preferred': ['Terraform', 'CI/CD', 'Linux', 'Networking', 'Security', 'Microservices'],
    },
    'Mobile Developer': {
        'core': ['Android', 'iOS', 'Git', 'REST API'],
        'preferred': ['Flutter', 'React Native', 'Kotlin', 'Swift', 'Firebase', 'Java'],
    },
    'Cybersecurity Analyst': {
        'core': ['Cybersecurity', 'Linux', 'Networking', 'Python'],
        'preferred': ['Penetration Testing', 'SIEM', 'Firewalls', 'Incident Response', 'Cloud Security'],
    },
    'Java Developer': {
        'core': ['Java', 'Spring', 'SQL', 'Git'],
        'preferred': ['Docker', 'Maven', 'Hibernate', 'REST API', 'Microservices', 'Kubernetes'],
    },
    'Python Developer': {
        'core': ['Python', 'SQL', 'Git', 'REST API'],
        'preferred': ['Django', 'Flask', 'FastAPI', 'Docker', 'AWS', 'PostgreSQL'],
    },
    'React Developer': {
        'core': ['React', 'JavaScript', 'TypeScript', 'Git'],
        'preferred': ['Node.js', 'REST API', 'Redux', 'Next.js', 'CSS', 'GraphQL'],
    },
    'Data Engineer': {
        'core': ['Python', 'SQL', 'Big Data', 'ETL', 'Git'],
        'preferred': ['Spark', 'Hadoop', 'AWS', 'Kafka', 'Docker', 'Airflow'],
    },
    'Business Analyst': {
        'core': ['SQL', 'Excel', 'Data Analysis', 'Communication'],
        'preferred': ['Tableau', 'Power BI', 'Statistics', 'Agile', 'JIRA', 'Requirements Gathering'],
    },
    'UI/UX Designer': {
        'core': ['Figma', 'Adobe XD', 'User Research', 'Wireframing'],
        'preferred': ['CSS', 'HTML', 'Prototyping', 'Design Systems', 'Accessibility'],
    },
    'Network Engineer': {
        'core': ['Networking', 'Linux', 'Cisco', 'Firewalls'],
        'preferred': ['AWS', 'Python', 'Cybersecurity', 'Monitoring', 'VPN', 'DNS'],
    },
    'Project Manager': {
        'core': ['Project Management', 'Agile', 'Communication', 'Leadership'],
        'preferred': ['JIRA', 'Scrum', 'Risk Management', 'Budgeting', 'Stakeholder Management'],
    },
    'Blockchain Developer': {
        'core': ['Solidity', 'Ethereum', 'Smart Contracts', 'JavaScript'],
        'preferred': ['Web3.js', 'Truffle', 'DeFi', 'NFT', 'Rust', 'Python'],
    },
    'AI/NLP Engineer': {
        'core': ['Python', 'NLP', 'Deep Learning', 'Machine Learning'],
        'preferred': ['TensorFlow', 'PyTorch', 'Transformers', 'Hugging Face', 'spaCy', 'Docker'],
    },
}


def compute_job_matches(extracted_skills: list, top_n: int = 5) -> list:
    """
    Compare extracted skill names against JOB_ROLE_REQUIREMENTS.
    Returns top_n roles sorted by match percentage.
    Scoring: core_match * 70% + preferred_match * 30%.
    """
    skill_names_lower = set()
    for s in extracted_skills:
        if isinstance(s, dict):
            skill_names_lower.add(s['text'].lower().strip())
        else:
            skill_names_lower.add(str(s).lower().strip())

    matches = []
    for role, reqs in JOB_ROLE_REQUIREMENTS.items():
        core = reqs['core']
        pref = reqs['preferred']

        matched_core = [s for s in core if s.lower() in skill_names_lower]
        matched_pref = [s for s in pref if s.lower() in skill_names_lower]

        core_pct = (len(matched_core) / len(core)) * 70 if core else 0
        pref_pct = (len(matched_pref) / len(pref)) * 30 if pref else 0
        total_pct = round(core_pct + pref_pct, 1)

        matches.append({
            'role': role,
            'match_pct': total_pct,
            'matched_skills': matched_core + matched_pref,
            'missing_core': [s for s in core if s.lower() not in skill_names_lower],
            'missing_preferred': [s for s in pref if s.lower() not in skill_names_lower],
            'total_required': len(core) + len(pref),
        })

    matches.sort(key=lambda x: x['match_pct'], reverse=True)
    return matches[:top_n]


def compute_skill_gaps(extracted_skills: list, target_roles: list = None) -> list:
    """
    For each target role (or top 3 matched roles), compute missing skills.
    Returns list of dicts: {role, missing_skills, missing_core_skills, message}
    """
    if target_roles is None:
        top_matches = compute_job_matches(extracted_skills, top_n=3)
        target_roles = [m['role'] for m in top_matches]

    skill_names_lower = set()
    for s in extracted_skills:
        if isinstance(s, dict):
            skill_names_lower.add(s['text'].lower().strip())
        else:
            skill_names_lower.add(str(s).lower().strip())

    gaps = []
    for role in target_roles:
        if role not in JOB_ROLE_REQUIREMENTS:
            continue
        reqs = JOB_ROLE_REQUIREMENTS[role]
        all_skills = reqs['core'] + reqs['preferred']
        missing = [s for s in all_skills if s.lower() not in skill_names_lower]
        missing_core = [s for s in reqs['core'] if s.lower() not in skill_names_lower]

        if missing_core:
            names = ', '.join(f"'{s}'" for s in missing_core[:3])
            msg = f"To become a {role}, you are missing {names}."
        elif missing:
            names = ', '.join(f"'{s}'" for s in missing[:2])
            msg = f"You have all core skills for {role}! Consider adding {names} to stand out."
        else:
            msg = f"You are fully qualified for {role} roles!"

        gaps.append({
            'role': role,
            'missing_skills': missing,
            'missing_core_skills': missing_core,
            'message': msg,
        })

    return gaps


# =============================================================================
# Project Ideas Database



def get_project_ideas(extracted_skills: list, max_projects: int = 4) -> list:
    """
    Generate personalized project ideas using Gemini 2.5 Flash API based on actual skills.
    Falls back to a small curated list if API call fails.

    Setup: Set GEMINI_API_KEY environment variable.
    Get key: https://aistudio.google.com/app/apikey
    """
    import os, json, urllib.request, urllib.error

    skill_names = []
    for s in extracted_skills:
        if isinstance(s, dict):
            skill_names.append(s['text'])
        else:
            skill_names.append(str(s))

    api_key = os.environ.get('GEMINI_API_KEY', '')

    if api_key and skill_names:
        try:
            prompt = (
                "You are a project idea generator for software developers.\n"
                f"Based on these skills: {', '.join(skill_names[:15])}\n\n"
                f"Generate exactly {max_projects} unique, practical project ideas that:\n"
                "- Use a good mix of the listed skills\n"
                "- Range from intermediate to advanced difficulty\n"
                "- Are portfolio-worthy and impressive to employers\n"
                "- Are specific (not generic)\n\n"
                "Respond ONLY with a JSON array, no markdown, no extra text:\n"
                '[\n'
                '  {\n'
                '    "name": "Project Name",\n'
                '    "description": "2-sentence description of what it does and why it\'s useful.",\n'
                '    "matched_skills": ["Skill1", "Skill2"],\n'
                '    "all_skills": ["Skill1", "Skill2", "Skill3"],\n'
                '    "difficulty": "Intermediate",\n'
                '    "relevance": 90.0\n'
                '  }\n'
                ']'
            )

            payload = json.dumps({
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": 1000,
                    "temperature": 0.7
                }
            }).encode()

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"content-type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data     = json.loads(resp.read())
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw_text = re.sub(r"```json|```", "", raw_text).strip()
                projects = json.loads(raw_text)
                if isinstance(projects, list) and projects:
                    return projects[:max_projects]

        except Exception as e:
            print(f"  Gemini API project ideas failed: {e} — using fallback")

    # ── Fallback: curated list filtered by skill match ────────────────────────
    FALLBACK_PROJECTS = [
        {"name": "AI Resume Analyzer",         "description": "Build an NLP system that parses resumes, extracts skills, and recommends jobs using ML classifiers and BERT NER.", "matched_skills": ["Python", "Machine Learning", "NLP"], "all_skills": ["Python", "Machine Learning", "NLP", "Flask"], "difficulty": "Advanced",     "relevance": 100.0},
        {"name": "Real-Time Chat App",          "description": "WebSocket-based chat application with React frontend and Node.js backend, featuring rooms and message history.",    "matched_skills": ["React", "Node.js", "JavaScript"],   "all_skills": ["React", "Node.js", "JavaScript", "SQL"],      "difficulty": "Intermediate", "relevance": 90.0},
        {"name": "E-Commerce REST API",         "description": "Full REST API with JWT auth, product catalog, cart, orders, and payment integration using Stripe.",                 "matched_skills": ["Python", "SQL", "REST API"],         "all_skills": ["Python", "SQL", "REST API", "Docker"],        "difficulty": "Intermediate", "relevance": 85.0},
        {"name": "ML Model Dashboard",          "description": "Deploy and monitor multiple ML models with a Flask dashboard showing predictions, confidence scores, and drift.",    "matched_skills": ["Python", "Machine Learning"],        "all_skills": ["Python", "Machine Learning", "Docker"],       "difficulty": "Advanced",     "relevance": 80.0},
        {"name": "DevOps Pipeline",             "description": "End-to-end CI/CD pipeline with GitHub Actions, Docker, Kubernetes deployment, and Prometheus monitoring.",          "matched_skills": ["Docker", "Kubernetes", "CI/CD"],     "all_skills": ["Docker", "Kubernetes", "CI/CD", "Linux"],     "difficulty": "Advanced",     "relevance": 80.0},
        {"name": "Fraud Detection System",      "description": "Real-time fraud detection using ensemble ML methods and Kafka streams, with an interactive analytics dashboard.",   "matched_skills": ["Python", "Machine Learning", "SQL"], "all_skills": ["Python", "Machine Learning", "SQL", "Kafka"], "difficulty": "Advanced",     "relevance": 85.0},
        {"name": "Portfolio Website Builder",   "description": "SaaS app that generates developer portfolio sites from GitHub profiles with drag-and-drop editor.",                 "matched_skills": ["React", "JavaScript"],               "all_skills": ["React", "JavaScript", "Node.js", "SQL"],      "difficulty": "Intermediate", "relevance": 75.0},
        {"name": "Inventory Management System", "description": "Warehouse inventory system with barcode scanning, CRUD operations, reporting, and low-stock alerts.",               "matched_skills": ["Java", "Spring", "SQL"],             "all_skills": ["Java", "Spring", "SQL", "REST API"],          "difficulty": "Intermediate", "relevance": 80.0},
    ]

    skill_lower = {(s['text'] if isinstance(s, dict) else str(s)).lower() for s in extracted_skills}
    scored = []
    for p in FALLBACK_PROJECTS:
        matched = [s for s in p["all_skills"] if s.lower() in skill_lower]
        if len(matched) >= len(p["all_skills"]) * 0.4:
            p = dict(p)
            p["matched_skills"] = matched
            p["relevance"]      = round(len(matched) / len(p["all_skills"]) * 100, 1)
            scored.append(p)

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:max_projects] if scored else FALLBACK_PROJECTS[:max_projects]