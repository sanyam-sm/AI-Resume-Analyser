"""
utils/parser.py
Resume text extraction, Gemini-powered entity extraction, ML domain classification,
job-role matching, skill-gap analysis, project ideas, and course recommendations.
"""
import os
import re
import io
import json
import urllib.request
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
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))

# ── Words that are NOT real technical skills ───────────────────────────────────
_SKILL_BLOCKLIST = {
    'skill', 'skills', 'technical', 'other', 'language', 'tools',
    'software', 'windows', 'environment', 'technologies', 'technology',
    'database', 'databases', 'framework', 'frameworks', 'platform',
    'team', 'individual', 'developer', 'size', 'languages', 'os',
    'interpersonal', 'competencies', 'interests', 'hobbies', 'details',
    'summary', 'objective', 'profile', 'education', 'experience',
    'training', 'project', 'projects', 'achievements', 'certifications',
}

# ── Words that look like names but are NOT names ───────────────────────────────
# Covers degree words, section headings, Indian city/institution names,
# and other common false positives from two-column PDF extraction.
_NAME_REJECTLIST = {
    # Degree / qualification words
    'bachelor', 'master', 'doctor', 'doctorate', 'diploma', 'intermediate',
    'matriculation', 'btech', 'mtech', 'mba', 'phd',
    # Resume section headings
    'education', 'skills', 'experience', 'summary', 'profile', 'objective',
    'certifications', 'achievements', 'projects', 'training', 'engineering',
    'technology', 'science', 'commerce', 'arts',
    # Generic non-name words
    'unknown', 'name', 'candidate', 'applicant', 'resume', 'curriculum',
    'vitae', 'contact', 'address', 'details', 'information', 'about',
    'professional', 'academic', 'personal', 'technical', 'interpersonal',
    # Indian cities / institutions that appear in college names
    'chandigarh', 'delhi', 'mumbai', 'bangalore', 'hyderabad', 'chennai',
    'pune', 'kolkata', 'group', 'college', 'university', 'institute',
    'school', 'landran', 'mohali', 'noida', 'gurugram', 'gurgaon',
    'lucknow', 'jaipur', 'ahmedabad', 'surat', 'bhopal', 'indore',
}


# =============================================================================
# PDF Extraction
# =============================================================================

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF.
    For two-column layouts, reads left column first then right column.
    """
    text = ""

    if PDF_BACKEND == 'pdfplumber':
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = _extract_page_smart(page)
                if page_text:
                    text += page_text + "\n"
                else:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"

    elif PDF_BACKEND == 'pdfminer3':
        from pdfminer3.layout import LAParams
        from pdfminer3.pdfpage import PDFPage
        from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer3.converter import TextConverter
        resource_manager = PDFResourceManager()
        fake_handle  = io.StringIO()
        converter    = TextConverter(resource_manager, fake_handle, laparams=LAParams())
        interpreter  = PDFPageInterpreter(resource_manager, converter)
        with open(file_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                interpreter.process_page(page)
        text = fake_handle.getvalue()
        converter.close()
        fake_handle.close()

    return text


def _extract_page_smart(page) -> str:
    """
    Smart page extraction that handles two-column resume layouts.
    Splits the page into left and right halves by x-coordinate,
    extracts each column separately, then joins left first, right second.
    Falls back to simple extraction if word extraction fails.
    """
    try:
        words = page.extract_words()
        if not words:
            return ""

        page_width = page.width
        mid_x      = page_width / 2

        left_words  = [w for w in words if float(w['x0']) < mid_x]
        right_words = [w for w in words if float(w['x0']) >= mid_x]

        total = len(words)
        if total > 0:
            left_ratio = len(left_words) / total
            # Single-column: >85% of words on one side
            if left_ratio > 0.85 or left_ratio < 0.15:
                return page.extract_text() or ""

        left_words.sort( key=lambda w: (round(float(w['top']), 1), float(w['x0'])))
        right_words.sort(key=lambda w: (round(float(w['top']), 1), float(w['x0'])))

        def words_to_text(word_list):
            if not word_list:
                return ""
            lines  = []
            line   = []
            prev_y = None
            for w in word_list:
                y = round(float(w['top']), 1)
                if prev_y is not None and abs(y - prev_y) > 5:
                    if line:
                        lines.append(' '.join(line))
                    line = []
                line.append(w['text'])
                prev_y = y
            if line:
                lines.append(' '.join(line))
            return '\n'.join(lines)

        left_text  = words_to_text(left_words)
        right_text = words_to_text(right_words)

        combined = ""
        if left_text:
            combined += left_text + "\n"
        if right_text:
            combined += right_text + "\n"
        return combined

    except Exception:
        return ""


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
# Text Cleaning  (used for TF-IDF ML classification)
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and preprocess resume text for ML inference."""
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+',            ' ', text)
    text = re.sub(r'\S+@\S+',                      ' ', text)
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', ' ', text)
    text = re.sub(r'[^\w\s]',                      ' ', text)
    text = re.sub(r'\d+',                          ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# =============================================================================
# Regex Helpers  (used as fallbacks & by Gemini normalizer)
# =============================================================================

def extract_email(text: str) -> str:
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return emails[0] if emails else ""


def extract_phone(text: str) -> str:
    """Extract phone — handles Indian (+91) and international formats."""
    patterns = [
        r'\+91[\s\-]?[6-9]\d{9}',
        r'\+91[\s\-]?\d{5}[\s\-]?\d{5}',
        r'\b[6-9]\d{9}\b',
        r'\+?\d[\d\s\-\.\(\)]{9,14}\d',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group().strip()
    return ""


def _is_valid_name(name: str) -> bool:
    """
    Returns True only if the string looks like a real person's name.

    Rejects:
    - All-caps strings       (EDUCATION, SKILLS, CHANDIGARH)
    - Degree/section words   (Bachelor, Master, Engineering)
    - City/institution names (Chandigarh, Delhi, Group, College)
    - Strings with special characters (@, +91, .com etc.)
    - Strings longer than 5 words
    - Single characters or empty strings
    """
    if not name or not name.strip():
        return False

    name  = name.strip()
    lower = name.lower()

    # Reject all-caps (EDUCATION, CHANDIGARH GROUP etc.)
    if name.isupper():
        return False

    # Reject if exactly matches a rejectlist word
    if lower in _NAME_REJECTLIST:
        return False

    # Reject if STARTS WITH a rejectlist word
    # e.g. "Bachelor of Technology", "Chandigarh Group of Colleges"
    for w in _NAME_REJECTLIST:
        if lower.startswith(w + ' ') or lower == w:
            return False

    # Reject if contains characters that indicate it is not a name
    if any(c in name for c in ['@', '.com', '+91', 'http', '/', '|', '–', ':', ',']):
        return False

    # Reject if too long (more than 5 words — not a person's name)
    if len(name.split()) > 5:
        return False

    # Reject single characters
    if len(name) < 2:
        return False

    return True


# Contact info pattern — used to anchor name search
_CONTACT_PATTERN = re.compile(
    r'(\+?[6-9]\d{9}'              # Indian mobile
    r'|\+91[\s\-]?\d{10}'          # +91 format
    r'|[\w\.-]+@[\w\.-]+\.\w+'     # email
    r'|\b\d{6}\b'                  # 6-digit pincode
    r'|linkedin\.com'              # linkedin
    r'|github\.com)',              # github
    re.IGNORECASE
)


def extract_name(text: str) -> str:
    """
    Smart contact-anchored name extractor.

    Strategy:
    On almost every resume, the candidate's name sits immediately ABOVE
    their phone number, email, or address. So we:
    1. Find the first contact-info line (phone/email/pincode/linkedin)
    2. Look at the 1-4 lines BEFORE it for a valid name
    3. Fall back to scanning the first 15 lines if the anchor method fails

    This reliably handles two-column PDFs where college names or city names
    appear at the same vertical level as the candidate's name but in the
    wrong column order.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # ── Step 1: Contact-anchor strategy ───────────────────────────────────────
    for i, line in enumerate(lines[:30]):
        if _CONTACT_PATTERN.search(line):
            # Look back up to 4 lines before the contact line
            start = max(0, i - 4)
            for j in range(start, i):
                candidate   = lines[j]
                words       = candidate.split()

                if not (1 <= len(words) <= 5):
                    continue
                if not _is_valid_name(candidate):
                    continue

                alpha_words = [w for w in words if w.isalpha()]
                if alpha_words and all(w[0].isupper() for w in alpha_words):
                    return candidate

            # Contact info found but no valid name above — stop anchor search
            break

    # ── Step 2: Fallback — direct line scan ───────────────────────────────────
    for line in lines[:15]:
        words = line.split()
        if not (1 <= len(words) <= 5):
            continue
        if not _is_valid_name(line):
            continue
        alpha_words = [w for w in words if w.isalpha()]
        if alpha_words and all(w[0].isupper() for w in alpha_words):
            return line

    return ""


def extract_experience(text: str) -> list:
    """
    Extract work experience entries using heuristics.
    Uses word boundaries to avoid matching substrings like 'engineering'.
    Returns list of dicts: {designation, company, duration, confidence}
    """
    entries    = []
    title_words = [
        r'\bengineer\b', r'\bdeveloper\b', r'\banalyst\b', r'\bmanager\b',
        r'\barchitect\b', r'\bdesigner\b', r'\bscientist\b', r'\blead\b',
        r'\bdirector\b', r'\bintern\b', r'\bconsultant\b', r'\bspecialist\b',
        r'\bassociate\b', r'\bofficer\b', r'\bexecutive\b', r'\bprogrammer\b',
        r'\badministrator\b', r'\bcoordinator\b',
    ]

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
        if (line and '|' not in line and len(line.split()) <= 8
                and any(re.search(p, line, re.IGNORECASE) for p in title_words)):
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
    Returns list of dicts: {degree, college, year, confidence}
    """
    entries = []

    section_match = re.search(
        r'(?:EDUCATION|ACADEMIC)[\s\S]{0,20}?\n([\s\S]+?)'
        r'(?:\n(?:PROJECTS|SKILLS|CERTIFICATIONS|WORK|EXPERIENCE|TECHNICAL|TRAINING)|$)',
        text, re.IGNORECASE
    )
    section = section_match.group(1) if section_match else text

    degree_keywords = [
        r'B\.?Tech', r'B\.?E\.?', r'B\.?Sc', r'B\.?Com', r'B\.?A\.?',
        r'M\.?Tech', r'M\.?E\.?', r'M\.?Sc', r'M\.?S\.?', r'MBA',
        r'Ph\.?D', r'Bachelor', r'Master', r'Doctor', r'Diploma',
        r'Intermediate', r'Matriculation', r'10th', r'12th',
    ]
    degree_pat = r'(?:' + '|'.join(degree_keywords) + r')[^\n]{0,80}'

    for m in re.finditer(degree_pat, section, re.IGNORECASE):
        degree_line = m.group().strip()
        start       = m.end()
        after       = section[start:start + 200]
        after_lines = [l.strip() for l in after.split('\n') if l.strip()]

        college = ''
        year    = ''

        for line in after_lines[:3]:
            yr_m = re.search(r'\b(19|20)\d{2}\b', line)
            if yr_m and not year:
                year = yr_m.group()
            if re.search(r'university|college|institute|school|iit|nit|bits', line, re.IGNORECASE):
                college = re.sub(r'[\|,].*', '', line).strip()
            elif '|' in line and not college:
                college = line.split('|')[0].strip()

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
# Gemini-Powered Resume Extraction  (main extraction path)
# =============================================================================

def extract_resume_with_gemini(raw_text: str) -> dict:
    """
    Send raw resume text to Gemini API and get back clean structured JSON.
    Handles any resume layout (single-column, two-column, complex).
    Falls back to regex extraction if API call fails or key is missing.
    """
    api_key = os.environ.get('GEMINI_API_KEY', '')

    if not api_key:
        print("  No GEMINI_API_KEY found — using regex fallback for extraction")
        return _regex_fallback_extraction(raw_text)

    prompt = f"""You are a professional resume parser. Extract information from the resume text below.

STRICT RULES:
- Return ONLY a valid JSON object. No markdown, no backticks, no explanation text.
- If a field is not found, use empty string "" or empty list [].
- For skills: only include TECHNICAL skills (programming languages, frameworks,
  databases, tools, cloud platforms).
  Do NOT include soft skills like "Team Leadership", "Problem Solving", "Adaptability".
  Do NOT include generic words like "Windows", "Environment", "Technology", "Languages".
- For experience_entries: only include REAL work experience or internships at companies.
  Do NOT include personal projects or academic training projects.
  If the person is a fresher/student with no real job experience, return empty list [].
- For education_entries: include ALL qualifications (B.Tech, 12th, 10th etc).
- Name: ONLY the person's actual first and last name.
  The name is typically found just ABOVE the phone number or email address in the resume.
  NEVER return a degree name like "Bachelor", "Master", "B.Tech" as the name.
  NEVER return college names like "Chandigarh", "Group of Colleges" as the name.
  NEVER return city names or section headings as the name.
- years_of_experience: if fresher/student with no job experience, return "".

Return exactly this JSON structure:
{{
  "name": "Pankaj",
  "email": "email@example.com",
  "phone": "+91-9992915596",
  "location": "Jhajjar, Haryana",
  "years_of_experience": "",
  "skills": ["Python", "Flask", "MySQL", "Machine Learning", "NumPy", "Pandas",
             "Git", "GitHub", "scikit-learn"],
  "experience_entries": [],
  "education_entries": [
    {{
      "degree": "Bachelor of Technology in Computer Science Engineering",
      "college": "Chandigarh Group of Colleges, Landran",
      "year": "2026",
      "confidence": 0.95
    }}
  ],
  "locations": [
    {{"text": "Jhajjar, Haryana", "confidence": 0.95}}
  ]
}}

RESUME TEXT:
{raw_text[:4500]}
"""

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 1500,
            "temperature": 0.1
        }
    }).encode()

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:generateContent?key={api_key}"
    )

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data     = json.loads(resp.read())
            raw_resp = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            raw_resp = re.sub(r"```json|```", "", raw_resp).strip()
            extracted = json.loads(raw_resp)

            # ── Strict name validation ─────────────────────────────────────
            name = extracted.get('name', '').strip()
            if not _is_valid_name(name):
                print(f"  Gemini name '{name}' rejected — using contact-anchor fallback")
                name = extract_name(raw_text)

            # ── Normalize skills — filter both blocklists ──────────────────
            raw_skills  = extracted.get('skills', [])
            skills      = []
            seen_skills = set()
            for s in raw_skills:
                if not isinstance(s, str):
                    continue
                s   = s.strip()
                key = s.lower()
                if (key
                        and key not in _SKILL_BLOCKLIST
                        and key not in _NAME_REJECTLIST
                        and len(key) > 1
                        and key not in seen_skills):
                    seen_skills.add(key)
                    skills.append({'text': s, 'confidence': 0.95})

            # ── Regex is more reliable than Gemini for phone/email ─────────
            phone = extracted.get('phone', '').strip() or extract_phone(raw_text)
            email = extracted.get('email', '').strip() or extract_email(raw_text)

            print(f"  Gemini extraction success — name: '{name}', skills: {len(skills)}")

            return {
                'name'               : name,
                'email'              : email,
                'phone'              : phone,
                'skills'             : skills,
                'years_of_experience': extracted.get('years_of_experience', ''),
                'locations'          : extracted.get('locations', []),
                'experience_entries' : extracted.get('experience_entries', []),
                'education_entries'  : extracted.get('education_entries', []),
                'all_entities'       : {},
            }

    except urllib.error.HTTPError as e:
        print(f"  Gemini API HTTP error {e.code}: {e.reason} — falling back to regex")
    except json.JSONDecodeError as e:
        print(f"  Gemini returned invalid JSON: {e} — falling back to regex")
    except Exception as e:
        print(f"  Gemini extraction failed: {e} — falling back to regex")

    return _regex_fallback_extraction(raw_text)


def _regex_fallback_extraction(raw_text: str) -> dict:
    """Pure regex fallback used when Gemini API is unavailable."""
    print("  Using regex fallback for extraction")
    return {
        'name'               : extract_name(raw_text),
        'email'              : extract_email(raw_text),
        'phone'              : extract_phone(raw_text),
        'skills'             : extract_skills_by_keyword(raw_text),
        'years_of_experience': '',
        'locations'          : [],
        'experience_entries' : extract_experience(raw_text),
        'education_entries'  : extract_education(raw_text),
        'all_entities'       : {},
    }


# =============================================================================
# Keyword-based Skill Extractor  (used alongside Gemini for extra coverage)
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
    """Scan resume text for known skills using whole-word matching."""
    found      = []
    seen       = set()
    text_lower = text.lower()
    for skill in _KNOWN_SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            key = skill.lower()
            if key not in seen and key not in _SKILL_BLOCKLIST:
                seen.add(key)
                found.append({'text': skill, 'confidence': 0.80})
    return found


def merge_skills(gemini_skills: list, keyword_skills: list) -> list:
    """
    Merge Gemini-extracted + keyword skills.
    Deduplicates by lowercase key. Filters blocklist. Gemini takes precedence.
    """
    merged = {}
    for s in gemini_skills:
        if not isinstance(s, dict):
            continue
        key = s.get('text', '').lower().strip()
        if key and key not in _SKILL_BLOCKLIST and len(key) > 1:
            merged[key] = s
    for s in keyword_skills:
        if not isinstance(s, dict):
            continue
        key = s.get('text', '').lower().strip()
        if key and key not in _SKILL_BLOCKLIST and key not in merged:
            merged[key] = s
    return sorted(merged.values(), key=lambda x: x['confidence'], reverse=True)


# =============================================================================
# Resume Scoring  (section-based heuristic, 0-100)
# =============================================================================

def score_resume(text: str) -> dict:
    """
    Score a resume across 9 sections (total 100 points).

    Experience uses exact regex phrases to avoid false positives from
    body text containing the word 'experience' (e.g. 'feature engineering').
    """
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
        'Contact Info': any(k in text_l for k in [
            'email', 'phone', 'linkedin', 'github', 'contact',
        ]),

        'Summary/Objective': any(k in text_l for k in [
            'professional summary', 'career objective', 'objective',
            'profile summary', 'about me',
        ]),

        'Education': any(k in text_l for k in [
            'education', 'university', 'college', 'degree',
            'bachelor', 'master', 'b.tech', 'b.e',
            'intermediate', 'matriculation',
        ]),

        # Exact phrases only — avoids false positive from body text
        # (e.g. "feature engineering using Python" contains "experience" indirectly)
        'Experience': (
            bool(re.search(r'\bwork\s+experience\b',         text_l)) or
            bool(re.search(r'\bemployment\s+history\b',      text_l)) or
            bool(re.search(r'\bprofessional\s+experience\b', text_l)) or
            bool(re.search(r'\bwork\s+history\b',            text_l)) or
            bool(re.search(r'\binternship\b',                text_l)) or
            bool(re.search(r'\bindustrial\s+training\b',     text_l)) or
            bool(re.search(r'\btraining\s+&\s+projects\b',   text_l))
        ),

        'Skills': any(k in text_l for k in [
            'technical skills', 'core competencies', 'technical competencies',
            'key skills', 'skills', 'competencies',
        ]),

        'Projects': any(k in text_l for k in [
            'projects', 'portfolio', 'training & projects',
            'personal projects', 'academic projects',
        ]),

        'Certifications': any(k in text_l for k in [
            'certifications', 'certification', 'certificate',
            'certified', 'completion badge', 'completion of',
        ]),

        'Achievements': any(k in text_l for k in [
            'achievements', 'academic achievements', 'award',
            'honors', 'recognition', 'finalist', 'hackathon',
        ]),

        'Hobbies/Interests': any(k in text_l for k in [
            'hobbies', 'interests & hobbies', 'interests',
            'activities', 'volunteer',
        ]),
    }

    total     = 0
    breakdown = {}
    for section, points in sections.items():
        got              = points if checks[section] else 0
        breakdown[section] = {'earned': got, 'max': points, 'present': checks[section]}
        total           += got

    return {'total': total, 'max': 100, 'breakdown': breakdown}


def detect_experience_level(text: str, pages: int) -> str:
    """Detect experience level from resume text and page count."""
    text_l    = text.lower()
    exp_years = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience', text_l)
    if exp_years:
        yrs = max(int(y) for y in exp_years)
        if yrs >= 5:   return 'Senior'
        elif yrs >= 2: return 'Mid-Level'
        else:          return 'Junior'

    if any(k in text_l for k in ['senior', 'lead ', 'principal', 'architect', 'head of']):
        return 'Senior'
    if any(k in text_l for k in [
        'internship', 'intern ', 'fresher', 'entry level', 'trainee',
        'aspiring', 'b.tech student', 'pursuing',
    ]):
        return 'Fresher/Intern'
    if re.search(r'\bwork\s+experience\b|\bemployment\b', text_l):
        return 'Experienced'
    if pages <= 1:
        return 'Fresher/Intern'
    return 'Mid-Level'


# =============================================================================
# Job Role Requirements Database
# =============================================================================

JOB_ROLE_REQUIREMENTS = {
    'Backend Developer': {
        'core'     : ['Python', 'Java', 'SQL', 'REST API', 'Git'],
        'preferred': ['Docker', 'Kubernetes', 'AWS', 'Redis', 'PostgreSQL', 'CI/CD', 'Linux', 'Node.js'],
    },
    'Frontend Developer': {
        'core'     : ['JavaScript', 'React', 'HTML', 'CSS', 'Git'],
        'preferred': ['TypeScript', 'Vue', 'Angular', 'Webpack', 'Figma', 'REST API', 'Next.js'],
    },
    'Full Stack Developer': {
        'core'     : ['JavaScript', 'Python', 'React', 'SQL', 'Git', 'REST API'],
        'preferred': ['Docker', 'Kubernetes', 'AWS', 'TypeScript', 'Node.js', 'CI/CD', 'MongoDB'],
    },
    'Data Scientist': {
        'core'     : ['Python', 'Machine Learning', 'SQL', 'Statistics', 'Data Analysis'],
        'preferred': ['Deep Learning', 'TensorFlow', 'PyTorch', 'NLP', 'Big Data', 'Data Visualization', 'R'],
    },
    'ML Engineer': {
        'core'     : ['Python', 'Machine Learning', 'Docker', 'SQL', 'Git'],
        'preferred': ['Deep Learning', 'Kubernetes', 'AWS', 'TensorFlow', 'PyTorch', 'CI/CD', 'MLflow'],
    },
    'Data Analyst': {
        'core'     : ['SQL', 'Excel', 'Data Analysis', 'Data Visualization', 'Statistics'],
        'preferred': ['Python', 'Tableau', 'Power BI', 'R', 'Machine Learning'],
    },
    'DevOps Engineer': {
        'core'     : ['Docker', 'Kubernetes', 'CI/CD', 'Linux', 'Git'],
        'preferred': ['AWS', 'Azure', 'GCP', 'Terraform', 'Ansible', 'Python', 'Monitoring'],
    },
    'Cloud Architect': {
        'core'     : ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'],
        'preferred': ['Terraform', 'CI/CD', 'Linux', 'Networking', 'Security', 'Microservices'],
    },
    'Mobile Developer': {
        'core'     : ['Android', 'iOS', 'Git', 'REST API'],
        'preferred': ['Flutter', 'React Native', 'Kotlin', 'Swift', 'Firebase', 'Java'],
    },
    'Cybersecurity Analyst': {
        'core'     : ['Cybersecurity', 'Linux', 'Networking', 'Python'],
        'preferred': ['Penetration Testing', 'SIEM', 'Firewalls', 'Incident Response', 'Cloud Security'],
    },
    'Java Developer': {
        'core'     : ['Java', 'Spring', 'SQL', 'Git'],
        'preferred': ['Docker', 'Maven', 'Hibernate', 'REST API', 'Microservices', 'Kubernetes'],
    },
    'Python Developer': {
        'core'     : ['Python', 'SQL', 'Git', 'REST API'],
        'preferred': ['Django', 'Flask', 'FastAPI', 'Docker', 'AWS', 'PostgreSQL'],
    },
    'React Developer': {
        'core'     : ['React', 'JavaScript', 'TypeScript', 'Git'],
        'preferred': ['Node.js', 'REST API', 'Redux', 'Next.js', 'CSS', 'GraphQL'],
    },
    'Data Engineer': {
        'core'     : ['Python', 'SQL', 'Big Data', 'ETL', 'Git'],
        'preferred': ['Spark', 'Hadoop', 'AWS', 'Kafka', 'Docker', 'Airflow'],
    },
    'Business Analyst': {
        'core'     : ['SQL', 'Excel', 'Data Analysis', 'Communication'],
        'preferred': ['Tableau', 'Power BI', 'Statistics', 'Agile', 'JIRA', 'Requirements Gathering'],
    },
    'UI/UX Designer': {
        'core'     : ['Figma', 'Adobe XD', 'User Research', 'Wireframing'],
        'preferred': ['CSS', 'HTML', 'Prototyping', 'Design Systems', 'Accessibility'],
    },
    'Network Engineer': {
        'core'     : ['Networking', 'Linux', 'Cisco', 'Firewalls'],
        'preferred': ['AWS', 'Python', 'Cybersecurity', 'Monitoring', 'VPN', 'DNS'],
    },
    'Project Manager': {
        'core'     : ['Project Management', 'Agile', 'Communication', 'Leadership'],
        'preferred': ['JIRA', 'Scrum', 'Risk Management', 'Budgeting', 'Stakeholder Management'],
    },
    'Blockchain Developer': {
        'core'     : ['Solidity', 'Ethereum', 'Smart Contracts', 'JavaScript'],
        'preferred': ['Web3.js', 'Truffle', 'DeFi', 'NFT', 'Rust', 'Python'],
    },
    'AI/NLP Engineer': {
        'core'     : ['Python', 'NLP', 'Deep Learning', 'Machine Learning'],
        'preferred': ['TensorFlow', 'PyTorch', 'Transformers', 'Hugging Face', 'spaCy', 'Docker'],
    },
}

_CATEGORY_ROLE_MAP = {
    'data science'          : ['Data Scientist', 'ML Engineer', 'Data Analyst', 'AI/NLP Engineer'],
    'engineering'           : ['Backend Developer', 'Full Stack Developer', 'Java Developer', 'Python Developer'],
    'information-technology': ['Backend Developer', 'DevOps Engineer', 'Full Stack Developer'],
    'python developer'      : ['Python Developer', 'Backend Developer', 'ML Engineer'],
    'java developer'        : ['Java Developer', 'Backend Developer'],
    'react developer'       : ['React Developer', 'Frontend Developer'],
    'devops engineer'       : ['DevOps Engineer', 'Cloud Architect'],
    'database'              : ['Data Engineer', 'Backend Developer', 'Data Analyst'],
    'business analyst'      : ['Business Analyst', 'Data Analyst'],
    'hr'                    : ['Project Manager', 'Business Analyst'],
    'designer'              : ['UI/UX Designer', 'Frontend Developer'],
}


def compute_job_matches(extracted_skills: list, top_n: int = 5,
                        predicted_category: str = '') -> list:
    """
    Compare extracted skills against JOB_ROLE_REQUIREMENTS.
    Scoring: core_match * 70% + preferred_match * 30%.
    Applies +8% boost to roles aligned with the ML-predicted category.
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

        core_pct  = (len(matched_core) / len(core)) * 70 if core else 0
        pref_pct  = (len(matched_pref) / len(pref)) * 30 if pref else 0
        total_pct = round(core_pct + pref_pct, 1)

        matches.append({
            'role'             : role,
            'match_pct'        : total_pct,
            'matched_skills'   : matched_core + matched_pref,
            'missing_core'     : [s for s in core if s.lower() not in skill_names_lower],
            'missing_preferred': [s for s in pref if s.lower() not in skill_names_lower],
            'total_required'   : len(core) + len(pref),
        })

    if predicted_category:
        cat_lower = predicted_category.lower().strip()
        for cat_key, boosted_roles in _CATEGORY_ROLE_MAP.items():
            if cat_key in cat_lower or cat_lower in cat_key:
                for m in matches:
                    if m['role'] in boosted_roles:
                        m['match_pct'] = min(100.0, round(m['match_pct'] + 8, 1))
                break

    matches.sort(key=lambda x: x['match_pct'], reverse=True)
    return matches[:top_n]


def compute_skill_gaps(extracted_skills: list, target_roles: list = None,
                       predicted_category: str = '') -> list:
    """
    For each target role (or top 3 matched roles), compute missing skills.
    Returns list of dicts: {role, missing_skills, missing_core_skills, message}
    """
    if target_roles is None:
        top_matches  = compute_job_matches(extracted_skills, top_n=3,
                                           predicted_category=predicted_category)
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
        reqs         = JOB_ROLE_REQUIREMENTS[role]
        all_skills   = reqs['core'] + reqs['preferred']
        missing      = [s for s in all_skills  if s.lower() not in skill_names_lower]
        missing_core = [s for s in reqs['core'] if s.lower() not in skill_names_lower]

        if missing_core:
            names = ', '.join(f"'{s}'" for s in missing_core[:3])
            msg   = f"To become a {role}, you are missing {names}."
        elif missing:
            names = ', '.join(f"'{s}'" for s in missing[:2])
            msg   = f"You have all core skills for {role}! Consider adding {names} to stand out."
        else:
            msg   = f"You are fully qualified for {role} roles!"

        gaps.append({
            'role'               : role,
            'missing_skills'     : missing,
            'missing_core_skills': missing_core,
            'message'            : msg,
        })

    return gaps


# =============================================================================
# Project Ideas  (Gemini API with curated fallback)
# =============================================================================

def get_project_ideas(extracted_skills: list, max_projects: int = 4,
                      experience_level: str = '') -> list:
    """
    Generate personalized project ideas using Gemini API.
    Calibrates difficulty based on experience level.
    Falls back to curated list if API call fails.
    """
    skill_names = [
        s['text'] if isinstance(s, dict) else str(s)
        for s in extracted_skills
    ]

    api_key = os.environ.get('GEMINI_API_KEY', '')

    if api_key and skill_names:
        level_lower = experience_level.lower()
        if 'fresher' in level_lower or 'intern' in level_lower:
            difficulty_guidance = "Range from Beginner to Intermediate only. Do NOT suggest Advanced projects."
        elif 'junior' in level_lower:
            difficulty_guidance = "Range from Beginner to Intermediate difficulty."
        elif 'senior' in level_lower:
            difficulty_guidance = "Range from Intermediate to Advanced difficulty."
        else:
            difficulty_guidance = "Include a mix of Beginner, Intermediate, and Advanced difficulty."

        try:
            prompt = (
                "You are a project idea generator for software developers.\n"
                f"Based on these skills: {', '.join(skill_names[:15])}\n"
                f"Experience level: {experience_level or 'Unknown'}\n\n"
                f"Generate exactly {max_projects} unique, practical project ideas that:\n"
                "- Use a good mix of the listed skills\n"
                f"- {difficulty_guidance}\n"
                "- Are portfolio-worthy and impressive to employers\n"
                "- Are specific (not generic)\n\n"
                "Respond ONLY with a JSON array, no markdown, no extra text:\n"
                '[\n'
                '  {\n'
                '    "name": "Project Name",\n'
                '    "description": "2-sentence description.",\n'
                '    "matched_skills": ["Skill1", "Skill2"],\n'
                '    "all_skills": ["Skill1", "Skill2", "Skill3"],\n'
                '    "difficulty": "Beginner",\n'
                '    "relevance": 90.0\n'
                '  }\n'
                ']'
            )

            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.7}
            }).encode()

            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/gemini-2.0-flash:generateContent?key={api_key}"
            )
            req = urllib.request.Request(
                url, data=payload,
                headers={"content-type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data     = json.loads(resp.read())
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw_text = re.sub(r"```json|```", "", raw_text).strip()
                projects = json.loads(raw_text)
                if isinstance(projects, list) and projects:
                    return projects[:max_projects]

        except Exception as e:
            print(f"  Gemini project ideas failed: {e} — using fallback")

    # ── Curated fallback with difficulty variety ──────────────────────────────
    FALLBACK_PROJECTS = [
        {
            "name"          : "Student Grade Tracker",
            "description"   : "A web app to track student grades, calculate GPA, and visualize performance over time.",
            "matched_skills": ["Python", "Flask", "MySQL"],
            "all_skills"    : ["Python", "Flask", "MySQL", "Excel"],
            "difficulty"    : "Beginner",
            "relevance"     : 100.0,
        },
        {
            "name"          : "Heart Disease Predictor",
            "description"   : "ML model that predicts heart disease risk from patient data with a clean Flask web interface.",
            "matched_skills": ["Python", "Machine Learning", "Flask"],
            "all_skills"    : ["Python", "Machine Learning", "Flask", "Pandas", "NumPy"],
            "difficulty"    : "Intermediate",
            "relevance"     : 100.0,
        },
        {
            "name"          : "AI Resume Analyzer",
            "description"   : "NLP system that parses resumes, extracts skills, and recommends jobs using ML classifiers.",
            "matched_skills": ["Python", "Machine Learning", "NLP"],
            "all_skills"    : ["Python", "Machine Learning", "NLP", "Flask"],
            "difficulty"    : "Advanced",
            "relevance"     : 100.0,
        },
        {
            "name"          : "Real-Time Chat App",
            "description"   : "WebSocket-based chat application with rooms, message history, and user authentication.",
            "matched_skills": ["Python", "Flask"],
            "all_skills"    : ["Python", "Flask", "JavaScript", "SQL"],
            "difficulty"    : "Intermediate",
            "relevance"     : 90.0,
        },
        {
            "name"          : "Expense Tracker",
            "description"   : "Personal finance app to log expenses, categorize spending, and visualize monthly budgets.",
            "matched_skills": ["Python", "MySQL"],
            "all_skills"    : ["Python", "MySQL", "Excel", "Data Visualization"],
            "difficulty"    : "Beginner",
            "relevance"     : 85.0,
        },
        {
            "name"          : "ML Model Dashboard",
            "description"   : "Deploy and monitor ML models with a Flask dashboard showing predictions and confidence scores.",
            "matched_skills": ["Python", "Machine Learning"],
            "all_skills"    : ["Python", "Machine Learning", "Docker", "Flask"],
            "difficulty"    : "Advanced",
            "relevance"     : 80.0,
        },
        {
            "name"          : "E-Commerce REST API",
            "description"   : "Full REST API with JWT auth, product catalog, cart, orders, and payment integration.",
            "matched_skills": ["Python", "MySQL"],
            "all_skills"    : ["Python", "SQL", "REST API", "Docker"],
            "difficulty"    : "Intermediate",
            "relevance"     : 85.0,
        },
        {
            "name"          : "Fraud Detection System",
            "description"   : "Real-time fraud detection using ensemble ML methods with an interactive analytics dashboard.",
            "matched_skills": ["Python", "Machine Learning"],
            "all_skills"    : ["Python", "Machine Learning", "SQL", "Kafka"],
            "difficulty"    : "Advanced",
            "relevance"     : 85.0,
        },
    ]

    skill_lower = {
        (s['text'] if isinstance(s, dict) else str(s)).lower()
        for s in extracted_skills
    }
    scored = []
    for p in FALLBACK_PROJECTS:
        matched = [s for s in p["all_skills"] if s.lower() in skill_lower]
        if len(matched) >= len(p["all_skills"]) * 0.4:
            p               = dict(p)
            p["matched_skills"] = matched
            p["relevance"]  = round(len(matched) / len(p["all_skills"]) * 100, 1)
            scored.append(p)

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:max_projects] if scored else FALLBACK_PROJECTS[:max_projects]