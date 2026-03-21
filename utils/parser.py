"""
utils/parser.py
Resume text extraction, BERT NER (primary) + Gemini (fallback) entity extraction,
ML domain classification, job-role matching, skill-gap analysis, project ideas.
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

# ── Skills that are NOT real technical skills ──────────────────────────────────
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
_NAME_REJECTLIST = {
    'bachelor', 'master', 'doctor', 'doctorate', 'diploma', 'intermediate',
    'matriculation', 'btech', 'mtech', 'mba', 'phd',
    'education', 'skills', 'experience', 'summary', 'profile', 'objective',
    'certifications', 'achievements', 'projects', 'training', 'engineering',
    'technology', 'science', 'commerce', 'arts',
    'unknown', 'name', 'candidate', 'applicant', 'resume', 'curriculum',
    'vitae', 'contact', 'address', 'details', 'information', 'about',
    'professional', 'academic', 'personal', 'technical', 'interpersonal',
    'chandigarh', 'delhi', 'mumbai', 'bangalore', 'hyderabad', 'chennai',
    'pune', 'kolkata', 'group', 'college', 'university', 'institute',
    'school', 'landran', 'mohali', 'noida', 'gurugram', 'gurgaon',
    'lucknow', 'jaipur', 'ahmedabad', 'surat', 'bhopal', 'indore',
}


# ── Tech/tool names — never valid as company names ────────────────────────────
_COMPANY_TECH_BLOCKLIST = {
    'flask', 'django', 'fastapi', 'react', 'angular', 'vue', 'next.js',
    'node.js', 'express', 'docker', 'kubernetes', 'terraform', 'ansible',
    'github', 'gitlab', 'github actions', 'bitbucket', 'jenkins', 'circleci',
    'aws', 'gcp', 'azure', 'ec2', 's3', 'lambda', 'sagemaker',
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
    'kafka', 'spark', 'hadoop', 'airflow', 'mlflow', 'ci/cd',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'scikit',
    'pandas', 'numpy', 'matplotlib', 'seaborn',
    'render', 'heroku', 'vercel', 'netlify', 'firebase', 'supabase',
    'html', 'css', 'bootstrap', 'tailwind', 'webpack',
    'git', 'linux', 'bash', 'postman', 'jira', 'figma',
    'xgboost', 'lightgbm', 'opencv', 'nltk', 'spacy',
}

_COMPANY_PROJECT_KEYWORDS = {
    'prediction', 'analysis', 'detection', 'classifier', 'model',
    'system', 'app', 'application', 'dashboard', 'pipeline',
    'project', 'tracker', 'generator', 'analyzer', 'bot',
}


def _is_valid_company(company: str, designation: str = '', duration: str = '') -> bool:
    """Returns True only if string looks like a real company name."""
    if not company or not company.strip():
        return False
    c  = company.strip()
    cl = c.lower()
    if c.startswith('@'):                                     return False
    if cl in _COMPANY_TECH_BLOCKLIST:                        return False
    if any(tech in cl for tech in _COMPANY_TECH_BLOCKLIST):  return False
    if any(kw in cl for kw in _COMPANY_PROJECT_KEYWORDS):    return False
    if len(c) < 3:                                           return False
    if not designation and not duration:                      return False
    if any(ch in c for ch in ['/', 'http', '.com', '.io']):  return False
    if designation and designation.lower() in _COMPANY_TECH_BLOCKLIST: return False
    return True




# =============================================================================
# PDF Extraction
# =============================================================================

def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text — handles single and two-column layouts."""
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

    # ── Clean PDF artifacts ────────────────────────────────────────────────────
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    text = re.sub(r'\x7f', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def _extract_page_smart(page) -> str:
    """Handle two-column layouts by splitting at page midpoint."""
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
            if left_ratio > 0.85 or left_ratio < 0.15:
                return page.extract_text() or ""
        for wlist in [left_words, right_words]:
            wlist.sort(key=lambda w: (round(float(w['top']), 1), float(w['x0'])))

        def words_to_text(word_list):
            if not word_list:
                return ""
            lines, line, prev_y = [], [], None
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

        combined = ""
        left_text  = words_to_text(left_words)
        right_text = words_to_text(right_words)
        if left_text:  combined += left_text  + "\n"
        if right_text: combined += right_text + "\n"
        return combined
    except Exception:
        return ""


def count_pages(file_path: str) -> int:
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
# Text Cleaning  (TF-IDF ML input)
# =============================================================================

def clean_text(text: str) -> str:
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
# Regex Helpers
# =============================================================================

def extract_email(text: str) -> str:
    """
    Extract email from raw text.
    - Uses strict pattern requiring @ and valid TLD
    - Filters social profile URLs that contain @
    - Scans full text not just first line (handles two-column layouts
      where email appears in right column)
    """
    emails = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
    # Filter social/platform URLs
    emails = [e for e in emails if not any(
        s in e.lower() for s in ['linkedin', 'github', 'twitter', 'instagram',
                                  'example.com', 'youremail', 'email.com']
    )]
    # Prefer emails with common domains (gmail, yahoo, outlook, edu, org)
    # over obscure ones — helps when multiple emails found
    preferred = [e for e in emails if any(
        d in e.lower() for d in ['gmail', 'yahoo', 'outlook', 'hotmail',
                                  '.edu', '.org', '.ac.in', '.co.in', 'iit', 'nit']
    )]
    return preferred[0] if preferred else (emails[0] if emails else "")


def extract_phone(text: str) -> str:
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
    if not name or not name.strip():
        return False
    name  = name.strip()
    lower = name.lower()
    if name.isupper():
        return False
    if lower in _NAME_REJECTLIST:
        return False
    for w in _NAME_REJECTLIST:
        if lower.startswith(w + ' ') or lower == w:
            return False
    if any(c in name for c in ['@', '.com', '+91', 'http', '/', '|', '–', ':', ',']):
        return False
    if len(name.split()) > 5:
        return False
    if len(name) < 2:
        return False
    return True


_CONTACT_PATTERN = re.compile(
    r'(\+?[6-9]\d{9}|\+91[\s\-]?\d{10}|[\w\.-]+@[\w\.-]+\.\w+'
    r'|\b\d{6}\b|linkedin\.com|github\.com)',
    re.IGNORECASE
)


def extract_name(text: str) -> str:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for i, line in enumerate(lines[:30]):
        if _CONTACT_PATTERN.search(line):
            start = max(0, i - 4)
            for j in range(start, i):
                candidate = lines[j]
                words     = candidate.split()
                if not (1 <= len(words) <= 5):
                    continue
                if not _is_valid_name(candidate):
                    continue
                alpha_words = [w for w in words if w.isalpha()]
                if alpha_words and all(w[0].isupper() for w in alpha_words):
                    return candidate
            break
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
    entries     = []
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
                company  = (info_line.split('|')[0] if '|' in info_line
                            else info_line.split(',')[0] if ',' in info_line
                            else info_line).strip()
                if company:
                    entries.append({
                        'designation': line, 'company': company,
                        'duration': duration, 'confidence': 0.85,
                    })
                i = j + 1
                continue
        i += 1
    return entries


def extract_education(text: str) -> list:
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
        after_lines = [l.strip() for l in section[m.end():m.end()+200].split('\n') if l.strip()]
        college, year = '', ''
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
            entries.append({'degree': degree_line, 'college': college,
                            'year': year, 'confidence': 0.85})
    return entries


# =============================================================================
# BERT NER Extractor  (Primary — Deep Learning)
# =============================================================================

class ResumeNERExtractor:
    """
    Primary entity extractor using yashpwr/resume-ner-bert-v2 BERT model.
    This is the DL component of the project.
    Falls back to regex if model unavailable.
    """
    MODEL_NAME = 'yashpwr/resume-ner-bert-v2'

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            'token-classification',
            model=self.MODEL_NAME,
            aggregation_strategy='simple',
            device=-1,
        )
        self._loaded = True
        print(f"  BERT NER loaded: {self.MODEL_NAME}")

    _MAX_WORDS = {
        'Name': 5, 'Email Address': 1, 'Phone': 1,
        'Skills': 6, 'Companies worked at': 8, 'Company': 8,
        'Designation': 8, 'Degree': 10, 'College Name': 10,
        'Graduation Year': 1, 'Years of Experience': 6, 'Location': 6,
    }

    # Per-entity confidence thresholds
    _THRESHOLDS = {
        'Name': 0.80, 'Email Address': 0.70, 'Phone': 0.70,
        'Skills': 0.60, 'Companies worked at': 0.65, 'Company': 0.65,
        'Designation': 0.65, 'Degree': 0.70, 'College Name': 0.65,
        'Graduation Year': 0.75, 'Years of Experience': 0.65, 'Location': 0.60,
    }

    _NOISE_WORDS = {'and', 'the', 'of', 'with', 'for', 'in', 'at', 'to',
                    'a', 'an', 'is', 'are', 'was', 'on', 'by', 'as'}

    def _clean_word(self, word: str) -> str:
        word = re.sub(r'\s*##', '', word)
        word = re.sub(r'^\W+$', '', word)
        return word.strip()

    def _is_noise(self, group: str, word: str) -> bool:
        if group in ('Companies worked at', 'Company'):
            if word.lower() in self._NOISE_WORDS: return True
            if len(word) < 2: return True
        if group == 'Degree':
            if len(word.split()) > 6: return True
        if group == 'Graduation Year':
            try:
                yr = int(re.sub(r'\D', '', word))
                if not (1980 <= yr <= 2030): return True
            except:
                return True
        return False

    def extract(self, text: str) -> dict:
        """
        Extraction strategy per field:
          BERT NER → name, skills, locations, years_of_experience  (token-level DL)
          Gemini   → education_entries, experience_entries only    (structured, context-aware)
          Regex    → phone always + email validation + edu/exp fallback

        Gemini is used ONLY for education and experience — not for name/skills/email.
        BERT handles all token-level extraction. Regex is always the safety net.
        """
        chunks = self._chunk_text(text, max_chars=1800, overlap_chars=200)
        all_raw = []
        for chunk in chunks:
            try:
                all_raw.extend(self.pipe(chunk))
            except Exception:
                continue

        grouped = {}
        for ent in all_raw:
            group = ent['entity_group']
            word  = self._clean_word(ent['word'])
            score = float(ent['score'])
            if not word or len(word) < 2:
                continue
            if len(word.split()) > self._MAX_WORDS.get(group, 10):
                continue
            if score < self._THRESHOLDS.get(group, 0.40):
                continue
            if self._is_noise(group, word):
                continue
            if group not in grouped:
                grouped[group] = []
            grouped[group].append({'text': word, 'confidence': round(score, 4)})

        # Deduplicate
        for group in grouped:
            seen = {}
            for item in grouped[group]:
                key = item['text'].lower().strip()
                if key not in seen or item['confidence'] > seen[key]['confidence']:
                    seen[key] = item
            grouped[group] = sorted(seen.values(), key=lambda x: x['confidence'], reverse=True)

        # ── Structured fields: Gemini (primary) → Regex → BERT ────────────────
        # Gemini understands full document context — gives clean {degree, college, year}
        # even from two-column, non-standard, or student resume layouts.
        # Regex is reliable fallback. BERT entity assembly is last resort.
        gemini_results     = None
        experience_entries = []
        education_entries  = []

        # ── Education + Experience: Gemini → Regex → BERT ───────────────────
        # Gemini called ONLY for these two structured fields.
        # All other extraction (name, skills, email, locations) done by BERT above.
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if api_key:
            gemini_result      = extract_resume_with_gemini(text)
            experience_entries = gemini_result.get('experience_entries', [])
            education_entries  = gemini_result.get('education_entries',  [])

        # Regex fallback for experience
        if not experience_entries:
            regex_exp = extract_experience(text)
            if regex_exp:
                experience_entries = regex_exp
            else:
                # BERT 'Companies worked at' entities — filter out tech names
                # BERT often tags tools from project descriptions as companies
                bert_companies = grouped.get('Companies worked at',
                                             grouped.get('Company', []))
                bert_desigs    = grouped.get('Designation', [])
                experience_entries = []
                for i, d in enumerate(bert_companies):
                    designation = bert_desigs[i]['text'] if i < len(bert_desigs) else ''
                    company     = d['text']
                    if _is_valid_company(company, designation, ''):
                        experience_entries.append({
                            'designation': designation,
                            'company'    : company,
                            'duration'   : '',
                            'confidence' : 0.5,
                        })

        # Regex fallback for education
        if not education_entries:
            regex_edu         = extract_education(text)
            education_entries = regex_edu if regex_edu else [
                {
                    'degree' : grouped.get('Degree',          [{}])[i]['text']
                               if i < len(grouped.get('Degree', []))          else '',
                    'college': grouped.get('College Name',    [{}])[i]['text']
                               if i < len(grouped.get('College Name', []))    else '',
                    'year'   : grouped.get('Graduation Year', [{}])[i]['text']
                               if i < len(grouped.get('Graduation Year', [])) else '',
                    'confidence': 0.5,
                }
                for i in range(max(
                    len(grouped.get('Degree', [])),
                    len(grouped.get('College Name', [])),
                    1
                ))
                if grouped.get('Degree') or grouped.get('College Name')
            ]

        # ── Token-level fields: BERT primary, regex fallback ──────────────────
        # BERT is best at: name spans, skill tokens, location mentions, YOE phrases
        phone     = extract_phone(text)
        bert_name = self._first(grouped.get('Name', []))
        name      = bert_name if _is_valid_name(bert_name) else extract_name(text)

        # Skills: BERT NER tokens (high confidence DL extraction)
        # Merged with keyword scan in app.py for full coverage
        skills = grouped.get('Skills', [])

        # Locations: BERT reliable for location spans
        locations = grouped.get('Location', [])

        # Years of experience: BERT reliable for "X years" phrases
        years_of_experience = self._first(grouped.get('Years of Experience', []))

        # Email: validate BERT result — BERT sometimes tags partial names/usernames
        # as Email Address entities (e.g. 'sanyam.' from a LinkedIn handle)
        # Only accept BERT email if it contains @ and matches proper email pattern
        _bert_email = self._first(grouped.get('Email Address', []))
        email = (
            _bert_email
            if _bert_email and re.match(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', _bert_email)
            else extract_email(text)
        )

        return {
            'name'               : name,
            'email'              : email,
            'phone'              : phone,
            'skills'             : skills,             # BERT (DL)
            'years_of_experience': years_of_experience, # BERT (DL)
            'locations'          : locations,           # BERT (DL)
            'experience_entries' : experience_entries,  # Gemini → Regex → BERT
            'education_entries'  : education_entries,   # Gemini → Regex → BERT
            'all_entities'       : grouped,
        }

    def _first(self, items):
        if not items:
            return ''
        return max(items, key=lambda x: x['confidence'])['text']

    def _chunk_text(self, text, max_chars=1800, overlap_chars=200):
        if len(text) <= max_chars:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap_chars
        return chunks


# =============================================================================
# Gemini Extraction  (Fallback when BERT unavailable)
# =============================================================================

def extract_resume_with_gemini(raw_text: str) -> dict:
    """
    Gemini 2.5 Flash Lite — extracts ONLY education and experience entries.
    Used exclusively inside ResumeNERExtractor.extract() for structured fields.

    Why only these two fields?
    - BERT NER handles name/skills/locations reliably (token-level DL)
    - Education and experience require document-level context understanding:
      e.g. correctly pairing degree + college + year even in messy layouts,
      distinguishing real jobs from student projects in experience.
    - Gemini reads the full document like a human and returns clean JSON.

    Falls back gracefully: if API unavailable → caller uses regex.
    """
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return {'education_entries': [], 'experience_entries': []}

    prompt = f"""You are a professional resume parser. Extract ONLY education and work experience from the resume below.

STRICT RULES:
- Return ONLY valid JSON. No markdown, no backticks, no explanation.
- experience_entries: ONLY paid jobs or formal internships at real companies.
  A valid experience entry has: a job title AND a company name AND a date range.
  Do NOT include: personal projects, academic projects, hackathons, training courses,
  certifications, freelance gigs, or any entry that does not have a real company name.
  Examples of INVALID entries: "@ GitHub Actions", "@ Flask", "House Price Prediction",
  "Python Training", "ML Project", anything starting with @ or a technology name.
  If the person is a student/fresher with no real paid job or internship, return [].
- education_entries: include ALL qualifications (B.Tech, M.Tech, 12th, 10th, MBA etc).
  Include the full college/university name exactly as written in the resume.

Return exactly this JSON structure:
{{
  "experience_entries": [],
  "education_entries": [
    {{
      "degree": "Bachelor of Technology in Computer Science",
      "college": "Chandigarh Engineering College - CGC Landran",
      "year": "2026",
      "confidence": 0.95
    }}
  ]
}}

RESUME TEXT:
{raw_text[:4000]}"""

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.1}
    }).encode()

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash-lite:generateContent?key={api_key}"
    )

    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data     = json.loads(resp.read())
            raw_resp  = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            raw_resp  = re.sub(r"```json|```", "", raw_resp).strip()
            # Extract JSON object boundaries robustly
            obj_start = raw_resp.find("{")
            obj_end   = raw_resp.rfind("}") + 1
            if obj_start >= 0 and obj_end > obj_start:
                raw_resp = raw_resp[obj_start:obj_end]
            result    = json.loads(raw_resp)

            edu = result.get('education_entries', [])
            raw_exp = result.get('experience_entries', [])

            # ── Post-process: filter out invalid experience entries ────────────
            # Gemini sometimes returns project technologies or tool names as
            # company names (e.g. "@ GitHub Actions", "@ Flask").
            # A valid experience entry must have:
            #   - company that does NOT start with @
            #   - company that is NOT a known technology/tool name
            #   - company length > 2 characters
            #   - either a designation OR a duration present
            # Filter using module-level _is_valid_company()
            exp = [
                e for e in raw_exp
                if _is_valid_company(
                    str(e.get('company', '')),
                    str(e.get('designation', '')),
                    str(e.get('duration', ''))
                )
            ]


            print(f"  Gemini edu/exp — edu: {len(edu)}, exp: {len(exp)} (filtered from {len(raw_exp)})")
            return {'education_entries': edu, 'experience_entries': exp}

    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore')
        if e.code == 429:
            if 'quota' in body.lower() or 'RESOURCE_EXHAUSTED' in body:
                print(f"  Gemini DAILY QUOTA exhausted — using regex for edu/exp")
            else:
                print(f"  Gemini rate limit — using regex for edu/exp")
        else:
            print(f"  Gemini HTTP {e.code} — using regex for edu/exp")
    except Exception as e:
        print(f"  Gemini edu/exp failed: {e} — using regex")

    return {'education_entries': [], 'experience_entries': []}


def _regex_fallback_extraction(raw_text: str) -> dict:
    """Pure regex — last resort when both BERT and Gemini unavailable."""
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
# Keyword Skill Extractor  (supplements BERT/Gemini)
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
    'XGBoost', 'LightGBM', 'OpenCV', 'NLTK', 'spaCy', 'FastText',
    'DVC', 'MLOps', 'SageMaker', 'Weights & Biases', 'Featuretools',
    'Power Automate', 'Power Apps', 'Tableau', 'Looker',
]


def extract_skills_by_keyword(text: str) -> list:
    found, seen = [], set()
    text_lower  = text.lower()
    for skill in _KNOWN_SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            key = skill.lower()
            if key not in seen and key not in _SKILL_BLOCKLIST:
                seen.add(key)
                found.append({'text': skill, 'confidence': 0.80})
    return found


def merge_skills(ner_skills: list, keyword_skills: list) -> list:
    """Merge BERT NER + keyword skills. BERT takes precedence."""
    merged = {}
    for s in ner_skills:
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
# Resume Scoring
# =============================================================================

def score_resume(text: str) -> dict:
    text_l   = text.lower()
    sections = {
        'Contact Info': 10, 'Summary/Objective': 8, 'Education': 15,
        'Experience': 20, 'Skills': 15, 'Projects': 12,
        'Certifications': 10, 'Achievements': 5, 'Hobbies/Interests': 5,
    }
    checks = {
        'Contact Info'     : any(k in text_l for k in ['email', 'phone', 'linkedin', 'github', 'contact']),
        'Summary/Objective': any(k in text_l for k in ['professional summary', 'career objective', 'objective', 'profile summary', 'about me']),
        'Education'        : any(k in text_l for k in ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'b.tech', 'b.e', 'intermediate', 'matriculation']),
        'Experience'       : (
            bool(re.search(r'\bwork\s+experience\b',         text_l)) or
            bool(re.search(r'\bemployment\s+history\b',      text_l)) or
            bool(re.search(r'\bprofessional\s+experience\b', text_l)) or
            bool(re.search(r'\binternship\b',                text_l)) or
            bool(re.search(r'\bindustrial\s+training\b',     text_l)) or
            bool(re.search(r'\btraining\s+&\s+experience\b', text_l))
        ),
        'Skills'           : any(k in text_l for k in ['technical skills', 'core competencies', 'key skills', 'skills', 'competencies']),
        'Projects'         : any(k in text_l for k in ['projects', 'portfolio', 'personal projects', 'academic projects']),
        'Certifications'   : any(k in text_l for k in ['certifications', 'certification', 'certificate', 'certified', 'completion badge', 'completion of']),
        'Achievements'     : any(k in text_l for k in ['achievements', 'award', 'honors', 'recognition', 'finalist', 'hackathon']),
        'Hobbies/Interests': any(k in text_l for k in ['hobbies', 'interests', 'activities', 'volunteer']),
    }
    total, breakdown = 0, {}
    for section, points in sections.items():
        got              = points if checks[section] else 0
        breakdown[section] = {'earned': got, 'max': points, 'present': checks[section]}
        total           += got
    return {'total': total, 'max': 100, 'breakdown': breakdown}


def detect_experience_level(text: str, pages: int) -> str:
    text_l    = text.lower()
    exp_years = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience', text_l)
    if exp_years:
        yrs = max(int(y) for y in exp_years)
        if yrs >= 5:   return 'Senior'
        elif yrs >= 2: return 'Mid-Level'
        else:          return 'Junior'
    if any(k in text_l for k in ['senior', 'lead ', 'principal', 'architect', 'head of']):
        return 'Senior'
    if any(k in text_l for k in ['internship', 'intern ', 'fresher', 'entry level',
                                   'trainee', 'aspiring', 'b.tech student', 'pursuing']):
        return 'Fresher/Intern'
    if re.search(r'\bwork\s+experience\b|\bprofessional\s+experience\b|\bemployment\b', text_l):
        return 'Experienced'
    if pages <= 1:
        return 'Fresher/Intern'
    return 'Mid-Level'


# =============================================================================
# Job Role Requirements
# =============================================================================

JOB_ROLE_REQUIREMENTS = {
    'Backend Developer'        : {'core': ['Python', 'Java', 'SQL', 'REST API', 'Git'],
                                   'preferred': ['Docker', 'Kubernetes', 'AWS', 'Redis', 'PostgreSQL', 'CI/CD', 'Linux', 'Node.js']},
    'Frontend Developer'       : {'core': ['JavaScript', 'React', 'HTML', 'CSS', 'Git'],
                                   'preferred': ['TypeScript', 'Vue', 'Angular', 'Webpack', 'Figma', 'REST API', 'Next.js']},
    'Full Stack Developer'     : {'core': ['JavaScript', 'Python', 'React', 'SQL', 'Git', 'REST API'],
                                   'preferred': ['Docker', 'Kubernetes', 'AWS', 'TypeScript', 'Node.js', 'CI/CD', 'MongoDB']},
    'Data Scientist'           : {'core': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'Data Analysis'],
                                   'preferred': ['Deep Learning', 'TensorFlow', 'PyTorch', 'NLP', 'Big Data', 'Data Visualization', 'R']},
    'ML Engineer'              : {'core': ['Python', 'Machine Learning', 'Docker', 'SQL', 'Git'],
                                   'preferred': ['Deep Learning', 'Kubernetes', 'AWS', 'TensorFlow', 'PyTorch', 'CI/CD', 'MLflow']},
    'Data Analyst'             : {'core': ['SQL', 'Excel', 'Data Analysis', 'Data Visualization', 'Statistics'],
                                   'preferred': ['Python', 'Tableau', 'Power BI', 'R', 'Machine Learning']},
    'DevOps Engineer'          : {'core': ['Docker', 'Kubernetes', 'CI/CD', 'Linux', 'Git'],
                                   'preferred': ['AWS', 'Azure', 'GCP', 'Terraform', 'Ansible', 'Python', 'Monitoring']},
    'Cloud Architect'          : {'core': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'],
                                   'preferred': ['Terraform', 'CI/CD', 'Linux', 'Networking', 'Security', 'Microservices']},
    'Mobile Developer'         : {'core': ['Android', 'iOS', 'Git', 'REST API'],
                                   'preferred': ['Flutter', 'React Native', 'Kotlin', 'Swift', 'Firebase', 'Java']},
    'Cybersecurity Analyst'    : {'core': ['Cybersecurity', 'Linux', 'Networking', 'Python'],
                                   'preferred': ['Penetration Testing', 'SIEM', 'Firewalls', 'Incident Response', 'Cloud Security']},
    'Java Developer'           : {'core': ['Java', 'Spring', 'SQL', 'Git'],
                                   'preferred': ['Docker', 'Maven', 'Hibernate', 'REST API', 'Microservices', 'Kubernetes']},
    'Python Developer'         : {'core': ['Python', 'SQL', 'Git', 'REST API'],
                                   'preferred': ['Django', 'Flask', 'FastAPI', 'Docker', 'AWS', 'PostgreSQL']},
    'React Developer'          : {'core': ['React', 'JavaScript', 'TypeScript', 'Git'],
                                   'preferred': ['Node.js', 'REST API', 'Redux', 'Next.js', 'CSS', 'GraphQL']},
    'Data Engineer'            : {'core': ['Python', 'SQL', 'Big Data', 'ETL', 'Git'],
                                   'preferred': ['Spark', 'Hadoop', 'AWS', 'Kafka', 'Docker', 'Airflow']},
    'Business Analyst'         : {'core': ['SQL', 'Excel', 'Data Analysis', 'Communication'],
                                   'preferred': ['Tableau', 'Power BI', 'Statistics', 'Agile', 'JIRA', 'Requirements Gathering']},
    'UI/UX Designer'           : {'core': ['Figma', 'Adobe XD', 'User Research', 'Wireframing'],
                                   'preferred': ['CSS', 'HTML', 'Prototyping', 'Design Systems', 'Accessibility']},
    'Network Engineer'         : {'core': ['Networking', 'Linux', 'Cisco', 'Firewalls'],
                                   'preferred': ['AWS', 'Python', 'Cybersecurity', 'Monitoring', 'VPN', 'DNS']},
    'Blockchain Developer'     : {'core': ['Solidity', 'Ethereum', 'Smart Contracts', 'JavaScript'],
                                   'preferred': ['Web3.js', 'Truffle', 'DeFi', 'NFT', 'Rust', 'Python']},
    'AI/NLP Engineer'          : {'core': ['Python', 'NLP', 'Deep Learning', 'Machine Learning'],
                                   'preferred': ['TensorFlow', 'PyTorch', 'Transformers', 'Hugging Face', 'spaCy', 'Docker']},
    'DevOps Engineer'          : {'core': ['Docker', 'Kubernetes', 'CI/CD', 'Linux', 'Git'],
                                   'preferred': ['AWS', 'Azure', 'GCP', 'Terraform', 'Ansible', 'Jenkins', 'Monitoring']},
    'ETL Developer'            : {'core': ['Python', 'SQL', 'ETL', 'Data Warehouse', 'Git'],
                                   'preferred': ['Informatica', 'Talend', 'SSIS', 'Spark', 'Airflow', 'AWS']},
    'DotNet Developer'         : {'core': ['C#', '.NET', 'ASP.NET', 'SQL', 'Git'],
                                   'preferred': ['Azure', 'Entity Framework', 'WPF', 'REST API', 'Docker']},
    'SAP Developer'            : {'core': ['SAP ABAP', 'SAP', 'SQL', 'Git'],
                                   'preferred': ['SAP HANA', 'SAP Fiori', 'SAP BW', 'SAP MM', 'SAP SD']},
    'Network Security Engineer': {'core': ['Networking', 'Cybersecurity', 'Linux', 'Firewalls'],
                                   'preferred': ['Penetration Testing', 'SIEM', 'Cisco', 'VPN', 'CISSP', 'Python']},
    'Testing'                  : {'core': ['Selenium', 'Python', 'SQL', 'Git'],
                                   'preferred': ['JMeter', 'Cypress', 'Postman', 'JIRA', 'REST API', 'Docker']},
}

_CATEGORY_ROLE_MAP = {
    'data science'             : ['Data Scientist', 'ML Engineer', 'AI/NLP Engineer'],
    'python developer'         : ['Python Developer', 'Backend Developer', 'ML Engineer'],
    'java developer'           : ['Java Developer', 'Backend Developer'],
    'react developer'          : ['React Developer', 'Frontend Developer'],
    'devops engineer'          : ['DevOps Engineer', 'Cloud Architect'],
    'database'                 : ['Data Engineer', 'Backend Developer', 'Data Analyst'],
    'business analyst'         : ['Business Analyst', 'Data Analyst'],
    'full stack developer'     : ['Full Stack Developer', 'Backend Developer', 'React Developer'],
    'information technology'   : ['Backend Developer', 'DevOps Engineer'],
    'testing'                  : ['Testing', 'Backend Developer'],
    'etl developer'            : ['ETL Developer', 'Data Engineer'],
    'network security engineer': ['Network Security Engineer', 'Cybersecurity Analyst'],
}


# Skill aliases — DIRECT tool equivalents only
# Only maps specific tools to their canonical name
# Does NOT map generic concepts (pandas→data analysis causes Data Analyst over-matching)
_SKILL_ALIASES = {
    # SQL variants — these ARE sql, just different engines
    'mysql'               : 'sql',
    'postgresql'          : 'sql',
    'sqlite'              : 'sql',
    'oracle'              : 'sql',
    'microsoft sql server': 'sql',
    'sql server'          : 'sql',
    # Git hosting — these require git knowledge
    'github'              : 'git',
    'gitlab'              : 'git',
    # Spring variants
    'spring boot'         : 'spring',
    'spring mvc'          : 'spring',
    # ML framework aliases
    'keras'               : 'tensorflow',
    # Cloud sub-services
    'ec2'                 : 'aws',
    's3'                  : 'aws',
    'sagemaker'           : 'aws',
    'lambda'              : 'aws',
}


def compute_job_matches(extracted_skills: list, top_n: int = 5,
                        predicted_category: str = '') -> list:
    skill_names_lower = set()
    for s in extracted_skills:
        skill_names_lower.add((s['text'] if isinstance(s, dict) else str(s)).lower().strip())

    # Expand with aliases — so 'mysql' also satisfies 'sql' requirement etc.
    expanded_skills = set(skill_names_lower)
    for skill in skill_names_lower:
        if skill in _SKILL_ALIASES:
            expanded_skills.add(_SKILL_ALIASES[skill])
    skill_names_lower = expanded_skills

    matches = []
    for role, reqs in JOB_ROLE_REQUIREMENTS.items():
        core, pref = reqs['core'], reqs['preferred']
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

    # Category boost — +8% for roles aligned with ML prediction
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
    if target_roles is None:
        top_matches  = compute_job_matches(extracted_skills, top_n=3,
                                           predicted_category=predicted_category)
        target_roles = [m['role'] for m in top_matches]

    skill_names_lower = set()
    for s in extracted_skills:
        skill_names_lower.add((s['text'] if isinstance(s, dict) else str(s)).lower().strip())

    # Expand with aliases — consistent with compute_job_matches
    expanded = set(skill_names_lower)
    for skill in skill_names_lower:
        if skill in _SKILL_ALIASES:
            expanded.add(_SKILL_ALIASES[skill])
    skill_names_lower = expanded

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
        gaps.append({'role': role, 'missing_skills': missing,
                     'missing_core_skills': missing_core, 'message': msg})
    return gaps


# =============================================================================
# Project Ideas  (Gemini API with curated fallback)
# =============================================================================

def get_project_ideas(extracted_skills: list, max_projects: int = 4,
                      experience_level: str = '') -> list:
    skill_names = [(s['text'] if isinstance(s, dict) else str(s)) for s in extracted_skills]
    api_key     = os.environ.get('GEMINI_API_KEY', '')

    if api_key and skill_names:
        level_lower = experience_level.lower()
        if 'fresher' in level_lower or 'intern' in level_lower:
            diff_guide = "Range from Beginner to Intermediate only. Do NOT suggest Advanced projects."
        elif 'senior' in level_lower:
            diff_guide = "Range from Intermediate to Advanced difficulty."
        else:
            diff_guide = "Include a mix of Beginner, Intermediate, and Advanced difficulty."

        try:
            # Simplified prompt — fewer fields = less chance of malformed JSON
            prompt = (
                f"Generate exactly {max_projects} project ideas for a developer "
                f"with these skills: {', '.join(skill_names[:12])}.\n"
                f"Experience level: {experience_level or 'Unknown'}.\n"
                f"{diff_guide}\n\n"
                f"Return ONLY a valid JSON array. No markdown, no explanation, no extra text.\n"
                f"Each object must have exactly these fields:\n"
                f"name (string), description (1 sentence max 20 words), "
                f"matched_skills (array of 2-3 strings), "
                f"difficulty (one of: Beginner, Intermediate, Advanced), "
                f"relevance (number 0-100).\n"
                f"Example: "
                f'[{{"name":"Price Predictor","description":"ML model predicting house prices using regression.",'
                f'"matched_skills":["Python","Machine Learning"],"difficulty":"Intermediate","relevance":95.0}}]'
            )
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 800,
                    "temperature"    : 0.5,    # lower = more consistent JSON
                    "responseMimeType": "application/json",   # force JSON output
                }
            }).encode()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
            req = urllib.request.Request(url, data=payload,
                                         headers={"content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data     = json.loads(resp.read())
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw_text = re.sub(r"```json|```", "", raw_text).strip()
                # Extract [ ... ] array robustly
                arr_start = raw_text.find("[")
                arr_end   = raw_text.rfind("]") + 1
                if arr_start >= 0 and arr_end > arr_start:
                    raw_text = raw_text[arr_start:arr_end]
                projects = json.loads(raw_text)
                if isinstance(projects, list) and projects:
                    # Normalise — ensure all_skills field exists
                    for p in projects:
                        if 'all_skills' not in p:
                            p['all_skills'] = p.get('matched_skills', [])
                    return projects[:max_projects]
        except Exception as e:
            print(f"  Gemini project ideas failed: {e} — using fallback")

    FALLBACK = [
        {"name": "Student Grade Tracker",    "description": "Web app to track grades, calculate GPA, and visualize performance over time.", "matched_skills": ["Python", "Flask", "MySQL"], "all_skills": ["Python", "Flask", "MySQL", "Excel"], "difficulty": "Beginner",      "relevance": 100.0},
        {"name": "Heart Disease Predictor",  "description": "ML model predicting heart disease risk with a clean Flask interface.", "matched_skills": ["Python", "Machine Learning", "Flask"], "all_skills": ["Python", "Machine Learning", "Flask", "Pandas", "NumPy"], "difficulty": "Intermediate", "relevance": 100.0},
        {"name": "AI Resume Analyzer",       "description": "NLP system parsing resumes, extracting skills, recommending jobs using ML classifiers.", "matched_skills": ["Python", "Machine Learning", "NLP"], "all_skills": ["Python", "Machine Learning", "NLP", "Flask"], "difficulty": "Advanced",     "relevance": 100.0},
        {"name": "Real-Time Chat App",       "description": "WebSocket-based chat with rooms, message history, and user authentication.", "matched_skills": ["Python", "Flask"], "all_skills": ["Python", "Flask", "JavaScript", "SQL"], "difficulty": "Intermediate", "relevance": 90.0},
        {"name": "Expense Tracker",          "description": "Personal finance app to log expenses, categorize spending, visualize monthly budgets.", "matched_skills": ["Python", "MySQL"], "all_skills": ["Python", "MySQL", "Excel"], "difficulty": "Beginner",      "relevance": 85.0},
        {"name": "ML Model Dashboard",       "description": "Deploy and monitor ML models with a Flask dashboard showing predictions and confidence.", "matched_skills": ["Python", "Machine Learning"], "all_skills": ["Python", "Machine Learning", "Docker", "Flask"], "difficulty": "Advanced",     "relevance": 80.0},
        {"name": "E-Commerce REST API",      "description": "Full REST API with JWT auth, product catalog, cart, orders, and payment integration.", "matched_skills": ["Python", "SQL", "REST API"], "all_skills": ["Python", "SQL", "REST API", "Docker"], "difficulty": "Intermediate", "relevance": 85.0},
        {"name": "Fraud Detection System",   "description": "Real-time fraud detection using ensemble ML with an interactive analytics dashboard.", "matched_skills": ["Python", "Machine Learning"], "all_skills": ["Python", "Machine Learning", "SQL", "Kafka"], "difficulty": "Advanced",     "relevance": 85.0},
    ]
    skill_lower = {(s['text'] if isinstance(s, dict) else str(s)).lower() for s in extracted_skills}
    scored = []
    for p in FALLBACK:
        matched = [s for s in p["all_skills"] if s.lower() in skill_lower]
        if len(matched) >= len(p["all_skills"]) * 0.4:
            p = dict(p)
            p["matched_skills"] = matched
            p["relevance"]      = round(len(matched) / len(p["all_skills"]) * 100, 1)
            scored.append(p)
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:max_projects] if scored else FALLBACK[:max_projects]