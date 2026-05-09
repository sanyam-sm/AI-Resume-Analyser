"""
cover_letter.py — Cover Letter Generator using Groq API

Takes structured resume data + JD text → returns a professional cover letter.
Uses Groq (free tier) with llama-3.3-70b-versatile model.
"""

import os
import json
import re
import requests
from datetime import date

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL   = 'llama-3.3-70b-versatile'


def _build_prompt(resume_data: dict, jd_text: str) -> str:
    name             = resume_data.get('name', '')
    email            = resume_data.get('email', '')
    phone            = resume_data.get('phone', '')
    category         = resume_data.get('category', '')
    experience_level = resume_data.get('experience_level', '')
    skills           = resume_data.get('skills', [])
    exp_entries      = resume_data.get('experience_entries', [])
    edu_entries      = resume_data.get('education_entries', [])

    # Build readable experience summary
    exp_summary = ''
    if exp_entries:
        lines = []
        for e in exp_entries[:3]:
            parts = [e.get('designation', ''), e.get('company', ''), e.get('duration', '')]
            line  = ' at '.join([p for p in parts[:2] if p])
            if parts[2]:
                line += f' ({parts[2]})'
            if line:
                lines.append(line)
        exp_summary = '; '.join(lines)

    edu_summary = ''
    if edu_entries:
        e = edu_entries[0]
        parts = [e.get('degree', ''), e.get('college', ''), e.get('year', '')]
        edu_summary = ', '.join([p for p in parts if p])

    skills_str = ', '.join(skills[:15]) if skills else 'various technical skills'

    prompt = f"""You are a professional career coach writing a cover letter.

CANDIDATE INFORMATION:
- Name: {name or 'Candidate'}
- Email: {email or ''}
- Phone: {phone or ''}
- Domain/Field: {category or 'Technology'}
- Experience Level: {experience_level or 'Professional'}
- Key Skills: {skills_str}
- Work Experience: {exp_summary or 'Relevant professional experience'}
- Education: {edu_summary or 'Relevant educational background'}

JOB DESCRIPTION:
{jd_text[:2000]}

TASK:
Write a professional cover letter for this candidate applying to the above role.

STRICT RULES:
1. Keep it between 250-320 words total
2. Do NOT use any filler phrases like "I am excited", "I am thrilled", "I am passionate", "dynamic", "leverage", "synergy"
3. DO use concrete skills from the candidate's profile that match the JD
4. Use a confident, direct tone — not overly eager
5. Structure: opening (1 short paragraph) + body (2 paragraphs) + closing (1 short paragraph)
6. If you can extract a company name from the JD, use it. Otherwise use "your organization"
7. Do NOT mention the salary, benefits, or personal hardships
8. Do NOT start with "Dear Sir/Madam" — use "Dear Hiring Manager" if no name is available
9. Return ONLY the cover letter body text — no subject line, no meta commentary
10. Do NOT include the candidate's name/contact at top or bottom — we handle that separately

Write the cover letter now:"""

    return prompt


def generate_cover_letter(resume_data: dict, jd_text: str) -> dict:
    """
    Main entry point. Returns dict with:
      - success: bool
      - cover_letter: str (the generated text)
      - error: str (if failed)
    """
    if not GROQ_API_KEY:
        return {
            'success': False,
            'error': 'GROQ_API_KEY is not configured. Add it to your .env file.'
        }

    if not jd_text or not jd_text.strip():
        return {'success': False, 'error': 'Job description is required.'}

    prompt = _build_prompt(resume_data, jd_text)

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': GROQ_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 600,
                'temperature': 0.7,
            },
            timeout=30
        )

        if response.status_code != 200:
            err = response.json().get('error', {}).get('message', 'Groq API error')
            return {'success': False, 'error': f'Groq API error: {err}'}

        result       = response.json()
        cover_letter = result['choices'][0]['message']['content'].strip()

        return {
            'success'      : True,
            'cover_letter' : cover_letter,
            'candidate'    : {
                'name' : resume_data.get('name', ''),
                'email': resume_data.get('email', ''),
                'phone': resume_data.get('phone', ''),
            },
            'date': date.today().strftime('%B %d, %Y'),
        }

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out. Please try again.'}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Network error: {str(e)}'}
    except (KeyError, IndexError) as e:
        return {'success': False, 'error': f'Unexpected response format: {str(e)}'}
