/* ── main.js — AI Resume Analyzer (BERT + NER) ──────────────── */

const uploadZone  = document.getElementById('upload-zone');
const fileInput   = document.getElementById('file-input');
const fileInfo    = document.getElementById('file-info');
const errorBox    = document.getElementById('error-box');
const loader      = document.getElementById('loader');
const results     = document.getElementById('results');
const resetBtn    = document.getElementById('reset-btn');
const demoBtn     = document.getElementById('demo-btn');

let scoreChart = null;

// ── Drag & Drop ──────────────────────────────────────────────────────────────
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave',() => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// ── Demo ──────────────────────────────────────────────────────────────────────
demoBtn.addEventListener('click', async () => {
  showLoading();
  try {
    const res  = await fetch('/api/demo');
    const data = await res.json();
    const meta = await fetchModelInfo();
    renderResults(data, meta);
  } catch {
    showError('Could not load demo. Is Flask running?');
  }
});

// ── Reset ─────────────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  results.style.display  = 'none';
  loader.style.display   = 'none';
  document.querySelector('.hero').style.display = 'block';
  errorBox.style.display = 'none';
  fileInput.value = '';
  fileInfo.style.display = 'none';
  fileInfo.textContent = '';
  if (scoreChart) { scoreChart.destroy(); scoreChart = null; }
});

// ── Handle File ───────────────────────────────────────────────────────────────
function handleFile(file) {
  if (file.type !== 'application/pdf') { showError('Please upload a PDF file.'); return; }
  if (file.size > 10 * 1024 * 1024)   { showError('File too large. Max 10 MB.'); return; }

  fileInfo.textContent = `\u{1F4CE} ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  fileInfo.style.display = 'block';
  analyzeResume(file);
}

async function analyzeResume(file) {
  showLoading();
  const form = new FormData();
  form.append('resume', file);
  try {
    const [res, meta] = await Promise.all([
      fetch('/api/analyze', { method: 'POST', body: form }),
      fetchModelInfo()
    ]);
    const data = await res.json();
    if (data.status !== 'success') { showError(data.message || 'Analysis failed.'); return; }
    renderResults(data, meta);
  } catch (err) {
    showError('Server error: ' + err.message);
  }
}

async function fetchModelInfo() {
  try {
    const r = await fetch('/api/model-info');
    return await r.json();
  } catch { return {}; }
}

// ── UI States ─────────────────────────────────────────────────────────────────
function showLoading() {
  document.querySelector('.hero').style.display = 'none';
  errorBox.style.display = 'none';
  loader.style.display   = 'block';
  results.style.display  = 'none';
}
function showError(msg) {
  loader.style.display        = 'none';
  document.querySelector('.hero').style.display = 'block';
  errorBox.style.display      = 'block';
  errorBox.textContent        = '\u26A0 ' + msg;
}

// ── Render All Results ────────────────────────────────────────────────────────
function renderResults(data, meta = {}) {
  loader.style.display  = 'none';
  results.style.display = 'block';

  renderModelBanner(meta, data);
  renderStats(data);
  renderProfile(data);
  renderCategory(data);
  renderScore(data);
  renderPredictions(data);
  renderSkills(data);
  renderNerEntities(data);
  renderJobMatches(data);
  renderSkillGaps(data);
  renderProjectIdeas(data);
  renderCourses(data);
  renderPdfPreview(data);

  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Model Banner ─────────────────────────────────────────────────────────────
function renderModelBanner(meta, data) {
  const el = document.getElementById('model-banner');
  const navBadge = document.getElementById('nav-badge');

  if (!meta || meta.status === 'error') {
    el.innerHTML = `<div class="model-banner-left"><span class="model-banner-icon">\u{1F916}</span>
      <div><div class="model-banner-name">AI Resume Analyzer</div>
      <div class="model-banner-sub">Run the Jupyter notebook to train models</div></div></div>`;
    return;
  }

  const mode = meta.model_mode || 'ml';
  const isBert = mode === 'bert';
  const hasNer = meta.ner_available;

  // Update nav badge
  if (navBadge) {
    navBadge.textContent = isBert ? 'BERT + NER Powered' : 'ML + NER Powered';
  }

  const modeName = isBert ? 'BERT Domain Classifier' : (meta.model_name || 'ML Model');
  const acc = isBert ? (meta.bert_accuracy || meta.accuracy || '\u2014') : (meta.accuracy || '\u2014');
  const f1 = isBert ? (meta.bert_f1 || meta.f1_weighted || '\u2014') : (meta.f1_weighted || '\u2014');

  el.innerHTML = `
    <div class="model-banner-left">
      <span class="model-banner-icon">\u{1F9E0}</span>
      <div>
        <div class="model-banner-name">${modeName}</div>
        <div class="model-banner-sub">${meta.num_classes || 24} resume categories \u00B7 ${(meta.train_samples || 0).toLocaleString()} training samples</div>
      </div>
    </div>
    <div class="model-pills">
      ${isBert ? '<span class="model-pill" style="background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.2);color:var(--amber)">BERT</span>' : ''}
      ${hasNer ? '<span class="model-pill" style="background:rgba(168,85,247,.1);border-color:rgba(168,85,247,.2);color:var(--violet2)">NER</span>' : ''}
      <span class="model-pill">Acc: ${acc}%</span>
      <span class="model-pill" style="background:rgba(6,182,212,.1);border-color:rgba(6,182,212,.2);color:var(--cyan)">F1: ${f1}%</span>
      <span class="model-pill" style="background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.2);color:var(--emerald)">CV: ${meta.cv_mean || '\u2014'}%</span>
    </div>`;
}

// ── Stats Row ─────────────────────────────────────────────────────────────────
function renderStats(data) {
  const e = data.extracted  || {};
  const s = data.score      || {};
  const jobMatches = data.job_matches || [];
  const topMatch = jobMatches.length > 0 ? jobMatches[0].match_pct : 0;

  const stats = [
    { val: s.total || 0,               label: 'Resume Score',     color: scoreColor(s.total), unit: '/100' },
    { val: data.skills?.current?.length || 0, label: 'Skills Detected', color: 'var(--violet2)', unit: '' },
    { val: Math.round(topMatch),       label: 'Best Job Match',   color: 'var(--emerald)',    unit: '%' },
    { val: e.pages || 0,               label: 'Pages',            color: 'var(--amber)',      unit: '' },
  ];

  document.getElementById('stat-grid').innerHTML = stats.map(s => `
    <div class="stat-card">
      <div class="stat-val" style="color:${s.color}">${typeof s.val === 'number' ? s.val.toLocaleString() : s.val}<span style="font-size:.9rem;color:var(--txt2)">${s.unit}</span></div>
      <div class="stat-label">${s.label}</div>
    </div>`).join('');
}

// ── Profile Card ──────────────────────────────────────────────────────────────
function renderProfile(data) {
  const e   = data.extracted  || {};
  const p   = data.prediction || {};
  const ner = data.ner_entities || {};
  const lvl = p.experience_level || 'Unknown';
  const initials = (e.name || 'RR').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  const badgeClass = lvl.includes('Senior') ? 'badge-senior' : lvl.includes('Mid') ? 'badge-mid' : lvl.includes('Junior') ? 'badge-junior' : 'badge-fresher';

  const location = ner.locations && ner.locations.length > 0
    ? (ner.locations[0].text || ner.locations[0]) : '';

  document.getElementById('card-profile').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>Candidate Profile</div>
    <div class="profile-avatar">${initials}</div>
    <div class="profile-name">${e.name || 'Name not detected'}</div>
    <div class="profile-meta">\u{1F4E7} ${e.email || 'Email not found'}</div>
    <div class="profile-meta">\u{1F4DE} ${e.phone || 'Phone not found'}</div>
    ${location ? `<div class="profile-meta">\u{1F4CD} ${location}</div>` : ''}
    <div class="profile-meta">\u{1F4C4} ${e.pages || 1} page${e.pages !== 1 ? 's' : ''} \u00B7 ${(e.word_count || 0).toLocaleString()} words</div>
    <span class="badge-level ${badgeClass}">\u2B21 ${lvl}</span>
  `;
}

// ── Category Card ─────────────────────────────────────────────────────────────
function renderCategory(data) {
  const p    = data.prediction || {};
  const top  = p.top_predictions?.[0] || {};
  const conf = top.confidence || 0;
  const mode = p.model_used || 'ml';
  const isBert = mode === 'bert';

  const modeLabel = isBert
    ? 'Classified by fine-tuned BERT model on resume text.'
    : 'Based on TF-IDF feature extraction and the best-performing ML classifier.';

  document.getElementById('card-category').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--violet2)"></span>${isBert ? 'BERT' : 'ML'} Prediction</div>
    <div class="category-name">${p.category || '\u2014'}</div>
    <div class="category-sub">Predicted job category \u00B7 ${conf.toFixed(1)}% confidence</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin-top:8px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="font-size:.82rem;color:var(--txt2)">Model Confidence</span>
        <span style="font-size:.82rem;font-weight:700;color:var(--violet2)">${conf.toFixed(1)}%</span>
      </div>
      <div class="pred-bar-bg"><div class="pred-bar-fill" style="width:0%;background:linear-gradient(90deg,var(--violet),var(--violet2))" data-target="${conf}"></div></div>
    </div>
    <div style="margin-top:16px;font-size:.82rem;color:var(--txt2);line-height:1.5;">
      ${modeLabel}
    </div>`;
  animateBars();
}

// ── Score Card ────────────────────────────────────────────────────────────────
function renderScore(data) {
  const s   = data.score || { total: 0, max: 100, breakdown: {} };

  const breakdownHtml = Object.entries(s.breakdown || {}).map(([name, info]) => {
    const pct2 = info.max > 0 ? Math.round((info.earned / info.max) * 100) : 0;
    const color = info.present ? 'var(--emerald)' : 'var(--txt3)';
    return `
      <div class="score-row">
        <span class="score-row-label" title="${name}">${name}</span>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width:0%;background:${color}" data-target="${pct2}"></div>
        </div>
        <span class="score-row-val" style="color:${color}">${info.earned}/${info.max}</span>
      </div>`;
  }).join('');

  document.getElementById('card-score').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--emerald)"></span>Resume Strength</div>
    <div class="score-wrap">
      <div class="score-ring-container">
        <canvas id="score-ring" width="140" height="140"></canvas>
        <div class="score-ring-label">
          <span class="score-ring-num" style="color:${scoreColor(s.total)}">${s.total}</span>
          <span class="score-ring-max">/ ${s.max}</span>
        </div>
      </div>
      <div class="score-sections">${breakdownHtml}</div>
    </div>`;

  renderRingChart(s.total, s.max);
  animateBars();
}

function renderRingChart(score, max) {
  const ctx = document.getElementById('score-ring').getContext('2d');
  if (scoreChart) scoreChart.destroy();
  const clr = scoreColor(score);
  scoreChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score, max - score],
        backgroundColor: [clr, 'rgba(255,255,255,.05)'],
        borderWidth: 0, borderRadius: 4,
      }]
    },
    options: {
      cutout: '78%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1200, easing: 'easeOutQuart' },
    }
  });
}

function scoreColor(score) {
  if (score >= 80) return 'var(--emerald)';
  if (score >= 55) return 'var(--cyan)';
  if (score >= 35) return 'var(--amber)';
  return 'var(--rose)';
}

// ── Top Predictions ───────────────────────────────────────────────────────────
function renderPredictions(data) {
  const preds = data.prediction?.top_predictions || [];
  const colors = [
    'linear-gradient(90deg,var(--violet),var(--violet2))',
    'linear-gradient(90deg,var(--cyan),#38bdf8)',
    'linear-gradient(90deg,var(--emerald),#34d399)',
    'linear-gradient(90deg,var(--amber),#fcd34d)',
    'linear-gradient(90deg,var(--rose),#fb7185)',
  ];

  const html = preds.map((p, i) => `
    <div class="pred-row">
      <div class="pred-meta">
        <span class="pred-label">${i === 0 ? '\u{1F3C6} ' : ''}${p.label}</span>
        <span class="pred-pct">${p.confidence.toFixed(1)}%</span>
      </div>
      <div class="pred-bar-bg">
        <div class="pred-bar-fill" style="width:0%;background:${colors[i % colors.length]}" data-target="${p.confidence}"></div>
      </div>
    </div>`).join('');

  document.getElementById('card-predictions').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--amber)"></span>Top Category Predictions</div>
    <div class="pred-list">${html || '<div class="empty-state">No predictions available</div>'}</div>`;
  animateBars();
}

// ── Skills Card (NER-extracted with confidence) ──────────────────────────────
function renderSkills(data) {
  const current    = data.skills?.current || [];
  const withConf   = data.skills?.current_with_confidence || [];
  const skillGaps  = data.skill_gaps || [];

  // Collect all missing skills from gaps for the "recommended" section
  const missingSet = new Set();
  skillGaps.forEach(gap => {
    (gap.missing_core_skills || []).forEach(s => missingSet.add(s));
    (gap.missing_skills || []).slice(0, 3).forEach(s => missingSet.add(s));
  });
  const recommended = [...missingSet].slice(0, 12);

  // Show skills with confidence if available
  let currentHtml = '';
  if (withConf.length > 0) {
    currentHtml = withConf.map((s, i) => {
      const text = typeof s === 'object' ? s.text : s;
      const conf = typeof s === 'object' && s.confidence ? ` (${(s.confidence * 100).toFixed(0)}%)` : '';
      return `<span class="skill-tag current" style="animation-delay:${i * 0.04}s" title="Confidence${conf}">${text}</span>`;
    }).join('');
  } else if (current.length > 0) {
    currentHtml = current.map((s, i) => `<span class="skill-tag current" style="animation-delay:${i * 0.04}s">${s}</span>`).join('');
  } else {
    currentHtml = '<span style="color:var(--txt2);font-size:.85rem">No skills detected</span>';
  }

  const recHtml = recommended.length > 0
    ? recommended.map((s, i) => `<span class="skill-tag missing" style="animation-delay:${i * 0.04}s">+ ${s}</span>`).join('')
    : '<span style="color:var(--txt2);font-size:.85rem">No skill gaps identified \u2014 your profile is strong!</span>';

  document.getElementById('card-skills').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--rose)"></span>Skills Analysis</div>
    <div style="margin-bottom:20px;">
      <div style="font-size:.82rem;font-weight:600;color:var(--txt2);margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em">NER-Extracted Skills (${current.length})</div>
      <div class="skills-grid">${currentHtml}</div>
    </div>
    <div>
      <div style="font-size:.82rem;font-weight:600;color:var(--txt2);margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em">Skills to Develop (${recommended.length})</div>
      <div class="skills-grid">${recHtml}</div>
      ${recommended.length ? `<p style="margin-top:12px;font-size:.8rem;color:var(--txt2)">\u{1F4A1} Adding these skills can boost your match for top roles in <strong style="color:var(--violet2)">${data.prediction?.category || 'your field'}</strong>.</p>` : ''}
    </div>`;
}

// ── NER Extracted Entities ───────────────────────────────────────────────────
function renderNerEntities(data) {
  const ner = data.ner_entities || {};
  const el  = document.getElementById('card-ner-entities');

  const entitySections = [
    { label: 'Companies',        items: ner.companies || [],         icon: '\u{1F3E2}' },
    { label: 'Designations',     items: ner.designations || [],      icon: '\u{1F4BC}' },
    { label: 'Degrees',          items: ner.degrees || [],           icon: '\u{1F393}' },
    { label: 'Colleges',         items: ner.college_names || [],     icon: '\u{1F3EB}' },
    { label: 'Graduation Years', items: ner.graduation_years || [],  icon: '\u{1F4C5}' },
    { label: 'Locations',        items: ner.locations || [],         icon: '\u{1F4CD}' },
  ];

  const yoe = ner.years_of_experience || '';

  // Filter only sections that have data
  const populated = entitySections.filter(s => s.items.length > 0);

  if (populated.length === 0 && !yoe) {
    el.innerHTML = `
      <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>NER Extracted Details</div>
      <div class="empty-state">No entities extracted by NER model.</div>`;
    return;
  }

  const rowsHtml = populated.map(section => {
    const values = section.items.map(item => {
      const text = typeof item === 'object' ? item.text : item;
      const conf = typeof item === 'object' && item.confidence
        ? `<span class="ner-confidence">(${(item.confidence * 100).toFixed(0)}%)</span>` : '';
      return `${escapeHtml(text)}${conf}`;
    }).join(', ');

    return `<tr>
      <td>${section.icon} ${section.label}</td>
      <td>${values}</td>
    </tr>`;
  }).join('');

  const yoeRow = yoe
    ? `<tr><td>\u{23F3} Experience</td><td>${escapeHtml(yoe)}</td></tr>` : '';

  el.innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>NER Extracted Details</div>
    <table class="ner-table">
      <thead><tr><th>Entity Type</th><th>Extracted Values</th></tr></thead>
      <tbody>${rowsHtml}${yoeRow}</tbody>
    </table>`;
}

// ── Job Role Matches ─────────────────────────────────────────────────────────
function renderJobMatches(data) {
  const matches = data.job_matches || [];
  const el = document.getElementById('card-job-matches');

  if (matches.length === 0) {
    el.innerHTML = `
      <div class="card-title"><span class="dot" style="background:var(--emerald)"></span>Job Role Match</div>
      <div class="empty-state">Upload a resume to see job role matching.</div>`;
    return;
  }

  const matchColors = [
    'linear-gradient(90deg, var(--emerald), #34d399)',
    'linear-gradient(90deg, var(--cyan), #38bdf8)',
    'linear-gradient(90deg, var(--violet), var(--violet2))',
    'linear-gradient(90deg, var(--amber), #fcd34d)',
    'linear-gradient(90deg, var(--rose), #fb7185)',
  ];

  const html = matches.map((m, i) => {
    const matchedCount = m.matched_skills ? m.matched_skills.length : 0;
    const totalReq     = m.total_required || (matchedCount + (m.missing_core || []).length + (m.missing_preferred || []).length);
    const desc = m.missing_core && m.missing_core.length > 0
      ? `Missing core: ${m.missing_core.join(', ')}`
      : `${matchedCount}/${totalReq} skills matched`;

    return `
      <div class="job-match-row">
        <div class="job-match-header">
          <span class="job-match-role">${i === 0 ? '\u{1F3AF} ' : ''}${m.role}</span>
          <span class="job-match-pct" style="color:${m.match_pct >= 70 ? 'var(--emerald)' : m.match_pct >= 40 ? 'var(--cyan)' : 'var(--rose)'}">${m.match_pct.toFixed(1)}%</span>
        </div>
        <div class="job-match-bar-bg">
          <div class="job-match-bar-fill" style="width:0%;background:${matchColors[i % matchColors.length]}" data-target="${m.match_pct}"></div>
        </div>
        <div class="job-match-desc">${desc}</div>
      </div>`;
  }).join('');

  el.innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--emerald)"></span>Job Role Match</div>
    ${html}`;
  animateBars();
}

// ── Skill Gaps ───────────────────────────────────────────────────────────────
function renderSkillGaps(data) {
  const gaps = data.skill_gaps || [];
  const el   = document.getElementById('card-skill-gaps');

  if (gaps.length === 0) {
    el.innerHTML = `
      <div class="card-title"><span class="dot" style="background:var(--rose)"></span>Skill Gaps</div>
      <div class="empty-state">No skill gaps identified.</div>`;
    return;
  }

  const html = gaps.map(gap => {
    const coreHtml = (gap.missing_core_skills || []).map(s =>
      `<span class="skill-tag missing">${s}</span>`
    ).join('');

    const prefHtml = (gap.missing_skills || []).filter(s =>
      !(gap.missing_core_skills || []).includes(s)
    ).slice(0, 5).map(s =>
      `<span class="skill-tag recommended">${s}</span>`
    ).join('');

    return `
      <div class="gap-section">
        <div class="gap-role-title">${gap.role}</div>
        <div class="gap-message">${escapeHtml(gap.message)}</div>
        ${coreHtml || prefHtml ? `<div class="skills-grid">${coreHtml}${prefHtml}</div>` : ''}
      </div>`;
  }).join('');

  el.innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--rose)"></span>Skill Gaps</div>
    ${html}`;
}

// ── Project Ideas ────────────────────────────────────────────────────────────
function renderProjectIdeas(data) {
  const projects = data.project_ideas || [];
  const el = document.getElementById('card-project-ideas');

  if (projects.length === 0) {
    el.innerHTML = `
      <div class="card-title"><span class="dot" style="background:var(--amber)"></span>Project Ideas</div>
      <div class="empty-state">No project ideas matched your skills.</div>`;
    return;
  }

  const html = projects.map(proj => {
    const matchedLower = new Set((proj.matched_skills || []).map(s => s.toLowerCase()));
    const skillTags = (proj.all_skills || []).map(s => {
      const isMatched = matchedLower.has(s.toLowerCase());
      return `<span class="project-skill-tag${isMatched ? '' : ' dim'}">${s}</span>`;
    }).join('');

    return `
      <div class="project-card">
        <div class="project-name">
          <span>${proj.name}</span>
          <span class="project-difficulty">${proj.difficulty || ''}</span>
        </div>
        <div class="project-desc">${escapeHtml(proj.description)}</div>
        <div class="project-skills">${skillTags}</div>
        ${proj.relevance != null ? `<div style="margin-top:8px;font-size:.72rem;color:var(--txt3)">Skill match: ${proj.relevance}%</div>` : ''}
      </div>`;
  }).join('');

  el.innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--amber)"></span>Project Ideas</div>
    ${html}`;
}

// ── Courses Card ─────────────────────────────────────────────────────────────
function renderCourses(data) {
  const courses = data.courses || [];
  const html = courses.map(c => `
    <div class="course-item">
      <div>
        <span class="course-name">\u{1F393} ${escapeHtml(c.name)}</span>
        ${c.reason ? `<div style="font-size:.72rem;color:var(--txt3);margin-top:2px">${escapeHtml(c.reason)}</div>` : ''}
      </div>
      <a class="course-link" href="${c.url}" target="_blank" rel="noopener">Enroll \u2192</a>
    </div>`).join('');

  document.getElementById('card-courses').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>Recommended Courses</div>
    <div class="course-list">${html || '<div class="empty-state">No courses found for this profile.</div>'}</div>`;
}

// ── PDF Preview ──────────────────────────────────────────────────────────────
function renderPdfPreview(data) {
  const cardPdf = document.getElementById('card-pdf');
  const frame   = document.getElementById('pdf-frame');
  const toggle  = document.getElementById('pdf-toggle');

  if (data.pdf_preview) {
    cardPdf.style.display = 'block';
    const src = `data:application/pdf;base64,${data.pdf_preview}`;
    // Remove old listeners by cloning
    const newToggle = toggle.cloneNode(true);
    toggle.parentNode.replaceChild(newToggle, toggle);
    newToggle.addEventListener('click', () => {
      if (frame.style.display === 'none') {
        frame.src = src;
        frame.style.display = 'block';
        newToggle.textContent = 'Hide PDF Preview';
      } else {
        frame.style.display = 'none';
        newToggle.textContent = 'Show PDF Preview';
      }
    });
  } else {
    cardPdf.style.display = 'none';
  }
}

// ── Utilities ────────────────────────────────────────────────────────────────
function animateBars() {
  requestAnimationFrame(() => {
    document.querySelectorAll('[data-target]').forEach(bar => {
      const target = parseFloat(bar.dataset.target);
      setTimeout(() => { bar.style.width = Math.min(target, 100) + '%'; }, 100);
    });
  });
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
