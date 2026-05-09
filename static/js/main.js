/* ── main.js — AI Resume Analyzer ──────────────── */

const uploadZone  = document.getElementById('upload-zone');
const fileInput   = document.getElementById('file-input');
const fileInfo    = document.getElementById('file-info');
const errorBox    = document.getElementById('error-box');
const loader      = document.getElementById('loader');
const results     = document.getElementById('results');
const resetBtn    = document.getElementById('reset-btn');
const demoBtn     = document.getElementById('demo-btn');

let scoreChart = null;
let jobsState = { linkedin: [], indeed: [], processed: { linkedin: 0, indeed: 0 } };
let userLocation = { city: '', country: 'in' };

// Store cover letter data for PDF download
let _coverLetterData = null;

// City to nearby cities mapping
const NEARBY_CITIES = {
  'hyderabad': ['Hyderabad', 'Bangalore', 'Chennai', 'Visakhapatnam', 'Pune'],
  'bangalore': ['Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Mysore'],
  'chennai': ['Chennai', 'Bangalore', 'Hyderabad', 'Coimbatore', 'Pune'],
  'delhi': ['Delhi', 'Noida', 'Gurgaon', 'Faridabad', 'Jaipur', 'Greater Noida'],
  'noida': ['Noida', 'Delhi', 'Gurgaon', 'Faridabad', 'Greater Noida', 'Jaipur'],
  'gurgaon': ['Gurgaon', 'Delhi', 'Noida', 'Faridabad', 'Jaipur', 'Greater Noida'],
  'mumbai': ['Mumbai', 'Pune', 'Ahmedabad', 'Surat', 'Nagpur'],
  'pune': ['Pune', 'Mumbai', 'Bangalore', 'Hyderabad', 'Nashik', 'Aurangabad'],
  'ahmedabad': ['Ahmedabad', 'Surat', 'Vadodara', 'Mumbai', 'Rajkot'],
  'kolkata': ['Kolkata', 'Bhubaneswar', 'Ranchi', 'Patna', 'Asansol'],
  'bangalore': ['Bangalore', 'Hyderabad', 'Mysore', 'Chennai', 'Pune', 'Madurai'],
  'mysore': ['Mysore', 'Bangalore', 'Coimbatore', 'Chennai', 'Hassan'],
  'coimbatore': ['Coimbatore', 'Chennai', 'Bangalore', 'Mysore', 'Salem'],
  'jaipur': ['Jaipur', 'Delhi', 'Agra', 'Udaipur', 'Ajmer'],
  'chandigarh': ['Chandigarh', 'Mohali', 'Panchkula', 'Zirakpur', 'Kharar'],
  'toronto': ['Toronto', 'Mississauga', 'Brampton', 'Hamilton', 'Markham', 'Vaughan'],
  'vancouver': ['Vancouver', 'Burnaby', 'Surrey', 'Richmond', 'Coquitlam'],
  'new york': ['New York', 'Newark', 'Jersey City', 'Yonkers', 'New Rochelle'],
  'los angeles': ['Los Angeles', 'Long Beach', 'Anaheim', 'Santa Ana', 'Irvine'],
  'london': ['London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol'],
  'san francisco': ['San Francisco', 'Oakland', 'San Jose', 'Palo Alto', 'Mountain View'],
  'seattle': ['Seattle', 'Bellevue', 'Redmond', 'Renton', 'Kent'],
  'austin': ['Austin', 'Round Rock', 'Cedar Park', 'Pflugerville', 'Georgetown'],
  'singapore': ['Singapore'],
  'dubai': ['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman'],
};

function getNearbyCities(city) {
  if (!city) return [];
  const normalized = city.toLowerCase().trim();

  // Direct match
  if (NEARBY_CITIES[normalized]) {
    return NEARBY_CITIES[normalized];
  }

  // Partial match (e.g., user types "Delhi" and we find "delhi")
  for (const [key, cities] of Object.entries(NEARBY_CITIES)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return cities;
    }
  }

  // If no matches, return the city itself
  return [city];
}

function inferCountryFromCity(city) {
  const c = (city || '').toLowerCase().trim();
  const IN_CITIES = [
    'hyderabad','bangalore','chennai','delhi','mumbai','pune','kolkata',
    'noida','gurgaon','ahmedabad','surat','jaipur','chandigarh','mysore',
    'coimbatore','visakhapatnam','kochi','nagpur','indore','bhopal',
    'lucknow','patna','ranchi','bhubaneswar','vadodara'
  ];
  const GB_CITIES = ['london','manchester','birmingham','leeds','bristol','glasgow','edinburgh'];
  const CA_CITIES = ['toronto','vancouver','montreal','calgary','ottawa','edmonton'];
  const AU_CITIES = ['sydney','melbourne','brisbane','perth','adelaide'];
  const SG_CITIES = ['singapore'];
  const AE_CITIES = ['dubai','abu dhabi','sharjah'];
  if (IN_CITIES.some(city => c.includes(city) || city.includes(c))) return 'in';
  if (GB_CITIES.some(city => c.includes(city) || city.includes(c))) return 'gb';
  if (CA_CITIES.some(city => c.includes(city) || city.includes(c))) return 'ca';
  if (AU_CITIES.some(city => c.includes(city) || city.includes(c))) return 'au';
  if (SG_CITIES.some(city => c.includes(city) || city.includes(c))) return 'sg';
  if (AE_CITIES.some(city => c.includes(city) || city.includes(c))) return 'ae';
  return 'us';
}

async function getUserLocation() {
  if (!navigator.geolocation) {
    console.log('[LOCATION] Geolocation not supported by browser');
    return null;
  }
  return new Promise(resolve => {
    navigator.geolocation.getCurrentPosition(
      async pos => {
        const { latitude, longitude } = pos.coords;
        console.log(`[LOCATION] Got coordinates: ${latitude}, ${longitude}`);
        try {
          const r = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`);
          const j = await r.json();
          const city = j?.address?.city || j?.address?.town || j?.address?.village || '';
          const countryCode = (j?.address?.country_code || 'us').toLowerCase();
          console.log(`[LOCATION] Detected: ${city}, ${countryCode}`);
          resolve({ city, country: countryCode });
        } catch (err) {
          console.warn('[LOCATION] Reverse geocoding failed:', err);
          resolve({ city: '', country: 'in' });
        }
      },
      err => {
        console.warn('[LOCATION] Permission denied or error:', err.message);
        resolve(null);
      },
      { enableHighAccuracy: false, timeout: 6000, maximumAge: 600000 }
    );
  });
}

// ── Location Modal Handler ────────────────────────────────────────────────────────
function initLocationModal() {
  const modal = document.getElementById('location-modal');
  const detectBtn = document.getElementById('detect-location-btn');
  const skipBtn = document.getElementById('skip-location-btn');
  const confirmBtn = document.getElementById('confirm-location-btn');
  const manualInput = document.getElementById('manual-location-input');

  if (!modal) return;

  // Don't show on load - only on demand via "Find Jobs" button
  modal.style.display = 'none';

  // Detect location button
  if (detectBtn) {
    detectBtn.addEventListener('click', async () => {
      detectBtn.disabled = true;
      detectBtn.textContent = '🔍 Detecting...';
      const loc = await getUserLocation();
      if (loc && loc.city) {
        manualInput.value = loc.city;
        userLocation = loc;
        const nearbyCities = getNearbyCities(loc.city);
        const displayCities = nearbyCities.slice(0, 5).join(', ') + (nearbyCities.length > 5 ? ', ...' : '');
        console.log(`[LOCATION] Detected: ${loc.city} → Searching in: ${displayCities}`);
        modal.style.display = 'none';
        startJobSearch();
      } else {
        detectBtn.disabled = false;
        detectBtn.textContent = '🎯 Detect Location';
        alert('Could not detect location. Please enter manually or skip.');
      }
    });
  }

  // Skip button
  if (skipBtn) {
    skipBtn.addEventListener('click', () => {
      console.log('[LOCATION] Skipped → Searching globally without location filter');
      userLocation = { city: '', country: 'in' };
      modal.style.display = 'none';
      startJobSearch();
    });
  }

  // Confirm button
  if (confirmBtn) {
    confirmBtn.addEventListener('click', () => {
      const location = manualInput.value.trim();
      if (location) {
        const nearbyCities = getNearbyCities(location);
        const displayCities = nearbyCities.slice(0, 5).join(', ') + (nearbyCities.length > 5 ? ', ...' : '');
        console.log(`[LOCATION] Confirmed: ${location} → Searching in: ${displayCities}`);
        userLocation = { city: location, country: inferCountryFromCity(location) };
      } else {
        userLocation = { city: '', country: 'in' };
      }
      modal.style.display = 'none';
      startJobSearch();
    });
  }

  // Allow Enter key to confirm
  if (manualInput) {
    manualInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && confirmBtn) confirmBtn.click();
    });
  }
}

// Show location modal on Find Jobs button click
function attachFindJobsButton() {
  const btn = document.getElementById('find-jobs-btn');
  if (btn) {
    btn.addEventListener('click', () => {
      const modal = document.getElementById('location-modal');
      if (modal) {
        modal.style.display = 'flex';
        document.getElementById('manual-location-input').value = userLocation.city || '';
      }
    });
  }
}

async function startJobSearch() {
  const jobsGrid = document.getElementById('jobs-grid');
  const feed = document.getElementById('apply-progress-feed');
  const findBtn = document.getElementById('find-jobs-btn');

  if (findBtn) findBtn.disabled = true;
  if (feed) feed.innerHTML = '<div>📍 Searching for jobs...</div>';
  if (jobsGrid) jobsGrid.innerHTML = Array.from({ length: 6 }).map(() => '<div class="skeleton"></div>').join('');

  const cityInput = userLocation?.city || '';
  const country = userLocation?.country || 'us';

  // Get nearby cities
  const nearbyCities = getNearbyCities(cityInput);
  const locationQuery = nearbyCities.join(' OR ');

  if (locationQuery) {
    if (feed) feed.innerHTML = `<div>🔌 Searching for jobs in ${nearbyCities.slice(0, 3).join(', ')}${nearbyCities.length > 3 ? ' & more...' : ''}...</div>`;
  } else {
    if (feed) feed.innerHTML = '<div>🔌 Searching jobs globally...</div>';
  }

  try {
    const res = await fetch(
      `/api/jobs?location=${encodeURIComponent(locationQuery)}&country=${encodeURIComponent(country)}`
    );
    const payload = await res.json();
    if (payload.status === 'setup_required') {
      const grid = document.getElementById('jobs-grid');
      if (grid) grid.innerHTML = `<div class="empty-state">${escapeHtml(payload.message)}</div>`;
      if (feed) feed.innerHTML = '<div>🛠 API key setup required.</div>';
      return;
    }
    if (payload.status !== 'success') throw new Error(payload.message || 'Could not fetch jobs');
    
    const jobs = payload.jobs || [];
    const linkedinCount = payload.linkedin_count || 0;
    renderJobs(jobs, linkedinCount);

    if (feed) feed.innerHTML = `<div>✅ Found ${jobs.length} jobs (${linkedinCount} LinkedIn). Click cards to select, then apply!</div>`;
  } catch (err) {
    const grid = document.getElementById('jobs-grid');
    if (grid) grid.innerHTML = `<div class="empty-state">Failed to load jobs: ${escapeHtml(err.message)}</div>`;
    if (feed) feed.innerHTML = `<div>❌ ${escapeHtml(err.message)}</div>`;
  } finally {
    if (findBtn) findBtn.disabled = false;
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  initLocationModal();
  attachFindJobsButton();
});

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

  // Reset JD section
  const jdSection = document.getElementById('jd-section');
  const jdTextarea = document.getElementById('jd-textarea');
  const jdResults = document.getElementById('card-jd-results');
  if (jdSection) jdSection.style.display = 'none';
  if (jdTextarea) jdTextarea.value = '';
  if (jdResults) jdResults.style.display = 'none';

  // Reset cover letter
  _coverLetterData = null;
  const clPreview = document.getElementById('cover-letter-preview');
  const clText    = document.getElementById('cover-letter-text');
  if (clPreview) clPreview.style.display = 'none';
  if (clText)    clText.textContent = '';
  const genBtn = document.getElementById('generate-cover-letter-btn');
  if (genBtn) { genBtn.disabled = false; genBtn.textContent = '✉ Generate Cover Letter'; }
  
  // Reset jobs section completely
  const jobsSection = document.getElementById('jobs-section');
  if (jobsSection) jobsSection.style.display = 'none';
  const jobsGrid = document.getElementById('jobs-grid');
  if (jobsGrid) jobsGrid.innerHTML = '';
  const selectionBar = document.getElementById('selection-bar');
  if (selectionBar) selectionBar.style.display = 'none';
  const progressFeed = document.getElementById('apply-progress-feed');
  if (progressFeed) progressFeed.innerHTML = '';
  const totalCount = document.getElementById('jobs-total-count');
  if (totalCount) totalCount.textContent = '0 jobs';
  const linkedinBadge = document.getElementById('linkedin-badge');
  if (linkedinBadge) linkedinBadge.textContent = '0 LinkedIn';
  
  // Reset jobs state
  jobsState = { allJobs: [], processed: {} };
});

// ── Handle File ───────────────────────────────────────────────────────────────
function handleFile(file) {
  if (file.type !== 'application/pdf') { showError('Please upload a PDF file.'); return; }
  if (file.size > 10 * 1024 * 1024)   { showError('File too large. Max 10 MB.'); return; }

  fileInfo.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
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
  errorBox.textContent        = '⚠ ' + msg;
}

// ── Render All Results ────────────────────────────────────────────────────────
function renderResults(data, meta = {}) {
  loader.style.display  = 'none';
  results.style.display = 'block';

  renderModelBanner(meta, data);
  renderStats(data);
  renderProfile(data);
  renderProjectField(data);
  renderScore(data);
  renderPredictions(data);
  renderSkills(data);
  renderNerEntities(data);
  renderJobMatches(data);
  renderSkillGaps(data);
  renderProjectIdeas(data);
  renderCourses(data);
  renderPdfPreview(data);
  loadJobsFromAnalysis(data);

  // Show JD Match section after resume is analyzed
  const jdSection = document.getElementById('jd-section');
  if (jdSection) jdSection.style.display = 'block';

  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Model Banner ─────────────────────────────────────────────────────────────
function renderModelBanner(meta, data) {
  const el       = document.getElementById('model-banner');
  const navBadge = document.getElementById('nav-badge');

  if (!meta || meta.status === 'error') {
    el.innerHTML = `<div class="model-banner-left"><span class="model-banner-icon">🤖</span>
      <div><div class="model-banner-name">AI Resume Analyzer</div>
      <div class="model-banner-sub">Run the Jupyter notebook to train models</div></div></div>`;
    return;
  }

  const mode   = meta.model_mode || 'ml';
  const isBert = mode === 'bert';
  const hasNer = meta.ner_available;

  if (navBadge) {
    navBadge.textContent = isBert ? 'BERT + NER Powered' : 'ML + NER Powered';
  }

  const modeName = isBert ? 'BERT Domain Classifier' : (meta.model_name || 'ML Model');
  const acc      = isBert ? (meta.bert_accuracy || meta.accuracy || '—') : (meta.accuracy || '—');
  const f1       = isBert ? (meta.bert_f1 || meta.f1_weighted || '—') : (meta.f1_weighted || '—');

  el.innerHTML = `
    <div class="model-banner-left">
      <span class="model-banner-icon">🧠</span>
      <div>
        <div class="model-banner-name">${modeName}</div>
        <div class="model-banner-sub">${meta.num_classes || 24} resume categories · ${(meta.train_samples || 0).toLocaleString()} training samples</div>
      </div>
    </div>
    <div class="model-pills">
      ${isBert ? '<span class="model-pill" style="background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.2);color:var(--amber)">BERT</span>' : ''}
      ${hasNer ? '<span class="model-pill" style="background:rgba(168,85,247,.1);border-color:rgba(168,85,247,.2);color:var(--violet2)">NER</span>' : ''}
      <span class="model-pill">Acc: ${acc}%</span>
      <span class="model-pill" style="background:rgba(6,182,212,.1);border-color:rgba(6,182,212,.2);color:var(--cyan)">F1: ${f1}%</span>
      <span class="model-pill" style="background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.2);color:var(--emerald)">CV: ${meta.cv_mean || '—'}%</span>
    </div>`;
}

// ── Stats Row ─────────────────────────────────────────────────────────────────
function renderStats(data) {
  const e          = data.extracted  || {};
  const s          = data.score      || {};
  const jobMatches = data.job_matches || [];
  const topMatch   = jobMatches.length > 0 ? jobMatches[0].match_pct : 0;

  const stats = [
    { val: s.total || 0,                      label: 'Resume Score',    color: scoreColor(s.total), unit: '/100' },
    { val: data.skills?.current?.length || 0, label: 'Skills Detected', color: 'var(--violet2)',    unit: '' },
    { val: Math.round(topMatch),              label: 'Best Job Match',  color: 'var(--emerald)',    unit: '%' },
    { val: e.pages || 0,                      label: 'Pages',           color: 'var(--amber)',      unit: '' },
  ];

  document.getElementById('stat-grid').innerHTML = stats.map(s => `
    <div class="stat-card">
      <div class="stat-val" style="color:${s.color}">${typeof s.val === 'number' ? s.val.toLocaleString() : s.val}<span style="font-size:.9rem;color:var(--txt2)">${s.unit}</span></div>
      <div class="stat-label">${s.label}</div>
    </div>`).join('');
}

// ── Profile Card ──────────────────────────────────────────────────────────────
function renderProfile(data) {
  const e   = data.extracted    || {};
  const p   = data.prediction   || {};
  const ner = data.ner_entities || {};
  const lvl = p.experience_level || 'Unknown';

  const initials   = (e.name || 'RR').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  const badgeClass = lvl.includes('Senior') ? 'badge-senior'
                   : lvl.includes('Mid')    ? 'badge-mid'
                   : lvl.includes('Junior') ? 'badge-junior'
                   : 'badge-fresher';

  const location = ner.locations && ner.locations.length > 0
    ? (ner.locations[0].text || ner.locations[0]) : '';

  document.getElementById('card-profile').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>Candidate Profile</div>
    <div class="profile-avatar">${initials}</div>
    <div class="profile-name">${e.name || 'Name not detected'}</div>
    <div class="profile-meta">📧 ${e.email || 'Email not found'}</div>
    <div class="profile-meta">📞 ${e.phone || 'Phone not found'}</div>
    ${location ? `<div class="profile-meta">📍 ${location}</div>` : ''}
    <div class="profile-meta">📄 ${e.pages || 1} page${e.pages !== 1 ? 's' : ''} · ${(e.word_count || 0).toLocaleString()} words</div>
    <span class="badge-level ${badgeClass}">⬡ ${lvl}</span>
  `;
}

// ── Project Field Card ────────────────────────────────────────────────────────
// Shows the best-matched job role from skill-based matching.
// Much more accurate than the raw ML category (which can be misleading —
// e.g. HEALTHCARE for a data science student who built a heart disease project).
function renderProjectField(data) {
  const jobMatches = data.job_matches || [];
  const el         = document.getElementById('card-category');

  if (!el) return;

  if (jobMatches.length === 0) {
    el.innerHTML = `
      <div class="card-title"><span class="dot" style="background:var(--violet2)"></span>Project Field</div>
      <div class="empty-state">Upload a resume to see your project field.</div>`;
    return;
  }

  // Best matched role = top result from skill-based job matching
  const topMatch      = jobMatches[0];
  const role          = topMatch.role          || '—';
  const matchPct      = topMatch.match_pct     || 0;
  const matchedSkills = topMatch.matched_skills || [];

  // Color based on match percentage
  const pctColor = matchPct >= 70 ? 'var(--emerald)'
                 : matchPct >= 40 ? 'var(--cyan)'
                 : 'var(--amber)';

  // Show top 4 matched skills as tags
  const skillTags = matchedSkills.slice(0, 4).map(s =>
    `<span style="font-size:.75rem;padding:3px 10px;border-radius:6px;
      background:rgba(124,58,237,.12);border:1px solid rgba(124,58,237,.2);
      color:var(--violet2)">${escapeHtml(s)}</span>`
  ).join('');

  el.innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--violet2)"></span>Project Field</div>
    <div class="category-name">${escapeHtml(role)}</div>
    <div class="category-sub">Best matched job role · ${matchPct.toFixed(1)}% skill match</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin-top:8px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="font-size:.82rem;color:var(--txt2)">Skill Match Score</span>
        <span style="font-size:.82rem;font-weight:700;color:${pctColor}">${matchPct.toFixed(1)}%</span>
      </div>
      <div class="pred-bar-bg">
        <div class="pred-bar-fill" style="width:0%;background:linear-gradient(90deg,var(--violet),var(--violet2))" data-target="${matchPct}"></div>
      </div>
    </div>
    ${skillTags ? `
    <div style="margin-top:14px;">
      <div style="font-size:.75rem;color:var(--txt3);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">Matched Skills</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px">${skillTags}</div>
    </div>` : ''}
    <div style="margin-top:14px;font-size:.82rem;color:var(--txt2);line-height:1.5;">
      Based on your technical skills and experience level.
    </div>`;
  animateBars();
}

// ── Score Card ────────────────────────────────────────────────────────────────
function renderScore(data) {
  const s = data.score || { total: 0, max: 100, breakdown: {} };

  const breakdownHtml = Object.entries(s.breakdown || {}).map(([name, info]) => {
    const pct2  = info.max > 0 ? Math.round((info.earned / info.max) * 100) : 0;
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
  const preds  = data.prediction?.top_predictions || [];
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
        <span class="pred-label">${i === 0 ? '🏆 ' : ''}${p.label}</span>
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

// ── Skills Card ───────────────────────────────────────────────────────────────
function renderSkills(data) {
  const current   = data.skills?.current || [];
  const withConf  = data.skills?.current_with_confidence || [];
  const skillGaps = data.skill_gaps || [];

  const missingSet = new Set();
  skillGaps.forEach(gap => {
    (gap.missing_core_skills || []).forEach(s => missingSet.add(s));
    (gap.missing_skills || []).slice(0, 3).forEach(s => missingSet.add(s));
  });
  const recommended = [...missingSet].slice(0, 12);

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
    : '<span style="color:var(--txt2);font-size:.85rem">No skill gaps identified — your profile is strong!</span>';

  document.getElementById('card-skills').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--rose)"></span>Skills Analysis</div>
    <div style="margin-bottom:20px;">
      <div style="font-size:.82rem;font-weight:600;color:var(--txt2);margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em">Extracted Skills (${current.length})</div>
      <div class="skills-grid">${currentHtml}</div>
    </div>
    <div>
      <div style="font-size:.82rem;font-weight:600;color:var(--txt2);margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em">Skills to Develop (${recommended.length})</div>
      <div class="skills-grid">${recHtml}</div>
      ${recommended.length ? `<p style="margin-top:12px;font-size:.8rem;color:var(--txt2)">💡 Adding these skills can boost your match for top roles in <strong style="color:var(--violet2)">${data.prediction?.category || 'your field'}</strong>.</p>` : ''}
    </div>`;
}

// ── NER Extracted Entities ────────────────────────────────────────────────────
function renderNerEntities(data) {
  const ner = data.ner_entities || {};
  const el  = document.getElementById('card-ner-entities');

  const expEntries = ner.experience_entries || [];
  const eduEntries = ner.education_entries  || [];
  const locations  = ner.locations          || [];
  const yoe        = ner.years_of_experience || '';

  // Work experience rows
  const expHtml = expEntries.length > 0
    ? expEntries.map(e => {
        const line = [
          e.designation ? `<strong>${escapeHtml(e.designation)}</strong>` : '',
          e.company     ? `@ ${escapeHtml(e.company)}` : '',
          e.duration    ? `<span style="color:var(--txt3)">(${escapeHtml(e.duration)})</span>` : '',
        ].filter(Boolean).join(' ');
        return `<div style="margin-bottom:4px">${line}</div>`;
      }).join('')
    : '<span style="color:var(--txt3);font-size:.82rem">Not detected</span>';

  // ── FIX: Education rows — degree, college and year on separate lines ───────
  // Previously all joined on one line with | which hid college/year if empty.
  // Now each piece is on its own line so all 3 always show when present.
  const eduHtml = eduEntries.length > 0
    ? eduEntries.map(e => {
        const degree  = e.degree
          ? `<div><strong>${escapeHtml(e.degree)}</strong></div>`
          : '';
        const college = e.college
          ? `<div style="font-size:.82rem;color:var(--txt2);margin-top:3px">🏫 ${escapeHtml(e.college)}</div>`
          : '';
        const year    = e.year
          ? `<div style="font-size:.78rem;color:var(--txt3);margin-top:2px">📅 ${escapeHtml(e.year)}</div>`
          : '';
        return `<div style="margin-bottom:12px">${degree}${college}${year}</div>`;
      }).join('')
    : '<span style="color:var(--txt3);font-size:.82rem">Not detected</span>';

  // Locations
  const locHtml = locations.length > 0
    ? locations.map(l => {
        const text = typeof l === 'object' ? l.text : l;
        const conf = typeof l === 'object' && l.confidence
          ? `<span class="ner-confidence">(${(l.confidence * 100).toFixed(0)}%)</span>` : '';
        return `${escapeHtml(text)}${conf}`;
      }).join(', ')
    : '<span style="color:var(--txt3);font-size:.82rem">Not detected</span>';

  const hasData = expEntries.length || eduEntries.length || locations.length || yoe;
  if (!hasData) {
    el.innerHTML = `
      <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>Extracted Details</div>
      <div class="empty-state">No entities extracted.</div>`;
    return;
  }

  el.innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>Extracted Details</div>
    <table class="ner-table">
      <thead><tr><th>Entity Type</th><th>Extracted Values</th></tr></thead>
      <tbody>
        <tr>
          <td>💼 Work Experience</td>
          <td>${expHtml}</td>
        </tr>
        <tr>
          <td>🎓 Education</td>
          <td>${eduHtml}</td>
        </tr>
        ${locations.length ? `<tr><td>📍 Locations</td><td>${locHtml}</td></tr>` : ''}
        ${yoe ? `<tr><td>⏳ Experience</td><td>${escapeHtml(yoe)}</td></tr>` : ''}
      </tbody>
    </table>`;
}

// ── Job Role Matches ──────────────────────────────────────────────────────────
function renderJobMatches(data) {
  const matches = data.job_matches || [];
  const el      = document.getElementById('card-job-matches');

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
    const desc         = m.missing_core && m.missing_core.length > 0
      ? `Missing core: ${m.missing_core.join(', ')}`
      : `${matchedCount}/${totalReq} skills matched`;

    return `
      <div class="job-match-row">
        <div class="job-match-header">
          <span class="job-match-role">${i === 0 ? '🎯 ' : ''}${m.role}</span>
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

// ── Skill Gaps ────────────────────────────────────────────────────────────────
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

// ── Project Ideas ─────────────────────────────────────────────────────────────
function renderProjectIdeas(data) {
  const projects = data.project_ideas || [];
  const el       = document.getElementById('card-project-ideas');

  if (projects.length === 0) {
    el.innerHTML = `
      <div class="card-title"><span class="dot" style="background:var(--amber)"></span>Project Ideas</div>
      <div class="empty-state">No project ideas matched your skills.</div>`;
    return;
  }

  const html = projects.map(proj => {
    const matchedLower = new Set((proj.matched_skills || []).map(s => s.toLowerCase()));
    const skillTags    = (proj.all_skills || []).map(s => {
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

// ── Courses Card ──────────────────────────────────────────────────────────────
function renderCourses(data) {
  const courses = data.courses || [];
  const html    = courses.map(c => `
    <div class="course-item">
      <div>
        <span class="course-name">🎓 ${escapeHtml(c.name)}</span>
        ${c.reason ? `<div style="font-size:.72rem;color:var(--txt3);margin-top:2px">${escapeHtml(c.reason)}</div>` : ''}
      </div>
      <a class="course-link" href="${c.url}" target="_blank" rel="noopener">Enroll →</a>
    </div>`).join('');

  document.getElementById('card-courses').innerHTML = `
    <div class="card-title"><span class="dot" style="background:var(--cyan)"></span>Recommended Courses</div>
    <div class="course-list">${html || '<div class="empty-state">No courses found for this profile.</div>'}</div>`;
}

// ── PDF Preview ───────────────────────────────────────────────────────────────
function renderPdfPreview(data) {
  const cardPdf = document.getElementById('card-pdf');
  const frame   = document.getElementById('pdf-frame');
  const toggle  = document.getElementById('pdf-toggle');

  if (data.pdf_preview) {
    cardPdf.style.display = 'block';
    const src       = `data:application/pdf;base64,${data.pdf_preview}`;
    const newToggle = toggle.cloneNode(true);
    toggle.parentNode.replaceChild(newToggle, toggle);
    newToggle.addEventListener('click', () => {
      if (frame.style.display === 'none') {
        frame.src             = src;
        frame.style.display   = 'block';
        newToggle.textContent = 'Hide PDF Preview';
      } else {
        frame.style.display   = 'none';
        newToggle.textContent = 'Show PDF Preview';
      }
    });
  } else {
    cardPdf.style.display = 'none';
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
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
  const div       = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ── JD Match Section ──────────────────────────────────────────────────────────
const jdAnalyzeBtn = document.getElementById('jd-analyze-btn');
const jdClearBtn   = document.getElementById('jd-clear-btn');
const jdTextarea   = document.getElementById('jd-textarea');
const jdResults    = document.getElementById('card-jd-results');

if (jdAnalyzeBtn) {
  jdAnalyzeBtn.addEventListener('click', async () => {
    const jdText = jdTextarea?.value?.trim();

    if (!jdText) {
      showError('Please paste a job description first.');
      return;
    }

    // Button loading state
    jdAnalyzeBtn.disabled    = true;
    jdAnalyzeBtn.textContent = 'Analyzing...';

    try {
      const res = await fetch('/api/jd-match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd_text: jdText })
      });

      const data = await res.json();

      if (data.status !== 'success') {
        showError(data.message || 'JD analysis failed.');
        jdAnalyzeBtn.disabled    = false;
        jdAnalyzeBtn.textContent = 'Analyze Match';
        return;
      }

      // Render results and show card
      renderJDMatch(data);
      jdResults.style.display = 'block';

      // Reset cover letter state when JD changes
      _coverLetterData = null;
      const clPreview = document.getElementById('cover-letter-preview');
      const clText    = document.getElementById('cover-letter-text');
      if (clPreview) clPreview.style.display = 'none';
      if (clText)    clText.textContent = '';
      const genBtn = document.getElementById('generate-cover-letter-btn');
      if (genBtn) { genBtn.disabled = false; genBtn.textContent = '✉ Generate Cover Letter'; }

      jdResults.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
      showError('Server error: ' + err.message);
    }

    jdAnalyzeBtn.disabled    = false;
    jdAnalyzeBtn.textContent = 'Analyze Match';
  });
}

if (jdClearBtn) {
  jdClearBtn.addEventListener('click', () => {
    if (jdTextarea) jdTextarea.value = '';
    if (jdResults) jdResults.style.display = 'none';
    if (jdAnalyzeBtn) jdAnalyzeBtn.textContent = 'Analyze Match';
    // Also clear cover letter
    _coverLetterData = null;
    const clPreview = document.getElementById('cover-letter-preview');
    if (clPreview) clPreview.style.display = 'none';
  });
}

function renderJDMatch(data) {
  const verdictEl   = document.getElementById('jd-verdict');
  const breakdownEl = document.getElementById('jd-breakdown');
  const quickWinEl  = document.getElementById('jd-quick-win');
  const inflationEl = document.getElementById('jd-inflation-flags');

  if (!verdictEl || !breakdownEl || !quickWinEl || !inflationEl) return;

  // Layer 1 — Verdict
  const verdictConfig = {
    strong_match:  { color: 'var(--emerald)', label: 'Strong Match',  desc: 'You meet most requirements and are a great fit for this role.' },
    partial_match: { color: 'var(--amber)',   label: 'Partial Match', desc: 'You have relevant experience but some gaps exist.' },
    stretch_role:  { color: 'var(--rose)',    label: 'Stretch Role',  desc: 'This role may require skills beyond your current profile.' },
    not_suitable:  { color: '#64748b',        label: 'Low Match',     desc: 'Your profile may not align with this position.' }
  };

  const v = verdictConfig[data.verdict] || verdictConfig.not_suitable;
  verdictEl.innerHTML = `
    <div class="jd-verdict-row">
      <span class="jd-verdict-dot" style="background:${v.color}"></span>
      <span class="jd-verdict-label" style="color:${v.color}; font-weight:bold; font-size: 1.2rem;">${v.label}</span>
    </div>
    <div class="jd-verdict-desc" style="margin-top: 8px;">${v.desc}</div>`;

  // Layer 2 — Breakdown
  const breakdown = data.breakdown || {};
  const directMatches  = breakdown.direct_matches  || [];
  const impliedMatches = breakdown.implied_matches || [];
  const genuineGaps    = breakdown.genuine_gaps    || [];
  const expAligned     = breakdown.exp_aligned;
  const resumeLevel    = breakdown.resume_level    || '—';
  const jdLevel        = breakdown.jd_level        || '—';

  let breakdownHtml = '';

  // Row 1: Core skills met
  if (directMatches.length > 0) {
    const pills = directMatches.map(s => `<span class="jd-pill green">${escapeHtml(s)}</span>`).join('');
    breakdownHtml += `
      <div class="jd-breakdown-row">
        <span class="jd-breakdown-icon" style="color:var(--emerald)">✓</span>
        <div class="jd-breakdown-content">
          <div class="jd-breakdown-label">Core skills met</div>
          <div>${pills}</div>
        </div>
      </div>`;
  }

  // Row 2: Implied but not stated
  if (impliedMatches.length > 0) {
    const impliedText = impliedMatches.slice(0, 3).map(s =>
      `Your resume implies "${escapeHtml(s)}" — consider adding it explicitly.`
    ).join(' ');
    breakdownHtml += `
      <div class="jd-breakdown-row">
        <span class="jd-breakdown-icon" style="color:var(--violet2)">ℹ</span>
        <div class="jd-breakdown-content">
          <div class="jd-breakdown-label">Implied but not stated</div>
          <div style="font-size:.84rem;color:var(--txt)">${impliedText}</div>
        </div>
      </div>`;
  }

  // Row 3: Genuine gaps
  if (genuineGaps.length > 0) {
    const pills = genuineGaps.map(s => `<span class="jd-pill rose">${escapeHtml(s)}</span>`).join('');
    breakdownHtml += `
      <div class="jd-breakdown-row">
        <span class="jd-breakdown-icon" style="color:var(--rose)">✗</span>
        <div class="jd-breakdown-content">
          <div class="jd-breakdown-label">Genuine gaps</div>
          <div>${pills}</div>
        </div>
      </div>`;
  }

  // Row 4: Experience alignment
  const expIcon  = expAligned ? '✓' : '⚠';
  const expColor = expAligned ? 'var(--emerald)' : 'var(--amber)';
  const expText  = expAligned
    ? `Your experience level (${resumeLevel}) aligns with the role (${jdLevel}).`
    : `Experience mismatch: Your level (${resumeLevel}) vs role requirement (${jdLevel}).`;
  breakdownHtml += `
    <div class="jd-breakdown-row">
      <span class="jd-breakdown-icon" style="color:${expColor}">${expIcon}</span>
      <div class="jd-breakdown-content">
        <div class="jd-breakdown-label">Experience alignment</div>
        <div style="font-size:.84rem;color:var(--txt)">${expText}</div>
      </div>
    </div>`;

  breakdownEl.innerHTML = breakdownHtml;

  // Layer 3 — Quick Win
  quickWinEl.innerHTML = `
    <div class="jd-quick-win">
      <div class="jd-quick-win-label">Quick Win</div>
      <div class="jd-quick-win-text">${escapeHtml(data.quick_win)}</div>
    </div>`;

  // Inflation flags (conditional)
  const inflationFlags = data.inflation_flags || [];
  if (inflationFlags.length > 0) {
    const flagItems = inflationFlags.map(f =>
      `<div class="jd-inflation-item">• ${escapeHtml(f.skill)}: ${f.required_years} years required, but ${escapeHtml(f.skill)} has only existed since ${f.tech_birth_year} (max ${f.max_possible} years).</div>`
    ).join('');
    inflationEl.innerHTML = `
      <div class="jd-inflation-card">
        <div class="jd-inflation-title">⚠ JD Quality Flags</div>
        ${flagItems}
        <div class="jd-inflation-note">Don't be discouraged — these requirements may be overstated.</div>
      </div>`;
  } else {
    inflationEl.innerHTML = '';
  }
}

async function loadJobsFromAnalysis(data) {
  const jobsSection = document.getElementById('jobs-section');
  if (!jobsSection) return;

  jobsSection.style.display = 'block';

  // Show the button to find jobs - don't auto-load
  const feed = document.getElementById('apply-progress-feed');
  if (feed) feed.innerHTML = '<div>📍 Click "Find Jobs Near You" to search for opportunities</div>';
}


function getPlatformClass(platform) {
  const p = (platform || '').toLowerCase();
  if (p.includes('linkedin')) return 'linkedin';
  if (p.includes('indeed')) return 'indeed';
  if (p.includes('glassdoor')) return 'glassdoor';
  if (p.includes('ziprecruiter')) return 'ziprecruiter';
  return 'other';
}

function updateSelectionBar() {
  const cards = [...document.querySelectorAll('.job-card.selected')];
  const count = cards.length;
  const linkedinCount = cards.filter(c => c.dataset.platform?.toLowerCase() === 'linkedin').length;
  
  const bar = document.getElementById('selection-bar');
  const countEl = document.getElementById('selection-count');
  
  if (count === 0) {
    bar.style.display = 'none';
  } else {
    bar.style.display = 'flex';
    const linkedinNote = linkedinCount > 0 ? ` (${linkedinCount} LinkedIn)` : '';
    countEl.textContent = `${count} job${count !== 1 ? 's' : ''} selected${linkedinNote}`;
  }
  
}

function toggleJobSelection(card) {
  card.classList.toggle('selected');
  updateSelectionBar();
}

function renderJobs(jobs, linkedinCount) {
  const grid = document.getElementById('jobs-grid');
  const totalEl = document.getElementById('jobs-total-count');
  const linkedinBadge = document.getElementById('linkedin-badge');
  
  if (!grid) return;
  
  totalEl.textContent = `${jobs.length} jobs`;
  linkedinBadge.textContent = `${linkedinCount} LinkedIn`;
  
  if (!jobs.length) {
    grid.innerHTML = '<div class="empty-state">No jobs found. Try different search criteria.</div>';
    return;
  }
  
  grid.innerHTML = jobs.map((job, idx) => {
    const platform = job.source_platform || 'Other';
    const platformClass = getPlatformClass(platform);
    return `
    <div class="job-card" data-index="${idx}" data-platform="${platform}" onclick="toggleJobSelection(this)">
      <div class="job-title">${escapeHtml(job.job_title)}</div>
      <div class="job-company">${escapeHtml(job.company_name)}</div>
      <div class="job-location">📍 ${escapeHtml(job.location)}</div>
      <div class="job-meta">
        <span class="platform-badge ${platformClass}">${escapeHtml(platform)}</span>
        <span class="applied-badge">✓ Applied</span>
      </div>
      <div class="job-actions" onclick="event.stopPropagation()">
        <a class="btn btn-outline btn-sm" href="${job.apply_url}" target="_blank" rel="noopener">Apply ↗</a>
      </div>
    </div>
  `}).join('');
  
  // Store jobs in state
  jobsState.allJobs = jobs;
  updateSelectionBar();
}

function getSelectedJobs() {
  const jobs = jobsState.allJobs || [];
  const cards = [...document.querySelectorAll('.job-card.selected')];
  return cards.map(card => jobs[parseInt(card.dataset.index)]).filter(Boolean);
}

function getSelectedLinkedInJobs() {
  return getSelectedJobs().filter(job => 
    (job.source_platform || '').toLowerCase() === 'linkedin'
  );
}

function showProgress(message, done = false) {
  const bar = document.getElementById('apply-progress-bar');
  const text = document.getElementById('apply-progress-text');
  if (!bar || !text) return;
  
  if (done) {
    bar.style.display = 'none';
  } else {
    bar.style.display = 'block';
    text.textContent = message;
  }
}

function markAppliedByTitleCompany(title, company) {
  const cards = [...document.querySelectorAll('.job-card')];
  cards.forEach(card => {
    const t = card.querySelector('.job-title')?.textContent?.trim();
    const c = card.querySelector('.job-company')?.textContent?.trim();
    if (t === title && c === company) {
      const badge = card.querySelector('.applied-badge');
      if (badge) badge.style.display = 'inline-flex';
      card.classList.remove('selected');
    }
  });
  updateSelectionBar();
}

async function openSelectedInTabs() {
  const jobs = getSelectedJobs();
  if (!jobs.length) {
    showProgress('⚠ No jobs selected.');
    setTimeout(() => showProgress('', true), 2000);
    return;
  }
  
  showProgress(`Opening ${jobs.length} jobs in new tabs...`);
  let opened = 0;
  for (const job of jobs) {
    if (job.apply_url) {
      window.open(job.apply_url, '_blank');
      opened++;
      await new Promise(r => setTimeout(r, 300));
    }
  }
  showProgress(`✅ Opened ${opened} jobs in your browser`);
  setTimeout(() => showProgress('', true), 3000);
}

function attachApplyButtons() {
  // Select ALL jobs
  const selectAllBtn = document.getElementById('select-all-jobs');
  if (selectAllBtn) {
    selectAllBtn.addEventListener('click', () => {
      document.querySelectorAll('.job-card').forEach(card => {
        card.classList.add('selected');
      });
      updateSelectionBar();
    });
  }
  
  // Select all LinkedIn jobs
  const selectAllLinkedinBtn = document.getElementById('select-all-linkedin');
  if (selectAllLinkedinBtn) {
    selectAllLinkedinBtn.addEventListener('click', () => {
      document.querySelectorAll('.job-card').forEach(card => {
        if (card.dataset.platform?.toLowerCase() === 'linkedin') {
          card.classList.add('selected');
        }
      });
      updateSelectionBar();
    });
  }
  
  // Clear selection
  const clearBtn = document.getElementById('clear-selection');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      document.querySelectorAll('.job-card.selected').forEach(card => {
        card.classList.remove('selected');
      });
      updateSelectionBar();
    });
  }
  
  // Open selected in tabs
  const openSelectedBtn = document.getElementById('open-selected-btn');
  if (openSelectedBtn) {
    openSelectedBtn.addEventListener('click', openSelectedInTabs);
  }
}

attachApplyButtons();


// ── Cover Letter Generation ───────────────────────────────────────────────────

const generateCoverLetterBtn   = document.getElementById('generate-cover-letter-btn');
const downloadCoverLetterBtn   = document.getElementById('download-cover-letter-btn');
const regenerateCoverLetterBtn = document.getElementById('regenerate-cover-letter-btn');
const coverLetterPreview       = document.getElementById('cover-letter-preview');
const coverLetterText          = document.getElementById('cover-letter-text');

async function generateCoverLetter() {
  if (!generateCoverLetterBtn) return;

  generateCoverLetterBtn.disabled    = true;
  generateCoverLetterBtn.textContent = '⏳ Generating...';

  try {
    const jdText = document.getElementById('jd-textarea')?.value?.trim() || '';

    const res = await fetch('/api/generate-cover-letter', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ jd_text: jdText }),
    });

    const data = await res.json();

    if (data.status !== 'success') {
      alert('Cover letter generation failed: ' + (data.message || 'Unknown error'));
      generateCoverLetterBtn.disabled    = false;
      generateCoverLetterBtn.textContent = '✉ Generate Cover Letter';
      return;
    }

    // Store for PDF download
    _coverLetterData = {
      cover_letter: data.cover_letter,
      candidate   : data.candidate,
      date        : data.date,
    };

    // Show preview
    if (coverLetterText)    coverLetterText.textContent   = data.cover_letter;
    if (coverLetterPreview) coverLetterPreview.style.display = 'block';

    coverLetterPreview.scrollIntoView({ behavior: 'smooth', block: 'start' });

    generateCoverLetterBtn.disabled    = false;
    generateCoverLetterBtn.textContent = '✉ Generate Cover Letter';

  } catch (err) {
    alert('Server error: ' + err.message);
    generateCoverLetterBtn.disabled    = false;
    generateCoverLetterBtn.textContent = '✉ Generate Cover Letter';
  }
}

async function downloadCoverLetterPDF() {
  if (!_coverLetterData) {
    alert('Please generate a cover letter first.');
    return;
  }

  if (!downloadCoverLetterBtn) return;

  downloadCoverLetterBtn.disabled    = true;
  downloadCoverLetterBtn.textContent = '⏳ Generating PDF...';

  try {
    const res = await fetch('/api/download-cover-letter', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(_coverLetterData),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: 'PDF generation failed' }));
      alert('PDF error: ' + (err.message || 'Unknown error'));
      downloadCoverLetterBtn.disabled    = false;
      downloadCoverLetterBtn.textContent = '⬇ Download as PDF';
      return;
    }

    // Trigger browser download
    const blob     = await res.blob();
    const url      = URL.createObjectURL(blob);
    const a        = document.createElement('a');
    const name     = (_coverLetterData.candidate?.name || 'Candidate').replace(/\s+/g, '_');
    a.href         = url;
    a.download     = `Cover_Letter_${name}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

  } catch (err) {
    alert('Download error: ' + err.message);
  }

  downloadCoverLetterBtn.disabled    = false;
  downloadCoverLetterBtn.textContent = '⬇ Download as PDF';
}

if (generateCoverLetterBtn) {
  generateCoverLetterBtn.addEventListener('click', generateCoverLetter);
}

if (regenerateCoverLetterBtn) {
  regenerateCoverLetterBtn.addEventListener('click', generateCoverLetter);
}

if (downloadCoverLetterBtn) {
  downloadCoverLetterBtn.addEventListener('click', downloadCoverLetterPDF);
}