'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const colors = { history: '#64766d', initial: '#8299ad', refined: '#3e855f', target: '#c67d56' };
let research;
let episodeStep = 0;
let replayTimer;
let metric = 'mse';
let family = 'all';
let codeTab = 'install';

const steps = [
  { title: 'Start with the observed history.', label: 'DIAGNOSTIC EVIDENCE', description: 'Statistical and temporal diagnostics characterize recurring fluctuations in the 96-hour history. Their summaries are retained in Analysis History for later decisions.' },
  { title: 'Choose a forecaster using the evidence.', label: 'MODEL ROUTING', description: 'In this episode, the agent selects Chronos using the diagnostic evidence. The forecasting tool returns a 96-step prediction, which is stored in Prediction Results.' },
  { title: 'Revisit the forecast. Refine its details.', label: 'REFLECTION & REFINEMENT', description: 'The same policy reviews the intermediate forecast against historical observations and retained evidence, then adjusts local fluctuations and peak amplitudes. Future targets remain unavailable.' },
  { title: 'Reveal the target for comparison.', label: 'AFTER-THE-FACT COMPARISON', description: 'The held-out target is now shown for visual comparison. This is a qualitative case from the paper; it illustrates the decision process rather than a guarantee that refinement always improves a forecast.' }
];

const commands = {
  install: { text: 'git clone https://github.com/ustc-time-series/Cast-R1.git\ncd Cast-R1\n\npython3.11 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt', note: 'Python 3.11. Full policy training and serving require a compatible CUDA environment.' },
  check: { text: '# From the installed repository\nOMP_NUM_THREADS=1 python -m pytest -q \\\n  tests/time_series_forecast/test_paper_logic.py \\\n  tests/time_series_forecast/test_paper_training_smoke.py', note: 'CPU integration checks exercise the pipeline with tiny models. They do not reproduce the manuscript’s benchmark values.' },
  evaluate: { text: '# Prepare data, trained policy, and forecasting tools first\nexport STUDENT_API_BASE="http://127.0.0.1:8000/v1"\n\npython -m recipe.time_series_forecast.evaluate_paper \\\n  --inputs datasets/PAPER_RAW/test.parquet \\\n  --api-base "$STUDENT_API_BASE" --model Cast-R1 \\\n  --output results/paper_test.json', note: 'Requires the trained student served at STUDENT_API_BASE, prepared benchmark inputs, and the forecasting-tool service. See the reproduction guide.' }
};

const svgText = (x, y, text, extra = '') => `<text x="${x}" y="${y}" ${extra}>${text}</text>`;
const polyline = (points, x, y, color, width = 1.7, dash = '') => `<path d="${points.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;

function renderHeroChart() {
  if (!research) return;
  const svg = $('#hero-chart');
  const W = Math.max(280, svg.clientWidth), H = Math.round(Math.max(180, W * .48));
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const m = { l: 5, r: 7, t: 25, b: 22 };
  const x = v => m.l + v / 192 * (W - m.l - m.r);
  const y = v => H - m.b - (v - 7) / 8.6 * (H - m.t - m.b);
  const cut = x(96);
  let out = `<title>Paper ETTh1 case: observed history and Cast-R1 forecast</title><rect x="${cut}" y="${m.t - 10}" width="${W - m.r - cut}" height="${H - m.t - m.b + 10}" fill="#dce8cc" opacity=".42" rx="3"/>`;
  for (let i = 0; i < 4; i++) out += `<line x1="0" x2="${W}" y1="${y(8 + i * 2)}" y2="${y(8 + i * 2)}" stroke="#ced8c5" stroke-width=".6" stroke-dasharray="2 4"/>`;
  out += `<line x1="${cut}" x2="${cut}" y1="${m.t - 10}" y2="${H - m.b}" stroke="#acbca2" stroke-dasharray="3 4" stroke-width=".8"/>`;
  out += svgText(5, 10, 'OBSERVED HISTORY', 'font-size="7" fill="#93a086" letter-spacing="1"');
  out += svgText(cut + 12, 10, 'FORECAST', 'font-size="7" fill="#6a875c" letter-spacing="1"');
  out += polyline(research.case.target.filter(p => p[0] < 96), x, y, colors.history, 1.55);
  out += polyline(research.case.refined, x, y, colors.refined, 1.85);
  out += svgText(5, H - 4, '−96 h', 'font-size="8" fill="#95a28b"');
  out += svgText(cut, H - 4, 'Forecast origin', 'font-size="8" fill="#8a9c7e" text-anchor="middle"');
  out += svgText(W - 6, H - 4, '+96 h', 'font-size="8" fill="#95a28b" text-anchor="end"');
  svg.innerHTML = out;
}

function renderEpisodeChart() {
  if (!research) return;
  const svg = $('#episode-chart');
  const W = Math.max(280, svg.clientWidth), H = Math.round(Math.max(225, Math.min(330, W * .47)));
  const m = { l: 29, r: 12, t: 30, b: 31 };
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const x = v => m.l + v / 192 * (W - m.l - m.r);
  const y = v => H - m.b - (v - 7) / 8.4 * (H - m.t - m.b);
  const cut = x(95.5);
  let out = `<title>${steps[episodeStep].title}</title><desc>Figure-based replay of 96 historical hours and a 96-hour forecasting horizon. ${steps[episodeStep].description}</desc>`;
  out += `<rect x="${cut}" y="${m.t - 10}" width="${W - m.r - cut}" height="${H - m.t - m.b + 10}" rx="3" fill="#edf2e5"/>`;
  [7, 9, 11, 13, 15].forEach(v => {
    out += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#dfe5d7" stroke-width=".65"/>`;
    out += svgText(m.l - 8, y(v) + 3, v, 'text-anchor="end" fill="#95a18d" font-size="8"');
  });
  [0, 48, 96, 144, 192].forEach(v => {
    out += svgText(x(v), H - 14, v, 'text-anchor="middle" fill="#95a18d" font-size="8"');
  });
  out += `<line x1="${cut}" x2="${cut}" y1="${m.t - 10}" y2="${H - m.b}" stroke="#b1bfa4" stroke-dasharray="3 4" stroke-width=".8"/>`;
  out += svgText(m.l + 2, 12, 'OBSERVED · 96 h', 'font-size="7" fill="#92a087" letter-spacing=".5"');
  out += svgText(cut + 10, 12, 'FORECAST · 96 h', 'font-size="7" fill="#78926b" letter-spacing=".5"');
  if (episodeStep === 0) out += svgText(cut + (W - m.r - cut) / 2, H / 2, 'Awaiting forecast', 'text-anchor="middle" font-size="10" fill="#a1af96"');
  if (episodeStep >= 3) out += polyline(research.case.target.filter(p => p[0] >= 95), x, y, colors.target, 1.35, '3 3');
  out += polyline(research.case.target.filter(p => p[0] < 96), x, y, colors.history, 1.55);
  if (episodeStep >= 1) out += polyline(research.case.initial, x, y, colors.initial, 1.5, '4 3');
  if (episodeStep >= 2) out += polyline(research.case.refined, x, y, colors.refined, 1.9);
  out += svgText(W - 11, H - 1, 'Hour', 'font-size="7" fill="#a1ae97" text-anchor="end"');
  svg.innerHTML = out;
  svg.setAttribute('aria-label', `Step ${episodeStep + 1}: ${steps[episodeStep].title}`);
  ['initial', 'refined', 'target'].forEach((name, i) => {
    $(`#legend-${name}`).classList.toggle('unavailable', episodeStep < i + 1);
  });
}

function setEpisode(step) {
  episodeStep = Math.max(0, Math.min(3, step));
  $$('.episode-step').forEach(b => {
    const active = Number(b.dataset.step) === episodeStep;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', String(active));
  });
  $('#episode-title').textContent = steps[episodeStep].title;
  $('#evidence-label').textContent = steps[episodeStep].label;
  $('#episode-description').textContent = steps[episodeStep].description;
  $('#step-count').textContent = `0${episodeStep + 1} / 04`;
  renderEpisodeChart();
}

function stopReplay() {
  clearInterval(replayTimer);
  replayTimer = undefined;
  $('#play-replay span').textContent = 'Play episode';
  $('#play-replay').setAttribute('aria-label', 'Play forecasting episode');
  $('#play-replay use').setAttribute('href', '#i-play');
}

function selectedIndices() {
  return research.datasets.map((_, i) => i).filter(i => family === 'all' || (family === 'ltsf' ? i < 5 : i >= 5));
}

function reductionFor(i) {
  const rows = research[metric];
  const ours = rows.find(r => r.name === 'Cast-R1');
  const best = rows.filter(r => r.name !== 'Cast-R1').reduce((a, b) => a.values[i] <= b.values[i] ? a : b);
  return { improvement: (best.values[i] - ours.values[i]) / best.values[i] * 100, best: best.name, bestValue: best.values[i], ours: ours.values[i] };
}

function renderResults() {
  if (!research) return;
  const indices = selectedIndices();
  $('#result-chart-title').textContent = `Relative ${metric.toUpperCase()} reduction`;
  const values = indices.map(i => ({ i, ...reductionFor(i) }));
  // The zero line stays fixed; left and right scales accommodate both signs.
  const positiveMax = Math.max(1, ...values.map(v => Math.max(v.improvement, 0)));
  const negativeMax = Math.max(4, ...values.map(v => Math.max(-v.improvement, 0)));
  const fragment = document.createDocumentFragment();
  values.forEach(v => {
    const row = document.createElement('div');
    row.className = 'result-bar-row'; row.tabIndex = 0;
    const neg = v.improvement < 0;
    const width = neg ? Math.abs(v.improvement) / negativeMax * 27 : v.improvement / positiveMax * 69;
    const tip = `Cast-R1: ${v.ours.toFixed(3)} · Best baseline: ${v.best} (${v.bestValue.toFixed(3)})`;
    row.setAttribute('aria-label', `${research.datasets[v.i]}: ${neg ? 'increase' : 'reduction'} of ${Math.abs(v.improvement).toFixed(2)} percent in ${metric.toUpperCase()}. ${tip}`);
    row.innerHTML = `<span class="bar-name">${research.datasets[v.i]}</span><span class="bar-track"><span class="bar-zero"></span><span class="bar-fill ${neg ? 'negative' : ''}" style="left:${neg ? 28 - width : 28}%;width:${width}%"></span></span><span class="bar-value ${neg ? 'negative' : ''}">${neg ? '−' : '+'}${Math.abs(v.improvement).toFixed(2)}%</span><span class="bar-tooltip">${tip}</span>`;
    fragment.appendChild(row);
  });
  $('#results-bars').replaceChildren(fragment);
  renderTable(indices);
}

function renderTable(indices) {
  const rows = research[metric];
  const rankings = indices.map(i => [...new Set(rows.map(r => r.values[i]))].sort((a, b) => a - b));
  const table = $('#benchmark-table');
  table.innerHTML = `<caption class="sr-only">${metric.toUpperCase()} results from the revised manuscript, raw scale. Lower is better.</caption><thead><tr><th scope="col">Model</th>${indices.map(i => `<th scope="col">${research.datasets[i]}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr class="${r.name === 'Cast-R1' ? 'cast-row' : ''}"><th scope="row">${r.name}</th>${indices.map((i, j) => `<td class="${r.values[i] === rankings[j][0] ? 'best' : r.values[i] === rankings[j][1] ? 'second' : ''}">${r.values[i].toFixed(3)}</td>`).join('')}</tr>`).join('')}</tbody>`;
  $('#table-caption').textContent = `${metric.toUpperCase()} · lower is better`;
}

function setCodeTab(name) {
  codeTab = name;
  $$('.code-tabs [role="tab"]').forEach(b => {
    const selected = b.dataset.code === name;
    b.setAttribute('aria-selected', String(selected)); b.tabIndex = selected ? 0 : -1;
  });
  $('#quickstart-code').setAttribute('aria-labelledby', `tab-${name}`);
  $('#command-text').textContent = commands[name].text;
  $('#command-note').textContent = commands[name].note;
}

let toastTimer;
function toast(text) {
  const el = $('#toast'); el.textContent = text; el.classList.add('visible');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('visible'), 2400);
}

async function copyText(text, button) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const area = document.createElement('textarea'); area.value = text;
      area.style.cssText = 'position:fixed;opacity:0;pointer-events:none';
      document.body.appendChild(area); area.select();
      const success = document.execCommand('copy'); area.remove();
      if (!success) throw new Error('Clipboard unavailable');
      button.focus();
    }
    toast('Copied to clipboard');
  } catch {
    toast('Select the text and copy it manually.');
  }
}

function setupInteractions() {
  $$('.episode-step').forEach(b => b.addEventListener('click', () => { stopReplay(); setEpisode(Number(b.dataset.step)); }));
  $('#play-replay').addEventListener('click', () => {
    if (replayTimer) { stopReplay(); return; }
    setEpisode(0);
    $('#play-replay span').textContent = 'Pause episode';
    $('#play-replay').setAttribute('aria-label', 'Pause forecasting episode');
    replayTimer = setInterval(() => {
      setEpisode(episodeStep + 1);
      if (episodeStep >= 3) stopReplay();
    }, 2600);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopReplay(); });
  $$('[data-metric]').forEach(b => b.addEventListener('click', () => {
    metric = b.dataset.metric;
    $$('[data-metric]').forEach(t => { t.classList.toggle('active', t === b); t.setAttribute('aria-pressed', String(t === b)); });
    renderResults();
  }));
  $$('[data-family]').forEach(b => b.addEventListener('click', () => {
    family = b.dataset.family;
    $$('[data-family]').forEach(t => { t.classList.toggle('active', t === b); t.setAttribute('aria-pressed', String(t === b)); });
    renderResults();
  }));
  $$('.code-tabs [role="tab"]').forEach(b => {
    b.addEventListener('click', () => setCodeTab(b.dataset.code));
    b.addEventListener('keydown', e => {
      const tabs = $$('.code-tabs [role="tab"]'); const i = tabs.indexOf(b);
      let n;
      if (e.key === 'ArrowRight') n = (i + 1) % tabs.length;
      if (e.key === 'ArrowLeft') n = (i + tabs.length - 1) % tabs.length;
      if (e.key === 'Home') n = 0;
      if (e.key === 'End') n = tabs.length - 1;
      if (n !== undefined) { e.preventDefault(); setCodeTab(tabs[n].dataset.code); tabs[n].focus(); }
    });
  });
  $('#copy-command').addEventListener('click', e => copyText(commands[codeTab].text, e.currentTarget));
  $('#copy-citation').addEventListener('click', e => copyText($('#bibtex').textContent, e.currentTarget));
  $('#download-results').addEventListener('click', () => {
    if (!research) return;
    const ids = selectedIndices();
    const lines = [['Model', ...ids.map(i => research.datasets[i])], ...research[metric].map(r => [r.name, ...ids.map(i => r.values[i])])];
    const csv = lines.map(row => row.map(v => `"${String(v).replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = `cast-r1-${metric}-${family}.csv`; document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast('Benchmark CSV downloaded');
  });
  const menu = $('.menu-toggle');
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open)); menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    $('#main-nav').classList.toggle('open', open);
  });
  $$('#main-nav a').forEach(a => a.addEventListener('click', () => { menu.setAttribute('aria-expanded', 'false'); menu.setAttribute('aria-label', 'Open navigation'); $('#main-nav').classList.remove('open'); }));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { menu.setAttribute('aria-expanded', 'false'); menu.setAttribute('aria-label', 'Open navigation'); $('#main-nav').classList.remove('open'); } });
  const updateProgress = () => {
    const height = document.documentElement.scrollHeight - window.innerHeight;
    $('.reading-progress').style.width = `${height > 0 ? window.scrollY / height * 100 : 0}%`;
  };
  window.addEventListener('scroll', updateProgress, { passive: true }); updateProgress();
  if ('IntersectionObserver' in window) {
    const sections = ['overview', 'method', 'results', 'resources'];
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => { if (entry.isIntersecting) $$('#main-nav a').forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${entry.target.id}`)); });
    }, { rootMargin: '-15% 0px -50% 0px' });
    sections.forEach(id => observer.observe(document.getElementById(id)));
  }
  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { renderHeroChart(); renderEpisodeChart(); }, 80); });
  setCodeTab('install');
}

async function init() {
  setupInteractions();
  try {
    research = window.CAST_R1_RESEARCH;
    if (!research) {
      const response = await fetch('assets/research-data.json');
      if (!response.ok) throw new Error(`Data request returned ${response.status}`);
      research = await response.json();
    }
    renderHeroChart(); setEpisode(0); renderResults();
    if (document.fonts?.ready) document.fonts.ready.then(() => { renderHeroChart(); renderEpisodeChart(); });
  } catch (error) {
    $('#results-bars').innerHTML = '<p class="data-error">The interactive data could not be loaded. <a href="assets/cast-r1-manuscript.pdf">View all results in the paper PDF ↗</a></p>';
    $('#episode-title').textContent = 'Explore the case in the original figure.';
    $('#episode-description').innerHTML = '<a href="assets/case-study.pdf">Open the original case figure ↗</a>';
    $('#play-replay').disabled = true;
    console.error('Research data unavailable:', error);
  }
}
init();
