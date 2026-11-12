(() => {
'use strict';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const store = {
  get(k, d) { try { const v = localStorage.getItem('gs.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('gs.' + k, JSON.stringify(v)); } catch {} },
};

/* ------------------------------------------------------------------ api */
async function req(method, url, body) {
  const opt = {method, headers: {}};
  if (body instanceof FormData) opt.body = body;
  else if (body !== undefined) { opt.body = JSON.stringify(body); opt.headers['content-type'] = 'application/json'; }
  const r = await fetch(url, opt);
  const t = await r.json().catch(() => ({}));
  if (!r.ok) {
    let msg = t.detail || t.error || `오류 (${r.status})`;
    if (Array.isArray(msg)) msg = msg.map(e => e.msg).join(', ');
    throw new Error(msg);
  }
  return t;
}
const api = {get: u => req('GET', u), post: (u, b) => req('POST', u, b ?? {}), del: u => req('DELETE', u)};
const imgUrl = (id, v) => `/api/images/${id}/file${v ? '?v=' + v : ''}`;
const thumbUrl = id => `/api/images/${id}/thumb`;

let tt; function toast(m, ms = 3000) { const t = $('#toast'); t.textContent = m; t.classList.add('show'); clearTimeout(tt); tt = setTimeout(() => t.classList.remove('show'), ms); }

async function pollJob(id, onTick, signal) {
  for (;;) {
    if (signal?.aborted) throw new Error('취소됨');
    const j = await api.get(`/api/jobs/${id}`);
    onTick?.(j);
    if (['done', 'error', 'canceled'].includes(j.status)) return j;
    await sleep(700);
  }
}

/* --------------------------------------------------------------- status */
let STATUS = null, MODELS = [];
async function refreshStatus() {
  try {
    [STATUS, MODELS] = await Promise.all([api.get('/api/status'), api.get('/api/models')]);
  } catch { STATUS = null; }
  if (!STATUS) { renderSys(); return; }
  fillModelSelect();
  renderBanner();
}
/* click the status bar: copy a diagnostic report to paste into a chat */
async function copyDiagnostics() {
  let text;
  try { text = await (await fetch('/api/diagnostics', {credentials: 'same-origin'})).text(); }
  catch (e) { toast('진단 정보를 가져오지 못했어요: ' + e.message, 5000); return; }
  try { await navigator.clipboard.writeText(text); toast('진단 정보를 복사했어요. 대화창에 붙여넣어 주세요.', 5000); return; } catch {}
  const d = document.createElement('dialog'); d.className = 'diag';
  d.innerHTML = `<p>아래 내용을 전부 복사해서 붙여넣어 주세요.</p><textarea readonly></textarea><div class="row"><button class="btn primary">닫기</button></div>`;
  d.querySelector('textarea').value = text;
  d.querySelector('button').onclick = () => d.remove();
  document.body.appendChild(d); d.showModal(); d.querySelector('textarea').select();
}
$('#status').addEventListener('click', copyDiagnostics);

/* live machine numbers in the top bar (GPU, VRAM, RAM, CPU, running job) */
let SYS = null;
async function pollSystem() {
  if (document.hidden) return;
  try { SYS = await api.get('/api/system'); } catch { SYS = null; }
  renderSys();
}
function meter(label, pct, text, cls = '') {
  return `<span class="m ${cls} ${pct >= 90 ? 'hot' : ''}"><b>${label}</b><i style="--p:${Math.max(0, Math.min(100, pct))}%"></i><em>${text}</em></span>`;
}
function renderSys() {
  const el = $('#status');
  if (!SYS) { el.innerHTML = '<i class="dot bad"></i><span>Gen Studio 서버 응답 없음</span>'; return; }
  if (!SYS.comfyui) { el.innerHTML = '<i class="dot bad"></i><span>ComfyUI 꺼짐</span>'; return; }
  const g = SYS.gpu || {}, run = SYS.queue.running, parts = [];
  if (run || SYS.queue.waiting)
    parts.push(`<span class="job">${run ? `<span class="ja">${esc(run.action)} </span>${Math.round(run.progress * 100)}%` : ''}<span class="ja">${run && SYS.queue.waiting ? ' · ' : ''}${SYS.queue.waiting ? `대기 ${SYS.queue.waiting}` : ''}</span>${!run && SYS.queue.waiting ? `<span class="jm">대기 ${SYS.queue.waiting}</span>` : ''}</span>`);
  if (g.util != null) parts.push(meter('GPU', g.util, `${Math.round(g.util)}%`, 'opt2'));
  if (g.vram_total) parts.push(meter('VRAM', g.vram_used / g.vram_total * 100, `${g.vram_used.toFixed(1)}/${Math.round(g.vram_total)}G<span class="ja">B</span>`));
  if (SYS.ram_total) parts.push(meter('RAM', SYS.ram_used / SYS.ram_total * 100, `${Math.round(SYS.ram_used)}/${Math.round(SYS.ram_total)}GB`, 'opt'));
  if (SYS.cpu != null) parts.push(meter('CPU', SYS.cpu, `${Math.round(SYS.cpu)}%`, 'opt'));
  if (g.temp != null) parts.push(`<span class="m opt ${g.temp >= 83 ? 'hot' : ''}"><b>온도</b><em>${Math.round(g.temp)}°</em></span>`);
  el.title = `${g.name ? g.name + ' · ' : ''}눌러서 진단 정보 복사`;
  el.innerHTML = `<i class="dot ok"></i>${parts.join('') || `<span>${esc(g.name || '연결됨')}</span>`}`;
}

function renderBanner() {
  let b = document.querySelector('.banner');
  const msgs = [];
  if (STATUS && !STATUS.comfyui.connected)
    msgs.push(`<b>ComfyUI가 꺼져 있어요.</b> 막 켰다면 잠시 기다려 주세요. 계속 꺼져 있으면 <a href="#settings">설정</a>에서 다시 켤 수 있어요.`);
  const missing = MODELS.filter(m => m.installed === false);
  if (missing.length === MODELS.length && MODELS.length)
    msgs.push(`<b>아직 받은 모델이 없어요.</b> <a href="#settings">설정</a>에서 모델 받기를 누르세요.`);
  else if (missing.length)
    msgs.push(`받지 않은 모델: ${missing.map(m => esc(m.name)).join(', ')}. <a href="#settings">설정</a>에서 받을 수 있어요.`);
  if (!msgs.length) { b?.remove(); return; }
  if (!b) { b = document.createElement('div'); b.className = 'banner'; $('.top').after(b); }
  b.innerHTML = `<div>${msgs.join('<br>')}</div>`;
}
function fillModelSelect(force) {
  const sel = $('#genModel');
  if (!force && sel.options.length === MODELS.length + 1) return;
  const keep = sel.value;
  while (sel.options.length > 1) sel.remove(1);
  for (const m of MODELS) sel.add(new Option(`${m.name} — ${m.best_for || ''}${m.installed === false ? ' (미설치)' : ''}`, m.id));
  sel.value = [...sel.options].some(o => o.value === keep) ? keep : 'auto';
}
function markRecommended(sel, recId, autoLabel = '자동') {
  // "★ 추천" on the recommended model; the auto entry says which one it will use
  for (const o of sel.options) {
    const base = o.dataset.base || (o.dataset.base = o.textContent);
    if (o.value === 'auto') o.textContent = recId ? `${autoLabel} (추천: ${(MODELS.find(m => m.id === recId) || {}).name || recId})` : base;
    else o.textContent = (o.value === recId ? '★ 추천 · ' : '') + base;
  }
}

/* --------------------------------------------------------------- router */
const VIEWS = {generate: $('#genView'), edit: $('#editView'), gallery: $('#galleryView'), loras: $('#lorasView'), settings: $('#settingsView')};
function route() {
  const h = location.hash.slice(1);
  let view = 'generate', arg = null;
  if (h.startsWith('edit=')) { view = 'edit'; arg = h.slice(5); }
  else if (h === 'gallery') view = 'gallery';
  else if (h === 'loras') view = 'loras';
  else if (h === 'settings') view = 'settings';
  for (const [k, el] of Object.entries(VIEWS)) el.hidden = k !== view;
  document.querySelectorAll('.tabs a').forEach(a => { if (a.dataset.view === view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  if (view === 'generate') Gen.show();
  if (view === 'gallery') Gallery.show();
  if (view === 'loras') LoraPage.show();
  Settings.active(view === 'settings');
  if (view === 'edit') Editor.open(arg);
}
window.addEventListener('hashchange', route);

function autoGrow(ta) { ta.style.height = 'auto'; ta.style.height = Math.min(160, ta.scrollHeight) + 'px'; }
function bindEnter(ta, form) {
  ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); } });
  ta.addEventListener('input', () => autoGrow(ta));
}

/* ============================================================ generate */
const Gen = (() => {
  const prompt = $('#genPrompt'), form = $('#genForm'), results = $('#results'), routeEl = $('#genRoute');
  const sels = {style: $('#genStyle'), aspect: $('#genAspect'), count: $('#genCount'), model: $('#genModel')};
  const lbl = {style: $('#styleLbl'), aspect: $('#aspectLbl'), count: $('#countLbl'), model: $('#modelLbl')};
  const jobs = [];      // {id, status, progress, message, result_ids, error, prompt}
  let recent = [], timer, refs = store.get('gen.refs', []), chars = [], loras = [], picked = store.get('gen.loras', []);
  const FAM = {zimage: 'Z-Image', illustrious: 'Illustrious', qwen21: 'Qwen 2.1', wan22: 'Wan 2.2'};
  async function loadLoras() { try { loras = (await api.get('/api/loras')).loras; } catch { loras = []; } picked = picked.filter(p => loras.some(l => l.id === p.id)); renderLoras(); }
  function savePicked() { store.set('gen.loras', picked); renderLoras(); preview(); }
  function renderLoras() {
    $('#genLoras').innerHTML = picked.map(p => { const l = loras.find(x => x.id === p.id); return l ? `<span class="lchip" title="${esc(FAM[l.family] || '')}용">${esc(l.name)}<input type="range" min="0.1" max="1.5" step="0.05" value="${p.strength}" data-lstr="${p.id}" aria-label="세기"><span>${Number(p.strength).toFixed(2)}</span><button type="button" data-lrm="${p.id}" aria-label="빼기">×</button></span>` : ''; }).join('');
  }
  function renderPick(fam) {
    const el = $('#loraPick');
    el.innerHTML = loras.length ? loras.map(l => `<button type="button" data-lpick="${l.id}" class="${fam && l.family && l.family !== fam ? 'dim' : ''}" ${l.family ? '' : 'disabled'}><b>${esc(l.name)}</b><span class="fam">${esc(FAM[l.family] || '계열 모름')}${l.triggers?.length ? ' · ' + esc(l.triggers.join(', ')) : ''}${fam && l.family && l.family !== fam ? ' · 고르면 모델이 바뀌어요' : ''}</span></button>`).join('')
      : '<span class="hint">등록된 LoRA가 없어요. 위쪽 <a href="#loras">LoRA</a> 탭에서 추가하세요.</span>';
  }
  $('#addLora').onclick = async () => { const el = $('#loraPick'); if (!el.hidden) { el.hidden = true; return; } await loadLoras(); const fam = (MODELS.find(m => m.id === lastRoute?.model_id) || {}).family; renderPick(fam); el.hidden = false; };
  $('#loraPick').addEventListener('click', e => { const b = e.target.closest('[data-lpick]'); if (!b) return; const l = loras.find(x => x.id === b.dataset.lpick); if (!picked.some(p => p.id === l.id)) picked.push({id: l.id, strength: l.strength || 0.8}); $('#loraPick').hidden = true; savePicked(); });
  $('#genLoras').addEventListener('input', e => { const id = e.target.dataset.lstr; if (!id) return; const p = picked.find(x => x.id === id); p.strength = +e.target.value; e.target.nextElementSibling.textContent = p.strength.toFixed(2); store.set('gen.loras', picked); });
  $('#genLoras').addEventListener('change', () => preview());
  $('#genLoras').addEventListener('click', e => { const b = e.target.closest('[data-lrm]'); if (b) { picked = picked.filter(p => p.id !== b.dataset.lrm); savePicked(); } });
  let lastRoute = null;
  const charSel = $('#genChar'), refsEl = $('#genRefs');
  async function loadChars() {
    try { chars = await api.get('/api/characters'); } catch { chars = []; }
    const keep = store.get('gen.character', '');
    charSel.innerHTML = '<option value="">없음</option>' + chars.map(c => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('');
    charSel.value = chars.some(c => c.id === keep) ? keep : '';
    $('#charLbl').textContent = charSel.selectedOptions[0]?.textContent || '없음';
  }
  charSel.addEventListener('change', () => { store.set('gen.character', charSel.value); $('#charLbl').textContent = charSel.selectedOptions[0].textContent; preview(); });
  function renderRefs() {
    refsEl.innerHTML = refs.map((id, i) => `<span class="ref"><img src="${thumbUrl(id)}" alt=""><i>${i + 1}</i><button type="button" data-unref="${id}" aria-label="참고 이미지 빼기">×</button></span>`).join('');
  }
  function setRefs(list) { refs = list.slice(0, 10); store.set('gen.refs', refs); renderRefs(); preview(); }
  refsEl.addEventListener('click', e => { const b = e.target.closest('[data-unref]'); if (b) setRefs(refs.filter(r => r !== b.dataset.unref)); });
  $('#addRef').onclick = () => $('#refIn').click();
  $('#refIn').onchange = async e => {
    for (const f of e.target.files) {
      const fd = new FormData(); fd.append('file', f, f.name);
      try { const r = await req('POST', '/api/import', fd); refs.push(r.id); } catch (err) { toast(err.message); }
    }
    e.target.value = ''; setRefs(refs);
  };
  /* ---- pose: copy the body pose of a photo (or a saved pose map) */
  let pose = store.get('gen.pose', null);   // {id, strength}
  const POSE_STR = [[0.6, '느슨하게'], [0.85, '보통'], [1.1, '정확하게']];
  function renderPose() {
    $('#addPose').hidden = !!pose;
    $('#addPoseEd').textContent = pose ? '뼈대 고치기' : '+ 뼈대';
    $('#genPose').innerHTML = pose ? `<span class="ref"><img src="${thumbUrl(pose.id)}" alt=""><i>포즈</i><button type="button" data-unpose aria-label="포즈 빼기">×</button></span>
      <span class="posestr">${POSE_STR.map(([v, n]) => `<button type="button" data-pstr="${v}" aria-pressed="${v === pose.strength}">${n}</button>`).join('')}</span>` : '';
  }
  function setPose(id, strength) { pose = id ? {id, strength: strength ?? pose?.strength ?? 0.85} : null; store.set('gen.pose', pose); renderPose(); preview(); }
  $('#addPose').onclick = () => $('#poseIn').click();
  $('#addPoseEd').onclick = async () => {
    let rec = null;
    if (pose) rec = await api.get(`/api/images/${pose.id}`).catch(() => null);
    const isMap = rec && rec.kind === 'pose';
    PoseEd.open({start: isMap ? rec.id : null, bg: rec && !isMap ? rec.id : null, autoRead: rec && !isMap,
                 useLabel: '이 자세로 그리기', onUse: id => { setPose(id); toast('이 자세로 그려요. 무엇을 그릴지 적어 주세요.'); }});
  };
  $('#poseIn').onchange = async e => {
    const f = e.target.files[0]; e.target.value = ''; if (!f) return;
    const fd = new FormData(); fd.append('file', f, f.name);
    try { const r = await req('POST', '/api/import', fd); setPose(r.id); toast('이 사진의 자세대로 그려요. 처음 한 번은 포즈를 읽느라 조금 더 걸려요.', 5000); } catch (err) { toast(err.message); }
  };
  $('#genPose').addEventListener('click', e => {
    if (e.target.closest('[data-unpose]')) setPose(null);
    const b = e.target.closest('[data-pstr]'); if (b) setPose(pose.id, +b.dataset.pstr);
  });
  renderPose();

  prompt.value = store.get('lastPrompt', '');
  for (const [k, s] of Object.entries(sels)) {
    s.value = store.get('gen.' + k, s.value) ?? s.value;
    s.addEventListener('change', () => { store.set('gen.' + k, s.value); syncLabels(); preview(); });
  }
  function syncLabels() {
    for (const k in sels) {
      const o = sels[k].selectedOptions[0];
      if (k === 'model') { const m = MODELS.find(x => x.id === sels.model.value); lbl.model.textContent = m ? m.name : '자동'; continue; }
      lbl[k].textContent = o?.textContent.replace(/\s*\(미설치\)/, '').split(' ')[0] || sels[k].value;
    }
  }
  bindEnter(prompt, form);
  prompt.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(preview, 250); store.set('lastPrompt', prompt.value); });

  function body() {
    const seed = parseInt($('#genSeed').value, 10);
    return {prompt: prompt.value.trim(), style: sels.style.value, aspect: sels.aspect.value, count: +sels.count.value,
            model: sels.model.value, character: charSel.value || null, references: refs,
            loras: picked.map(p => ({name: p.id, strength: p.strength})),
            negative: $('#genNeg').value.trim(), seed: Number.isFinite(seed) && seed >= 0 ? seed : -1,
            pose: pose ? pose.id : null, pose_strength: pose ? pose.strength : 0.85,
            regions: regions.filter(r => r.prompt.trim()).map(r => ({box: r.box.map(v => +v.toFixed(4)), prompt: r.prompt.trim()}))};
  }

  /* ---- advanced: regions (boxes drawn on a canvas of the chosen ratio), negative, seed */
  let regions = store.get('gen.regions', []);
  const board = $('#regBoard'), regList = $('#regList'), advBtn = $('#advBtn');
  const ASPECT_WH = {square: [1, 1], portrait: [2, 3], landscape: [3, 2], tall: [9, 16], wide: [16, 9]};
  function saveRegions() { store.set('gen.regions', regions); $('#advBadge').textContent = regions.length || ''; }
  function boardRatio() {
    const [w, h] = lastRoute ? [lastRoute.width, lastRoute.height] : (ASPECT_WH[sels.aspect.value] || [1, 1]);
    board.style.aspectRatio = `${w} / ${h}`;
    board.style.width = `min(100%, ${300 * w / h}px)`;
  }
  function drawRegions(draft) {
    boardRatio();
    board.querySelectorAll('.rg').forEach(x => x.remove());
    [...regions, ...(draft ? [{box: draft, draft: true}] : [])].forEach((r, i) => {
      const d = document.createElement('div'); d.className = 'rg' + (r.draft ? ' draft' : '');
      const [x0, y0, x1, y1] = r.box;
      Object.assign(d.style, {left: x0 * 100 + '%', top: y0 * 100 + '%', width: (x1 - x0) * 100 + '%', height: (y1 - y0) * 100 + '%'});
      if (!r.draft) d.innerHTML = `<i>${i + 1}</i>`;
      board.appendChild(d);
    });
    board.querySelector('.bh').hidden = regions.length > 0 || !!draft;
  }
  function listRegions(focusLast) {
    regList.innerHTML = regions.map((r, i) => `<div class="rr"><i>${i + 1}</i><input data-ri="${i}" value="${esc(r.prompt)}" placeholder="${i + 1}번 영역에 그릴 것 (예: 빨간 드레스를 입은 여자)"><button type="button" data-rdel="${i}" aria-label="영역 지우기">×</button></div>`).join('')
      || '<p class="hint" style="margin:0">아직 영역이 없어요. 왼쪽 그림에서 끌어서 상자를 그리세요. 최대 6개.</p>';
    if (focusLast) regList.querySelector(`[data-ri="${regions.length - 1}"]`)?.focus();
  }
  let dragFrom = null;
  const rel = e => { const r = board.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))]; };
  board.addEventListener('pointerdown', e => { if (regions.length >= 6) { toast('영역은 6개까지예요'); return; } dragFrom = rel(e); board.setPointerCapture(e.pointerId); });
  board.addEventListener('pointermove', e => { if (!dragFrom) return; const p = rel(e); drawRegions([Math.min(dragFrom[0], p[0]), Math.min(dragFrom[1], p[1]), Math.max(dragFrom[0], p[0]), Math.max(dragFrom[1], p[1])]); });
  board.addEventListener('pointerup', e => {
    if (!dragFrom) return; const p = rel(e), a = dragFrom; dragFrom = null;
    const box = [Math.min(a[0], p[0]), Math.min(a[1], p[1]), Math.max(a[0], p[0]), Math.max(a[1], p[1])];
    if ((box[2] - box[0]) * (box[3] - box[1]) < 0.01) { drawRegions(); return; }
    regions.push({box, prompt: ''}); saveRegions(); drawRegions(); listRegions(true);
  });
  regList.addEventListener('input', e => { const i = e.target.dataset.ri; if (i !== undefined) { regions[+i].prompt = e.target.value; saveRegions(); } });
  regList.addEventListener('click', e => { const b = e.target.closest('[data-rdel]'); if (b) { regions.splice(+b.dataset.rdel, 1); saveRegions(); drawRegions(); listRegions(); } });
  advBtn.onclick = () => {
    const open = $('#genAdv').hidden; $('#genAdv').hidden = !open; advBtn.setAttribute('aria-expanded', open); store.set('gen.adv', open);
    if (open) { drawRegions(); listRegions(); }
  };
  if (store.get('gen.adv', false)) { $('#genAdv').hidden = false; advBtn.setAttribute('aria-expanded', true); setTimeout(() => { drawRegions(); listRegions(); }, 0); }
  saveRegions();
  $('#genEnhance').onclick = async () => {
    const b = body(), btn = $('#genEnhance');
    if (!b.prompt) { prompt.focus(); toast('먼저 대충이라도 적어 주세요'); return; }
    if (btn.classList.contains('busy')) return;
    btn.classList.add('busy');
    toast('다듬는 중… 처음 한 번은 모델을 불러오느라 20~40초 걸릴 수 있어요', 4000);
    try {
      const r = await api.post('/api/enhance', b);
      prompt.focus(); prompt.select();
      // insertText keeps the browser's undo history, so Ctrl+Z brings the short version back
      if (!document.execCommand('insertText', false, r.prompt)) prompt.value = r.prompt;
      // the rewrite drops words like "반실사"; keep the style/model that was picked for the short version
      if (sels.style.value === 'auto' && r.style && [...sels.style.options].some(o => o.value === r.style)) { sels.style.value = r.style; store.set('gen.style', r.style); }
      if (sels.aspect.value === 'auto' && r.aspect && r.aspect !== 'square' && [...sels.aspect.options].some(o => o.value === r.aspect)) { sels.aspect.value = r.aspect; store.set('gen.aspect', r.aspect); }
      if (sels.model.value === 'auto' && r.family === 'wan22') { sels.model.value = r.model_id; store.set('gen.model', r.model_id); }
      syncLabels();
      autoGrow(prompt); store.set('lastPrompt', prompt.value); preview();
      toast(`${r.model_name}에 맞게 다듬었어요. 마음에 안 들면 Ctrl+Z`, 4000);
    } catch (e) { toast(e.message, 5000); }
    finally { btn.classList.remove('busy'); }
  };
  async function preview() {
    const b = body();
    if (b.character || b.references.length) {
      const c = chars.find(x => x.id === b.character);
      const what = `${c ? `캐릭터 '${esc(c.name)}' 유지` : ''}${c && refs.length ? ' · ' : ''}${refs.length ? `참고 이미지 ${refs.length}장` : ''}`;
      try {
        const d = await api.post('/api/route', {...b, prompt: b.prompt || ' '});
        lastRoute = d;
        routeEl.innerHTML = d.restyle
          ? `<span class="pill auto">${b.model === 'auto' ? '자동 선택' : '직접 선택'} → <b>${esc(d.model_name)}</b> · ${what} · ${d.width}×${d.height}</span><span class="hint">참고 이미지를 ${({semi: '반실사', text: '글자·디자인'})[d.style] || '실사'}로 다시 그려요. 사람·포즈·구도는 따라가고 그림체만 바뀝니다.</span>`
          : `<span class="pill auto">자동 선택 → <b>${esc(d.model_name)}</b> · ${what} <span class="hint">(프롬프트에서 "image 1", "image 2"로 가리킬 수 있어요)</span></span>${refs.length === 1 && !c ? '<span class="hint">그림체만 바꾸려면 "반실사로", "실사화"처럼 쓰거나 스타일을 반실사로 고르세요.</span>' : ''}`;
      } catch (e) { routeEl.innerHTML = `<span class="hint">${esc(e.message)}</span>`; }
      return;
    }
    if (!b.prompt) { routeEl.innerHTML = `<span class="hint">한국어로 써도 됩니다. "실사", "반실사", "포스터", "세로" 같은 말을 넣으면 그에 맞춰 고릅니다. 이미지에 넣을 글자는 "따옴표"로 감싸세요.</span>`; return; }
    try {
      const d = await api.post('/api/route', {...b, model: 'auto'});
      lastRoute = d;
      if (!$('#genAdv').hidden) drawRegions();
      markRecommended(sels.model, d.model_id);
      const lf = [...new Set(picked.map(p => (loras.find(l => l.id === p.id) || {}).family).filter(Boolean))];
      if (lf.length && b.model === 'auto' && (MODELS.find(m => m.id === d.model_id) || {}).family !== lf[0]) {
        const m = MODELS.find(x => x.family === lf[0] && x.installed !== false) || MODELS.find(x => x.family === lf[0]);
        routeEl.innerHTML = `<span class="pill auto">LoRA에 맞춰 → <b>${esc(m ? m.name : lf[0])}</b> · ${d.width}×${d.height}</span><span class="hint">LoRA가 없으면 ${esc(d.model_name)}을(를) 골랐을 거예요</span>`;
        return;
      }
      if (b.model !== 'auto' && b.model !== d.model_id) {
        const chosen = MODELS.find(m => m.id === b.model);
        routeEl.innerHTML = `<span class="pill auto">직접 선택 → <b>${esc(chosen ? chosen.name : b.model)}</b></span><span class="hint">이 요청의 추천은 ${esc(d.model_name)} (${esc(d.reasons.join(', '))})</span>`;
        return;
      }
      routeEl.innerHTML = `<span class="pill auto">${b.model === 'auto' ? '자동 선택' : '직접 선택 · 추천과 같음'} → <b>${esc(d.model_name)}</b> · ${({semi: '반실사', text: '글자·디자인'})[d.style] || '실사'} · ${d.width}×${d.height}${d.reasons.length ? ` <span class="hint">(${esc(d.reasons.join(', '))})</span>` : ''}</span>`;
    } catch (e) { routeEl.innerHTML = `<span class="hint">${esc(e.message)}</span>`; }
  }
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const b = body(); if (!b.prompt) { prompt.focus(); return; }
    try {
      const j = await api.post('/api/generate', b);
      const entry = {...j, prompt: b.prompt}; jobs.unshift(entry); render();
      pollJob(j.id, x => { Object.assign(entry, x); render(); }).then(x => {
        if (x.status === 'done') { loadRecent(); if (x.result_ids.length === 1 && !VIEWS.generate.hidden && jobs[0] === entry) toast('완료! 이미지를 누르면 편집기로 열려요'); }
        if (x.status === 'error') toast('생성 실패: ' + x.error, 6000);
      }).catch(err => toast(err.message));
    } catch (err) { toast(err.message, 5000); }
  });
  async function loadRecent() { try { recent = await api.get('/api/images?limit=40'); } catch { recent = []; } render(); }
  function render() {
    const active = jobs.filter(j => j.status === 'queued' || j.status === 'running' || j.status === 'error');
    let h = '';
    if (active.length) {
      h += `<h3>진행 중</h3><div class="grid">` + active.map(j => `<div class="card job">${j.status === 'error'
        ? `<div><b class="err">실패</b><p>${esc(j.error)}</p></div>`
        : `<div style="display:grid;gap:10px;justify-items:center"><span>${esc(j.message)}</span><i class="bar" style="--p:${Math.round((j.progress || 0) * 100)}%"></i><span class="hint">${esc(j.decision?.model_name || '')}</span><button class="hint" data-cancel="${j.id}">취소</button></div>`}</div>`).join('') + `</div><div style="height:18px"></div>`;
    }
    if (recent.length) {
      h += `<h3>최근 이미지</h3><div class="grid">` + recent.map(r => `<a class="card" href="#edit=${r.id}" title="${esc(r.prompt || '')}"><img loading="lazy" src="${thumbUrl(r.id)}" alt=""><span class="cap">${esc(r.label && r.label !== '원본' ? r.label : (r.prompt || r.original_name || ''))}</span></a>`).join('') + `</div>`;
    } else if (!active.length) {
      h += `<div class="empty"><b>첫 이미지를 만들어 보세요</b><span>아래에 원하는 장면을 적고 Enter를 누르면 됩니다.<br>실사·반실사와 모델은 자동으로 골라요.</span></div>`;
    }
    results.innerHTML = h;
  }
  results.addEventListener('click', async e => {
    const c = e.target.closest('[data-cancel]'); if (!c) return;
    await api.post(`/api/jobs/${c.dataset.cancel}/cancel`).catch(() => {}); toast('취소를 요청했어요');
  });
  return {show() { syncLabels(); loadChars().then(preview); loadLoras(); renderRefs(); loadRecent(); autoGrow(prompt); },
          addRef(id) { if (!refs.includes(id)) setRefs([...refs, id]); },
          reloadChars() { return loadChars(); },
          setPose(id) { setPose(id); }};
})();

/* ============================================================== gallery */
const Gallery = (() => {
  const grid = $('#galGrid');
  async function show() {
    let items = [];
    try { items = await api.get('/api/images?limit=500&roots_only=true'); } catch (e) { toast(e.message); }
    $('#galCount').textContent = items.length ? `${items.length}개` : '';
    grid.innerHTML = items.length ? items.map(r => `<span class="cw"><a class="card" href="#edit=${r.id}"><img loading="lazy" src="${thumbUrl(r.id)}" alt=""><span class="cap">${esc(r.kind === 'import' ? '불러온 이미지 · ' + (r.original_name || '') : r.prompt || '')}</span></a><button class="vx" data-gdel="${r.id}" title="이 이미지와 모든 버전 삭제" aria-label="삭제">×</button></span>`).join('')
      : `<div class="empty" style="grid-column:1/-1"><b>아직 이미지가 없어요</b><span>생성 탭에서 만들거나, 이미지를 이 창에 끌어다 놓으세요.</span></div>`;
  }
  return {show};
})();

/* ================================================================ LoRA */
const LoraPage = (() => {
  const FAM = {zimage: 'Z-Image (실사)', illustrious: 'Illustrious·SDXL (반실사)', qwen21: 'Qwen-Image 2.1', wan22: 'Wan 2.2'};
  let poll = null;
  async function show() {
    let d; try { d = await api.get('/api/loras'); } catch (e) { toast(e.message); return; }
    $('#loraCount').textContent = d.loras.length ? `${d.loras.length}개` : '';
    $('#loraDl').innerHTML = d.downloads.filter(x => x.status !== 'done').map(x => `<p class="hint">${x.status === 'error' ? `⚠ ${esc(x.file)}: ${esc(x.error)}` : `받는 중: ${esc(x.file)} ${x.total ? Math.round(x.done / x.total * 100) : 0}%`}</p>`).join('');
    $('#loraGrid').innerHTML = d.loras.map(l => `<div class="lcard" data-id="${esc(l.id)}">
      <div class="pv" style="${l.preview ? `background-image:url('${esc(l.preview)}')` : ''}"></div>
      <div><h4>${esc(l.name)}</h4><span class="fam ${l.family ? '' : 'unk'}">${esc(FAM[l.family] || '계열을 골라 주세요')}</span>
        ${l.present === false ? '<p class="hint" style="color:var(--danger)">파일이 폴더에 없어요</p>' : ''}
        <label>계열<select data-f="family"><option value="">모름</option>${Object.entries(FAM).map(([k, v]) => `<option value="${k}" ${k === l.family ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
        <label>트리거 단어 (쉼표로 구분)<input type="text" data-f="triggers" value="${esc((l.triggers || []).join(', '))}"></label>
        <label>기본 세기 <b>${Number(l.strength).toFixed(2)}</b><input type="range" min="0.1" max="1.5" step="0.05" data-f="strength" value="${l.strength}"></label>
        <div class="row2">${l.civitai?.url ? `<a class="hint" href="${esc(l.civitai.url)}" target="_blank" rel="noopener">Civitai</a>` : ''}<span class="hint">${esc(l.note || '')}</span><div class="spacer"></div><button class="ghost" data-del>삭제</button></div>
      </div></div>`).join('') || '<div class="empty" style="grid-column:1/-1"><b>아직 LoRA가 없어요</b><span>Civitai 주소를 붙여넣거나, .safetensors 파일을 넣거나, ComfyUI/models/loras 폴더에 넣고 "폴더 다시 읽기"를 누르세요.</span></div>';
    clearTimeout(poll);
    if (d.downloads.some(x => x.status === 'running')) poll = setTimeout(show, 1000);
  }
  $('#loraAdd').onclick = async () => {
    const url = $('#loraUrl').value.trim(); if (!url) return;
    try { await api.post('/api/loras/civitai', {url}); $('#loraUrl').value = ''; toast('받기 시작했어요'); show(); } catch (e) { toast(e.message, 5000); }
  };
  $('#loraScan').onclick = async () => { const r = await api.post('/api/loras/scan'); toast(r.added.length ? `${r.added.length}개를 새로 찾았어요` : '새 파일이 없어요'); show(); };
  $('#loraFileBtn').onclick = () => $('#loraFile').click();
  $('#loraFile').onchange = async e => {
    for (const f of e.target.files) { const fd = new FormData(); fd.append('file', f, f.name); try { await req('POST', '/api/loras/upload', fd); } catch (err) { toast(err.message, 5000); } }
    e.target.value = ''; show();
  };
  $('#loraGrid').addEventListener('change', async e => {
    const card = e.target.closest('[data-id]'), f = e.target.dataset.f; if (!card || !f) return;
    let v = e.target.value;
    if (f === 'triggers') v = v.split(',').map(x => x.trim()).filter(Boolean);
    if (f === 'strength') v = +v;
    try { await req('PATCH', `/api/loras/${card.dataset.id}`, {[f]: v || null}); show(); } catch (err) { toast(err.message); }
  });
  /* ---- training */
  let trSel = new Set(), trChars = [], trLoaded = false;
  async function loadTrain() {
    if (trLoaded) return; trLoaded = true;
    try {
      const [imgs, chars] = await Promise.all([api.get('/api/images?limit=120'), api.get('/api/characters')]);
      trChars = chars;
      $('#trChar').innerHTML = '<option value="">캐릭터 사진 쓰기: 없음</option>' + chars.map(c => `<option value="${esc(c.id)}">캐릭터 '${esc(c.name)}' 사진 ${c.ref_ids.length}장 포함</option>`).join('');
      $('#trPick').innerHTML = imgs.map(r => `<button type="button" data-tr="${esc(r.id)}" aria-pressed="false" title="${esc(r.label || '')}" style="background-image:url('${thumbUrl(r.id)}')"></button>`).join('') || '<span class="hint">갤러리에 이미지가 없어요</span>';
    } catch (e) { toast(e.message); trLoaded = false; }
  }
  function trCount() {
    const c = trChars.find(x => x.id === $('#trChar').value);
    $('#trCount').textContent = new Set([...trSel, ...(c ? c.ref_ids : [])]).size;
  }
  $('#trainBox').addEventListener('toggle', () => { if ($('#trainBox').open) loadTrain(); });
  $('#trPick').addEventListener('click', e => {
    const b = e.target.closest('[data-tr]'); if (!b) return;
    const on = !trSel.has(b.dataset.tr); on ? trSel.add(b.dataset.tr) : trSel.delete(b.dataset.tr);
    b.setAttribute('aria-pressed', on); trCount();
  });
  $('#trChar').onchange = trCount;
  async function watchTrain(id) {
    store.set('trainJob', id);
    try {
      const j = await pollJob(id, x => { $('#trStatus').textContent = `${x.status === 'queued' ? '대기 중' : x.message} · ${Math.round(x.progress * 100)}%`; });
      store.set('trainJob', '');
      if (j.status === 'done') { $('#trStatus').textContent = `완료: '${j.result.lora.name}' (트리거 ${j.decision.trigger})`; toast('LoRA 학습이 끝났어요. 생성 화면의 + LoRA에서 고르세요.', 6000); show(); }
      else $('#trStatus').textContent = j.status === 'error' ? `실패: ${j.error}` : '취소했어요';
    } catch { store.set('trainJob', ''); }
  }
  $('#trStart').onclick = async () => {
    const name = $('#trName').value.trim();
    if (!name) { $('#trName').focus(); toast('LoRA 이름을 적어 주세요'); return; }
    try {
      const j = await api.post('/api/loras/train', {name, image_ids: [...trSel], character: $('#trChar').value || null,
                                                    kind: $('#trKind').value, family: $('#trFam').value});
      toast(`학습을 시작했어요: ${j.decision.reasons.join(' · ')}`, 6000);
      watchTrain(j.id);
    } catch (e) { toast(e.message, 6000); }
  };
  const resumeTrain = store.get('trainJob', '');
  if (resumeTrain) { $('#trainBox').open = true; watchTrain(resumeTrain); }

  let delArm = '';
  $('#loraGrid').addEventListener('click', async e => {
    if (!e.target.closest('[data-del]')) return;
    const id = e.target.closest('[data-id]').dataset.id;
    if (delArm !== id) { delArm = id; toast('한 번 더 누르면 목록과 파일에서 삭제해요'); setTimeout(() => delArm = '', 3000); return; }
    await api.del(`/api/loras/${id}?delete_file=true`); toast('삭제했어요'); show();
  });
  let armed = '';
  $('#galGrid').addEventListener('click', async e => {
    const x = e.target.closest('[data-gdel]'); if (!x) return;
    e.preventDefault();
    if (armed !== x.dataset.gdel) { armed = x.dataset.gdel; toast('한 번 더 누르면 이 이미지와 모든 버전을 삭제해요'); setTimeout(() => armed = '', 3000); return; }
    armed = '';
    try { const r = await api.del(`/api/images/${x.dataset.gdel}?all_versions=true`); toast(`${r.deleted}장 삭제했어요`); } catch (err) { toast(err.message); }
    Gallery.show();
  });
  return {show};
})();

/* =============================================================== settings */
const Settings = (() => {
  const body = $('#setBody');
  let info = null, latest = null, timer = null, on = false, busyAct = '';
  const human = n => n >= 1e9 ? (n / 1073741824).toFixed(1) + 'GB' : n >= 1e6 ? (n / 1048576).toFixed(0) + 'MB' : (n / 1024).toFixed(0) + 'KB';
  async function load() {
    try {
      info = await api.get('/api/app/info');
      [info.unused, info.added, info.library] = await Promise.all([api.get('/api/app/models/unused').catch(() => null),
                                                    api.get('/api/app/models/added').then(r => r.models).catch(() => []),
                                                    api.get('/api/app/library').catch(() => null)]);
    }
    catch (e) { body.innerHTML = `<div class="scard"><h3>설정은 PC에서만</h3><p class="hint">${esc(e.message)}</p></div>`; return; }
    paint();
    clearTimeout(timer);
    if (on) timer = setTimeout(load, info.downloads.running || /starting|updating/.test(info.comfy.state) ? 1000 : 4000);
  }
  async function checkUpdate() {
    busyAct = 'check'; paint();
    try { latest = await api.get('/api/app/latest'); latest.checked = Date.now(); } catch (e) { toast(e.message); }
    busyAct = ''; paint();
  }
  function card(title, inner, cls = '') { return `<div class="scard ${cls}"><h3>${title}</h3>${inner}</div>`; }
  function paint() {
    if (!info) return;
    $('#setVer').textContent = `v${info.version}${info.commit ? ' · ' + info.commit.slice(0, 7) : ''}`;
    const d = info.downloads, mm = info.models.missing, c = info.comfy, r = info.remote;
    const upd = latest
      ? (latest.update_available ? `<p><b>새 버전이 있어요.</b> ${esc(latest.latest.message)} <span class="hint">(${new Date(latest.latest.date).toLocaleString()})</span></p>
          <div class="row"><button class="btn primary" data-act="update">지금 업데이트</button><span class="hint">앱이 잠깐 꺼졌다 다시 켜져요 (1~2분)</span></div>`
        : (latest.latest ? `<p>최신 버전이에요. <span class="hint">(${new Date(latest.checked).toLocaleTimeString()} 확인)</span></p>
            <div class="row"><button class="btn" data-act="check" ${busyAct === 'check' ? 'disabled' : ''}>${busyAct === 'check' ? '확인 중…' : '다시 확인'}</button></div>`
          : '<p class="hint">GitHub에 연결하지 못했어요. 인터넷 연결을 확인하세요.</p><div class="row"><button class="btn" data-act="check">다시 확인</button></div>'))
      : `<div class="row"><button class="btn" data-act="check" ${busyAct === 'check' ? 'disabled' : ''}>${busyAct === 'check' ? '확인 중…' : '업데이트 확인'}</button></div>`;
    const pct = d.total ? Math.round(d.done / d.total * 100) : 0;
    const eta = d.speed > 0 ? Math.max(0, Math.round((d.total - d.done) / d.speed / 60)) : null;
    const models = d.running
      ? `<p>받는 중 <b>${d.index}/${d.count}</b> · ${esc(d.file)}</p><div class="pbar"><i style="width:${pct}%"></i></div>
         <p class="hint">${pct}% · ${human(d.done)} / ${human(d.total)} · ${human(d.speed)}/s${eta !== null ? ` · 남은 시간 약 ${eta}분` : ''}</p>
         <div class="row"><button class="btn" data-act="dl-cancel">멈추기</button><span class="hint">멈춰도 다음에 이어서 받아요</span></div>`
      : (mm.length
        ? `<p>받지 않은 파일 <b>${mm.length}개</b> · 합계 <b>${human(info.models.missing_bytes)}</b></p>
           <details><summary class="hint">목록 보기</summary><ul class="mlist">${mm.map(m => `<li>${esc(m.file)} <span class="hint">${human(m.size)} · ${esc(m.label)}</span></li>`).join('')}</ul></details>
           <div class="row"><button class="btn primary" data-act="dl">모두 받기</button></div>`
        : '<p>필요한 모델을 모두 받았어요. ✓</p>')
      + (d.finished && d.failed && d.failed.length ? `<p class="warn">받지 못한 파일: ${d.failed.map(esc).join('<br>')}</p>` : '')
      + (d.error ? `<p class="warn">${esc(d.error)}</p><div class="row"><input id="civKey" type="password" placeholder="Civitai API 키 붙여넣기"><button class="btn" data-act="civkey">저장</button></div>
         <p class="hint">civitai.com 로그인 → 오른쪽 위 프로필 → Settings → API Keys → Add API key</p>` : '');
    const added = info.added || [];
    const addHtml = `<div class="addm"><p class="hint" style="margin-top:12px">Civitai 체크포인트 주소를 붙여넣으면 목록에 추가하고 바로 받아요. Z-Image Turbo, Illustrious·SDXL·Pony, Qwen 2.1 계열만 돼요.</p>
      <div class="row"><input id="addModelUrl" placeholder="https://civitai.red/models/…"><button class="btn" data-act="add-model">추가</button></div>
      ${added.length ? `<ul class="mlist">${added.map(m => `<li>${esc(m.name)} <span class="hint">${m.installed ? '받음' : '받는 중/대기'}${m.note ? ' · ' + esc(m.note) : ''}</span> <button class="ghost" data-act="rm-model" data-id="${esc(m.id)}">빼기</button></li>`).join('')}</ul>` : ''}</div>`;
    const un = info.unused;
    const unusedHtml = un && un.files.length
      ? `<details class="unused"><summary>안 쓰는 모델 파일 <b>${un.files.length}개</b> · ${human(un.bytes)} <span class="hint">(예전 모델·받다 만 파일)</span></summary>
          <ul class="mlist">${un.files.map(f => `<li>${esc(f.path)} <span class="hint">${human(f.size)}${f.partial ? ' · 받다 만 파일' : ''}</span></li>`).join('')}</ul>
          <div class="row"><button class="btn" data-act="unused-del">모두 지우기</button><span class="hint">지운 파일은 휴지통에 가지 않아요. 다시 쓰려면 다시 받아야 해요.</span></div></details>`
      : '';
    const cs = {running: ['ok', '켜져 있음'], starting: ['', '켜는 중…'], updating: ['', '업데이트 중… (몇 분)'], stopped: ['bad', '꺼짐'],
                missing: ['bad', '설치 안 됨'], error: ['bad', '문제 있음'], unknown: ['', '확인 중']}[c.alive ? 'running' : c.state] || ['', c.state];
    const comfy = `<p><i class="dot ${cs[0]}"></i> ${cs[1]}${c.message ? ` <span class="hint">${esc(c.message)}</span>` : ''}</p>
      <div class="row"><button class="btn" data-act="comfy-restart">다시 켜기</button><button class="btn" data-act="comfy-update">ComfyUI 업데이트</button></div>
      <p class="hint">새 기능(배경 제거, LoRA 학습, 포즈, 영상)이 안 되면 ComfyUI 업데이트를 눌러 보세요. 몇 분 걸리고, 그동안 생성은 멈춰요.</p>`;
    const remote = !r.tailscale
      ? `<p>Tailscale이 설치돼 있지 않아요.</p><p class="hint">tailscale.com 에서 설치하고 같은 계정으로 폰·노트북에도 로그인하면, 밖에서도 이 PC의 Gen Studio를 쓸 수 있어요.</p>`
      : `<label class="switch"><input type="checkbox" id="remoteOn" ${r.on ? 'checked' : ''}><span>폰·노트북에서 접속 허용</span></label>
         ${!r.logged_in ? '<p class="warn">Tailscale에 로그인돼 있지 않아요. 작업 표시줄의 Tailscale 아이콘에서 로그인하세요.</p>' : ''}
         ${r.on && r.link ? `<div class="qr"><img src="/api/app/remote/qr?t=${Date.now() >> 14}" alt="접속 QR 코드"><div>
            <p>폰 카메라로 QR을 찍거나 아래 주소를 여세요. 한 번 열면 그 기기는 기억돼요.</p>
            <div class="row"><input readonly value="${esc(r.link)}" id="remoteLink"><button class="btn" data-act="copy-link">복사</button></div>
            <p class="hint">이 주소에는 접속 열쇠가 들어 있어요. 다른 사람에게 보내지 마세요.</p></div></div>` : ''}`;
    const codex = `<p>${info.codex.connected ? '<i class="dot ok"></i> 연결돼 있어요' : '<i class="dot"></i> 아직 연결 안 됨'}</p>
      <div class="row"><button class="btn ${info.codex.connected ? '' : 'primary'}" data-act="codex">${info.codex.connected ? '다시 연결' : 'Codex에 연결하기'}</button></div>
      <p class="hint">연결한 뒤 Codex 앱을 완전히 껐다 켜면 "Gen Studio" 도구가 보여요. 주소: ${esc(info.codex.url)}</p>`;
    const lib = info.library;
    const libHtml = lib ? `<p>이미지·영상 <b>${lib.count}개</b> · ${human(lib.bytes + lib.comfy_bytes)} <span class="hint">(ComfyUI 사본 ${human(lib.comfy_bytes)} 포함)</span></p>
      <p class="hint">저장 폴더: ${esc(lib.folder)}</p>
      <div class="row"><button class="btn danger" data-act="lib-clear" ${lib.count || lib.comfy_bytes ? '' : 'disabled'}>모든 이미지·영상 지우기</button></div>
      <p class="hint">모든 버전, 세그먼트, 썸네일, 등록한 캐릭터, ComfyUI가 따로 남긴 사본까지 지워요. 휴지통에 가지 않고 되돌릴 수 없어요. 모델·LoRA·내 칩·설정은 그대로예요. 다운로드 폴더로 받아둔 파일은 건드리지 않아요.</p>` : '';
    const help = `<div class="row"><button class="btn" data-act="diag">진단 정보 복사</button>${info.desktop ? '<button class="btn" data-act="logs">로그 폴더 열기</button>' : ''}</div>
      <p class="hint">문제가 생기면 진단 정보를 복사해 대화창에 붙여넣어 주세요. 로그: ${esc(info.logs)}</p>`;
    body.innerHTML = [
      card('업데이트', (info.last_error ? `<p class="warn">지난 업데이트 실패: ${esc(info.last_error)}</p>` : '') + upd),
      card('모델', models + addHtml + unusedHtml), card('ComfyUI', comfy), card('원격 접속', remote), card('Codex', codex), card('저장된 이미지', libHtml), card('문제 해결', help),
    ].join('');
  }
  body.addEventListener('change', async e => {
    if (e.target.id !== 'remoteOn') return;
    e.target.disabled = true;
    try { const r = await api.post('/api/app/remote', {on: e.target.checked}); if (!r.ok && r.message) toast(r.message, 8000); } catch (err) { toast(err.message, 6000); }
    load();
  });
  body.addEventListener('click', async e => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const act = b.dataset.act;
    try {
      if (act === 'check') return checkUpdate();
      if (act === 'update') {
        if (info.downloads.running && b.dataset.armed !== '1') { b.dataset.armed = '1'; b.textContent = '받는 중인 모델은 멈췄다 이어 받아요. 한 번 더 누르면 업데이트'; return; }
        const r = await api.post('/api/app/update'); toast(r.message, 15000); b.disabled = true; return;
      }
      if (act === 'lib-clear') {
        const n = info.library.count;
        if (b.dataset.armed !== '1') { b.dataset.armed = '1'; b.textContent = `정말 ${n}개를 전부 지울까요? 한 번 더 누르기`; return; }
        b.disabled = true; b.textContent = '지우는 중…';
        const r = await api.post('/api/app/library/clear', {confirm: '모두 지우기'});
        toast(`${r.deleted}개를 지웠어요 (${human(r.bytes)} 비움)`, 6000);
        return load();
      }
      if (act === 'add-model') {
        const url = $('#addModelUrl').value.trim(); if (!url) return;
        b.disabled = true;
        try { const r = await api.post('/api/app/models/add', {url}); toast(`'${r.model.name}'을(를) 추가했어요.${r.downloading ? ' 받기 시작했어요.' : ''}`, 6000); MODELS = await api.get('/api/models'); fillModelSelect(true); }
        finally { b.disabled = false; }
        return load();
      }
      if (act === 'rm-model') { await api.del(`/api/app/models/added/${b.dataset.id}`); MODELS = await api.get('/api/models'); fillModelSelect(true); return load(); }
      if (act === 'unused-del') {
        if (b.dataset.armed !== '1') { b.dataset.armed = '1'; b.textContent = '정말 지울까요? 한 번 더 누르기'; return; }
        const r = await api.post('/api/app/models/unused/delete', {paths: info.unused.files.map(f => f.path)});
        toast(`${r.deleted}개를 지웠어요`); return load();
      }
      if (act === 'dl') { await api.post('/api/app/models/download'); return load(); }
      if (act === 'dl-cancel') { await api.post('/api/app/models/cancel'); return load(); }
      if (act === 'civkey') { await api.post('/api/app/civitai-key', {key: $('#civKey').value}); toast('저장했어요. 모두 받기를 다시 누르세요.'); return load(); }
      if (act === 'comfy-restart' || act === 'comfy-update') { await api.post(`/api/app/comfy/${act.slice(6)}`); return load(); }
      if (act === 'codex') { const r = await api.post('/api/app/codex'); toast(r.message, 7000); return load(); }
      if (act === 'copy-link') { await navigator.clipboard.writeText($('#remoteLink').value).catch(() => {}); toast('복사했어요'); return; }
      if (act === 'diag') return copyDiagnostics();
      if (act === 'logs') return api.post('/api/app/open-logs');
    } catch (err) { toast(err.message, 6000); }
  });
  return {active(v) { on = v; clearTimeout(timer); if (v) { load(); if (!latest || Date.now() - latest.checked > 60000) checkUpdate(); } }};
})();

async function importFile(file) {
  if (!file || !file.type.startsWith('image/')) { toast('이미지 파일만 불러올 수 있어요'); return; }
  const fd = new FormData(); fd.append('file', file, file.name || 'pasted.png');
  try { const rec = await req('POST', '/api/import', fd); location.hash = 'edit=' + rec.id; toast('불러왔어요'); }
  catch (e) { toast(e.message, 5000); }
}
$('#importBtn').onclick = () => $('#fileIn').click();
$('#fileIn').onchange = e => { importFile(e.target.files[0]); e.target.value = ''; };
let dragDepth = 0;
addEventListener('dragenter', e => { if ([...(e.dataTransfer?.types || [])].includes('Files')) { dragDepth++; $('#drop').hidden = false; } });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('#drop').hidden = true; } });
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; $('#drop').hidden = true; importFile(e.dataTransfer.files[0]); });
addEventListener('paste', e => { const f = [...(e.clipboardData?.files || [])][0]; if (f && f.type.startsWith('image/')) importFile(f); });

/* ========================================================= skeleton editor */
/* Drag the joints of one or more people to make a pose, with hands, face, feet and a camera you can turn.
   Saved as a pose map (/api/poses) that drawing, "자세 바꾸기" and pose videos use like a pose read from a photo.
   Model space: body joints [x, y, z] (x, y in 0..1 of the canvas, z in canvas heights, + = away); what is
   drawn is that seen from the camera. The maths matches genstudio/poses.py (make_pose for Codex). */
const PoseEd = (() => {
  let META = null;
  const ASPECTS = [['2:3', 2, 3, '세로'], ['3:2', 3, 2, '가로'], ['1:1', 1, 1, '정사각'], ['9:16', 9, 16, '세로 길게'], ['16:9', 16, 9, '가로 길게']];
  const ARMS = {right: [2, 3, 4], left: [5, 6, 7]}, LEGS = {right: [8, 9, 10], left: [11, 12, 13]};
  const canvasSize = (a, b) => { const k = Math.sqrt(1024 * 1024 / (a * b)); return [Math.max(256, Math.round(a * k / 64) * 64), Math.max(256, Math.round(b * k / 64) * 64)]; };
  const r4 = v => Math.round(v * 10000) / 10000;
  const clone = v => JSON.parse(JSON.stringify(v));
  async function meta() { if (!META) META = await api.get('/api/poses'); return META; }

  function preset3d(id, W, H) {
    const p = META.presets.find(x => x.id === id) || META.presets[0];
    const s = Math.min(W / 2, H / 3), k = 3 * s / H;
    return p.joints.map(([x, y], i) => [r4(0.5 + (x - 0.5) * 2 * s / W), r4(0.5 + (y - 0.5) * 3 * s / H), r4(p.z[i] * k)]);
  }
  function sceneCenter(people, W, H) {
    const hips = people.flatMap(p => [p.body[8], p.body[11]]).filter(Boolean);
    if (!hips.length) return [W / 2, H / 2];
    return [hips.reduce((a, j) => a + j[0], 0) / hips.length * W, hips.reduce((a, j) => a + j[1], 0) / hips.length * H];
  }
  function rot(cam) {
    const y = cam.yaw * Math.PI / 180, p = cam.pitch * Math.PI / 180;
    return {cy: Math.cos(y), sy: Math.sin(y), cp: Math.cos(p), sp: Math.sin(p)};
  }
  function project(j, W, H, cam, c) {               // model -> [x, y, depth] seen from the camera
    if (!j) return null;
    const R = rot(cam), x = j[0] * W - c[0], y = j[1] * H - c[1], z = (j[2] || 0) * H;
    const x1 = x * R.cy + z * R.sy, z1 = -x * R.sy + z * R.cy;
    const y1 = y * R.cp - z1 * R.sp, z2 = y * R.sp + z1 * R.cp;
    return [(c[0] + x1) / W, (c[1] + y1) / H, z2 / H];
  }
  function unproject(q, depth, W, H, cam, c) {      // screen point + its depth -> model
    const R = rot(cam), x1 = q[0] * W - c[0], y1 = q[1] * H - c[1], z2 = depth * H;
    const y = y1 * R.cp + z2 * R.sp, z1 = -y1 * R.sp + z2 * R.cp;
    const x = x1 * R.cy - z1 * R.sy, z = x1 * R.sy + z1 * R.cy;
    return [r4((c[0] + x) / W), r4((c[1] + y) / H), r4(z / H)];
  }
  function handFrame(body, side, W, H) {
    const s = body[ARMS[side][0]], e = body[ARMS[side][1]], w = body[ARMS[side][2]];
    if (!e || !w) return null;
    const ex = e[0] * W, ey = e[1] * H, wx = w[0] * W, wy = w[1] * H, L = Math.hypot(wx - ex, wy - ey);
    if (L < 1) return null;
    const ux = (wx - ex) / L, uy = (wy - ey) / L; let px = -uy, py = ux;
    const ref = body[1] || s;
    if (ref && px * (ref[0] * W - wx) + py * (ref[1] * H - wy) < 0) { px = -px; py = -py; }
    return {wx, wy, L, ux, uy, px, py};
  }
  function handPoints(shape, body, side, W, H) {
    const f = handFrame(body, side, W, H); if (!f || !shape) return null;
    const local = typeof shape === 'string' ? (META.hands.find(h => h.id === shape) || META.hands[0]).points : shape;
    return local.map(([lx, ly]) => [r4((f.wx + f.L * (lx * f.px + ly * f.ux)) / W), r4((f.wy + f.L * (lx * f.py + ly * f.uy)) / H)]);
  }
  function handLocal(pt, body, side, W, H) {
    const f = handFrame(body, side, W, H); if (!f) return null;
    const vx = pt[0] * W - f.wx, vy = pt[1] * H - f.wy;
    return [r4((vx * f.px + vy * f.py) / f.L), r4((vx * f.ux + vy * f.uy) / f.L)];
  }
  function facePoints(body, W, H) {
    const n = body[0], er = body[14], el = body[15];
    if (!n || !er || !el) return null;
    const [t1, t2] = META.face_eyes;
    const ax = er[0] * W, ay = er[1] * H, bx = el[0] * W, by = el[1] * H;
    const dx = t2[0] - t1[0], dy = t2[1] - t1[1], den = dx * dx + dy * dy;
    const kr = ((bx - ax) * dx + (by - ay) * dy) / den, ki = ((by - ay) * dx - (bx - ax) * dy) / den;
    const pts = META.face.map(([x, y]) => { const vx = x - t1[0], vy = y - t1[1]; return [ax + vx * kr - vy * ki, ay + vx * ki + vy * kr]; });
    const ddx = n[0] * W - pts[30][0], ddy = n[1] * H - pts[30][1];
    const wgt = i => ({27: .25, 28: .5, 29: .75, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1})[i] ?? (i >= 48 ? .5 : i <= 16 ? .2 : 0);
    return pts.map(([x, y], i) => [r4((x + ddx * wgt(i)) / W), r4((y + ddy * wgt(i)) / H)]);
  }
  function feet3d(body, W, H) {
    const out = [];
    for (const [side, inward] of [['left', -1], ['right', 1]]) {
      const kn = body[LEGS[side][1]], an = body[LEGS[side][2]];
      if (!kn || !an) { out.push(null, null, null); continue; }
      const L = Math.max(1, Math.hypot((kn[0] - an[0]) * W, (kn[1] - an[1]) * H, ((kn[2] || 0) - (an[2] || 0)) * H));
      for (const [dx, dy, dz] of [[inward * .06, .18, -.38], [-inward * .08, .18, -.3], [0, .1, .12]])
        out.push([an[0] + dx * L / W, an[1] + dy * L / H, (an[2] || 0) + dz * L / H]);
    }
    return out;
  }
  /* what is drawn: per person {body, right_hand, left_hand, face, feet} in 0..1, as the camera sees it */
  function compose(S) {
    return S.people.map(p => {
      const b = p.body.map(j => project(j, S.W, S.H, S.cam, S.center));
      const [farEye, farEar] = S.cam.yaw > 0 ? [14, 16] : [15, 17];      // the far side goes behind the head
      if (Math.abs(S.cam.yaw) >= 30) b[farEar] = null;
      if (Math.abs(S.cam.yaw) >= 55) b[farEye] = null;
      const body = b.map(j => j && [r4(j[0]), r4(j[1])]);
      const o = {body, right_hand: null, left_hand: null, face: null, feet: null};
      for (const side of ['right', 'left']) if (p.hands[side]) o[side + '_hand'] = handPoints(p.hands[side], body, side, S.W, S.H);
      if (S.face) o.face = facePoints(body, S.W, S.H);
      if (S.feet) o.feet = feet3d(p.body, S.W, S.H).map(j => { const q = project(j, S.W, S.H, S.cam, S.center); return q && [r4(q[0]), r4(q[1])]; });
      return o;
    });
  }
  function fromSaved(kp, W, H) {             // a saved pose or one read from a photo -> editor people
    if (kp?.edit?.people3d) return kp.edit.people3d.map(p => ({body: clone(p.body), hands: clone(p.hands || {right: 'relaxed', left: 'relaxed'})}));
    return (kp?.people || []).map(p => {
      const d = Array.isArray(p) ? {body: p} : p;
      const body = d.body.map(j => j && [j[0], j[1], 0]);
      const hands = {};
      for (const side of ['right', 'left']) {
        const pts = d[side + '_hand'];
        hands[side] = pts && pts.every(Boolean) ? pts.map(q => handLocal(q, d.body, side, W, H)) : 'relaxed';
        if (Array.isArray(hands[side]) && hands[side].some(v => !v)) hands[side] = 'relaxed';
      }
      return {body, hands};
    });
  }

  async function open(opts = {}) {
    try { await meta(); } catch (e) { toast(e.message); return; }
    const S = {W: 832, H: 1216, people: [], cam: {yaw: 0, pitch: 0}, center: null, face: true, feet: true};
    let sel = 0, showBg = true, history = [], aspect = '2:3', busyRead = false, ik = store.get('pose.ik', true);
    let fingers = false, handSide = 'both', bg = opts.bg || null, bgRec = null, name = '', savedAs = null;
    if (bg) {
      try { bgRec = await api.get(`/api/images/${bg}`); } catch { bg = null; }
      if (bgRec) { [S.W, S.H] = canvasSize(bgRec.width, bgRec.height); aspect = null; }
    }
    const start = opts.start ? await api.get(`/api/images/${opts.start}`).catch(() => null) : null;
    if (start?.keypoints?.people?.length) {
      if (!bg) { [S.W, S.H] = canvasSize(start.width, start.height); aspect = null; }
      S.people = fromSaved(start.keypoints, S.W, S.H);
      const ed = start.keypoints.edit; if (ed) { S.cam = {...S.cam, ...(ed.camera || {})}; S.face = ed.face ?? true; S.feet = ed.feet ?? true; }
    } else S.people = [{body: preset3d('stand', S.W, S.H), hands: {right: 'relaxed', left: 'relaxed'}}];
    const recenter = () => { S.center = sceneCenter(S.people, S.W, S.H); };
    recenter();

    const d = document.createElement('dialog'); d.className = 'posedlg';
    d.innerHTML = `<div class="phead"><h3>${esc(opts.title || '뼈대로 자세 만들기')}</h3><button type="button" class="px" data-close aria-label="닫기">×</button></div>
      <div class="pbody"><div class="pstagewrap"><div class="pstage"><img alt="" hidden><svg xmlns="http://www.w3.org/2000/svg"></svg></div></div>
      <div class="pside"></div></div>`;
    document.body.appendChild(d);
    const stage = d.querySelector('.pstage'), svg = d.querySelector('svg'), bgImg = d.querySelector('.pstage img'), side = d.querySelector('.pside');
    const wrap = d.querySelector('.pstagewrap');
    if (bg) bgImg.src = imgUrl(bg);
    const close = () => { d.close(); d.remove(); };
    d.addEventListener('close', () => d.remove());
    const snap = () => { history.push(clone({people: S.people, W: S.W, H: S.H, cam: S.cam, sel})); if (history.length > 60) history.shift(); };

    function fitStage() {
      const cs = getComputedStyle(wrap);
      const aw = wrap.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      const ah = wrap.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
      const k = Math.max(0.01, Math.min(aw / S.W, ah / S.H));
      stage.style.width = Math.floor(S.W * k) + 'px'; stage.style.height = Math.floor(S.H * k) + 'px';
    }
    new ResizeObserver(fitStage).observe(wrap);
    function placeholder(body2d) {
      const t = preset3d('stand', S.W, S.H).map(j => project(j, S.W, S.H, S.cam, S.center));
      const ref = body2d[1] ? 1 : body2d.findIndex(j => j);
      if (ref < 0) return t;
      const dx = body2d[ref][0] - t[ref][0], dy = body2d[ref][1] - t[ref][1];
      return t.map(([x, y]) => [x + dx, y + dy]);
    }
    let drawn = [];
    function paintStage() {
      fitStage();
      bgImg.hidden = !bg || !showBg;
      svg.setAttribute('viewBox', `0 0 ${S.W} ${S.H}`);
      drawn = compose(S);
      const W = S.W, H = S.H, r = Math.max(W, H) / 95, sw = Math.max(W, H) / 110, hr = Math.max(W, H) / 260;
      const rgb = c => `rgb(${c.join(',')})`;
      let h = '';
      drawn.forEach((p, pi) => {
        const on = pi === sel;
        h += `<g class="pp ${on ? 'on' : ''}">`;
        META.limbs.forEach(([a, b], i) => {
          const A = p.body[a - 1], B = p.body[b - 1]; if (!A || !B) return;
          h += `<line data-p="${pi}" data-limb="1" x1="${A[0] * W}" y1="${A[1] * H}" x2="${B[0] * W}" y2="${B[1] * H}" stroke="${rgb(META.colors[i % 18])}" stroke-width="${sw}" stroke-linecap="round"/>`;
        });
        for (const sd of ['right_hand', 'left_hand']) {
          const hp = p[sd]; if (!hp) continue;
          META.hand_edges.forEach(([a, b], i) => { h += `<line class="hl" x1="${hp[a][0] * W}" y1="${hp[a][1] * H}" x2="${hp[b][0] * W}" y2="${hp[b][1] * H}" stroke="${rgb(META.hand_colors[i])}" stroke-width="${hr}"/>`; });
          hp.forEach((q, k) => {
            h += `<circle class="hd" cx="${q[0] * W}" cy="${q[1] * H}" r="${hr * (on && fingers ? 1.3 : 0.9)}"/>`;
            if (on && fingers && k) h += `<circle class="hit fh" data-p="${pi}" data-hand="${sd.split('_')[0]}" data-k="${k}" cx="${q[0] * W}" cy="${q[1] * H}" r="${hr * 4}"/>`;
          });
        }
        (p.feet || []).forEach((q, i) => { if (q) h += `<circle class="ft" cx="${q[0] * W}" cy="${q[1] * H}" r="${r * .6}" fill="${rgb(META.colors[i % 18])}"/>`; });
        (p.face || []).forEach(q => { h += `<circle class="fc" cx="${q[0] * W}" cy="${q[1] * H}" r="${hr}"/>`; });
        if (on && !fingers) placeholder(p.body).forEach((q, j) => { if (!p.body[j] && !S.people[pi].body[j]) h += `<circle class="miss" data-p="${pi}" data-j="${j}" cx="${q[0] * W}" cy="${q[1] * H}" r="${r}"><title>${esc(META.joints_ko[j])} (빠짐: 끌어서 넣기)</title></circle>`; });
        p.body.forEach((q, j) => {
          if (!q) return;
          h += `<circle class="jt" cx="${q[0] * W}" cy="${q[1] * H}" r="${r}" fill="${rgb(META.colors[j % 18])}"><title>${esc(META.joints_ko[j])}</title></circle>`;
          if (!(on && fingers)) h += `<circle class="hit" data-p="${pi}" data-j="${j}" cx="${q[0] * W}" cy="${q[1] * H}" r="${r * 2.4}"/>`;
        });
        h += '</g>';
      });
      svg.innerHTML = h;
    }
    function handButtons(sideKey) {
      const p = S.people[sel]; if (!p) return '';
      const cur = sideKey === 'both' ? (p.hands.right === p.hands.left ? p.hands.right : null) : p.hands[sideKey];
      return META.hands.map(hd => `<button data-hshape="${hd.id}" aria-pressed="${cur === hd.id}">${esc(hd.name)}</button>`).join('')
        + `<button data-hshape="" aria-pressed="${cur === null && sideKey !== 'both' ? 'true' : 'false'}">손 없음</button>`;
    }
    function paintSide() {
      const saved = META.saved || [];
      side.innerHTML = `
        <p class="note">관절(점)을 끌어서 자세를 만드세요. 선을 끌면 사람 전체가 움직이고, 점을 두 번 누르면 숨겨져요.</p>
        ${bg ? `<div class="field"><div class="seg2"><button data-pact="read" ${busyRead ? 'disabled' : ''}>${busyRead ? '읽는 중…' : '사진에서 뼈대 읽기'}</button>
          <button data-pact="bg" aria-pressed="${showBg}">사진 ${showBg ? '보임' : '숨김'}</button></div></div>` : ''}
        <div class="field"><span>모양 (${S.people.length ? `${sel + 1}번 사람` : '사람 없음'})</span><div class="seg2">${META.presets.map(p => `<button data-preset="${p.id}">${esc(p.name)}</button>`).join('')}</div></div>
        <div class="field"><span>사람</span><div class="seg2">
          ${S.people.map((_, i) => `<button data-psel="${i}" aria-pressed="${i === sel}">${i + 1}번</button>`).join('')}
          <button data-pact="add">+ 사람</button><button data-pact="del" ${S.people.length ? '' : 'disabled'}>지우기</button>
          <button data-pact="flip" ${S.people.length ? '' : 'disabled'}>좌우 뒤집기</button><button data-pact="undo" ${history.length ? '' : 'disabled'}>되돌리기</button></div></div>
        <div class="field"><span>팔다리</span><div class="seg2">
          <button data-pact="ik" aria-pressed="${ik}">손발 끌면 팔다리 따라오기</button>
          <button data-copy="left">왼쪽 → 오른쪽 복사</button><button data-copy="right">오른쪽 → 왼쪽 복사</button></div></div>
        <div class="field"><span>손</span><div class="seg2">${[['both', '양손'], ['right', '오른손'], ['left', '왼손']].map(([k, n]) => `<button data-hside="${k}" aria-pressed="${k === handSide}">${n}</button>`).join('')}
          <button data-pact="fingers" aria-pressed="${fingers}">손가락 끌기</button></div>
          <div class="seg2" style="margin-top:6px">${handButtons(handSide)}</div></div>
        <div class="field"><span>카메라 각도 · 옆 ${Math.round(S.cam.yaw)}° · 위 ${Math.round(S.cam.pitch)}°</span>
          <input type="range" id="pYaw" min="-90" max="90" step="5" value="${S.cam.yaw}" aria-label="옆에서 보기">
          <input type="range" id="pPitch" min="-60" max="60" step="5" value="${S.cam.pitch}" aria-label="위에서 보기">
          <div class="seg2"><button data-cam="0,0">정면</button><button data-cam="45,0">비스듬히</button><button data-cam="90,0">옆</button><button data-cam="0,35">위에서</button><button data-cam="0,-25">아래에서</button></div></div>
        <label class="switch"><input type="checkbox" id="pFace" ${S.face ? 'checked' : ''}><span>얼굴 (눈·코 방향을 따라감)</span></label>
        <label class="switch" style="margin-bottom:14px"><input type="checkbox" id="pFeet" ${S.feet ? 'checked' : ''}><span>발</span></label>
        ${bg ? '' : `<div class="field"><span>비율</span><div class="seg2">${ASPECTS.map(([k, , , n]) => `<button data-asp="${k}" aria-pressed="${k === aspect}">${n} ${k}</button>`).join('')}</div></div>`}
        ${saved.length ? `<div class="field"><span>저장된 자세</span><div class="vpick">${saved.map(s => `<button data-saved="${s.id}" title="${esc(s.name)}"><img src="${thumbUrl(s.id)}" alt="${esc(s.name)}"></button>`).join('')}</div></div>` : ''}
        <div class="field"><input id="pName" placeholder="이름 붙여 저장 (예: 의자에 기대기)" value="${esc(name)}" maxlength="60"></div>
        <div class="field"><div class="seg2"><button data-pact="use" aria-pressed="true" ${S.people.length ? '' : 'disabled'}>${esc(opts.useLabel || '이 자세로 그리기')}</button>
          <button data-pact="save" ${S.people.length ? '' : 'disabled'}>저장만</button></div></div>`;
    }
    const paint = () => { paintStage(); paintSide(); };

    function setAspect(k) {
      const a = ASPECTS.find(x => x[0] === k); if (!a) return;
      snap();
      const [nW, nH] = canvasSize(a[1], a[2]); const s = Math.min(nW / S.W, nH / S.H);
      S.people.forEach(p => { p.body = p.body.map(j => j && [r4(0.5 + (j[0] * S.W - S.W / 2) * s / nW), r4(0.5 + (j[1] * S.H - S.H / 2) * s / nH), r4((j[2] || 0) * S.H * s / nH)]); });
      S.W = nW; S.H = nH; aspect = k; recenter(); paint();
    }
    function addPerson() {
      snap();
      const b = preset3d('stand', S.W, S.H);
      const xs = S.people.flatMap(p => p.body).filter(Boolean).map(j => j[0]);
      const mean = xs.length ? xs.reduce((a, v) => a + v, 0) / xs.length : 0.5;
      const dx = S.people.length ? (mean > 0.5 ? 0.27 : 0.73) - b[1][0] : 0;
      const k = S.people.length ? 0.9 : 1, cx = b[1][0], cy = b[1][1];
      S.people.push({body: b.map(([x, y, z]) => [r4(cx + (x - cx) * k + dx), r4(cy + (y - cy) * k), r4(z * k)]), hands: {right: 'relaxed', left: 'relaxed'}});
      sel = S.people.length - 1; recenter(); paint();
    }
    function copySide(src) {
      const dst = src === 'left' ? 'right' : 'left', p = S.people[sel]; if (!p) return;
      snap();
      for (const g of [ARMS, LEGS]) {
        const a = g[src], b = g[dst]; if (!p.body[a[0]] || !p.body[b[0]]) continue;
        for (let n = 1; n < 3; n++) {
          const js = p.body[a[n]];
          p.body[b[n]] = js && [r4(p.body[b[0]][0] - (js[0] - p.body[a[0]][0])), js[1], js[2] || 0];
        }
      }
      p.hands[dst] = clone(p.hands[src]);
      paint();
    }
    async function readPhoto() {
      if (!bg || busyRead) return;
      busyRead = true; paintSide();
      try {
        const j = await pollJob((await api.post(`/api/images/${bg}/read-pose`)).id);
        if (j.status !== 'done') throw new Error(j.error || '뼈대를 읽지 못했어요');
        snap(); S.people = fromSaved(j.result.keypoints, S.W, S.H); S.cam = {yaw: 0, pitch: 0}; sel = 0; recenter();
        toast(`${S.people.length}명의 뼈대를 읽었어요`);
      } catch (e) { toast(e.message, 6000); }
      busyRead = false; paint();
    }
    async function save(use) {
      name = (d.querySelector('#pName')?.value || '').trim();
      if (!use && !name) { d.querySelector('#pName').focus(); toast('저장할 이름을 적어 주세요'); return; }
      const people = compose(S);
      const edit = {people3d: clone(S.people), camera: S.cam, face: S.face, feet: S.feet};
      const key = JSON.stringify([people, S.W, S.H, name]);
      if (savedAs && savedAs.key === key) {                 // nothing changed since "저장만": use that one
        if (use) { close(); await opts.onUse?.(savedAs.id); } else toast('이미 저장했어요');
        return;
      }
      try {
        const rec = await api.post('/api/poses', {people, width: S.W, height: S.H, name, parent_id: bg, edit});
        savedAs = {id: rec.id, key};
        META = null; await meta().catch(() => {});
        if (use) { close(); await opts.onUse?.(rec.id); } else { toast(`'${name}' 자세를 저장했어요`); paintSide(); }
      } catch (e) { toast(e.message, 6000); }
    }

    /* dragging: a joint moves itself (with IK the elbow/knee follows a wrist/ankle), a limb moves the person,
       in finger mode the hand points move. Everything happens on screen and goes back to model space. */
    let drag = null;
    const toCanvas = e => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const q = pt.matrixTransform(svg.getScreenCTM().inverse()); return [q.x / S.W, q.y / S.H]; };
    const scr = (pi, j) => project(S.people[pi].body[j], S.W, S.H, S.cam, S.center);
    const setScr = (pi, j, q, depth) => { S.people[pi].body[j] = unproject(q, depth, S.W, S.H, S.cam, S.center); };
    function ikMove(pi, j, q) {
      const limb = [...Object.values(ARMS), ...Object.values(LEGS)].find(l => l[2] === j);
      const A = limb && scr(pi, limb[0]), B = limb && scr(pi, limb[1]), C = scr(pi, j);
      if (!A || !B || !C) return false;
      const W = S.W, H = S.H, P = v => [v[0] * W, v[1] * H];
      const [ax, ay] = P(A), [bx, by] = P(B), [cx, cy] = P(C), [tx, ty] = P(q);
      const l1 = Math.hypot(bx - ax, by - ay), l2 = Math.hypot(cx - bx, cy - by);
      let dd = Math.hypot(tx - ax, ty - ay); if (dd < 1e-3 || l1 < 1 || l2 < 1) return false;
      dd = Math.min(l1 + l2 - 0.01, Math.max(Math.abs(l1 - l2) + 0.01, dd));
      const ux = (tx - ax) / Math.hypot(tx - ax, ty - ay), uy = (ty - ay) / Math.hypot(tx - ax, ty - ay);
      const bend = Math.sign((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) || 1;
      const ang = Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + dd * dd - l2 * l2) / (2 * l1 * dd))));
      const ca = Math.cos(-bend * ang), sa = Math.sin(-bend * ang);
      const nb = [(ax + l1 * (ux * ca - uy * sa)) / W, (ay + l1 * (ux * sa + uy * ca)) / H];
      setScr(pi, limb[1], nb, B[2]);
      setScr(pi, j, [(ax + ux * dd) / W, (ay + uy * dd) / H], C[2]);
      return true;
    }
    svg.addEventListener('pointerdown', e => {
      const t = e.target.closest('[data-p]'); if (!t) return;
      e.preventDefault();
      const pi = +t.dataset.p; snap(); sel = pi;
      const q = toCanvas(e);
      if (t.dataset.hand) {
        const sd = t.dataset.hand, p = S.people[pi], body2d = drawn[pi].body;
        if (typeof p.hands[sd] === 'string') p.hands[sd] = clone((META.hands.find(x => x.id === p.hands[sd]) || META.hands[0]).points);
        drag = {pi, hand: sd, k: +t.dataset.k, body2d};
      } else if (t.dataset.j !== undefined) {
        const j = +t.dataset.j;
        if (!S.people[pi].body[j]) { const n = scr(pi, 1); setScr(pi, j, q, n ? n[2] : 0); }
        drag = {pi, j, depth: scr(pi, j)[2]};
      } else if (t.dataset.limb) drag = {pi, move: q};
      svg.setPointerCapture(e.pointerId); paint();
    });
    svg.addEventListener('pointermove', e => {
      if (!drag) return;
      const q = toCanvas(e).map(v => Math.min(1.2, Math.max(-0.2, v)));
      const p = S.people[drag.pi];
      if (drag.hand) { const lp = handLocal(q, drag.body2d, drag.hand, S.W, S.H); if (lp) p.hands[drag.hand][drag.k] = lp; }
      else if (drag.j !== undefined) { if (!(ik && [4, 7, 10, 13].includes(drag.j) && ikMove(drag.pi, drag.j, q))) setScr(drag.pi, drag.j, q, drag.depth); }
      else {
        const dx = q[0] - drag.move[0], dy = q[1] - drag.move[1]; drag.move = q;
        p.body = p.body.map(j => { if (!j) return j; const s = project(j, S.W, S.H, S.cam, S.center); return unproject([s[0] + dx, s[1] + dy], s[2], S.W, S.H, S.cam, S.center); });
      }
      paintStage();
    });
    const end = () => { if (drag) { drag = null; paintSide(); } };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
    svg.addEventListener('dblclick', e => {
      const t = e.target.closest('[data-j]'); if (!t || !S.people[+t.dataset.p].body[+t.dataset.j]) return;
      snap(); S.people[+t.dataset.p].body[+t.dataset.j] = null; paint();
    });

    d.addEventListener('input', e => {
      if (e.target.id === 'pName') name = e.target.value;
      if (e.target.id === 'pYaw' || e.target.id === 'pPitch') {
        S.cam = {yaw: +d.querySelector('#pYaw').value, pitch: +d.querySelector('#pPitch').value};
        paintStage(); d.querySelector('.pside .field:has(#pYaw) > span').textContent = `카메라 각도 · 옆 ${S.cam.yaw}° · 위 ${S.cam.pitch}°`;
      }
    });
    d.addEventListener('change', e => {
      if (e.target.id === 'pYaw' || e.target.id === 'pPitch') snap();
      if (e.target.id === 'pFace') { S.face = e.target.checked; paintStage(); }
      if (e.target.id === 'pFeet') { S.feet = e.target.checked; paintStage(); }
    });
    d.addEventListener('click', async e => {
      if (e.target.closest('[data-close]')) { close(); return; }
      const pr = e.target.closest('[data-preset]');
      if (pr) {
        snap();
        const np = preset3d(pr.dataset.preset, S.W, S.H);
        if (!S.people.length) { S.people.push({body: np, hands: {right: 'relaxed', left: 'relaxed'}}); sel = 0; }
        else {                                           // keep the person where they stand
          const old = S.people[sel].body, ref = old[1] ? 1 : old.findIndex(j => j);
          const dx = ref >= 0 ? old[ref][0] - np[ref][0] : 0;
          S.people[sel].body = np.map(([x, y, z]) => [r4(x + dx), y, z]);
        }
        recenter(); paint(); return;
      }
      const ps = e.target.closest('[data-psel]'); if (ps) { sel = +ps.dataset.psel; paint(); return; }
      const as = e.target.closest('[data-asp]'); if (as) { setAspect(as.dataset.asp); return; }
      const cp = e.target.closest('[data-copy]'); if (cp) { copySide(cp.dataset.copy); return; }
      const hs = e.target.closest('[data-hside]'); if (hs) { handSide = hs.dataset.hside; paintSide(); return; }
      const hsh = e.target.closest('[data-hshape]');
      if (hsh && S.people[sel]) {
        snap(); const v = hsh.dataset.hshape || null;
        for (const s of handSide === 'both' ? ['right', 'left'] : [handSide]) S.people[sel].hands[s] = v;
        paint(); return;
      }
      const cm = e.target.closest('[data-cam]');
      if (cm) { snap(); const [y, p] = cm.dataset.cam.split(',').map(Number); S.cam = {yaw: y, pitch: p}; paint(); return; }
      const sv = e.target.closest('[data-saved]');
      if (sv) {
        const s = META.saved.find(x => x.id === sv.dataset.saved); if (!s?.keypoints) return;
        snap();
        if (!bg) { S.W = s.width; S.H = s.height; aspect = null; }
        S.people = fromSaved(s.keypoints, S.W, S.H); sel = 0; name = s.name || '';
        const ed = s.keypoints.edit; S.cam = {yaw: 0, pitch: 0, ...(ed?.camera || {})};
        if (ed) { S.face = ed.face ?? true; S.feet = ed.feet ?? true; }
        recenter(); paint(); return;
      }
      const a = e.target.closest('[data-pact]'); if (!a) return;
      const act = a.dataset.pact;
      if (act === 'add') addPerson();
      if (act === 'del' && S.people.length) { snap(); S.people.splice(sel, 1); sel = Math.max(0, sel - 1); recenter(); paint(); }
      if (act === 'flip' && S.people.length) {
        snap(); const m = META.mirror, p = S.people[sel];
        p.body = p.body.map((_, i) => { const j = p.body[m[i]]; return j && [r4(1 - j[0]), j[1], j[2] || 0]; });
        p.hands = {right: p.hands.left, left: p.hands.right}; paint();
      }
      if (act === 'undo' && history.length) { const h = history.pop(); S.people = h.people; S.W = h.W; S.H = h.H; S.cam = h.cam; sel = Math.min(h.sel, Math.max(0, S.people.length - 1)); recenter(); paint(); }
      if (act === 'ik') { ik = !ik; store.set('pose.ik', ik); paintSide(); }
      if (act === 'fingers') { fingers = !fingers; paint(); }
      if (act === 'bg') { showBg = !showBg; paint(); }
      if (act === 'read') readPhoto();
      if (act === 'use') save(true);
      if (act === 'save') save(false);
    });
    paint(); d.showModal();
    if (opts.autoRead && bg && !start?.keypoints) readPhoto();
  }
  return {open};
})();

/* =============================================================== editor */
const Editor = (() => {
  const view = $('#editView'), img = $('#edImg'), cv = $('#mask'), cx = cv.getContext('2d');
  const area = $('#area'), frame = $('#frame'), imgbox = $('#imgbox'), rail = $('#rail'), pane = $('#pane');
  const promptEl = $('#edPrompt'), form = $('#edForm'), routeEl = $('#edRoute');
  let rec = null, versions = [], tool = store.get('tool', 'seg'), zoom = 1, painted = false, pending = false, busy = null;
  if (!['seg', 'brush', 'crop'].includes(tool)) tool = 'seg';
  let segs = [], segSel = [], segHover = null, segState = 'idle', segHits = {}, segOpen = {};   // segState: idle|running|done|none
  const segLayer = $('#segLayer');
  const hasSel = () => painted || segSel.length > 0;
  let brushSize = store.get('brush', 40), erasing = false, strength = store.get('strength', 'mid');
  let autoFill = store.get('autoFill', false);
  let detStrength = store.get('detStrength', 0.4), upRefine = store.get('upRefine', true);
  let vidSec = store.get('vidSec', 5), vidQ = store.get('vidQ', '720p'), reposeId = null, vidText = '', reposeText = '';
  let vidSmooth = store.get('vidSmooth', true), vidKeys = {mid: null, end: null}, vidPick = null, vidRoot = null, joinSel = [];
  $('#vidKeyIn').onchange = async e => {
    const f = e.target.files[0]; e.target.value = ''; if (!f || !vidPick) return;
    const fd = new FormData(); fd.append('file', f, f.name);
    try { const r = await req('POST', '/api/import', fd); vidKeys[vidPick] = r.id; vidPick = null; paintPane(); }
    catch (err) { toast(err.message); }
  };
  $('#reposeIn').onchange = async e => {
    const f = e.target.files[0]; e.target.value = ''; if (!f) return;
    const fd = new FormData(); fd.append('file', f, f.name);
    try { const r = await req('POST', '/api/import', fd); reposeId = r.id; accOpen.add('repose'); paintPane(); toast('이 사진의 자세로 바꿔요. "자세 바꾸기"를 누르세요.'); }
    catch (err) { toast(err.message); }
  };
  const STRENGTH = {low: [0.35, '약하게'], mid: [0.55, '보통'], high: [0.75, '강하게']};
  let editMode = store.get('editMode', 'instruction');     // instruction (Qwen) | redraw (same model)
  let expandAspect = '3:4', expandAnchor = 'top';
  const qwenReady = () => MODELS.some(m => m.family === 'qwen21' && m.installed !== false);
  const REMOVE = /지워|지우|제거|없애|빼\s*줘|remove|erase|delete/i;
  const ENLARGE_RX = /키워|키우|크게|커지|늘려|늘리|길게|굵게|두껍|풍성|볼륨|넓게|부풀|bigger|larger|longer|thicker|fuller|wider|taller|enlarge|increase|grow/i;
  const EXPAND_RX = /전신|발끝|발까지|신발까지|다리까지|무릎까지|허벅지까지|줌\s*아웃|멀리서|더 넓게|full[\s-]?body|zoom(ed)?[\s-]?out|wider shot/i;
  bindEnter(promptEl, form);
  promptEl.addEventListener('input', paintRoute);
  /* ---- 내 칩: the owner's quick-insert sentences (data/chips.json) */
  let chips = [];
  async function loadChips() { try { chips = (await api.get('/api/chips')).chips; } catch { chips = []; } paintChips(); }
  function paintChips() {
    $('#edChips').innerHTML = chips.map((c, i) => `<button type="button" class="chip" data-chip="${i}" title="${esc(c.text)}">${esc(c.label)}</button>`).join('')
      + `<button type="button" class="chip chip-edit" data-chips-edit title="칩 추가·수정">＋ 칩</button>`;
  }
  $('#edChips').addEventListener('click', e => {
    if (e.target.closest('[data-chips-edit]')) { editChips(); return; }
    const b = e.target.closest('[data-chip]'); if (!b) return;
    const text = chips[+b.dataset.chip].text, cur = promptEl.value.trim();
    if (cur.includes(text)) { toast('이미 들어 있어요', 2000); return; }
    promptEl.value = cur ? `${cur.replace(/[,，]\s*$/, '')}, ${text}` : text;
    autoGrow(promptEl); promptEl.focus();
    promptEl.setSelectionRange(promptEl.value.length, promptEl.value.length);
    paintRoute();
  });
  function editChips() {
    const d = document.createElement('dialog'); d.className = 'chipdlg';
    let list = chips.map(c => ({...c}));
    const rows = () => list.map((c, i) => `<div class="crow" data-i="${i}">
        <input data-f="label" value="${esc(c.label)}" maxlength="24" placeholder="칩 이름">
        <input data-f="text" value="${esc(c.text)}" maxlength="600" placeholder="입력창에 들어갈 문장">
        <button type="button" data-mv="-1" aria-label="위로">↑</button><button type="button" data-mv="1" aria-label="아래로">↓</button><button type="button" data-rm aria-label="지우기">×</button></div>`).join('')
      || '<p class="hint">칩이 없어요. 아래에 추가하세요.</p>';
    const paint = () => { d.querySelector('.clist').innerHTML = rows(); };
    d.innerHTML = `<h3>내 칩</h3><p class="hint">칩을 누르면 문장이 편집 입력창에 이어 붙어요. 이 PC에만 저장돼요.</p>
      <div class="clist"></div>
      <div class="row"><button type="button" class="btn" data-add>＋ 하나 추가</button></div>
      <label class="hint" for="chipBulk">한꺼번에 붙여넣기 (한 줄에 하나: <b>칩 이름 | 문장</b>)</label>
      <textarea id="chipBulk" rows="4" placeholder="검은 티로 | 선택한 옷을 검은 티셔츠로 바꾸기&#10;조명 맞추기 | 원래 조명과 그림자에 맞추기"></textarea>
      <div class="row end"><button type="button" class="btn" data-close>취소</button><button type="button" class="btn primary" data-save>저장</button></div>`;
    document.body.appendChild(d); paint(); d.showModal();
    d.addEventListener('input', e => { const r = e.target.closest('[data-i]'); if (r && e.target.dataset.f) list[+r.dataset.i][e.target.dataset.f] = e.target.value; });
    d.addEventListener('click', async e => {
      const r = e.target.closest('[data-i]'), i = r ? +r.dataset.i : -1;
      if (e.target.closest('[data-rm]')) { list.splice(i, 1); paint(); return; }
      const mv = e.target.closest('[data-mv]');
      if (mv) { const j = i + +mv.dataset.mv; if (j >= 0 && j < list.length) { [list[i], list[j]] = [list[j], list[i]]; paint(); } return; }
      if (e.target.closest('[data-add]')) { list.push({label: '', text: ''}); paint(); d.querySelector('.crow:last-child input')?.focus(); return; }
      if (e.target.closest('[data-close]')) { d.remove(); return; }
      if (e.target.closest('[data-save]')) {
        try {
          const r2 = await req('PUT', '/api/chips', {chips: list, bulk: d.querySelector('#chipBulk').value});
          chips = r2.chips; paintChips(); d.remove(); toast(`칩 ${chips.length}개를 저장했어요`);
        } catch (err) { toast(err.message, 5000); }
      }
    });
    d.addEventListener('close', () => d.remove());
  }
  loadChips();
  $('#edEnhance').onclick = async () => {
    const t = promptEl.value.trim();
    if (!t) { promptEl.focus(); toast('먼저 대충이라도 적어 주세요'); return; }
    const btn = $('#edEnhance');
    if (btn.classList.contains('busy')) return;
    btn.classList.add('busy');
    toast('다듬는 중… 처음 한 번은 모델을 불러오느라 20~40초 걸릴 수 있어요', 4000);
    try {
      const segs = segSel.length ? segSel : [];
      const r = await api.post('/api/enhance-edit', {prompt: t, segments: segs, image_id: rec ? rec.id : null});
      promptEl.focus(); promptEl.select();
      if (!document.execCommand('insertText', false, r.prompt)) promptEl.value = r.prompt;
      autoGrow(promptEl);
      toast('자세히 다듬었어요. 마음에 안 들면 Ctrl+Z', 4000);
    } catch (e) { toast(e.message, 5000); }
    finally { btn.classList.remove('busy'); }
  };

  async function open(id) {
    if (!id) { location.hash = 'gallery'; return; }
    if (rec && rec.id === id && img.src) { fit(); return; }
    let data;
    try { data = await api.get(`/api/images/${id}`); } catch (e) { toast(e.message); location.hash = 'gallery'; return; }
    rec = data; versions = data.versions || [rec];
    if (rec.root_id !== vidRoot) { vidRoot = rec.root_id; vidKeys = {mid: null, end: null}; vidPick = null; joinSel = []; }
    pending = false; clearMask(false);
    $('#edTitle').textContent = rec.kind === 'import' ? '불러온 이미지' : (rec.prompt || '이미지');
    const model = MODELS.find(m => m.id === rec.model_id);
    $('#edSub').textContent = [rec.label, model?.name, `${rec.width}×${rec.height}`].filter(Boolean).join(' · ');
    img.onload = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; painted = false; fit(); };
    const isVideo = rec.media === 'video';
    view.classList.toggle('is-video', isVideo);
    imgbox.querySelector('video')?.remove();
    img.src = isVideo ? thumbUrl(rec.id) + '?v=' + Math.round(rec.created_at) : imgUrl(rec.id, Math.round(rec.created_at));
    if (isVideo) {
      const vid = document.createElement('video');
      Object.assign(vid, {src: imgUrl(rec.id, Math.round(rec.created_at)), controls: true, loop: true, autoplay: true, muted: true, playsInline: true});
      imgbox.appendChild(vid);
    }
    $('#saveBtn').href = `/api/images/${rec.id}/file?download=1`;
    $('#saveBtn').querySelector('span') && ($('#saveBtn').querySelector('span').textContent = isVideo ? 'MP4로 저장' : 'PNG로 저장');
    $('#regenBtn').disabled = rec.kind === 'import';
    segs = []; segSel = []; segHover = null; segState = 'idle'; segHits = {}; paintSegLayer();
    paintRail(); paintPane(); paintRoute();
    if (!isVideo) loadSegments(true);
  }

  function paintRail() {
    rail.innerHTML = versions.map((v, i) => `<span class="vw"><button class="vt ${v.id === rec.id ? 'cur' : ''}" data-id="${v.id}" title="${i + 1}. ${esc(v.label || '')}" aria-label="버전 ${i + 1}: ${esc(v.label || '')}"><img src="${thumbUrl(v.id)}" alt="">${v.media === 'video' ? '<b class="play">▶</b>' : ''}</button><button class="vx" data-vdel="${v.id}" title="이 버전 삭제" aria-label="버전 ${i + 1} 삭제">×</button></span>`).join('');
  }
  rail.addEventListener('click', async e => {
    const x = e.target.closest('[data-vdel]');
    if (x) {
      if (busy) return;
      const id = x.dataset.vdel, idx = versions.findIndex(v => v.id === id);
      try { await api.del(`/api/images/${id}`); } catch (err) { toast(err.message); return; }
      versions = versions.filter(v => v.id !== id);
      toast('삭제했어요');
      if (!versions.length) { rec = null; location.hash = 'gallery'; return; }
      if (rec && rec.id === id) { rec = null; location.hash = 'edit=' + versions[Math.max(0, idx - 1)].id; return; }
      paintRail(); paintPane();
      return;
    }
    const b = e.target.closest('[data-id]'); if (b && !busy) location.hash = 'edit=' + b.dataset.id;
  });

  /* ------------ side pane */
  function paintPane() {
    if (rec && rec.media === 'video') {
      pane.innerHTML = `<h2>영상</h2><dl class="kv">
        <dt>길이</dt><dd>${Number(rec.duration).toFixed(1)}초 · ${rec.fps}fps · ${rec.frames}프레임</dd><dt>크기</dt><dd>${rec.width}×${rec.height}</dd>
        ${rec.edit_prompt ? `<dt>움직임</dt><dd>${esc(rec.edit_prompt)}</dd>` : ''}${rec.seed !== undefined ? `<dt>시드</dt><dd>${rec.seed}</dd>` : ''}</dl>
        <p class="note">영상은 편집할 수 없어요. 왼쪽 버전 목록에서 원본 이미지를 골라 다시 만들거나, 아래 재생성으로 새 시드로 다시 만들 수 있어요.</p>
        <h2>장면 꺼내기</h2>
        <p class="note">영상의 한 장면을 사진으로 저장해요. 마지막 장면에서 다음 영상을 시작하면 이야기를 이어 갈 수 있어요.</p>
        <div class="field"><div class="seg2"><button data-vframe="first">첫 장면</button><button data-vframe="last">마지막 장면</button></div></div>`;
      const others = versions.filter(v => v.media === 'video' && v.id !== rec.id);
      if (others.length) pane.innerHTML += `<h2>이어 붙이기</h2>
        <p class="note">이 영상 뒤에 붙일 영상을 순서대로 고르세요.</p>
        <div class="vpick">${others.map(v => { const n = joinSel.indexOf(v.id); return `<button data-vjoin="${v.id}" aria-pressed="${n >= 0}"><img src="${thumbUrl(v.id)}" alt="">${n >= 0 ? `<i>${n + 2}</i>` : ''}</button>`; }).join('')}</div>
        <div class="field"><div class="seg2"><button id="doJoin" aria-pressed="true" ${joinSel.length ? '' : 'disabled'}>${joinSel.length + 1}개 이어 붙이기</button></div></div>`;
      return;
    }
    let h = '';
    if (tool === 'seg') {
      h = segPaneHtml();
    } else if (tool === 'brush') {
      h = `<h2>마스크 브러시</h2>
        <p class="note">바꿀 부분을 칠한 뒤 아래에 무엇으로 바꿀지 입력하세요. 칠한 부분만 다시 그리고 나머지는 그대로 둡니다.</p>
        <div class="field"><label for="bSize">브러시 크기 <b id="bSizeV">${brushSize}</b>px</label><input type="range" id="bSize" min="6" max="160" value="${brushSize}"></div>
        <div class="field"><span>모드</span><div class="seg2"><button data-bm="paint" aria-pressed="${!erasing}">칠하기</button><button data-bm="erase" aria-pressed="${erasing}">지우개</button><button id="bClear">모두 지우기</button></div></div>
        <div class="field"><div class="seg2"><button id="bDelete" ${painted ? '' : 'disabled'}>칠한 부분 삭제하기</button></div></div>`;
    } else if (tool === 'crop') {
      const ASP = ['3:4', '2:3', '9:16', '4:5', '1:1', '4:3', '3:2', '16:9'];
      const ANC = [['top', '위에 붙이기'], ['center', '가운데'], ['bottom', '아래에 붙이기'], ['left', '왼쪽'], ['right', '오른쪽']];
      h = `<h2>캔버스 늘리기</h2>
        <p class="note">그림 바깥을 늘리고 새로 생긴 부분을 AI가 채웁니다. 얼굴만 나온 초상화를 허벅지까지 보이게 하려면 <b>3:4 · 위에 붙이기</b>를 고르세요. 아래 입력칸에 늘어난 부분에 그릴 내용을 적어도 됩니다.</p>
        <div class="field"><span>새 비율 (지금 ${rec ? `${rec.width}×${rec.height}` : ''})</span><div class="seg2">${ASP.map(a => `<button data-asp="${a}" aria-pressed="${a === expandAspect}">${a}</button>`).join('')}</div></div>
        <div class="field"><span>원래 그림 위치</span><div class="seg2">${ANC.map(([k, n]) => `<button data-anc="${k}" aria-pressed="${k === expandAnchor}">${n}</button>`).join('')}</div></div>
        <div class="field"><div class="seg2"><button id="doExpand" aria-pressed="true">늘리기</button></div></div>`;
    }
    if (tool === 'seg' || tool === 'brush') h += `<label class="switch" style="margin:4px 0 16px"><input type="checkbox" id="autoFillToggle" ${autoFill ? 'checked' : ''}><span>삭제하면 묻지 않고 모델이 알아서 채우기</span></label>`;
    if (rec) {
      const model = MODELS.find(m => m.id === rec.model_id);
      h += `<div class="accs">`
        + acc('upscale', '업스케일', '2K·4K로 키우기', `
          <p class="note">크게 키운 뒤 피부·머리카락·옷감 디테일을 살짝 다시 그립니다. 새 버전으로 저장돼요. (지금 ${rec.width}×${rec.height})</p>
          <div class="field"><div class="seg2"><button data-up="2">2배 (약 2K)</button><button data-up="4">4배 (약 4K)</button></div></div>
          <label class="switch"><input type="checkbox" id="upRefine" ${upRefine ? 'checked' : ''}><span>디테일 다시 그리기 (끄면 선명하게만 키움)</span></label>`)
        + acc('detail', '디테일 보정', '작은 얼굴·눈·손 살리기', `
          <p class="note">작게 나온 얼굴·눈·손을 찾아 크게 다시 그린 뒤 제자리에 붙입니다. 전신이나 여러 명 나온 사진에 좋아요. 아래 입력칸에 "미소", "푸른 눈"처럼 덧붙일 수 있어요.</p>
          <div class="field"><div class="seg2"><button data-det="face">얼굴</button><button data-det="face,eyes">얼굴+눈</button><button data-det="hands">손</button><button data-det="face,eyes,hands">전부</button></div></div>
          <div class="field"><span>세기</span><div class="seg2">${[[0.3, '약하게'], [0.4, '보통'], [0.55, '강하게']].map(([v, n]) => `<button data-dstr="${v}" aria-pressed="${v === detStrength}">${n}</button>`).join('')}</div></div>`)
        + acc('bg', '배경 제거', '투명·단색 배경', `
          <p class="note">사람이나 물건만 남기고 배경을 지웁니다. 새 버전으로 저장되고, PNG로 받으면 투명하게 남아요.</p>
          <div class="field"><div class="seg2"><button data-bg="transparent">투명</button><button data-bg="white">흰색</button><button data-bg="gray">회색</button><button data-bg="black">검정</button></div></div>`)
        + acc('video', '영상 만들기', '움직이는 영상 · 처음·가운데·끝 장면', `
          <textarea class="acct" id="vidText" rows="2" placeholder="어떻게 움직일지 적으세요. 예: 고개를 돌리며 웃는다, 머리카락이 바람에 날린다">${esc(vidText)}</textarea>
          <p class="note">비워두면 자연스럽게 살짝 움직여요. 몇 분 걸려요. 지금 사진이 첫 장면이고, 가운데·마지막 장면을 고르면 그 사이를 이어서 움직여요.</p>
          <div class="field"><span>장면</span><div class="vkeys">${[['mid', '가운데'], ['end', '마지막']].map(([k, n]) => `<div class="vkey"><span>${n}</span>${vidKeys[k] ? `<span class="ref"><img src="${thumbUrl(vidKeys[k])}" alt=""><button type="button" data-vclear="${k}" aria-label="${n} 장면 빼기">×</button></span>` : ''}<button data-vpick="${k}" aria-pressed="${vidPick === k}">${vidKeys[k] ? '바꾸기' : '고르기'}</button></div>`).join('')}</div>
            ${vidPick ? `<div class="vpick">${versions.filter(v => v.media !== 'video' && v.id !== rec.id).map(v => `<button data-vset="${v.id}" title="${esc(v.label || '')}"><img src="${thumbUrl(v.id)}" alt=""></button>`).join('')}<button data-vup>사진 넣기</button></div>` : ''}</div>
          <div class="field"><span>길이</span><div class="seg2">${[3, 5, 8, 10].map(n => `<button data-vsec="${n}" aria-pressed="${n === vidSec}">${n}초</button>`).join('')}</div>${vidSec > 5 && !vidKeys.mid ? '<p class="note" style="margin-top:6px">5초보다 길면 가운데 장면을 같이 고르는 게 자연스러워요.</p>' : ''}</div>
          <div class="field"><span>화질</span><div class="seg2">${['480p', '720p'].map(q => `<button data-vq="${q}" aria-pressed="${q === vidQ}">${q}${q === '720p' ? ' (느림)' : ''}</button>`).join('')}</div></div>
          <label class="switch" style="margin:0 0 14px"><input type="checkbox" id="vidSmooth" ${vidSmooth ? 'checked' : ''}><span>부드럽게 (프레임 보간 16→32fps)</span></label>
          <div class="field"><div class="seg2"><button id="makeVideo" aria-pressed="true">영상 만들기</button></div></div>`)
        + acc('repose', '자세 바꾸기', '같은 사람·옷·배경, 자세만', `
          <textarea class="acct" id="reposeText" rows="2" placeholder="어떤 자세로 바꿀지 적으세요. 예: 침대에 앉아 다리를 꼬고 있다">${esc(reposeText)}</textarea>
          <p class="note">원하는 자세가 담긴 사진을 넣어도 돼요. 글과 사진을 같이 써도 됩니다. 사진이 있으면 더 정확해요.</p>
          <div class="field"><div class="seg2">${reposeId ? `<span class="ref"><img src="${thumbUrl(reposeId)}" alt=""><button type="button" data-act2="repose-clear" aria-label="자세 사진 빼기">×</button></span>` : ''}<button data-act2="repose-pick">${reposeId ? '다른 사진' : '자세 사진 넣기'}</button><button data-act2="repose-bones">뼈대로 자세 잡기</button></div></div>
          <div class="field"><div class="seg2"><button id="doRepose" aria-pressed="true">자세 바꾸기</button></div></div>`)
        + acc('person', '이 인물·자세 다시 쓰기', '캐릭터 등록, 포즈, 참고 이미지', `
          <div class="field"><div class="inline"><input id="charName" placeholder="캐릭터 이름 (예: 플라이트 디렉터)"><button class="btn" id="saveChar">이 사진만 등록</button></div></div>
          <div class="field"><div class="seg2"><button id="buildChar" aria-pressed="true">이 사진 1장으로 캐릭터 만들기</button></div>
            <p class="note" style="margin-top:6px">얼굴 정면·전신·3/4 측면 사진과 인물 특징을 만들어 함께 등록해요. 같은 사람으로 더 잘 그려집니다.</p></div>
          <div class="field"><div class="seg2"><button id="useAsPose">이 자세로 새로 그리기</button><button id="useAsRef">참고 이미지로 보내기</button></div></div>`)
        + acc('info', '정보', `${rec.width}×${rec.height} · 버전 ${versions.findIndex(v => v.id === rec.id) + 1}/${versions.length}`, `<dl class="kv">
          ${rec.prompt ? `<dt>프롬프트</dt><dd>${esc(rec.prompt)}</dd>` : ''}
          ${rec.prompt_en ? `<dt>영어로 변환</dt><dd>${esc(rec.prompt_en)}</dd>` : ''}
          ${rec.edit_prompt ? `<dt>편집</dt><dd>${esc(rec.edit_prompt)}</dd>` : ''}
          ${rec.edit_prompt_en ? `<dt>영어로 변환</dt><dd>${esc(rec.edit_prompt_en)}</dd>` : ''}
          <dt>모델</dt><dd>${esc(model?.name || rec.model_id || (rec.embedded?.tool ? rec.embedded.tool + ' (외부)' : '외부 이미지'))}</dd>
          <dt>크기</dt><dd>${rec.width}×${rec.height}</dd>
          ${rec.seed !== undefined ? `<dt>시드</dt><dd>${rec.seed}</dd>` : ''}
          <dt>버전</dt><dd>${versions.findIndex(v => v.id === rec.id) + 1} / ${versions.length}</dd></dl>`)
        + `</div>`;
    }
    const top = pane.scrollTop;
    pane.innerHTML = h;
    pane.scrollTop = top;
  }
  const accOpen = new Set(store.get('acc', []));
  function acc(key, title, sub, inner) {
    return `<details class="acc" data-acc="${key}" ${accOpen.has(key) ? 'open' : ''}><summary><b>${title}</b><span>${esc(sub)}</span></summary><div class="accb">${inner}</div></details>`;
  }
  pane.addEventListener('toggle', e => {
    const d = e.target.closest?.('[data-acc]'); if (!d) return;
    d.open ? accOpen.add(d.dataset.acc) : accOpen.delete(d.dataset.acc); store.set('acc', [...accOpen]);
  }, true);
  pane.addEventListener('pointerover', e => { const r = e.target.closest('[data-skey]'); const k = r ? r.dataset.skey : null; if (k !== segHover) { segHover = k; paintSegLayer(); } });
  pane.addEventListener('pointerleave', () => { if (segHover) { segHover = null; paintSegLayer(); } });
  pane.addEventListener('keydown', e => { if (e.target.id === 'segFind' && e.key === 'Enter') { e.preventDefault(); $('#findSeg').click(); } });
  pane.addEventListener('input', e => {
    if (e.target.id === 'vidText') vidText = e.target.value;
    if (e.target.id === 'reposeText') reposeText = e.target.value;
  });
  pane.addEventListener('input', e => { if (e.target.id === 'bSize') { brushSize = +e.target.value; $('#bSizeV').textContent = brushSize; store.set('brush', brushSize); } });
  pane.addEventListener('change', e => {
    if (e.target.id === 'autoFillToggle') setAutoFill(e.target.checked);
    if (e.target.id === 'upRefine') { upRefine = e.target.checked; store.set('upRefine', upRefine); }
    if (e.target.id === 'vidSmooth') { vidSmooth = e.target.checked; store.set('vidSmooth', vidSmooth); }
  });
  pane.addEventListener('click', e => {
    const g = e.target.closest('[data-goto]'); if (g) setTool(g.dataset.goto);
    const sd = e.target.closest('[data-sdel]'); if (sd) { selectSeg(sd.dataset.sdel, false); startDelete(); return; }
    const chev = e.target.closest('[data-sopen]'); if (chev) { segOpen[chev.dataset.sopen] = !segOpen[chev.dataset.sopen]; paintPane(); return; }
    const sr = e.target.closest('[data-skey]'); if (sr) { selectSeg(sr.dataset.skey, e.shiftKey || e.ctrlKey || e.metaKey); return; }
    if (e.target.closest('#reSeg')) { analyze(true); return; }
    if (e.target.closest('#findSeg')) { const v = $('#segFind').value.trim(); if (v) analyze(true, v.split(',').map(x => x.trim()).filter(Boolean)); return; }
    const bm = e.target.closest('[data-bm]'); if (bm) { erasing = bm.dataset.bm === 'erase'; paintPane(); }
    if (e.target.closest('#bClear')) clearMask();
    if (e.target.closest('#bDelete')) startDelete();
    const asp = e.target.closest('[data-asp]'); if (asp) { expandAspect = asp.dataset.asp; paintPane(); }
    const anc = e.target.closest('[data-anc]'); if (anc) { expandAnchor = anc.dataset.anc; paintPane(); }
    if (e.target.closest('#doExpand')) submitEdit({mode: 'outpaint', aspect: expandAspect, anchor: expandAnchor, prompt: promptEl.value.trim()});
    const a2 = e.target.closest('[data-act2]');
    if (a2 && a2.dataset.act2 === 'repose-pick') { $('#reposeIn').click(); return; }
    if (a2 && a2.dataset.act2 === 'repose-clear') { reposeId = null; paintPane(); return; }
    if (a2 && a2.dataset.act2 === 'repose-bones' && rec) {
      const target = rec;
      PoseEd.open({bg: target.id, autoRead: true, title: '뼈대로 자세 잡기', useLabel: '이 자세로 바꾸기',
        onUse: id => {
          reposeId = id; accOpen.add('repose'); paintPane();
          if (busy || !rec || rec.id !== target.id) { toast('자세를 넣었어요. "자세 바꾸기"를 누르세요.'); return; }
          api.post(`/api/images/${target.id}/repose`, {pose: id, prompt: reposeText.trim()})
            .then(j => runJob(j, `자세 바꾸기 · ${j.decision.model_name}`)).catch(err => toast(err.message, 6000));
        }});
      return;
    }
    if (e.target.closest('#doRepose') && rec && !busy) {
      api.post(`/api/images/${rec.id}/repose`, {pose: reposeId, prompt: (reposeText || promptEl.value).trim()})
        .then(j => runJob(j, `자세 바꾸기 · ${j.decision.model_name}`)).catch(err => toast(err.message, 6000));
      return;
    }
    const vs = e.target.closest('[data-vsec]'); if (vs) { vidSec = +vs.dataset.vsec; store.set('vidSec', vidSec); paintPane(); return; }
    const vq = e.target.closest('[data-vq]'); if (vq) { vidQ = vq.dataset.vq; store.set('vidQ', vidQ); paintPane(); return; }
    if (e.target.closest('#makeVideo')) { makeVideo(); return; }
    const vp = e.target.closest('[data-vpick]'); if (vp) { vidPick = vidPick === vp.dataset.vpick ? null : vp.dataset.vpick; paintPane(); return; }
    const vset = e.target.closest('[data-vset]'); if (vset && vidPick) { vidKeys[vidPick] = vset.dataset.vset; vidPick = null; paintPane(); return; }
    if (e.target.closest('[data-vup]')) { $('#vidKeyIn').click(); return; }
    const vc = e.target.closest('[data-vclear]'); if (vc) { vidKeys[vc.dataset.vclear] = null; paintPane(); return; }
    const vf = e.target.closest('[data-vframe]');
    if (vf && rec && !busy) {
      api.post(`/api/videos/${rec.id}/frame`, {at: vf.dataset.vframe}).then(j => runJob(j, j.decision.action)).catch(err => toast(err.message, 6000));
      return;
    }
    const vj = e.target.closest('[data-vjoin]');
    if (vj) { const i = joinSel.indexOf(vj.dataset.vjoin); i >= 0 ? joinSel.splice(i, 1) : joinSel.push(vj.dataset.vjoin); paintPane(); return; }
    if (e.target.closest('#doJoin') && rec && !busy && joinSel.length) {
      api.post('/api/videos/join', {video_ids: [rec.id, ...joinSel]})
        .then(j => { joinSel = []; return runJob(j, j.decision.action); }).catch(err => toast(err.message, 6000));
      return;
    }
    const up = e.target.closest('[data-up]');
    if (up && rec && !busy) {
      api.post(`/api/images/${rec.id}/upscale`, {scale: +up.dataset.up, refine: upRefine, prompt: promptEl.value.trim()})
        .then(j => runJob(j, `${j.decision.action} · ${j.decision.width}×${j.decision.height}`)).catch(err => toast(err.message, 6000));
      return;
    }
    const ds = e.target.closest('[data-dstr]'); if (ds) { detStrength = +ds.dataset.dstr; store.set('detStrength', detStrength); paintPane(); return; }
    const det = e.target.closest('[data-det]');
    if (det && rec && !busy) {
      api.post(`/api/images/${rec.id}/detail`, {targets: det.dataset.det.split(','), strength: detStrength, prompt: promptEl.value.trim()})
        .then(j => runJob(j, `${j.decision.action} · ${j.decision.model_name}`)).catch(err => toast(err.message, 6000));
      return;
    }
    const bgb = e.target.closest('[data-bg]');
    if (bgb && rec && !busy) {
      api.post(`/api/images/${rec.id}/remove-background`, {background: bgb.dataset.bg})
        .then(j => runJob(j, '배경 제거 · BiRefNet')).catch(err => toast(err.message, 6000));
      return;
    }
    if (e.target.closest('#useAsPose') && rec) { Gen.setPose(rec.id); location.hash = 'generate'; toast('이 사진의 자세로 그려요. 무엇을 그릴지 적어 주세요.'); return; }
    if (e.target.closest('#useAsRef') && rec) { Gen.addRef(rec.id); toast('생성 화면의 참고 이미지에 넣었어요'); }
    if (e.target.closest('#buildChar') && rec && !busy) {
      const name = $('#charName').value.trim();
      if (!name) { $('#charName').focus(); toast('캐릭터 이름을 먼저 적어 주세요'); return; }
      api.post('/api/characters/build', {name, image_id: rec.id})
        .then(j => runJob(j, `캐릭터 만들기 · ${name}`).then(done => {
          if (!done || done.status !== 'done') return;
          Gen.reloadChars();
          toast(`'${name}' 캐릭터를 만들었어요. 생성 화면의 캐릭터 메뉴에서 고르면 같은 인물로 그립니다.`, 6000);
        }))
        .catch(err => toast(err.message, 6000));
      return;
    }
    if (e.target.closest('#saveChar') && rec) {
      const name = $('#charName').value.trim();
      if (!name) { $('#charName').focus(); return; }
      api.post('/api/characters', {name, ref_ids: [rec.id], description: ''})
        .then(c => toast(`'${c.name}' 캐릭터로 등록했어요. 생성 화면의 캐릭터 메뉴에서 고르면 같은 인물로 그립니다.`, 5000))
        .catch(err => toast(err.message, 5000));
    }
  });
  function setAutoFill(on) {
    autoFill = on; store.set('autoFill', on);
    document.querySelectorAll('#autoFillToggle,#askDefault').forEach(x => x.checked = on);
    toast(on ? '이제 삭제하면 바로 모델이 알아서 채웁니다' : '이제 삭제하면 채울 내용을 물어봅니다');
  }
  function setTool(t) {
    tool = t;
    document.querySelectorAll('[data-tool]').forEach(b => b.setAttribute('aria-pressed', b.dataset.tool === t));
    view.classList.toggle('tool-brush', t === 'brush');
    view.classList.toggle('tool-seg', t === 'seg');
    store.set('tool', t);
    paintPane();
    if (t === 'seg' && rec && rec.media !== 'video' && segState === 'idle') analyze();
  }
  document.querySelectorAll('[data-tool]').forEach(b => b.onclick = () => setTool(b.dataset.tool));
  setTimeout(() => setTool(tool), 0);

  /* ------------ segments */
  const ICON_DL = '<svg class="i" viewBox="0 0 24 24" style="width:15px;height:15px"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>';
  const ICON_DEL = '<svg class="i" viewBox="0 0 24 24" style="width:16px;height:16px"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/></svg>';
  function segRows() {
    // groups: several instances of one label share a row that expands into each instance
    const rows = [], seen = new Set();
    for (const s of segs) {
      if (s.group) {
        if (seen.has(s.group)) continue;
        seen.add(s.group);
        rows.push({key: s.group, label: s.label, ko: s.ko, members: segs.filter(x => x.group === s.group), group: true});
      } else rows.push({key: s.id, label: s.label, ko: s.ko, members: [s]});
    }
    return rows;
  }
  const keyIds = k => { const r = segRows().find(x => x.key === k); return r ? r.members.map(m => m.id) : segs.filter(s => s.id === k).map(s => s.id); };
  function segPaneHtml() {
    if (!rec) return '';
    if (segState === 'running') return `<h2>세그먼트</h2><p class="note">이미지를 분석하는 중이에요… 사람, 머리카락, 옷 같은 부분을 찾고 있어요.</p><i class="bar" style="--p:${segProgress}%"></i>`;
    if (segState === 'missing') return `<h2>세그먼트</h2><p class="note">${esc(segError)}</p><div class="seg2"><button data-goto="brush" aria-pressed="true">브러시로 칠하기</button></div>`;
    const row = (r, child) => `<li><button class="srow ${segSel.includes(r.key) ? 'on' : ''}" data-skey="${esc(r.key)}">
        <span class="sthumb"><img src="/api/images/${rec.id}/segments/${encodeURIComponent(r.key)}/cutout?size=96" alt=""></span>
        <span class="lbl"><b>${esc(r.ko || r.label)}${r.group ? ` <small>(${r.members.length})</small>` : ''}</b>${r.ko ? `<small>${esc(r.label)}</small>` : ''}</span>
        ${r.group ? `<span class="hint" data-sopen="${esc(r.key)}" style="padding:6px">${segOpen[r.key] ? '▲' : '▼'}</span>` : ''}</button>
        ${child ? '' : `<button class="sdel" data-sdel="${esc(r.key)}" title="삭제" aria-label="${esc(r.ko || r.label)} 삭제">${ICON_DEL}</button>`}
        ${r.group && segOpen[r.key] ? `<ul>${r.members.map((m, i) => row({key: m.id, label: m.label, ko: `${m.ko || m.label} ${i + 1}`, members: [m]}, true)).join('')}</ul>` : ''}</li>`;
    const list = segRows();
    return `<h2>세그먼트 <span style="display:flex;gap:6px"><button class="ghost" id="reSeg" title="다시 분석">↻</button>${list.length ? `<a class="ghost" href="/api/images/${rec.id}/segments.zip" download>${ICON_DL}모두 다운로드</a>` : ''}</span></h2>
      ${list.length ? `<ul class="segs">${list.map(r => row(r)).join('')}</ul>` : '<p class="note">찾은 부분이 없어요. 이미지를 직접 클릭하거나, 아래에 찾을 대상을 입력해 보세요.</p>'}
      <p class="note" style="margin-top:12px">목록에 없는 부분은 <b>이미지를 클릭</b>하면 잡아요. 여러 개는 Shift를 누른 채 고르세요.</p>
      <div class="field"><div class="inline"><input id="segFind" placeholder="찾을 대상 (예: 스타킹, 목걸이, epaulets)"><button class="btn" id="findSeg">찾기</button></div></div>`;
  }
  let segProgress = 0, segError = '';
  async function loadSegments(autoRun) {
    if (!rec) return;
    const id = rec.id;
    try { segs = await api.get(`/api/images/${id}/segments`); } catch { segs = []; }
    if (!rec || rec.id !== id) return;
    segState = segs.length ? 'done' : 'idle';
    prepareHits(); paintPane(); paintSegLayer();
    if (autoRun && !segs.length && tool === 'seg') analyze();
  }
  async function analyze(force = false, labels = null, points = null) {
    if (!rec || segState === 'running') return;
    const id = rec.id;
    segState = 'running'; segProgress = 0; if (tool === 'seg') paintPane();
    try {
      const j = await api.post(`/api/images/${id}/segment`, {labels, points});
      const done = await pollJob(j.id, x => { segProgress = Math.round(x.progress * 100); const b = pane.querySelector('.bar'); if (b) b.style.setProperty('--p', segProgress + '%'); });
      if (!rec || rec.id !== id) return;
      if (done.status !== 'done') { segState = segs.length ? 'done' : 'idle'; toast(done.error || '세그먼트를 찾지 못했어요', 5000); }
      else {
        segs = done.result.segments || [];
        segState = 'done';
        if (done.result.new) { segSel = done.result.new; if (painted) clearMask(false); }
      }
    } catch (e) {
      segState = 'missing'; segError = e.message;
    }
    prepareHits(); paintPane(); paintSegLayer(); paintRoute();
  }
  function prepareHits() {
    segHits = {};
    for (const s of segs) {
      const im = new Image();
      im.onload = () => { const c = document.createElement('canvas'); c.width = 160; c.height = Math.round(160 * im.height / im.width); c.getContext('2d').drawImage(im, 0, 0, c.width, c.height); segHits[s.id] = c; };
      im.src = `/api/images/${rec.id}/segments/${s.id}/mask`;
    }
  }
  function segAt(fx, fy) {
    // smallest segment under the point wins (hair before woman)
    let best = null;
    for (const s of segs) {
      const c = segHits[s.id]; if (!c) continue;
      const a = c.getContext('2d').getImageData(Math.min(c.width - 1, fx * c.width | 0), Math.min(c.height - 1, fy * c.height | 0), 1, 1).data[3];
      if (a > 127 && (!best || s.area < best.area)) best = s;
    }
    return best;
  }
  function keyFor(s) { return s.group ? s.group : s.id; }
  function selectSeg(key, add) {
    if (painted) clearMask(false);
    if (add) segSel = segSel.includes(key) ? segSel.filter(k => k !== key) : [...segSel, key];
    else segSel = segSel.length === 1 && segSel[0] === key ? [] : [key];
    pending = false;
    paintPane(); paintSegLayer(); paintRoute();
  }
  function paintSegLayer() {
    if (!rec) { segLayer.innerHTML = ''; return; }
    const ov = (sid, cls) => `<div class="seg-ov ${cls}" style="-webkit-mask-image:url('/api/images/${rec.id}/segments/${sid}/mask');mask-image:url('/api/images/${rec.id}/segments/${sid}/mask')"></div>`;
    let h = '';
    for (const k of segSel) for (const id of keyIds(k)) h += ov(id, 'sel');
    if (segHover && !segSel.includes(segHover)) for (const id of keyIds(segHover)) h += ov(id, '');
    segLayer.innerHTML = h;
  }
  imgbox.addEventListener('click', e => {
    if (tool !== 'seg' || busy || !rec || e.target.closest('.busy')) return;
    const r = img.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const hit = segAt(fx, fy);
    if (hit) { selectSeg(keyFor(hit), e.shiftKey || e.ctrlKey || e.metaKey); return; }
    analyze(true, null, [{x: Math.round(fx * rec.width), y: Math.round(fy * rec.height)}]);
  });
  imgbox.addEventListener('pointermove', e => {
    if (tool !== 'seg' || !segs.length) return;
    const r = img.getBoundingClientRect();
    const hit = segAt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    const k = hit ? keyFor(hit) : null;
    if (k !== segHover) { segHover = k; paintSegLayer(); pane.querySelectorAll('[data-skey]').forEach(x => x.classList.toggle('hl', x.dataset.skey === k)); }
  });
  imgbox.addEventListener('pointerleave', () => { if (segHover) { segHover = null; paintSegLayer(); } });
  const selName = () => {
    if (painted) return '칠한 영역';
    const names = segSel.map(k => { const r = segRows().find(x => x.key === k); const s = segs.find(x => x.id === k); return (r && (r.ko || r.label)) || (s && (s.ko || s.label)) || k; });
    return names.join(', ');
  };

  /* ------------ model picker (per task) */
  const recCache = {}; let edModel = {};          // edModel[task] = chosen model id ('auto' default)
  function currentTask() {
    if (tool === 'crop') return 'expand';
    if (pending || hasSel()) return 'inpaint';
    if (!promptEl.value.trim()) return null;
    return qwenReady() && editMode === 'instruction' ? 'instruction' : 'redraw';
  }
  function modelPill(task) {
    if (!task || !rec) return '';
    const key = `${rec.id}:${task}`, r = recCache[key];
    if (!r) { api.post('/api/recommend', {task, image_id: rec.id}).then(x => { recCache[key] = x; paintRoute(); }).catch(() => {}); return ''; }
    const cur = edModel[task] || 'auto';
    const recName = (r.models.find(m => m.recommended) || {}).name || '';
    const label = cur === 'auto' ? `자동 · 추천 ${recName}` : (r.models.find(m => m.id === cur) || {}).name;
    const opts = [`<option value="auto" ${cur === 'auto' ? 'selected' : ''}>자동 (추천: ${esc(recName)})</option>`]
      .concat(r.models.map(m => `<option value="${m.id}" ${m.id === cur ? 'selected' : ''} ${m.compatible && m.installed !== false ? '' : 'disabled'}>${m.recommended ? '★ 추천 · ' : ''}${esc(m.name)} — ${esc(m.compatible ? m.best_for : m.note)}${m.installed === false ? ' (미설치)' : ''}</option>`));
    return `<label class="sel-pill" title="${esc(r.reason)}">모델 <b>${esc(label)}</b><select data-edmodel="${task}" aria-label="모델">${opts.join('')}</select></label>`;
  }

  /* ------------ prompt / routing */
  const SUG_MASK = ['지워줘', '빨간색으로', '검은 가죽 재킷'];
  function paintRoute() {
    const t = promptEl.value.trim();
    const model = MODELS.find(m => m.id === rec?.model_id);
    const modelName = model?.name || '기본 실사 모델';
    let h = '';
    if (pending) {
      h = `<div class="ask"><b>${esc(selName())} 삭제 · 그 자리를 어떻게 채울까요? 아래에 입력하거나 골라 주세요.</b>
        <div class="opts"><button type="button" class="pill main" data-fill="auto">모델이 알아서 채우기</button>
        <button type="button" class="pill plain" data-fill="clear">비워두기 (배경으로 메우기)</button>
        <button type="button" class="pill sug" data-cancel-ask>삭제 취소</button></div>
        <label class="switch"><input type="checkbox" id="askDefault" ${autoFill ? 'checked' : ''}><span>앞으로는 묻지 않고 항상 모델이 알아서 채우기</span></label>
        ${t ? `<span class="pill auto">→ <b>삭제한 자리 다시 그리기</b> · ${esc(modelName)}</span>` : ''}</div>`;
    } else if (hasSel()) {
      h += `<span class="pill sel">@${esc(selName())}<button type="button" data-clear-mask aria-label="칠한 영역 지우기"><svg class="i" viewBox="0 0 24 24" style="width:14px;height:14px"><path d="M6 6l12 12M18 6 6 18"/></svg></button></span>`;
      if (!t) h += SUG_MASK.map(s => `<button type="button" class="pill sug" data-sug="${s}">${s}</button>`).join('');
      else if (REMOVE.test(t)) h += `<span class="pill auto">→ <b>삭제</b> 후 채울 내용을 물어봅니다</span>`;
      else h += `<span class="pill auto">자동 선택 → <b>부분 다시 그리기</b> · ${esc(modelName)} <span class="hint">(원본과 같은 모델)</span></span>`
        + (ENLARGE_RX.test(t) ? `<span class="hint">커지는 요청이라 선택 영역을 주변까지 넉넉하게 넓혀서 그려요</span>` : '');
    } else if (t && EXPAND_RX.test(t)) {
      h += `<span class="pill auto">자동 선택 → <b>캔버스 늘리기</b> · 원본은 위에 두고 아래로 늘려서 그려요</span><span class="hint">화면 밖을 보여달라는 요청은 같은 크기 안에서 고칠 수 없어서 캔버스를 늘립니다</span>`;
    } else if (t) {
      const mode = qwenReady() ? editMode : 'redraw';
      if (qwenReady()) h += `<button type="button" class="pill ${mode === 'instruction' ? 'plain' : 'sug'}" data-emode="instruction">문장 편집 · Qwen</button><button type="button" class="pill ${mode === 'redraw' ? 'plain' : 'sug'}" data-emode="redraw">전체 다시 그리기 · 원본 모델</button>`;
      if (mode === 'instruction') h += `<span class="hint">인물은 유지하고 말한 부분만 바꿔요. NSFW는 '다시 그리기'나 브러시가 맞아요.</span>`;
      else h += Object.entries(STRENGTH).map(([k, [, name]]) => `<button type="button" class="pill ${k === strength ? 'plain' : 'sug'}" data-strength="${k}">${name}</button>`).join('') + `<span class="hint">${esc(modelName)}</span>`;
    } else {
      h = `<span class="hint">바꿀 부분을 브러시로 칠하세요. 칠하지 않고 "배경을 우주선 함교로"처럼 입력하면 문장대로 편집합니다.</span>`;
    }
    const task = currentTask();
    if (task && !pending) h += modelPill(task);
    routeEl.innerHTML = h;
    if (pending && task) { const ask = routeEl.querySelector('.ask .opts'); if (ask) ask.insertAdjacentHTML('beforeend', modelPill(task)); }
    promptEl.placeholder = pending ? '삭제한 자리에 무엇을 그릴지 입력하세요' : (hasSel() ? `${selName()}을(를) 무엇으로 바꿀까요?` : '바꿀 부분을 고르거나 칠하고, 무엇으로 바꿀지 입력하세요');
  }
  routeEl.addEventListener('click', e => {
    const f = e.target.closest('[data-fill]'); if (f) { submitEdit({mode: 'inpaint', fill: f.dataset.fill}); return; }
    if (e.target.closest('[data-cancel-ask]')) { pending = false; paintRoute(); toast('삭제를 취소했어요'); return; }
    if (e.target.closest('[data-clear-mask]')) { segSel = []; paintSegLayer(); clearMask(); return; }
    const s = e.target.closest('[data-sug]'); if (s) { promptEl.value = s.dataset.sug; paintRoute(); promptEl.focus(); return; }
    const st = e.target.closest('[data-strength]'); if (st) { strength = st.dataset.strength; store.set('strength', strength); paintRoute(); }
    const em = e.target.closest('[data-emode]'); if (em) { editMode = em.dataset.emode; store.set('editMode', editMode); paintRoute(); }
  });
  routeEl.addEventListener('change', e => {
    if (e.target.id === 'askDefault') setAutoFill(e.target.checked);
    if (e.target.dataset.edmodel) { edModel[e.target.dataset.edmodel] = e.target.value; paintRoute(); }
  });

  function startDelete() {
    if (!hasSel()) { toast('삭제할 부분을 먼저 고르거나 브러시로 칠하세요'); return; }
    promptEl.value = '';
    if (autoFill) { submitEdit({mode: 'inpaint', fill: 'auto'}); return; }
    pending = true; paintRoute(); promptEl.focus();
  }

  form.addEventListener('submit', e => {
    e.preventDefault();
    const t = promptEl.value.trim();
    if (pending) { if (t) submitEdit({mode: 'inpaint', fill: 'prompt', prompt: t}); return; }
    if (hasSel() && REMOVE.test(t) && t.length <= 12) { startDelete(); return; }
    if (hasSel()) { if (t) submitEdit({mode: 'inpaint', fill: 'prompt', prompt: t}); else promptEl.focus(); return; }
    if (t && qwenReady() && editMode === 'instruction') submitEdit({mode: 'reference', prompt: t});
    else if (t) submitEdit({mode: 'variation', prompt: t, denoise: STRENGTH[strength][0]});
  });

  async function submitEdit(opts) {
    if (busy || !rec) return;
    const task = {inpaint: 'inpaint', variation: 'redraw', reference: 'instruction', outpaint: 'expand'}[opts.mode];
    const body = {image_id: rec.id, prompt: opts.prompt || '', mode: opts.mode, fill: opts.fill || 'prompt', denoise: opts.denoise ?? null,
                  model: edModel[task] || 'auto',
                  aspect: opts.aspect || null, anchor: opts.anchor || 'center'};
    if (opts.mode === 'inpaint') {
      if (painted) body.mask_png = cv.toDataURL('image/png');
      else body.segments = segSel.flatMap(keyIds);
    }
    try {
      const j = await api.post('/api/edit', body);
      await runJob(j, `${j.decision.action} · ${j.decision.model_name}`);
    } catch (e) { toast(e.message, 5000); }
  }
  async function runJob(j, title) {
    const ov = document.createElement('div'); ov.className = 'busy';
    ov.innerHTML = `<div><span class="t">${esc(title)}</span><i class="bar"></i><span class="hint m">대기 중</span><button class="cancel">취소</button></div>`;
    imgbox.appendChild(ov); busy = j.id;
    ov.querySelector('.cancel').onclick = () => api.post(`/api/jobs/${j.id}/cancel`).catch(() => {});
    try {
      const done = await pollJob(j.id, x => { ov.querySelector('.bar').style.setProperty('--p', Math.round(x.progress * 100) + '%'); ov.querySelector('.m').textContent = x.status === 'queued' && x.queue_position > 1 ? `앞에 ${x.queue_position - 1}개 대기 중` : x.message; });
      if (done.status === 'done') {
        pending = false; promptEl.value = ''; autoGrow(promptEl); clearMask(false); segSel = [];
        if (done.result && done.result.fallback) toast(`처음 결과가 거의 그대로여서 ${done.result.fallback}(으)로 그 부분을 다시 그렸어요`, 6000);
        location.hash = 'edit=' + done.result_ids[0];
      } else if (done.status === 'error') toast('실패: ' + done.error + ' — 오른쪽 위 상태 표시를 누르면 진단 정보가 복사돼요', 9000);
      else toast('취소했어요');
      return done;
    } finally { ov.remove(); busy = null; }
  }

  /* ------------ brush */
  let drawing = false, last = null;
  function pos(e) { const r = cv.getBoundingClientRect(); const k = cv.width / r.width; return [(e.clientX - r.left) * k, (e.clientY - r.top) * k, k]; }
  function stroke(e) {
    const [x, y, k] = pos(e);
    cx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
    cx.strokeStyle = cx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#4d78f5';
    cx.lineWidth = brushSize * k; cx.lineCap = cx.lineJoin = 'round';
    cx.beginPath();
    if (last) { cx.moveTo(last[0], last[1]); cx.lineTo(x, y); cx.stroke(); }
    else { cx.arc(x, y, brushSize * k / 2, 0, Math.PI * 2); cx.fill(); }
    last = [x, y];
  }
  cv.addEventListener('pointerdown', e => { if (tool !== 'brush' || busy) return; if (segSel.length) { segSel = []; paintSegLayer(); } drawing = true; last = null; cv.setPointerCapture(e.pointerId); stroke(e); });
  cv.addEventListener('pointermove', e => { if (drawing) stroke(e); });
  const endStroke = () => { if (!drawing) return; drawing = false; last = null; const was = painted; painted = hasPaint(); if (was !== painted) { paintRoute(); if (tool === 'brush') paintPane(); } };
  cv.addEventListener('pointerup', endStroke); cv.addEventListener('pointercancel', endStroke);
  function hasPaint() {
    const d = cx.getImageData(0, 0, cv.width, cv.height).data;
    for (let i = 3; i < d.length; i += 64) if (d[i] > 20) return true;
    return false;
  }
  function clearMask(repaint = true) { cx.clearRect(0, 0, cv.width, cv.height); painted = false; pending = false; if (repaint) { paintRoute(); paintPane(); } }

  /* ------------ layout */
  function fit() {
    if (!img.naturalWidth || view.hidden) return;
    const cs = getComputedStyle(area);
    const aw = area.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const ah = area.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    const r = img.naturalWidth / img.naturalHeight;
    let w = Math.min(aw, ah * r), h = w / r;
    w *= zoom; h *= zoom;
    frame.style.width = w + 'px'; frame.style.height = h + 'px';
  }
  new ResizeObserver(fit).observe(area);
  function setZoom(z) { zoom = Math.max(.5, Math.min(4, Math.round(z * 10) / 10)); $('#zVal').textContent = Math.round(zoom * 100) + '%'; fit(); }
  $('#zIn').onclick = () => setZoom(zoom + .1);
  $('#zOut').onclick = () => setZoom(zoom - .1);
  area.addEventListener('wheel', e => { if (e.ctrlKey || e.metaKey) { e.preventDefault(); setZoom(zoom + (e.deltaY < 0 ? .1 : -.1)); } }, {passive: false});
  $('#panelBtn').onclick = () => { view.classList.toggle('no-panel'); fit(); };

  /* ------------ actions */
  $('#regenBtn').onclick = async () => {
    if (!rec || busy) return;
    try { const j = await api.post(`/api/images/${rec.id}/regenerate`); await runJob(j, `같은 설정, 새 시드로 다시 만드는 중 · ${j.decision.model_name}`); }
    catch (e) { toast(e.message, 5000); }
  };
  async function makeVideo() {
    if (!rec || busy) return;
    if (rec.media === 'video') { toast('영상에서 다시 영상을 만들 수는 없어요. 원본 이미지를 고르세요.'); return; }
    try {
      if (vidKeys.mid && !vidKeys.end) { toast('가운데 장면을 쓰려면 마지막 장면도 골라 주세요'); return; }
      const j = await api.post(`/api/images/${rec.id}/video`, {prompt: (vidText || promptEl.value).trim(), seconds: vidSec, quality: vidQ,
        smooth: vidSmooth, end_image_id: vidKeys.end, mid_image_id: vidKeys.mid});
      await runJob(j, `${j.decision.action} · ${j.decision.width}×${j.decision.height} · Wan 2.2`);
    } catch (e) { toast(e.message, 6000); }
  }
  $('#videoBtn').onclick = () => {
    if (!rec) return;
    if (rec.media === 'video') { toast('영상에서 다시 영상을 만들 수는 없어요. 원본 이미지를 고르세요.'); return; }
    accOpen.add('video'); store.set('acc', [...accOpen]); paintPane();
    pane.querySelector('[data-acc="video"]')?.scrollIntoView({behavior: 'smooth', block: 'start'});
    setTimeout(() => pane.querySelector('#vidText')?.focus(), 250);
  };
  $('#arBtn').onclick = () => setTool('crop');
  $('#copyBtn').onclick = async () => {
    const text = rec?.edit_prompt || rec?.prompt || '';
    try { await navigator.clipboard.writeText(text); toast('프롬프트를 복사했어요'); } catch { toast(text || '복사할 프롬프트가 없어요'); }
  };
  $('#openBtn').onclick = () => rec && window.open(imgUrl(rec.id), '_blank');
  let delArmed = 0;
  $('#delBtn').onclick = async () => {
    if (!rec) return;
    if (Date.now() - delArmed > 3000) { delArmed = Date.now(); toast('한 번 더 누르면 이 버전을 삭제해요'); return; }
    delArmed = 0;
    const idx = versions.findIndex(v => v.id === rec.id);
    await api.del(`/api/images/${rec.id}`).catch(e => toast(e.message));
    const rest = versions.filter(v => v.id !== rec.id);
    rec = null;
    location.hash = rest.length ? 'edit=' + rest[Math.max(0, idx - 1)].id : 'gallery';
    toast('삭제했어요');
  };
  document.addEventListener('keydown', e => {
    if (view.hidden) return;
    if (e.key === 'Escape' && pending) { pending = false; paintRoute(); return; }
    if (e.target === promptEl) return;
    if ((e.key === 'Delete' || e.key === 'Backspace') && hasSel() && !pending) { e.preventDefault(); startDelete(); }
    if (e.key === 'Escape' && segSel.length) { segSel = []; paintSegLayer(); paintPane(); paintRoute(); }
    if (e.key === '[') { brushSize = Math.max(6, brushSize - 6); paintPane(); }
    if (e.key === ']') { brushSize = Math.min(160, brushSize + 6); paintPane(); }
  });

  return {open};
})();

refreshStatus().then(route);
setInterval(refreshStatus, 5000);
pollSystem();
setInterval(pollSystem, 2000);
document.addEventListener('visibilitychange', pollSystem);
})();
