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
  const el = $('#status');
  if (!STATUS) { el.innerHTML = '<i class="dot bad"></i><span>Gen Studio 서버 응답 없음</span>'; return; }
  const c = STATUS.comfyui;
  const q = STATUS.queue.waiting + (STATUS.queue.running ? 1 : 0);
  el.innerHTML = c.connected
    ? `<i class="dot ok"></i><span>${esc((c.devices[0] || 'GPU').replace(/^cuda:\d+\s*/, '').split(':')[0])}${q ? ` · 작업 ${q}개` : ''}</span>`
    : '<i class="dot bad"></i><span>ComfyUI 꺼짐</span>';
  fillModelSelect();
  renderBanner();
}
function renderBanner() {
  let b = document.querySelector('.banner');
  const msgs = [];
  if (STATUS && !STATUS.comfyui.connected)
    msgs.push(`<b>ComfyUI가 꺼져 있어요.</b> 생성하려면 <code>start.bat</code>으로 실행하세요. (${esc(STATUS.comfyui.url)})`);
  const missing = MODELS.filter(m => m.installed === false);
  if (missing.length === MODELS.length && MODELS.length)
    msgs.push(`<b>아직 받은 모델이 없어요.</b> <code>install.bat</code>을 실행하면 모델을 자동으로 받습니다.`);
  else if (missing.length)
    msgs.push(`받지 않은 모델: ${missing.map(m => esc(m.name)).join(', ')}. 자동 선택은 받은 모델 중에서 고릅니다.`);
  if (!msgs.length) { b?.remove(); return; }
  if (!b) { b = document.createElement('div'); b.className = 'banner'; $('.top').after(b); }
  b.innerHTML = `<div>${msgs.join('<br>')}</div>`;
}
function fillModelSelect() {
  const sel = $('#genModel'); if (sel.options.length > 1) return;
  for (const m of MODELS) sel.add(new Option(`${m.name}${m.installed === false ? ' (미설치)' : ''}`, m.id));
}

/* --------------------------------------------------------------- router */
const VIEWS = {generate: $('#genView'), edit: $('#editView'), gallery: $('#galleryView')};
function route() {
  const h = location.hash.slice(1);
  let view = 'generate', arg = null;
  if (h.startsWith('edit=')) { view = 'edit'; arg = h.slice(5); }
  else if (h === 'gallery') view = 'gallery';
  for (const [k, el] of Object.entries(VIEWS)) el.hidden = k !== view;
  document.querySelectorAll('.tabs a').forEach(a => { if (a.dataset.view === view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  if (view === 'generate') Gen.show();
  if (view === 'gallery') Gallery.show();
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
  let recent = [], timer, refs = store.get('gen.refs', []), chars = [];
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
  prompt.value = store.get('lastPrompt', '');
  for (const [k, s] of Object.entries(sels)) {
    s.value = store.get('gen.' + k, s.value) ?? s.value;
    s.addEventListener('change', () => { store.set('gen.' + k, s.value); syncLabels(); preview(); });
  }
  function syncLabels() { for (const k in sels) lbl[k].textContent = sels[k].selectedOptions[0]?.textContent.replace(/\s*\(미설치\)/, '').split(' ')[0] || sels[k].value; }
  bindEnter(prompt, form);
  prompt.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(preview, 250); store.set('lastPrompt', prompt.value); });

  function body() {
    return {prompt: prompt.value.trim(), style: sels.style.value, aspect: sels.aspect.value, count: +sels.count.value,
            model: sels.model.value, character: charSel.value || null, references: refs};
  }
  async function preview() {
    const b = body();
    if (b.character || b.references.length) {
      const c = chars.find(x => x.id === b.character);
      routeEl.innerHTML = `<span class="pill auto">자동 선택 → <b>Qwen-Image 2.1</b> · ${c ? `캐릭터 '${esc(c.name)}' 유지` : ''}${c && refs.length ? ' · ' : ''}${refs.length ? `참고 이미지 ${refs.length}장` : ''} <span class="hint">(프롬프트에서 "image 1", "image 2"로 가리킬 수 있어요)</span></span>`;
      return;
    }
    if (!b.prompt) { routeEl.innerHTML = `<span class="hint">한국어로 써도 됩니다. "실사", "반실사", "포스터", "세로" 같은 말을 넣으면 그에 맞춰 고릅니다. 이미지에 넣을 글자는 "따옴표"로 감싸세요.</span>`; return; }
    try {
      const d = await api.post('/api/route', b);
      routeEl.innerHTML = `<span class="pill auto">자동 선택 → <b>${esc(d.model_name)}</b> · ${({semi: '반실사', text: '글자·디자인'})[d.style] || '실사'} · ${d.width}×${d.height}${d.reasons.length ? ` <span class="hint">(${esc(d.reasons.join(', '))})</span>` : ''}</span>`;
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
  return {show() { syncLabels(); loadChars().then(preview); renderRefs(); loadRecent(); autoGrow(prompt); },
          addRef(id) { if (!refs.includes(id)) setRefs([...refs, id]); }};
})();

/* ============================================================== gallery */
const Gallery = (() => {
  const grid = $('#galGrid');
  async function show() {
    let items = [];
    try { items = await api.get('/api/images?limit=500&roots_only=true'); } catch (e) { toast(e.message); }
    $('#galCount').textContent = items.length ? `${items.length}개` : '';
    grid.innerHTML = items.length ? items.map(r => `<a class="card" href="#edit=${r.id}"><img loading="lazy" src="${thumbUrl(r.id)}" alt=""><span class="cap">${esc(r.kind === 'import' ? '불러온 이미지 · ' + (r.original_name || '') : r.prompt || '')}</span></a>`).join('')
      : `<div class="empty" style="grid-column:1/-1"><b>아직 이미지가 없어요</b><span>생성 탭에서 만들거나, 이미지를 이 창에 끌어다 놓으세요.</span></div>`;
  }
  return {show};
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

/* =============================================================== editor */
const Editor = (() => {
  const view = $('#editView'), img = $('#edImg'), cv = $('#mask'), cx = cv.getContext('2d');
  const area = $('#area'), frame = $('#frame'), imgbox = $('#imgbox'), rail = $('#rail'), pane = $('#pane');
  const promptEl = $('#edPrompt'), form = $('#edForm'), routeEl = $('#edRoute');
  let rec = null, versions = [], tool = 'brush', zoom = 1, painted = false, pending = false, busy = null;
  let brushSize = store.get('brush', 40), erasing = false, strength = store.get('strength', 'mid');
  let autoFill = store.get('autoFill', false);
  const STRENGTH = {low: [0.35, '약하게'], mid: [0.55, '보통'], high: [0.75, '강하게']};
  let editMode = store.get('editMode', 'instruction');     // instruction (Qwen) | redraw (same model)
  let expandAspect = '3:4', expandAnchor = 'top';
  const qwenReady = () => MODELS.some(m => m.family === 'qwen21' && m.installed !== false);
  const REMOVE = /지워|지우|제거|없애|빼\s*줘|remove|erase|delete/i;
  bindEnter(promptEl, form);
  promptEl.addEventListener('input', paintRoute);

  async function open(id) {
    if (!id) { location.hash = 'gallery'; return; }
    if (rec && rec.id === id && img.src) { fit(); return; }
    let data;
    try { data = await api.get(`/api/images/${id}`); } catch (e) { toast(e.message); location.hash = 'gallery'; return; }
    rec = data; versions = data.versions || [rec];
    pending = false; clearMask(false);
    $('#edTitle').textContent = rec.kind === 'import' ? '불러온 이미지' : (rec.prompt || '이미지');
    const model = MODELS.find(m => m.id === rec.model_id);
    $('#edSub').textContent = [rec.label, model?.name, `${rec.width}×${rec.height}`].filter(Boolean).join(' · ');
    img.onload = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; painted = false; fit(); };
    img.src = imgUrl(rec.id, Math.round(rec.created_at));
    $('#saveBtn').href = `/api/images/${rec.id}/file?download=1`;
    $('#regenBtn').disabled = rec.kind === 'import';
    paintRail(); paintPane(); paintRoute();
  }

  function paintRail() {
    rail.innerHTML = versions.map((v, i) => `<button class="${v.id === rec.id ? 'cur' : ''}" data-id="${v.id}" title="${i + 1}. ${esc(v.label || '')}" aria-label="버전 ${i + 1}: ${esc(v.label || '')}"><img src="${thumbUrl(v.id)}" alt=""></button>`).join('');
  }
  rail.addEventListener('click', e => { const b = e.target.closest('[data-id]'); if (b && !busy) location.hash = 'edit=' + b.dataset.id; });

  /* ------------ side pane */
  function paintPane() {
    let h = '';
    if (tool === 'seg') {
      h = `<h2>세그먼트 <span class="badge">곧 추가</span></h2>
        <p class="note">사람, 머리카락, 옷 같은 부분을 자동으로 나눠서 목록으로 보여주는 기능은 지금 만드는 중입니다. 그전까지는 <b>브러시</b>로 바꿀 부분을 칠하세요.</p>
        <div class="seg2"><button data-goto="brush" aria-pressed="true">브러시로 칠하기</button></div>`;
    } else if (tool === 'brush') {
      h = `<h2>마스크 브러시</h2>
        <p class="note">바꿀 부분을 칠한 뒤 아래에 무엇으로 바꿀지 입력하세요. 칠한 부분만 다시 그리고 나머지는 그대로 둡니다.</p>
        <div class="field"><label for="bSize">브러시 크기 <b id="bSizeV">${brushSize}</b>px</label><input type="range" id="bSize" min="6" max="160" value="${brushSize}"></div>
        <div class="field"><span>모드</span><div class="seg2"><button data-bm="paint" aria-pressed="${!erasing}">칠하기</button><button data-bm="erase" aria-pressed="${erasing}">지우개</button><button id="bClear">모두 지우기</button></div></div>
        <div class="field"><div class="seg2"><button id="bDelete" ${painted ? '' : 'disabled'}>칠한 부분 삭제하기</button></div></div>`;
    } else if (tool === 'color') {
      h = `<h2>색상 <span class="badge">2단계</span></h2><p class="note">선택한 부분의 색만 바로 바꾸는 기능은 세그먼트와 함께 2단계에서 추가됩니다. 지금은 칠한 뒤 "빨간색으로"처럼 입력하면 됩니다.</p>`;
    } else if (tool === 'crop') {
      const ASP = ['3:4', '2:3', '9:16', '4:5', '1:1', '4:3', '3:2', '16:9'];
      const ANC = [['top', '위에 붙이기'], ['center', '가운데'], ['bottom', '아래에 붙이기'], ['left', '왼쪽'], ['right', '오른쪽']];
      h = `<h2>캔버스 늘리기</h2>
        <p class="note">그림 바깥을 늘리고 새로 생긴 부분을 AI가 채웁니다. 얼굴만 나온 초상화를 허벅지까지 보이게 하려면 <b>3:4 · 위에 붙이기</b>를 고르세요. 아래 입력칸에 늘어난 부분에 그릴 내용을 적어도 됩니다.</p>
        <div class="field"><span>새 비율 (지금 ${rec ? `${rec.width}×${rec.height}` : ''})</span><div class="seg2">${ASP.map(a => `<button data-asp="${a}" aria-pressed="${a === expandAspect}">${a}</button>`).join('')}</div></div>
        <div class="field"><span>원래 그림 위치</span><div class="seg2">${ANC.map(([k, n]) => `<button data-anc="${k}" aria-pressed="${k === expandAnchor}">${n}</button>`).join('')}</div></div>
        <div class="field"><div class="seg2"><button id="doExpand" aria-pressed="true">늘리기</button></div></div>`;
    } else {
      h = `<h2>그려서 지시 <span class="badge">3단계</span></h2><p class="note">이미지 위에 대충 그리고 "여기에 가방 추가"처럼 입력하면 그 모양대로 그려 넣는 기능입니다. 3단계에서 추가됩니다.</p>`;
    }
    h += `<label class="switch" style="margin:4px 0 16px"><input type="checkbox" id="autoFillToggle" ${autoFill ? 'checked' : ''}><span>삭제하면 묻지 않고 모델이 알아서 채우기</span></label>`;
    if (rec) {
      const model = MODELS.find(m => m.id === rec.model_id);
      h += `<h2>정보</h2><dl class="kv">
        ${rec.prompt ? `<dt>프롬프트</dt><dd>${esc(rec.prompt)}</dd>` : ''}
        ${rec.edit_prompt ? `<dt>편집</dt><dd>${esc(rec.edit_prompt)}</dd>` : ''}
        <dt>모델</dt><dd>${esc(model?.name || rec.model_id || (rec.embedded?.tool ? rec.embedded.tool + ' (외부)' : '외부 이미지'))}</dd>
        <dt>크기</dt><dd>${rec.width}×${rec.height}</dd>
        ${rec.seed !== undefined ? `<dt>시드</dt><dd>${rec.seed}</dd>` : ''}
        <dt>버전</dt><dd>${versions.findIndex(v => v.id === rec.id) + 1} / ${versions.length}</dd></dl>
        <h2 style="margin-top:18px">이 인물 다시 쓰기</h2>
        <div class="field"><div class="inline"><input id="charName" placeholder="캐릭터 이름 (예: 플라이트 디렉터)"><button class="btn" id="saveChar">캐릭터로 등록</button></div></div>
        <div class="field"><div class="seg2"><button id="useAsRef">생성 화면의 참고 이미지로 보내기</button></div></div>`;
    }
    pane.innerHTML = h;
  }
  pane.addEventListener('input', e => { if (e.target.id === 'bSize') { brushSize = +e.target.value; $('#bSizeV').textContent = brushSize; store.set('brush', brushSize); } });
  pane.addEventListener('change', e => { if (e.target.id === 'autoFillToggle') setAutoFill(e.target.checked); });
  pane.addEventListener('click', e => {
    const g = e.target.closest('[data-goto]'); if (g) setTool(g.dataset.goto);
    const bm = e.target.closest('[data-bm]'); if (bm) { erasing = bm.dataset.bm === 'erase'; paintPane(); }
    if (e.target.closest('#bClear')) clearMask();
    if (e.target.closest('#bDelete')) startDelete();
    const asp = e.target.closest('[data-asp]'); if (asp) { expandAspect = asp.dataset.asp; paintPane(); }
    const anc = e.target.closest('[data-anc]'); if (anc) { expandAnchor = anc.dataset.anc; paintPane(); }
    if (e.target.closest('#doExpand')) submitEdit({mode: 'outpaint', aspect: expandAspect, anchor: expandAnchor, prompt: promptEl.value.trim()});
    if (e.target.closest('#useAsRef') && rec) { Gen.addRef(rec.id); toast('생성 화면의 참고 이미지에 넣었어요'); }
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
    paintPane();
  }
  document.querySelectorAll('[data-tool]').forEach(b => b.onclick = () => setTool(b.dataset.tool));

  /* ------------ prompt / routing */
  const SUG_MASK = ['지워줘', '빨간색으로', '검은 가죽 재킷'];
  function paintRoute() {
    const t = promptEl.value.trim();
    const model = MODELS.find(m => m.id === rec?.model_id);
    const modelName = model?.name || '기본 실사 모델';
    let h = '';
    if (pending) {
      h = `<div class="ask"><b>칠한 부분 삭제 · 그 자리를 어떻게 채울까요? 아래에 입력하거나 골라 주세요.</b>
        <div class="opts"><button type="button" class="pill main" data-fill="auto">모델이 알아서 채우기</button>
        <button type="button" class="pill plain" data-fill="clear">비워두기 (배경으로 메우기)</button>
        <button type="button" class="pill sug" data-cancel-ask>삭제 취소</button></div>
        <label class="switch"><input type="checkbox" id="askDefault" ${autoFill ? 'checked' : ''}><span>앞으로는 묻지 않고 항상 모델이 알아서 채우기</span></label>
        ${t ? `<span class="pill auto">→ <b>삭제한 자리 다시 그리기</b> · ${esc(modelName)}</span>` : ''}</div>`;
    } else if (painted) {
      h += `<span class="pill sel">@칠한 영역<button type="button" data-clear-mask aria-label="칠한 영역 지우기"><svg class="i" viewBox="0 0 24 24" style="width:14px;height:14px"><path d="M6 6l12 12M18 6 6 18"/></svg></button></span>`;
      if (!t) h += SUG_MASK.map(s => `<button type="button" class="pill sug" data-sug="${s}">${s}</button>`).join('');
      else if (REMOVE.test(t)) h += `<span class="pill auto">→ <b>삭제</b> 후 채울 내용을 물어봅니다</span>`;
      else h += `<span class="pill auto">자동 선택 → <b>부분 다시 그리기</b> · ${esc(modelName)} <span class="hint">(원본과 같은 모델)</span></span>`;
    } else if (t) {
      const mode = qwenReady() ? editMode : 'redraw';
      if (qwenReady()) h += `<button type="button" class="pill ${mode === 'instruction' ? 'plain' : 'sug'}" data-emode="instruction">문장 편집 · Qwen</button><button type="button" class="pill ${mode === 'redraw' ? 'plain' : 'sug'}" data-emode="redraw">전체 다시 그리기 · 원본 모델</button>`;
      if (mode === 'instruction') h += `<span class="hint">인물은 유지하고 말한 부분만 바꿔요. NSFW는 '다시 그리기'나 브러시가 맞아요.</span>`;
      else h += Object.entries(STRENGTH).map(([k, [, name]]) => `<button type="button" class="pill ${k === strength ? 'plain' : 'sug'}" data-strength="${k}">${name}</button>`).join('') + `<span class="hint">${esc(modelName)}</span>`;
    } else {
      h = `<span class="hint">바꿀 부분을 브러시로 칠하세요. 칠하지 않고 "배경을 우주선 함교로"처럼 입력하면 문장대로 편집합니다.</span>`;
    }
    routeEl.innerHTML = h;
    promptEl.placeholder = pending ? '삭제한 자리에 무엇을 그릴지 입력하세요' : (painted ? '칠한 부분을 무엇으로 바꿀까요?' : '바꿀 부분을 브러시로 칠하고 무엇으로 바꿀지 입력하세요');
  }
  routeEl.addEventListener('click', e => {
    const f = e.target.closest('[data-fill]'); if (f) { submitEdit({mode: 'inpaint', fill: f.dataset.fill}); return; }
    if (e.target.closest('[data-cancel-ask]')) { pending = false; paintRoute(); toast('삭제를 취소했어요'); return; }
    if (e.target.closest('[data-clear-mask]')) { clearMask(); return; }
    const s = e.target.closest('[data-sug]'); if (s) { promptEl.value = s.dataset.sug; paintRoute(); promptEl.focus(); return; }
    const st = e.target.closest('[data-strength]'); if (st) { strength = st.dataset.strength; store.set('strength', strength); paintRoute(); }
    const em = e.target.closest('[data-emode]'); if (em) { editMode = em.dataset.emode; store.set('editMode', editMode); paintRoute(); }
  });
  routeEl.addEventListener('change', e => { if (e.target.id === 'askDefault') setAutoFill(e.target.checked); });

  function startDelete() {
    if (!painted) { toast('삭제할 부분을 먼저 브러시로 칠하세요'); return; }
    promptEl.value = '';
    if (autoFill) { submitEdit({mode: 'inpaint', fill: 'auto'}); return; }
    pending = true; paintRoute(); promptEl.focus();
  }

  form.addEventListener('submit', e => {
    e.preventDefault();
    const t = promptEl.value.trim();
    if (pending) { if (t) submitEdit({mode: 'inpaint', fill: 'prompt', prompt: t}); return; }
    if (painted && REMOVE.test(t) && t.length <= 12) { startDelete(); return; }
    if (painted) { if (t) submitEdit({mode: 'inpaint', fill: 'prompt', prompt: t}); else promptEl.focus(); return; }
    if (t && qwenReady() && editMode === 'instruction') submitEdit({mode: 'reference', prompt: t});
    else if (t) submitEdit({mode: 'variation', prompt: t, denoise: STRENGTH[strength][0]});
  });

  async function submitEdit(opts) {
    if (busy || !rec) return;
    const body = {image_id: rec.id, prompt: opts.prompt || '', mode: opts.mode, fill: opts.fill || 'prompt', denoise: opts.denoise ?? null,
                  aspect: opts.aspect || null, anchor: opts.anchor || 'center'};
    if (opts.mode === 'inpaint') body.mask_png = cv.toDataURL('image/png');
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
        pending = false; promptEl.value = ''; autoGrow(promptEl); clearMask(false);
        location.hash = 'edit=' + done.result_ids[0];
      } else if (done.status === 'error') toast('실패: ' + done.error, 7000);
      else toast('취소했어요');
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
  cv.addEventListener('pointerdown', e => { if (tool !== 'brush' || busy) return; drawing = true; last = null; cv.setPointerCapture(e.pointerId); stroke(e); });
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
  $('#videoBtn').onclick = () => toast('동영상 만들기는 4단계에서 추가됩니다');
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
    if ((e.key === 'Delete' || e.key === 'Backspace') && painted && !pending) { e.preventDefault(); startDelete(); }
    if (e.key === '[') { brushSize = Math.max(6, brushSize - 6); paintPane(); }
    if (e.key === ']') { brushSize = Math.min(160, brushSize + 6); paintPane(); }
  });

  return {open};
})();

refreshStatus().then(route);
setInterval(refreshStatus, 5000);
})();
