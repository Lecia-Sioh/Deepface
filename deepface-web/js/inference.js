/* Public page — no auth guard */
DF.renderNav('inference');

const $ = id => document.getElementById(id);
const settings = DF.store.settings();
const persons = DF.store.persons();
const COLORS = { Blacklisted: '#ff2d55', Authorised: '#1fff8f', Unknown: '#8b8b99' };

$('spec').innerHTML = `<span>MODEL <b>${settings.model}</b></span><span>METRIC <b>${settings.metric}</b></span><span>THRESHOLD <b>${settings.threshold}</b></span>`;
$('nth').value = settings.frameSkip; $('nVal').textContent = settings.frameSkip;

/* ---------- sidebar ---------- */
function renderProfiles() {
  $('profiles').innerHTML = persons.length ? persons.map(p => `
    <div class="profile ${p.status === 'Blacklisted' ? 'bl' : ''}" id="pf-${p.id}">
      ${DF.avatar(p)}
      <div>
        <div class="nm">${DF.esc(p.name)}</div>
        <div class="meta">🪪 ID: ${DF.esc(p.idNumber) || '—'}</div>
        <div style="margin:4px 0">${DF.statusTag(p.status)}</div>
        <div class="meta">${p.photoCount} photo(s) · ${p.vectors} vectors</div>
      </div>
    </div>`).join('') : '<div class="empty">No profiles registered</div>';
}
renderProfiles();
function flashProfile(name) {
  const p = persons.find(x => x.name === name); if (!p) return;
  const el = $('pf-' + p.id); if (!el) return;
  el.classList.remove('hit'); void el.offsetWidth; el.classList.add('hit');
}

/* ---------- source switch ---------- */
$('srcSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $('srcSeg').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  $('src-webcam').classList.toggle('hidden', b.dataset.v !== 'webcam');
  $('src-video').classList.toggle('hidden', b.dataset.v !== 'video');
  if (b.dataset.v !== 'webcam') stopCam();
});

/* ---------- shared drawing ---------- */
function drawFaces(ctx, faces) {
  ctx.lineWidth = 3;
  ctx.font = 'bold 15px "JetBrains Mono", monospace';
  faces.forEach(f => {
    const [x, y, w, h] = f.box;
    const c = COLORS[f.status] || COLORS.Unknown;
    ctx.strokeStyle = c; ctx.strokeRect(x, y, w, h);
    const label = f.status === 'Unknown' ? 'UNKNOWN' : `${f.name} ${(f.similarity * 100).toFixed(1)}%`;
    const tw = ctx.measureText(label).width + 14;
    ctx.fillStyle = c; ctx.fillRect(x - 1.5, y - 26, tw, 24);
    ctx.fillStyle = f.status === 'Blacklisted' ? '#fff' : '#000';
    ctx.fillText(label, x + 6, y - 9);
  });
}

function alertHTML(faces) {
  if (!faces.length) return '';
  const bl = faces.find(f => f.status === 'Blacklisted');
  if (bl) return `<div class="alert alert-bl">🚨 BLACKLISTED: ${DF.esc(bl.name)} | Similarity: ${(bl.similarity * 100).toFixed(1)}%</div>`;
  const ok = faces.find(f => f.status === 'Authorised');
  if (ok) return `<div class="alert alert-ok">✅ AUTHORISED: ${DF.esc(ok.name)} | Similarity: ${(ok.similarity * 100).toFixed(1)}%</div>`;
  return `<div class="alert alert-unk">❔ UNKNOWN face detected — no match under threshold ${settings.threshold}</div>`;
}

/* ---------- DEMO recogniser ----------
   Without a backend we can't run ArcFace in the browser, so we simulate a
   stable identity "track": the same person is reported for a few seconds,
   similarity jitters realistically, then it may switch. */
const sim = {
  current: null, until: 0,
  pick(now) {
    if (now < sim.until && sim.current) return sim.current;
    const pool = [...persons.map(p => ({ name: p.name, status: p.status, idNumber: p.idNumber })), { name: 'UNKNOWN', status: 'Unknown', idNumber: '' }];
    const t = pool[Math.floor(Math.random() * pool.length)];
    sim.current = { ...t, base: t.status === 'Unknown' ? 0 : 0.6 + Math.random() * 0.15 };
    sim.until = now + 3000 + Math.random() * 4000;
    return sim.current;
  },
  face(now, W, H, box) {
    const t = sim.pick(now);
    const wob = Math.sin(now / 400) * 6;
    return {
      box: box || [W * 0.35 + wob, H * 0.2, W * 0.3, H * 0.5],
      name: t.name, status: t.status, idNumber: t.idNumber,
      similarity: t.status === 'Unknown' ? 0 : Math.min(0.95, t.base + (Math.random() - 0.5) * 0.04),
    };
  },
};
/* ---------- real face *detection* in the browser (box position) ----------
   Uses MediaPipe BlazeFace (loaded from CDN on first use). Falls back to the
   native FaceDetector API, and finally to a fixed centre box. */
const MP_VER = '0.10.14';
let mpDetector = null, mpLoading = null, mpFailed = false;
const nativeDetector = 'FaceDetector' in window ? new window.FaceDetector({ fastMode: true, maxDetectedFaces: 5 }) : null;

function loadMediaPipe() {
  if (mpDetector || mpFailed) return Promise.resolve(mpDetector);
  return mpLoading ||= (async () => {
    try {
      const { FaceDetector, FilesetResolver } = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`);
      mpDetector = await FaceDetector.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
          delegate: 'GPU',
        },
        runningMode: 'IMAGE',
        minDetectionConfidence: 0.5,
      });
    } catch (e) {
      console.warn('MediaPipe face detector failed to load', e);
      mpFailed = true;
      DF.toast('Face detector could not load (offline?), so the box position is simulated', 'danger');
    }
    return mpDetector;
  })();
}

// Returns an array of [x,y,w,h] boxes, or null if no real detector is available.
async function detectBoxes(src) {
  const mp = await loadMediaPipe();
  if (mp) {
    try { return mp.detect(src).detections.map(d => { const b = d.boundingBox; return [b.originX, b.originY, b.width, b.height]; }); }
    catch (e) { console.warn(e); }
  }
  if (nativeDetector) {
    try { return (await nativeDetector.detect(src)).map(d => { const b = d.boundingBox; return [b.x, b.y, b.width, b.height]; }); }
    catch {}
  }
  return null;
}

// First face gets the simulated identity; any extra faces are reported as Unknown.
function demoFaces(now, W, H, boxes) {
  if (!boxes) return [sim.face(now, W, H, null)];
  return boxes.map((box, i) => i === 0
    ? sim.face(now, W, H, box)
    : { box, name: 'UNKNOWN', status: 'Unknown', idNumber: '', similarity: 0 });
}

/* ---------- blacklist logging (throttled) ---------- */
const lastLogged = {};
function logFaces(faces, source) {
  const now = Date.now();
  faces.forEach(f => {
    if (f.status === 'Unknown') return;
    if (now - (lastLogged[f.name] || 0) < 10000) return;
    lastLogged[f.name] = now;
    DF.store.addLog({
      type: f.status === 'Blacklisted' ? 'alert' : 'match',
      msg: `${f.status === 'Blacklisted' ? '🚨 BLACKLISTED' : 'Authorised'}: ${f.name} · ${(f.similarity * 100).toFixed(1)}% · ${source}`,
    });
    flashProfile(f.name);
  });
}

/* ================= WEBCAM ================= */
let stream = null, raf = null, inflight = false, lastSend = 0, liveFaces = [], frames = 0, fpsT = performance.now();

async function startCam() {
  if (!DF.api.live()) loadMediaPipe(); // start downloading the detector model in parallel
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
  } catch { return DF.toast('Camera blocked or unavailable', 'danger'); }
  const v = $('cam'); v.srcObject = stream;
  await new Promise(r => v.onloadedmetadata = r);
  $('overlay').width = v.videoWidth; $('overlay').height = v.videoHeight;
  $('viewer').classList.add('on'); $('ph').classList.add('hidden');
  $('camBtn').textContent = '■ Stop Camera Feed'; $('camBtn').className = 'btn btn-danger';
  loop();
}
function stopCam() {
  cancelAnimationFrame(raf);
  stream?.getTracks().forEach(t => t.stop()); stream = null;
  $('cam').srcObject = null;
  const o = $('overlay'); o.getContext('2d').clearRect(0, 0, o.width, o.height);
  $('viewer').classList.remove('on'); $('ph').classList.remove('hidden');
  $('camBtn').textContent = '▶ Start Camera Feed'; $('camBtn').className = 'btn btn-primary';
  $('liveAlert').innerHTML = ''; $('fps').textContent = '';
}
$('camBtn').onclick = () => stream ? stopCam() : startCam();

async function loop() {
  if (!stream) return;
  const v = $('cam'), o = $('overlay'), ctx = o.getContext('2d');
  const now = performance.now();

  // non-blocking: only send a new frame when the previous request has returned
  if (!inflight && now - lastSend > 250) {
    lastSend = now; inflight = true;
    recogniseFrame(v).then(f => { liveFaces = f; $('liveAlert').innerHTML = alertHTML(f); logFaces(f, 'webcam'); })
      .catch(err => { DF.toast('Backend error: ' + err.message, 'danger'); })
      .finally(() => inflight = false);
  }
  ctx.clearRect(0, 0, o.width, o.height);
  drawFaces(ctx, liveFaces);

  frames++;
  if (now - fpsT > 1000) { $('fps').textContent = `${frames} fps · ${DF.api.live() ? 'live API' : 'demo'}`; frames = 0; fpsT = now; }
  raf = requestAnimationFrame(loop);
}

async function recogniseFrame(v) {
  if (DF.api.live()) {
    const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.8));
    const fd = new FormData(); fd.append('frame', blob, 'frame.jpg'); fd.append('threshold', settings.threshold);
    return (await DF.api.post('/recognize', fd)).faces || [];
  }
  const boxes = await detectBoxes(v);
  return demoFaces(performance.now(), v.videoWidth, v.videoHeight, boxes);
}

/* ================= VIDEO ================= */
let vfile = null, lastReport = null;
$('nth').oninput = e => $('nVal').textContent = e.target.value;

const vDrop = $('vDrop'), vInput = $('vInput');
vDrop.onclick = () => vInput.click();
vInput.onchange = () => setVideo(vInput.files[0]);
vDrop.ondragover = e => { e.preventDefault(); vDrop.classList.add('over'); };
vDrop.ondragleave = () => vDrop.classList.remove('over');
vDrop.ondrop = e => { e.preventDefault(); vDrop.classList.remove('over'); setVideo(e.dataTransfer.files[0]); };

function setVideo(f) {
  if (!f || !f.type.startsWith('video/')) return DF.toast('Please choose a video file', 'danger');
  if (f.size > 200 * 1024 * 1024) return DF.toast('Limit is 200MB per file', 'danger');
  vfile = f;
  $('vFile').textContent = `🎞️ ${f.name} · ${(f.size / 1024 / 1024).toFixed(1)}MB`;
  $('analyseBtn').disabled = false;
}

$('analyseBtn').onclick = async () => {
  if (!vfile) return;
  const btn = $('analyseBtn'); btn.disabled = true; btn.textContent = '⏳ Analysing…';
  $('report').classList.add('hidden'); $('vAlert').innerHTML = '';
  const n = +$('nth').value;
  try {
    if (DF.api.live()) await analyseLive(n); else await analyseDemo(n);
  } catch (e) { DF.toast('Analysis failed: ' + e.message, 'danger'); }
  btn.disabled = false; btn.textContent = '▶ Analyse Video';
};

async function analyseLive(n) {
  $('vStatus').textContent = 'Uploading to backend…';
  const fd = new FormData(); fd.append('video', vfile); fd.append('every_n', n);
  const res = await DF.api.post('/analyse-video', fd);
  $('vStatus').textContent = `Done · ${res.frames} frames analysed`;
  showReport(res.report, res.frames, res.duration, res.annotated_url ? DF.api.base() + res.annotated_url : null);
}

async function analyseDemo(n) {
  const v = document.createElement('video');
  v.muted = true; v.src = URL.createObjectURL(vfile);
  await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = rej; });
  const FPS = 24, total = Math.floor(v.duration * FPS), step = n / FPS;
  const c = $('vCanvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  const ctx = c.getContext('2d');
  $('vViewer').classList.remove('hidden'); $('vViewer').classList.add('on');
  $('vProg').classList.remove('hidden');

  const tracks = {}; let analysed = 0;
  sim.current = null;
  for (let t = 0; t < v.duration; t += step) {
    v.currentTime = t;
    await new Promise(r => v.onseeked = r);
    ctx.drawImage(v, 0, 0);
    const boxes = await detectBoxes(c);
    const faces = demoFaces(t * 1000 * 1.8, c.width, c.height, boxes);
    drawFaces(ctx, faces);
    $('vAlert').innerHTML = alertHTML(faces);
    faces.forEach(f => {
      const k = f.name;
      const tr = tracks[k] ||= { name: f.name, idNumber: f.idNumber, status: f.status, first: t, last: t, det: 0, best: 0 };
      tr.last = t; tr.det++; tr.best = Math.max(tr.best, f.similarity);
    });
    analysed++;
    $('vProg').firstElementChild.style.width = `${(t / v.duration) * 100}%`;
    $('vStatus').textContent = `Frame ${Math.round(t * FPS)} / ${total}`;
    await new Promise(r => setTimeout(r, 15));
  }
  $('vProg').firstElementChild.style.width = '100%';
  $('vStatus').textContent = `Done · ${analysed} frames analysed`;
  $('vViewer').classList.remove('on');
  URL.revokeObjectURL(v.src);

  const report = Object.values(tracks).map(r => ({
    name: r.name, id_number: r.idNumber, status: r.status,
    first_seen: r.first, last_seen: r.last, detections: r.det, best_similarity: +(r.best * 100).toFixed(1),
  }));
  showReport(report, analysed, v.duration, null);
}

function showReport(report, frames, duration, annotatedUrl) {
  lastReport = report;
  $('report').classList.remove('hidden');
  $('repMeta').textContent = `${vfile.name} · ${frames} frames · ${DF.fmtTime(duration)} long`;
  const bl = report.filter(r => r.status === 'Blacklisted');
  $('repAlert').innerHTML = bl.length
    ? `<div class="alert alert-bl">🚨 Blacklisted person(s) appeared in this video: ${bl.map(r => DF.esc(r.name)).join(', ')}</div>`
    : `<div class="alert alert-ok">✅ No blacklisted persons detected</div>`;
  $('repRows').innerHTML = report.sort((a, b) => a.first_seen - b.first_seen).map(r => `
    <tr>
      <td><b>${DF.esc(r.name)}</b></td>
      <td class="mono">${DF.esc(r.id_number) || '<span class="muted">—</span>'}</td>
      <td>${DF.statusTag(r.status)}</td>
      <td class="mono">${DF.fmtTime(r.first_seen)}</td>
      <td class="mono">${DF.fmtTime(r.last_seen)}</td>
      <td class="mono">${r.detections}</td>
      <td class="mono">${r.best_similarity.toFixed(1)}</td>
    </tr>`).join('');
  const ann = $('annBtn');
  ann.classList.toggle('hidden', !annotatedUrl);
  if (annotatedUrl) ann.href = annotatedUrl;
  bl.forEach(r => DF.store.addLog({ type: 'alert', msg: `🚨 BLACKLISTED: ${r.name} in ${vfile.name} · ${r.best_similarity}%` }));
  $('report').scrollIntoView({ behavior: 'smooth' });
}

$('csvBtn').onclick = () => {
  if (!lastReport) return;
  const head = ['Name', 'ID Number', 'Status', 'First Seen', 'Last Seen', 'Detections', 'Best Similarity (%)'];
  const rows = lastReport.map(r => [r.name, r.id_number, r.status, DF.fmtTime(r.first_seen), DF.fmtTime(r.last_seen), r.detections, r.best_similarity]);
  const csv = [head, ...rows].map(r => r.map(x => `"${String(x ?? '').replace(/"/g, '""')}"`).join(',')).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `report_${vfile.name.replace(/\.\w+$/, '')}.csv`; a.click();
};

window.addEventListener('beforeunload', stopCam);
