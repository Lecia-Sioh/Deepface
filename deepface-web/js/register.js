DF.auth.guard('user');
DF.renderNav('register');

const $ = id => document.getElementById(id);
const state = { mode: 'new', status: 'Authorised', method: 'photo', shots: [] /* {thumb, blob} */ };
let stream = null;

/* ---------- segmented controls ---------- */
function seg(el, onChange) {
  el.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    el.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    onChange(b.dataset.v);
  });
}
seg($('modeSeg'), v => {
  state.mode = v;
  $('newFields').classList.toggle('hidden', v !== 'new');
  $('existingFields').classList.toggle('hidden', v !== 'existing');
  if (v === 'existing') fillExisting();
});
seg($('statusSeg'), v => state.status = v);
seg($('methodSeg'), v => {
  state.method = v;
  ['photo', 'video', 'webcam'].forEach(m => $('m-' + m).classList.toggle('hidden', m !== v));
  if (v !== 'webcam') stopCam();
});

function fillExisting() {
  const persons = DF.store.persons();
  $('existingSel').innerHTML = persons.length
    ? persons.map(p => `<option value="${p.id}">${DF.esc(p.name)} ${p.idNumber ? '· ' + DF.esc(p.idNumber) : ''}</option>`).join('')
    : '<option value="">No persons registered yet</option>';
  showExistingInfo();
}
function showExistingInfo() {
  const p = DF.store.persons().find(x => x.id === $('existingSel').value);
  $('existingInfo').innerHTML = p ? `${DF.statusTag(p.status)} &nbsp; ${p.photoCount} photo(s) · ${p.vectors} vectors currently stored` : '';
}
$('existingSel').addEventListener('change', showExistingInfo);

/* ---------- thumbnails ---------- */
function renderThumbs() {
  $('thumbs').innerHTML = state.shots.map((s, i) =>
    `<div class="t"><img src="${s.thumb}" alt=""><button data-i="${i}" title="Remove">×</button></div>`).join('');
  $('count').textContent = state.shots.length;
  $('vecCount').textContent = state.shots.length * DF.VECTORS_PER_PHOTO;
}
$('thumbs').addEventListener('click', e => {
  const i = e.target.dataset.i; if (i === undefined) return;
  state.shots.splice(+i, 1); renderThumbs();
});
$('clearBtn').onclick = () => { state.shots = []; renderThumbs(); };

/* ---------- drop zones ---------- */
function dropZone(zone, input, handler) {
  zone.onclick = () => input.click();
  input.onchange = () => { handler([...input.files]); input.value = ''; };
  zone.ondragover = e => { e.preventDefault(); zone.classList.add('over'); };
  zone.ondragleave = () => zone.classList.remove('over');
  zone.ondrop = e => { e.preventDefault(); zone.classList.remove('over'); handler([...e.dataTransfer.files]); };
}

dropZone($('photoDrop'), $('photoInput'), async files => {
  const imgs = files.filter(f => /image\/(png|jpe?g)/.test(f.type));
  if (!imgs.length) return DF.toast('Only JPG / PNG files please', 'danger');
  for (const f of imgs) state.shots.push({ thumb: await DF.fileToThumb(f), blob: f });
  renderThumbs();
  DF.toast(`${imgs.length} photo(s) added`, 'success');
});

/* ---------- video: extract evenly-spaced frames ---------- */
$('frameCount').oninput = e => $('fcVal').textContent = e.target.value;
dropZone($('videoDrop'), $('videoInput'), async files => {
  const f = files.find(x => x.type.startsWith('video/'));
  if (!f) return DF.toast('Please drop a video file', 'danger');
  const n = +$('frameCount').value;
  const v = document.createElement('video');
  v.muted = true; v.src = URL.createObjectURL(f);
  await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(); }).catch(() => null);
  if (!v.duration) return DF.toast('Could not read that video', 'danger');

  const prog = $('vProg'); prog.classList.remove('hidden');
  for (let i = 0; i < n; i++) {
    v.currentTime = (v.duration * (i + 0.5)) / n;
    await new Promise(r => v.onseeked = r);
    const thumb = DF.canvasThumb(v, 320);
    const blob = await (await fetch(thumb)).blob();
    state.shots.push({ thumb: DF.canvasThumb(v), blob });
    prog.firstElementChild.style.width = `${((i + 1) / n) * 100}%`;
    renderThumbs();
  }
  URL.revokeObjectURL(v.src);
  setTimeout(() => prog.classList.add('hidden'), 600);
  DF.toast(`Extracted ${n} frames from ${f.name}`, 'success');
});

/* ---------- webcam ---------- */
async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    $('cam').srcObject = stream;
    $('camViewer').classList.add('on'); $('camPh').classList.add('hidden');
    $('snapBtn').disabled = false; $('camToggle').textContent = 'Stop camera';
  } catch { DF.toast('Camera blocked or unavailable', 'danger'); }
}
function stopCam() {
  stream?.getTracks().forEach(t => t.stop()); stream = null;
  $('cam').srcObject = null;
  $('camViewer').classList.remove('on'); $('camPh').classList.remove('hidden');
  $('snapBtn').disabled = true; $('camToggle').textContent = 'Start camera';
}
$('camToggle').onclick = () => stream ? stopCam() : startCam();
$('snapBtn').onclick = async () => {
  const v = $('cam');
  const full = DF.canvasThumb(v, 640);
  state.shots.push({ thumb: DF.canvasThumb(v), blob: await (await fetch(full)).blob() });
  renderThumbs();
  $('camViewer').animate([{ filter: 'brightness(3)' }, { filter: 'none' }], 250);
};

/* ---------- submit ---------- */
$('registerBtn').onclick = async () => {
  if (!state.shots.length) return DF.toast('Add at least one photo first', 'danger');
  const persons = DF.store.persons();
  let person;

  if (state.mode === 'new') {
    const name = $('pname').value.trim();
    if (!name) return DF.toast('Enter the person\'s name', 'danger');
    if (persons.some(p => p.name.toLowerCase() === name.toLowerCase()))
      return DF.toast(`"${name}" already exists — use "Add photos to existing"`, 'danger');
    person = { id: DF.uid(), name, idNumber: $('pid').value.trim(), status: state.status, photos: [], photoCount: 0, vectors: 0, created: Date.now() };
  } else {
    person = persons.find(p => p.id === $('existingSel').value);
    if (!person) return DF.toast('Pick a person', 'danger');
  }

  const btn = $('registerBtn');
  btn.disabled = true; btn.textContent = '⏳ Extracting embeddings…';

  let backendVectors = null;
  try {
    if (DF.api.live()) {
      const fd = new FormData();
      fd.append('name', person.name);
      fd.append('id_number', person.idNumber || '');
      fd.append('status', person.status);
      fd.append('mode', state.mode);
      state.shots.forEach((s, i) => fd.append('files', s.blob, `face_${i}.jpg`));
      const res = await DF.api.post('/register', fd);
      backendVectors = res.vectors; // backend returns this person's total vector count
    } else {
      await new Promise(r => setTimeout(r, 900 + state.shots.length * 120)); // simulate training
    }

    const added = state.shots.length;
    person.photoCount += added;
    person.vectors = backendVectors ?? person.vectors + added * DF.VECTORS_PER_PHOTO;
    person.photos = [...person.photos, ...state.shots.map(s => s.thumb)].slice(-6); // keep a few thumbs locally

    const list = state.mode === 'new' ? [...persons, person] : persons.map(p => p.id === person.id ? person : p);
    if (!DF.store.savePersons(list)) throw new Error('storage');
    DF.store.addLog({ type: 'admin', msg: `${DF.auth.current().username} registered ${person.name} (+${added} photos, ${person.status})` });

    DF.toast(`✅ ${person.name} registered · ${person.photoCount} photo(s) · ${person.vectors} vectors`, 'success', 4200);
    state.shots = []; renderThumbs();
    $('pname').value = ''; $('pid').value = '';
    if (state.mode === 'existing') fillExisting();
  } catch (e) {
    if (e.message !== 'storage') DF.toast('Registration failed: ' + e.message, 'danger');
  } finally {
    btn.disabled = false; btn.textContent = 'Register & Train Model →';
  }
};

window.addEventListener('beforeunload', stopCam);
renderThumbs();
