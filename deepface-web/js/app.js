/* ============================================================
   DeepFace Watchlist — shared core
   Storage, auth, nav, API bridge, toasts, modals
   ============================================================ */

const DF = (() => {
  const KEYS = {
    persons: 'df_persons',
    accounts: 'df_accounts',
    settings: 'df_settings',
    logs: 'df_logs',
    session: 'df_session',
  };

  const DEFAULT_SETTINGS = {
    model: 'ArcFace',
    detector: 'yunet',
    metric: 'euclidean_l2',
    threshold: 1.13,         // ArcFace + euclidean_l2 (DeepFace's tuned value)
    frameSkip: 3,
    augmentations: { flip: true, brightness: true, rotation: true, blur: false },
    apiBase: '',            // e.g. http://localhost:8000  (empty = demo mode)
  };

  const DEFAULT_ACCOUNTS = [
    { username: 'admin', password: 'admin123', role: 'admin', created: Date.now() },
    { username: 'user',  password: 'user123',  role: 'user',  created: Date.now() },
  ];

  const DEFAULT_PERSONS = []; // start empty: only people you actually register appear

  const VECTORS_PER_PHOTO = 14; // 1 original + 13 augmentations (matches the Streamlit app)

  /* ---------- helpers ---------- */
  function uid() { return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4); }
  function read(key, fallback) {
    try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
    catch { return fallback; }
  }
  function write(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); return true; }
    catch (e) { toast('Storage full — try fewer / smaller photos', 'danger'); return false; }
  }
  function esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function fmtTime(sec) {
    const m = Math.floor(sec / 60), s = (sec % 60).toFixed(2).padStart(5, '0');
    return `${String(m).padStart(2, '0')}:${s}`;
  }
  function fmtDate(ts) {
    return new Date(ts).toLocaleString('en-SG', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  /* ---------- seed ---------- */
  if (!read(KEYS.accounts)) write(KEYS.accounts, DEFAULT_ACCOUNTS);
  if (!read(KEYS.persons))  write(KEYS.persons, DEFAULT_PERSONS);
  if (!read(KEYS.settings)) write(KEYS.settings, DEFAULT_SETTINGS);
  if (!read(KEYS.logs))     write(KEYS.logs, []);

  // One-time cleanup: remove the old fake example people (never trained, no photos)
  if (!localStorage.getItem('df_seed_removed')) {
    const FAKE = ['leeaa', 'brad', 'sh'];
    write(KEYS.persons, read(KEYS.persons, []).filter(p => !(FAKE.includes(p.name) && !p.photos?.length)));
    try { localStorage.setItem('df_seed_removed', '1'); } catch {}
  }

  /* ---------- data stores ---------- */
  const store = {
    persons:  () => read(KEYS.persons, []),
    savePersons: p => write(KEYS.persons, p),
    accounts: () => read(KEYS.accounts, []),
    saveAccounts: a => write(KEYS.accounts, a),
    settings: () => ({ ...DEFAULT_SETTINGS, ...read(KEYS.settings, {}) }),
    saveSettings: s => write(KEYS.settings, s),
    logs: () => read(KEYS.logs, []),
    addLog(entry) {
      const logs = read(KEYS.logs, []);
      logs.unshift({ ts: Date.now(), ...entry });
      write(KEYS.logs, logs.slice(0, 300));
    },
    clearLogs: () => write(KEYS.logs, []),
  };

  /* ---------- auth ---------- */
  const auth = {
    current: () => { try { return JSON.parse(sessionStorage.getItem(KEYS.session)); } catch { return null; } },
    login(username, password, role) {
      const acc = store.accounts().find(a => a.username.toLowerCase() === username.trim().toLowerCase() && a.password === password);
      if (!acc) return { ok: false, msg: 'Wrong username or password' };
      if (role === 'admin' && acc.role !== 'admin') return { ok: false, msg: 'This account has no admin access' };
      const session = { username: acc.username, role: acc.role, at: Date.now() };
      sessionStorage.setItem(KEYS.session, JSON.stringify(session));
      store.addLog({ type: 'auth', msg: `${acc.username} logged in (${acc.role})` });
      return { ok: true, session };
    },
    logout() {
      const s = auth.current();
      if (s) store.addLog({ type: 'auth', msg: `${s.username} logged out` });
      sessionStorage.removeItem(KEYS.session);
      location.href = 'index.html';
    },
    /** Redirect to login if not allowed. level: 'user' | 'admin' */
    guard(level) {
      const s = auth.current();
      const ok = s && (level === 'user' || s.role === 'admin');
      if (!ok) {
        const next = encodeURIComponent(location.pathname.split('/').pop());
        location.replace(`index.html?next=${next}${level === 'admin' ? '&need=admin' : ''}`);
        throw new Error('redirecting');
      }
      return s;
    },
  };

  /* ---------- API bridge (optional real backend) ----------
     If settings.apiBase is set, pages call your Python backend.
     Expected endpoints (see README.md):
       POST   /register          multipart: name, id_number, status, mode, files[]
       DELETE /persons/{name}
       POST   /recognize         multipart: frame (jpeg)  -> { faces: [{box:[x,y,w,h], name, status, similarity}] }
       POST   /analyse-video     multipart: video, every_n -> { report: [...], frames, duration, annotated_url }
       POST   /retrain
  ---------------------------------------------------------- */
  const api = {
    base: () => (store.settings().apiBase || window.DF_API_BASE || '').replace(/\/$/, ''),
    live: () => !!api.base(),
    async post(path, formData) {
      const r = await fetch(api.base() + path, { method: 'POST', body: formData });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    },
    async del(path) {
      const r = await fetch(api.base() + path, { method: 'DELETE' });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json().catch(() => ({}));
    },
  };

  /* ---------- LIVE mode: the backend's faces/ folder is the source of truth ----------
     Pull the registered people (and their photo URLs) from the backend, and
     reload the page once if the local copy was out of date. */
  if (api.live()) {
    fetch(api.base() + '/persons').then(r => r.ok ? r.json() : null).then(remote => {
      if (!Array.isArray(remote)) return;
      const local = store.persons();
      const merged = remote.map(r => {
        const old = local.find(p => p.name === r.name) || {};
        return {
          id: old.id || uid(), name: r.name, idNumber: r.id_number || '', status: r.status,
          photos: r.photos.slice(-6).map(u => api.base() + u),
          photoCount: r.photos.length, vectors: r.vectors, created: old.created || Date.now(),
        };
      });
      if (JSON.stringify(merged) !== JSON.stringify(local)) { store.savePersons(merged); location.reload(); }
    }).catch(() => toast('Cannot reach backend at ' + api.base(), 'danger', 5000));
  }

  /* ---------- UI: nav ---------- */
  function renderNav(active) {
    const s = auth.current();
    const links = [
      { href: 'inference.html',  key: 'inference',  label: 'Inference',   icon: '📹', show: true },
      { href: 'register.html',   key: 'register',   label: 'Register',    icon: '👤', show: !!s },
      { href: 'deregister.html', key: 'deregister', label: 'De-register', icon: '🗑️', show: !!s },
      { href: 'admin.html',      key: 'admin',      label: 'Admin',       icon: '⚙️', show: s?.role === 'admin' },
    ];
    const el = document.getElementById('nav');
    if (!el) return;
    el.innerHTML = `
      <a class="brand" href="${s ? 'register.html' : 'index.html'}">
        <span class="brand-mark">🛡️</span>
        <span class="brand-text">DEEPFACE<b>WATCHLIST</b></span>
      </a>
      <button class="nav-burger" aria-label="Menu" onclick="document.body.classList.toggle('nav-open')">☰</button>
      <div class="nav-links">
        ${links.filter(l => l.show).map(l =>
          `<a href="${l.href}" class="${l.key === active ? 'active' : ''}"><span>${l.icon}</span>${l.label}</a>`).join('')}
      </div>
      <div class="nav-user">
        <span class="mode-pill ${api.live() ? 'live' : ''}" title="${api.live() ? 'Connected to ' + esc(api.base()) : 'No backend set — results are simulated'}">
          ${api.live() ? '● LIVE API' : '● DEMO MODE'}
        </span>
        ${s
          ? `<span class="who"><b>${esc(s.username)}</b><small>${s.role}</small></span>
             <button class="btn btn-ghost btn-sm" onclick="DF.auth.logout()">Log out</button>`
          : `<a class="btn btn-primary btn-sm" href="index.html">Log in</a>`}
      </div>`;
  }

  /* ---------- UI: toast ---------- */
  function toast(msg, kind = 'info', ms = 3200) {
    let wrap = document.getElementById('toasts');
    if (!wrap) { wrap = document.createElement('div'); wrap.id = 'toasts'; document.body.appendChild(wrap); }
    const t = document.createElement('div');
    t.className = `toast toast-${kind}`;
    t.innerHTML = esc(msg);
    wrap.appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));
    setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 300); }, ms);
  }

  /* ---------- UI: confirm modal (no browser dialogs) ---------- */
  function confirmBox({ title, body, confirmText = 'Confirm', danger = false }) {
    return new Promise(resolve => {
      const m = document.createElement('div');
      m.className = 'modal-back';
      m.innerHTML = `
        <div class="modal ${danger ? 'modal-danger' : ''}">
          <h3>${esc(title)}</h3>
          <p>${body}</p>
          <div class="modal-actions">
            <button class="btn btn-ghost" data-v="0">Cancel</button>
            <button class="btn ${danger ? 'btn-danger' : 'btn-primary'}" data-v="1">${esc(confirmText)}</button>
          </div>
        </div>`;
      m.addEventListener('click', e => {
        const v = e.target.dataset?.v;
        if (v !== undefined || e.target === m) { m.remove(); resolve(v === '1'); }
      });
      document.body.appendChild(m);
    });
  }

  /* ---------- image helpers ---------- */
  function fileToThumb(file, max = 160) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => {
        const sc = Math.min(1, max / Math.max(img.width, img.height));
        const c = document.createElement('canvas');
        c.width = img.width * sc; c.height = img.height * sc;
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        URL.revokeObjectURL(img.src);
        res(c.toDataURL('image/jpeg', 0.75));
      };
      img.onerror = rej;
      img.src = URL.createObjectURL(file);
    });
  }
  function canvasThumb(source, max = 160) {
    const w = source.videoWidth || source.width, h = source.videoHeight || source.height;
    const sc = Math.min(1, max / Math.max(w, h));
    const c = document.createElement('canvas');
    c.width = w * sc; c.height = h * sc;
    c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.75);
  }

  function avatar(p) {
    if (p.photos?.length) return `<img class="avatar" src="${p.photos[0]}" alt="">`;
    return `<span class="avatar avatar-init ${p.status === 'Blacklisted' ? 'bl' : ''}">${esc(p.name.slice(0, 2).toUpperCase())}</span>`;
  }
  function statusTag(status) {
    const cls = status === 'Blacklisted' ? 'tag-bl' : status === 'Authorised' ? 'tag-ok' : 'tag-unk';
    const dot = status === 'Blacklisted' ? '🔴' : status === 'Authorised' ? '🟢' : '⚪';
    return `<span class="tag ${cls}">${dot} ${esc(status)}</span>`;
  }

  return { store, auth, api, renderNav, toast, confirmBox, fileToThumb, canvasThumb, avatar, statusTag, esc, uid, fmtTime, fmtDate, VECTORS_PER_PHOTO };
})();
