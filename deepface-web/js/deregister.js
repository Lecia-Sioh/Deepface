DF.auth.guard('user');
DF.renderNav('deregister');

const $ = id => document.getElementById(id);
let selected = null;

function render() {
  const persons = DF.store.persons();
  const q = $('search').value.trim().toLowerCase();
  const shown = persons.filter(p => !q || p.name.toLowerCase().includes(q) || (p.idNumber || '').toLowerCase().includes(q));
  if (!persons.some(p => p.id === selected)) selected = persons[0]?.id ?? null;

  $('total').textContent = persons.length;
  $('rows').innerHTML = shown.length ? shown.map(p => `
    <tr class="${p.id === selected ? 'sel' : ''}" data-id="${p.id}" style="cursor:pointer">
      <td><div class="person-cell">${DF.avatar(p)}${DF.esc(p.name)}</div></td>
      <td class="mono">${DF.esc(p.idNumber) || '<span class="muted">—</span>'}</td>
      <td>${DF.statusTag(p.status)}</td>
      <td class="mono">${p.photoCount}</td>
      <td class="mono">${p.vectors}</td>
      <td><button class="btn btn-ghost btn-sm" data-pick="${p.id}">Select</button></td>
    </tr>`).join('')
    : `<tr><td colspan="6" class="empty">${persons.length ? 'No matches' : 'No one registered yet — <a href="register.html" style="color:var(--accent)">register someone →</a>'}</td></tr>`;

  $('sel').innerHTML = persons.length
    ? persons.map(p => `<option value="${p.id}" ${p.id === selected ? 'selected' : ''}>${DF.esc(p.name)}</option>`).join('')
    : '<option>—</option>';
  $('removeBtn').disabled = !persons.length;

  const p = persons.find(x => x.id === selected);
  $('preview').innerHTML = p ? `
    <div class="profile ${p.status === 'Blacklisted' ? 'bl' : ''}">
      ${DF.avatar(p)}
      <div>
        <div class="nm">${DF.esc(p.name)}</div>
        <div class="meta">🪪 ${DF.esc(p.idNumber) || '—'}</div>
        <div class="meta">${p.photoCount} photo(s) · ${p.vectors} vectors</div>
      </div>
    </div>
    ${p.photos?.length ? `<div class="thumbs">${p.photos.map(t => `<div class="t"><img src="${t}"></div>`).join('')}</div>` : ''}` : '';
}

$('rows').addEventListener('click', e => {
  const tr = e.target.closest('tr[data-id]'); if (!tr) return;
  selected = tr.dataset.id; render();
});
$('sel').onchange = e => { selected = e.target.value; render(); };
$('search').oninput = render;

$('removeBtn').onclick = async () => {
  const p = DF.store.persons().find(x => x.id === selected);
  if (!p) return;
  const ok = await DF.confirmBox({
    title: `Remove ${p.name}?`,
    body: `This deletes <b>${p.photoCount} photo(s)</b> and <b>${p.vectors} embedding vectors</b> for <b>${DF.esc(p.name)}</b>. They will no longer be recognised.`,
    confirmText: 'Yes, remove', danger: true,
  });
  if (!ok) return;
  try {
    if (DF.api.live()) await DF.api.del('/persons/' + encodeURIComponent(p.name));
    DF.store.savePersons(DF.store.persons().filter(x => x.id !== p.id));
    DF.store.addLog({ type: 'admin', msg: `${DF.auth.current().username} de-registered ${p.name}` });
    DF.toast(`🗑️ ${p.name} removed from the watchlist`, 'success');
    selected = null; render();
  } catch (e) { DF.toast('Remove failed: ' + e.message, 'danger'); }
};

render();
