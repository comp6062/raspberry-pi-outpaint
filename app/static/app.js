const form = document.getElementById('form');
const statusEl = document.getElementById('status');
const preview = document.getElementById('preview');

function setStatus(msg) {
  statusEl.textContent = msg;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('file').files[0];
  if (!file) { alert('Please select an image'); return; }

  const fd = new FormData(form);
  fd.set('file', file);

  setStatus('Uploading and processing...');
  preview.innerHTML = '';

  try {
    const res = await fetch('/api/outpaint', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(()=>({error: res.statusText}));
      throw new Error(err.error || 'Request failed');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const img = document.createElement('img');
    img.src = url;
    preview.appendChild(img);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'outpaint.png';
    a.textContent = 'Download result';
    a.style.display = 'inline-block';
    a.style.marginTop = '10px';
    preview.appendChild(a);

    setStatus('Done.');
  } catch (err) {
    setStatus('Error: ' + err.message);
  }
});
