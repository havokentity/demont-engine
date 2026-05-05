// PathTracer console UI -- vanilla JS, no build step.
(function () {
  const out        = document.getElementById('output');
  const form       = document.getElementById('input-form');
  const input      = document.getElementById('input');
  const status     = document.getElementById('status');
  const cvarsPanel = document.getElementById('cvars');

  let ws = null;
  let nextId = 1;
  const pending = new Map();

  // History (Up/Down keys).
  const history = [];
  let histPos = -1;

  // Tab-completion state. allNames is a sorted array of cvar+command
  // identifiers; pressing Tab once completes to the longest common prefix,
  // pressing it again prints the candidate list.
  let allNames = [];
  let lastTabState = null;  // {prefix, matches, shownList} for Tab cycling

  // ---------- helpers --------------------------------------------------------
  function escape(s) {
    return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  }
  function ts() {
    const d = new Date();
    return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
  }
  function append(html) {
    const div = document.createElement('div');
    div.innerHTML = html;
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
  }

  function send(obj) {
    if (!ws || ws.readyState !== 1) return null;
    const id = String(nextId++);
    obj.id = id;
    ws.send(JSON.stringify(obj));
    return id;
  }
  function sendAndWait(obj) {
    return new Promise((resolve) => {
      const id = send(obj);
      if (!id) return resolve(null);
      pending.set(id, resolve);
    });
  }

  function renderResult(line, r) {
    const head = `<span class="ts">${ts()}</span><span class="you">&gt; ${escape(line)}</span>`;
    let body = '';
    if (r) {
      if (r.ok && r.output) body += `<div class="out">${escape(r.output)}</div>`;
      if (!r.ok) body += `<div class="err">error: ${escape(r.error || 'unknown')}</div>`;
    } else {
      body = `<div class="err">no response</div>`;
    }
    append(head + body);
  }

  async function exec(line) {
    if (!line.trim()) return;
    history.push(line);
    if (history.length > 200) history.shift();
    histPos = history.length;
    const r = await sendAndWait({ type: 'exec', line });
    renderResult(line, r);
    refreshCvars();
  }

  function onMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'result' && msg.id && pending.has(msg.id)) {
      const cb = pending.get(msg.id); pending.delete(msg.id); cb(msg);
      return;
    }
    if (msg.type === 'event' && msg.topic === 'log') {
      const lvl  = (msg.data && msg.data.level)   || 'info';
      const text = (msg.data && msg.data.message) || '';
      append(`<span class="ts">${ts()}</span><span class="log-${escape(lvl)}">[${escape(lvl).toUpperCase()}] ${escape(text)}</span>`);
      return;
    }
  }

  async function refreshCvars() {
    const r = await sendAndWait({ type: 'list_cvars' });
    if (!r || !r.ok || !r.cvars) return;
    cvarsPanel.innerHTML = '';
    for (const v of r.cvars) {
      const row = document.createElement('div');
      row.className = 'kv';
      row.innerHTML = `<span class="k" title="${escape(v.description||'')}">${escape(v.name)}</span><span class="v">${escape(v.value)}</span>`;
      cvarsPanel.appendChild(row);
    }
  }

  async function refreshNames() {
    const [c, k] = await Promise.all([
      sendAndWait({ type: 'list_cvars' }),
      sendAndWait({ type: 'list_commands' }),
    ]);
    const names = new Set();
    if (c && c.ok && c.cvars)    for (const v of c.cvars)    names.add(v.name);
    if (k && k.ok && k.commands) for (const v of k.commands) names.add(v.name);
    allNames = Array.from(names).sort();
  }

  // ---------- Tab completion -------------------------------------------------
  function commonPrefix(strs) {
    if (strs.length === 0) return '';
    let p = strs[0];
    for (let i = 1; i < strs.length; ++i) {
      let j = 0;
      while (j < p.length && j < strs[i].length && p[j] === strs[i][j]) ++j;
      p = p.slice(0, j);
      if (!p) break;
    }
    return p;
  }

  function handleTab() {
    // Only complete the FIRST token (the command/cvar name).  Value
    // completion (e.g. r_backend <Tab> -> software|metal|...) is a
    // future enhancement.
    const value = input.value;
    const cursor = input.selectionStart;
    if (cursor !== value.length) return;        // only complete at end
    const beforeCursor = value.slice(0, cursor);
    if (/\s/.test(beforeCursor)) return;        // already past first token

    const prefix = beforeCursor;
    const matches = allNames.filter(n => n.startsWith(prefix));

    if (matches.length === 0) return;

    if (matches.length === 1) {
      input.value = matches[0] + ' ';
      input.setSelectionRange(input.value.length, input.value.length);
      lastTabState = null;
      return;
    }

    const common = commonPrefix(matches);
    if (common.length > prefix.length) {
      input.value = common;
      input.setSelectionRange(common.length, common.length);
      lastTabState = { prefix: common, matches, shownList: false };
      return;
    }

    // Same prefix as before: show the candidate list once.
    if (lastTabState && lastTabState.prefix === prefix && !lastTabState.shownList) {
      append(`<span class="ts">${ts()}</span><span class="out">${escape(matches.join('  '))}</span>`);
      lastTabState.shownList = true;
    } else if (!lastTabState || lastTabState.prefix !== prefix) {
      lastTabState = { prefix, matches, shownList: false };
    }
  }

  // ---------- Connect / wiring ----------------------------------------------
  function connect() {
    const url = `ws://${location.host}/ws`;
    status.textContent = 'connecting…';
    status.className = 'status';
    ws = new WebSocket(url);
    ws.onopen = async () => {
      status.textContent = `connected · ${location.host}`;
      status.className = 'status ok';
      send({ type: 'subscribe', topics: ['log'] });
      append(`<span class="ts">${ts()}</span><span class="out">connected. type "list_commands", "sys_info", or hit Tab to complete.</span>`);
      await refreshNames();
      refreshCvars();
    };
    ws.onmessage = onMessage;
    ws.onclose = () => {
      status.textContent = 'disconnected';
      status.className = 'status bad';
      setTimeout(connect, 1500);
    };
    ws.onerror = () => {
      status.textContent = 'error';
      status.className = 'status bad';
    };
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const line = input.value;
    input.value = '';
    lastTabState = null;
    exec(line);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      handleTab();
      return;
    }
    if (e.key !== 'Tab') lastTabState = null;
    if (e.key === 'ArrowUp') {
      if (histPos > 0) { histPos--; input.value = history[histPos] || ''; }
      e.preventDefault();
    } else if (e.key === 'ArrowDown') {
      if (histPos < history.length) {
        histPos++;
        input.value = histPos === history.length ? '' : history[histPos];
      }
      e.preventDefault();
    }
  });

  // Refresh names occasionally so newly-registered cvars show up.
  setInterval(refreshNames, 10_000);

  connect();
})();
