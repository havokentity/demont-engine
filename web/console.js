// DeMonT PathTracer console UI -- vanilla JS, no build step.
(function () {
  const out        = document.getElementById('output');
  const form       = document.getElementById('input-form');
  const input      = document.getElementById('input');
  const status     = document.getElementById('status');
  const cvarsPanel = document.getElementById('cvars');
  const sideSearch = document.getElementById('side-search');
  const themeSelect = document.getElementById('theme-select');

  // Cached side-panel data so the search input can re-render without
  // refetching from the engine. Refilled by refreshCvars().
  let sidePanelData = null;

  // Theme: source of truth is the r_theme cvar on the engine.
  // localStorage is a quick UI cache for the moment before WS connects.
  function applyTheme(t) {
    if (!t) return;
    document.documentElement.setAttribute('data-theme', t);
    if (themeSelect && themeSelect.value !== t) themeSelect.value = t;
    localStorage.setItem('demont.theme', t);
  }
  applyTheme(localStorage.getItem('demont.theme') || 'hardcore');
  if (themeSelect) {
    themeSelect.addEventListener('change', () => {
      const t = themeSelect.value;
      applyTheme(t);
      // Push to engine; cvar's on_change will then BroadcastEvent
      // back so other connected clients also flip.
      send({ type: 'exec', line: 'r_theme ' + t });
    });
  }

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
  let cvarMeta = {};   // name -> { allowed_values: [...] }
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
    if (msg.type === 'event' && msg.topic === 'theme_change') {
      applyTheme(msg.data && msg.data.name);
      return;
    }
  }

  // Group both cvars and commands by their prefix (everything before
  // the first underscore). Items with no underscore go into "general".
  // Within each group, cvars list first, then commands. Click any row
  // to prefill the input with `<name> ` and focus it.
  function groupOf(name) {
    const u = name.indexOf('_');
    return u === -1 ? 'general' : name.slice(0, u);
  }

  async function refreshCvars() {
    const [c, k] = await Promise.all([
      sendAndWait({ type: 'list_cvars' }),
      sendAndWait({ type: 'list_commands' }),
    ]);
    if (!c || !c.ok) return;

    const groups = new Map();
    const ensure = (g) => {
      if (!groups.has(g)) groups.set(g, { cvars: [], commands: [] });
      return groups.get(g);
    };
    for (const v of (c.cvars || []))     ensure(groupOf(v.name)).cvars.push(v);
    for (const v of (k && k.commands) || []) ensure(groupOf(v.name)).commands.push(v);

    // Stable order: alpha groups, with "general" pushed to the end.
    const names = Array.from(groups.keys()).sort();
    const gi = names.indexOf('general');
    if (gi !== -1) { names.splice(gi, 1); names.push('general'); }

    sidePanelData = { names, groups };
    renderSidePanel();
  }

  // Re-render the side panel using `sidePanelData` and the current
  // search filter. Filter is a case-insensitive substring match against
  // the cvar / command name; groups whose items all get filtered out
  // are hidden. Cheap enough that we re-render on every keystroke.
  function renderSidePanel() {
    if (!sidePanelData) return;
    const filter = (sideSearch && sideSearch.value || '').trim().toLowerCase();
    const matches = (n) => !filter || n.toLowerCase().includes(filter);

    cvarsPanel.innerHTML = '';
    const fillInput = (name) => {
      input.value = name + ' ';
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    };

    for (const g of sidePanelData.names) {
      const items = sidePanelData.groups.get(g);
      const cvarsHere = items.cvars.filter(v => matches(v.name));
      const cmdsHere  = items.commands.filter(v => matches(v.name));
      if (cvarsHere.length === 0 && cmdsHere.length === 0) continue;

      const head = document.createElement('div');
      head.className = 'grp-head';
      head.textContent = g;
      cvarsPanel.appendChild(head);

      for (const v of cvarsHere) {
        const row = document.createElement('div');
        row.className = 'kv';
        row.title = v.description || '';
        row.innerHTML = `<span class="k">${escape(v.name)}</span><span class="v">${escape(v.value)}</span>`;
        row.addEventListener('click', () => fillInput(v.name));
        cvarsPanel.appendChild(row);
      }
      for (const cmd of cmdsHere) {
        const row = document.createElement('div');
        row.className = 'kv kv-cmd';
        row.title = cmd.description || '';
        row.innerHTML = `<span class="k">${escape(cmd.name)}</span><span class="v">cmd</span>`;
        row.addEventListener('click', () => fillInput(cmd.name));
        cvarsPanel.appendChild(row);
      }
    }
  }

  async function refreshNames() {
    const [c, k] = await Promise.all([
      sendAndWait({ type: 'list_cvars' }),
      sendAndWait({ type: 'list_commands' }),
    ]);
    const names = new Set();
    cvarMeta = {};
    if (c && c.ok && c.cvars) {
      for (const v of c.cvars) {
        names.add(v.name);
        cvarMeta[v.name] = { allowed_values: v.allowed_values || [] };
      }
    }
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
    const value = input.value;
    const cursor = input.selectionStart;
    if (cursor !== value.length) return;
    const beforeCursor = value.slice(0, cursor);
    const lastSpace = beforeCursor.lastIndexOf(' ');

    let prefix, candidates;
    if (lastSpace === -1) {
      // Token 0: cvar / command name.
      prefix = beforeCursor;
      candidates = allNames;
    } else {
      // Value position: special-case `toggle` -- second token is a cvar
      // name, only those with allowed_values are useful candidates.
      const firstTok = beforeCursor.split(/\s+/)[0];
      if (firstTok === 'toggle') {
        candidates = Object.keys(cvarMeta).filter(
          n => cvarMeta[n].allowed_values && cvarMeta[n].allowed_values.length > 0
        ).sort();
        prefix = beforeCursor.slice(lastSpace + 1);
      } else {
        // Default: complete from the named cvar's allowed_values.
        const meta = cvarMeta[firstTok];
        if (!meta || !meta.allowed_values || meta.allowed_values.length === 0) return;
        prefix = beforeCursor.slice(lastSpace + 1);
        candidates = meta.allowed_values;
      }
    }

    const matches = candidates.filter(n => n.startsWith(prefix));
    if (matches.length === 0) return;

    if (matches.length === 1) {
      const replaced = beforeCursor.slice(0, lastSpace + 1) + matches[0]
                     + (lastSpace === -1 ? ' ' : '');
      input.value = replaced;
      input.setSelectionRange(input.value.length, input.value.length);
      lastTabState = null;
      return;
    }

    const common = commonPrefix(matches);
    if (common.length > prefix.length) {
      const replaced = beforeCursor.slice(0, lastSpace + 1) + common;
      input.value = replaced;
      input.setSelectionRange(replaced.length, replaced.length);
      lastTabState = { prefix: common, matches, shownList: false };
      return;
    }

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
      send({ type: 'subscribe', topics: ['log', 'theme_change'] });
      // Pull the engine's current r_theme and apply.
      sendAndWait({ type: 'get_cvar', name: 'r_theme' }).then((r) => {
        if (r && r.ok && r.cvar && r.cvar.value) applyTheme(r.cvar.value);
      });
      // Hex banner with chevroned inner box and bouncing-ray scaffold.
      // Cyan frame, hot-pink ray + hit dots, bright letters.
      const banner = [
        ['frame',   '        ░▒▓██████████▓▒░'],
        ['frame',   '     ░▒▓██╔═══════════╗▓▒░'],
        ['letters', '   ░▓██╔═╝   D · M · T   ╚═╗██▓░'],
        ['ray',     '  ▒█░  ╔╝  ╲     ◉     ╱  ╚╗  ░█▒'],
        ['ray',     '  ▓█░ ║    ╲   ◉│◉   ╱    ║ ░█▓'],
        ['ray',     '  █░  ║     ╲  ─•─  ╱     ║  ░█'],
        ['ray',     '  ▓█░ ║      ╳  •  ╳      ║ ░█▓'],
        ['ray',     '  █░  ║     ╱  ─•─  ╲     ║  ░█'],
        ['ray',     '  ▓█░ ║    ╱   ◉│◉   ╲    ║ ░█▓'],
        ['ray',     '  ▒█░  ╚╗  ╱     ◉     ╲  ╔╝  ░█▒'],
        ['letters', '   ░▓██╚═╗   P · A · T   ╔═╝██▓░'],
        ['frame',   '     ░▒▓██╚═══════════╝▓▒░'],
        ['frame',   '        ░▒▓██████████▓▒░'],
      ];
      let block = '';
      for (const [kind, ln] of banner) {
        block += `<div class="banner-${kind}">${escape(ln)}</div>`;
      }
      block += `<div class="banner-tag">DeMonT PathTracer · v0.1.0 · De Monte Carlo-esque Tracer</div>`;
      block += `<div class="out"><span class="ts">${ts()}</span>console attached. type "list_commands", "sys_info", or hit Tab to complete.</div>`;
      append(block);
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

  // Paste-to-multiline: if the clipboard text spans multiple lines, treat
  // each line as its own command and run them in order. Whatever follows
  // the last newline (the "trailing partial") stays in the input so the
  // user can keep editing before hitting Enter on it.
  input.addEventListener('paste', (e) => {
    const text = (e.clipboardData || window.clipboardData)?.getData('text') || '';
    if (text.indexOf('\n') === -1) return;          // single-line paste: default behaviour
    e.preventDefault();

    const start = input.selectionStart ?? input.value.length;
    const end   = input.selectionEnd   ?? input.value.length;
    const combined = input.value.slice(0, start) + text + input.value.slice(end);
    const parts = combined.split(/\r?\n/);
    const trailing = parts.pop() ?? '';

    lastTabState = null;
    (async () => {
      for (const line of parts) {
        await exec(line);                            // exec() no-ops on blank lines
      }
      input.value = trailing;
      input.setSelectionRange(trailing.length, trailing.length);
    })();
  });

  if (sideSearch) {
    sideSearch.addEventListener('input', renderSidePanel);
    // Esc inside the search clears the filter and refocuses the main input.
    sideSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        sideSearch.value = '';
        renderSidePanel();
        input.focus();
      }
    });
  }

  // Hard reload button: rebuilds embed an updated index.html / console.js /
  // console.css into the binary, but the browser will happily serve the
  // cached copy. A cache-busting query string forces a fresh fetch.
  const reloadBtn = document.getElementById('reload-btn');
  if (reloadBtn) {
    reloadBtn.addEventListener('click', () => {
      location.replace(location.pathname + '?_=' + Date.now());
    });
  }

  // Refresh names occasionally so newly-registered cvars show up.
  setInterval(refreshNames, 10_000);

  connect();
})();
