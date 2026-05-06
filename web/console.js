// DeMonT PathTracer console UI -- vanilla JS, no build step.
(function () {
  const out        = document.getElementById('output');
  const form       = document.getElementById('input-form');
  const input      = document.getElementById('input');
  const status     = document.getElementById('status');
  const cvarsPanel = document.getElementById('cvars');
  const sideSearch = document.getElementById('side-search');
  const sidePanel  = document.getElementById('side-panel');
  const themeSelect = document.getElementById('theme-select');

  // Cached side-panel data so the search input can re-render without
  // refetching from the engine. Refilled by refreshCvars().
  let sidePanelData = null;

  // Tab + mode + pinned state. All persisted in localStorage.
  //   mode    : "modern" (tabs) or "classic" (single scrollable list)
  //   activeTab : one of pinned | cvars | commands | stats (modern only)
  //   pinned  : Set<string> of cvar/command names
  let activeTab = localStorage.getItem('demont.activeTab') || 'pinned';
  const pinned  = new Set(JSON.parse(localStorage.getItem('demont.pinned') || '[]'));
  const savePinned = () =>
    localStorage.setItem('demont.pinned', JSON.stringify(Array.from(pinned)));
  // Latest frame_stats payload (broadcast at ~10 Hz by the engine).
  let lastStats = null;

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
    if (msg.type === 'event' && msg.topic === 'frame_stats') {
      lastStats = msg.data || null;
      // Only re-render the panel if the user is actually looking at
      // the stats tab (otherwise we'd thrash the DOM at 10 Hz).
      const mode = sidePanel?.getAttribute('data-mode') || 'modern';
      if (mode === 'modern' && activeTab === 'stats') renderSidePanel();
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

  // CVar flag bits (must match src/console/Console.h CVarFlag enum).
  const CVAR_ARCHIVE  = 1 << 0;
  const CVAR_READONLY = 1 << 1;
  const CVAR_CHEAT    = 1 << 2;

  // Build an interactive row for a cvar. Detects the cvar's "shape"
  // from its allowed_values + flags and chooses the right widget:
  //   - read-only   -> static value text
  //   - allowed = ["0","1"]              -> toggle switch
  //   - allowed_values.length >= 2       -> <select> dropdown
  //   - no allowed_values, value is num  -> number input
  //   - everything else                  -> text input
  // Setting the widget value writes the cvar via the WS exec channel.
  // Build a small pin button for a cvar/command name. Toggles membership
  // in the pinned set; the panel re-renders to reflect the change.
  function makePinButton(name) {
    const btn = document.createElement('button');
    btn.className = 'pin-btn' + (pinned.has(name) ? ' pinned' : '');
    btn.title = pinned.has(name) ? 'Unpin from quick access' : 'Pin for quick access';
    btn.textContent = pinned.has(name) ? '★' : '☆';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (pinned.has(name)) pinned.delete(name);
      else                  pinned.add(name);
      savePinned();
      renderSidePanel();
    });
    return btn;
  }

  function renderCvarRow(v, fillInput) {
    const row = document.createElement('div');
    row.className = 'kv';
    if (v.description) row.title = v.description;

    row.appendChild(makePinButton(v.name));

    // Name (clickable to prefill the main input -- preserves the prior
    // "tap to inspect" UX even now that there's an inline editor).
    const k = document.createElement('span');
    k.className = 'k';
    k.textContent = v.name;
    k.addEventListener('click', () => fillInput(v.name));
    row.appendChild(k);

    const setCvar = (newVal) => {
      send({ type: 'exec', line: `${v.name} ${newVal}` });
      v.value = String(newVal);   // optimistic; will be re-synced on next refresh
    };

    const flags = (v.flags || 0) >>> 0;
    const isReadOnly = (flags & CVAR_READONLY) !== 0;
    const allowed = v.allowed_values || [];
    const isBoolean =
      allowed.length === 2 &&
      ((allowed[0] === '0' && allowed[1] === '1') ||
       (allowed[0] === '1' && allowed[1] === '0'));

    let widget;
    if (isReadOnly) {
      widget = document.createElement('span');
      widget.className = 'v v-readonly';
      widget.textContent = v.value;
    } else if (isBoolean) {
      widget = document.createElement('div');
      widget.className = 'v toggle' + (v.value === '1' ? ' on' : '');
      widget.setAttribute('role', 'switch');
      widget.setAttribute('aria-checked', v.value === '1' ? 'true' : 'false');
      widget.addEventListener('click', (e) => {
        e.stopPropagation();
        const next = (v.value === '1') ? '0' : '1';
        widget.classList.toggle('on', next === '1');
        widget.setAttribute('aria-checked', next === '1' ? 'true' : 'false');
        setCvar(next);
      });
    } else if (allowed.length >= 2) {
      widget = document.createElement('select');
      widget.className = 'v v-select';
      for (const opt of allowed) {
        const o = document.createElement('option');
        o.value = opt; o.textContent = opt;
        if (opt === v.value) o.selected = true;
        widget.appendChild(o);
      }
      widget.addEventListener('change', (e) => {
        e.stopPropagation();
        setCvar(widget.value);
      });
      widget.addEventListener('click', (e) => e.stopPropagation());
    } else {
      // Free-form: text input that commits on Enter or blur.
      widget = document.createElement('input');
      widget.type  = 'text';
      widget.className = 'v v-input';
      widget.value = v.value;
      widget.spellcheck = false;
      widget.autocomplete = 'off';
      const commit = () => {
        if (widget.value !== v.value) setCvar(widget.value);
      };
      widget.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { commit(); widget.blur(); }
        if (e.key === 'Escape') { widget.value = v.value; widget.blur(); }
      });
      widget.addEventListener('blur',  commit);
      widget.addEventListener('click', (e) => e.stopPropagation());
    }
    row.appendChild(widget);
    return row;
  }

  function renderCommandRow(cmd, fillInput) {
    const row = document.createElement('div');
    row.className = 'kv kv-cmd';
    row.title = cmd.description || '';
    row.appendChild(makePinButton(cmd.name));
    const k = document.createElement('span');
    k.className = 'k';
    k.textContent = cmd.name;
    k.addEventListener('click', () => fillInput(cmd.name));
    row.appendChild(k);
    const v = document.createElement('span');
    v.className = 'v';
    v.textContent = 'cmd';
    row.appendChild(v);
    return row;
  }

  // Render a flat (no grouping) list of pinned cvars + commands.
  function renderPinned(fillInput) {
    if (pinned.size === 0) {
      const empty = document.createElement('div');
      empty.className = 'grp-head';
      empty.textContent = 'pin items via the ☆ button';
      cvarsPanel.appendChild(empty);
      return;
    }
    if (!sidePanelData) return;
    const allCvars = [];
    const allCmds  = [];
    for (const g of sidePanelData.names) {
      const items = sidePanelData.groups.get(g);
      for (const v of items.cvars)    if (pinned.has(v.name))    allCvars.push(v);
      for (const c of items.commands) if (pinned.has(c.name))    allCmds.push(c);
    }
    const filter = (sideSearch && sideSearch.value || '').trim().toLowerCase();
    const ok = (n) => !filter || n.toLowerCase().includes(filter);
    for (const v of allCvars) if (ok(v.name)) cvarsPanel.appendChild(renderCvarRow(v, fillInput));
    for (const c of allCmds)  if (ok(c.name)) cvarsPanel.appendChild(renderCommandRow(c, fillInput));
  }

  // Render the live engine telemetry tab.
  function renderStats() {
    const wrap = document.createElement('div');
    wrap.className = 'stats';

    const block = (label, value, sub) => {
      const b = document.createElement('div'); b.className = 'stat-block';
      const l = document.createElement('div'); l.className = 'stat-label'; l.textContent = label;
      const v = document.createElement('div'); v.className = 'stat-value'; v.textContent = value;
      b.appendChild(l); b.appendChild(v);
      if (sub) {
        const s = document.createElement('div'); s.className = 'stat-sub'; s.textContent = sub;
        b.appendChild(s);
      }
      return b;
    };

    if (!lastStats) {
      wrap.appendChild(block('FPS', '—', 'waiting for frame_stats…'));
    } else {
      const row = document.createElement('div'); row.className = 'stat-row';
      row.appendChild(block('FPS',      lastStats.fps?.toFixed?.(1) ?? '—'));
      row.appendChild(block('FRAME MS', lastStats.frame_ms?.toFixed?.(2) ?? '—'));
      wrap.appendChild(row);
      wrap.appendChild(block('BACKEND',    lastStats.backend ?? '—',
        `trace ${lastStats.trace_ms?.toFixed?.(2) ?? '—'} ms`));
      const r = lastStats.resolution || [];
      wrap.appendChild(block('RESOLUTION', r.length === 2 ? `${r[0]} × ${r[1]}` : '—'));
    }
    cvarsPanel.appendChild(wrap);
  }

  // Re-render the side panel using `sidePanelData`, the current search
  // filter, the active tab, and the current mode (modern vs classic).
  // - classic: one big scrollable list grouped by prefix (no tabs).
  // - modern + pinned   : flat list of pinned items.
  // - modern + cvars    : grouped cvars only (no commands).
  // - modern + commands : grouped commands only (no cvars).
  // - modern + stats    : live frame_stats blocks.
  function renderSidePanel() {
    if (!sidePanelData) return;
    const mode = sidePanel?.getAttribute('data-mode') || 'modern';
    const filter = (sideSearch && sideSearch.value || '').trim().toLowerCase();
    const matches = (n) => !filter || n.toLowerCase().includes(filter);

    cvarsPanel.innerHTML = '';
    const fillInput = (name) => {
      input.value = name + ' ';
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    };

    if (mode === 'modern' && activeTab === 'stats') {
      renderStats();
      return;
    }
    if (mode === 'modern' && activeTab === 'pinned') {
      renderPinned(fillInput);
      return;
    }

    // For modern cvars/commands tabs we suppress the other type.
    const showCvars    = (mode === 'classic') || activeTab === 'cvars';
    const showCommands = (mode === 'classic') || activeTab === 'commands';

    for (const g of sidePanelData.names) {
      const items = sidePanelData.groups.get(g);
      const cvarsHere = showCvars    ? items.cvars.filter(v => matches(v.name))    : [];
      const cmdsHere  = showCommands ? items.commands.filter(v => matches(v.name)) : [];
      if (cvarsHere.length === 0 && cmdsHere.length === 0) continue;

      const head = document.createElement('div');
      head.className = 'grp-head';
      head.textContent = g;
      cvarsPanel.appendChild(head);

      for (const v of cvarsHere)   cvarsPanel.appendChild(renderCvarRow(v, fillInput));
      for (const cmd of cmdsHere)  cvarsPanel.appendChild(renderCommandRow(cmd, fillInput));
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
      send({ type: 'subscribe', topics: ['log', 'theme_change', 'frame_stats'] });
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
      block += `<div class="banner-tag">DeMonT Engine · v0.1.0 · non-rasterized · path-traced</div>`;
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

  // Density toggle (BMC-inspired): two settings, persisted via localStorage.
  // Compact mode crunches scope rows so more cvars fit on screen at once;
  // comfy is the default with normal padding.
  const densityBtn = document.getElementById('density-toggle');
  const applyDensity = (d) => {
    if (sidePanel) sidePanel.setAttribute('data-density', d);
    localStorage.setItem('demont.density', d);
  };
  applyDensity(localStorage.getItem('demont.density') || 'comfy');
  if (densityBtn) {
    densityBtn.addEventListener('click', () => {
      const cur = sidePanel?.getAttribute('data-density') || 'comfy';
      applyDensity(cur === 'compact' ? 'comfy' : 'compact');
    });
  }

  // Mode toggle: classic (single scrollable list) vs modern (tabbed).
  const modeBtn = document.getElementById('mode-toggle');
  const applyMode = (m) => {
    if (sidePanel) sidePanel.setAttribute('data-mode', m);
    localStorage.setItem('demont.mode', m);
    renderSidePanel();
  };
  applyMode(localStorage.getItem('demont.mode') || 'modern');
  if (modeBtn) {
    modeBtn.addEventListener('click', () => {
      const cur = sidePanel?.getAttribute('data-mode') || 'modern';
      applyMode(cur === 'modern' ? 'classic' : 'modern');
    });
  }

  // Tab clicks (modern mode). Active state is reflected on the buttons
  // and on the activeTab variable; renderSidePanel branches on it.
  const sideTabs = document.querySelectorAll('.side-tab');
  const setActiveTab = (name) => {
    activeTab = name;
    localStorage.setItem('demont.activeTab', name);
    sideTabs.forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    renderSidePanel();
  };
  sideTabs.forEach(t => t.addEventListener('click', () => setActiveTab(t.dataset.tab)));
  setActiveTab(activeTab);   // sync visual state on first load

  // Refresh names occasionally so newly-registered cvars show up.
  setInterval(refreshNames, 10_000);

  connect();
})();
