// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
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
  //   mode      : "modern" (split + tabs) or "classic" (single scrollable list)
  //   activeTab : one of cvars | commands | stats (modern only;
  //               pinned is its own always-visible left column now,
  //               not a tab)
  //   pinned    : Set<string> of cvar/command names
  let activeTab = localStorage.getItem('demont.activeTab') || 'cvars';
  if (activeTab === 'pinned') activeTab = 'cvars';   // migrate old pref
  // First-run default pins. Stuff that's almost always handy: the
  // sky-animation toggle + rate, the time-of-day slider, and exposure.
  // Once the user changes their pin set this gets persisted and the
  // default no longer applies.
  const DEFAULT_PINS = [
    'r_sky_animate', 'r_sky_animate_rate', 'r_sky_hour',
    'r_sky_year', 'r_sky_month', 'r_sky_day',
    'r_sky_use_astronomical', 'r_sky_city',
    'r_exposure', 'r_auto_exposure', 'r_eye_model',
    'r_quality', 'r_dof', 'r_dof_aperture', 'r_dof_focal_distance',
    'r_bloom', 'r_bloom_intensity', 'r_bloom_threshold',
    'r_lens_flare', 'r_lens_flare_intensity',
    'r_volumetric', 'r_volumetric_density', 'r_volumetric_intensity',
    'r_clouds', 'r_clouds_preset', 'r_clouds_coverage', 'r_clouds_density',
    'r_clouds_seed',
  ];
  const pinnedRaw = localStorage.getItem('demont.pinned');
  const pinned = new Set(pinnedRaw === null
    ? DEFAULT_PINS
    : JSON.parse(pinnedRaw));
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
  // identifiers. First Tab on an ambiguous prefix extends to the
  // longest common prefix AND activates ghostState: the first remaining
  // match is shown after the cursor in dim colour. Subsequent Tabs
  // cycle (Shift+Tab back); Right-arrow at end / End commits;
  // Esc / typing dismisses.
  let allNames    = [];
  let cvarMeta    = {};   // name -> { allowed_values, value, default_value, ... }
  let commandMeta = {};   // name -> { default_args }
  // ghostState shape:
  //   { matches: [...], index, before, prefix, isToken0,
  //     isMeta:    bool   -- matches = [current, default] of a free-form cvar
  //     annotation: str   -- "default: X" / "current: Y" when isMeta
  //   }
  let ghostState = null;
  const ghostTyped      = document.querySelector('#input-ghost .ghost-typed');
  const ghostTail       = document.querySelector('#input-ghost .ghost-tail');
  const ghostAnnotation = document.querySelector('#input-ghost .ghost-annotation');

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
      // Only redraw the stats section -- a full renderSidePanel() also
      // rebuilds the pinned column, and at 10 Hz that yanks pinned
      // widgets out from under the user mid-drag/mid-pick.
      const mode = sidePanel?.getAttribute('data-mode') || 'modern';
      if (mode === 'modern' && activeTab === 'stats') renderStats();
      return;
    }
  }

  // Group both cvars and commands by their prefix (everything before
  // the first underscore). Items with no underscore go into "general".
  // The renderer prefix `r` is huge (sky, camera, integrator, denoiser,
  // ...), so we further split it into themed sub-groups for sanity.
  // Click any row to prefill the input with `<name> ` and focus it.
  function groupOf(name) {
    const u = name.indexOf('_');
    if (u === -1) return { top: 'general', sub: '' };
    const top = name.slice(0, u);
    if (top !== 'r') return { top, sub: '' };
    const tail = name.slice(u + 1);
    if (/^sky_/.test(tail))                                                 return { top: 'r', sub: 'sky' };
    if (/^sun_/.test(tail))                                                 return { top: 'r', sub: 'sky' };
    if (/^stars_|^show_stars$/.test(tail))                                  return { top: 'r', sub: 'stars' };
    if (/^env_/.test(tail))                                                 return { top: 'r', sub: 'env' };
    if (/^exposure|^auto_exposure$|^eye_/.test(tail))                       return { top: 'r', sub: 'camera' };
    if (/^caustics$|^refract|^quality$|^spp$|^max_bounces$|^volumetric|^clouds/.test(tail)) return { top: 'r', sub: 'integrator' };
    if (/^denoiser$|^hdr_|^bloom|^lens_flare/.test(tail))                   return { top: 'r', sub: 'post' };
    return { top: 'r', sub: 'display' };
  }

  async function refreshCvars() {
    const [c, k] = await Promise.all([
      sendAndWait({ type: 'list_cvars' }),
      sendAndWait({ type: 'list_commands' }),
    ]);
    if (!c || !c.ok) return;

    // Two-level grouping. groups: top -> { subs: Map<sub, {cvars, cmds}> }.
    // Top-level keys are the cvar prefix (sky, app, net, sys, dev, r,
    // cam, ...). Items inside the renderer prefix `r` get a second
    // level (sky / stars / camera / integrator / post / env / display)
    // so the panel doesn't dump all 30+ render cvars into one wall.
    const groups = new Map();
    const ensure = (top, sub) => {
      if (!groups.has(top)) groups.set(top, { subs: new Map() });
      const g = groups.get(top);
      if (!g.subs.has(sub)) g.subs.set(sub, { cvars: [], commands: [] });
      return g.subs.get(sub);
    };
    for (const v of (c.cvars || [])) {
      const { top, sub } = groupOf(v.name);
      ensure(top, sub).cvars.push(v);
    }
    for (const v of (k && k.commands) || []) {
      const { top, sub } = groupOf(v.name);
      ensure(top, sub).commands.push(v);
    }

    // Stable top-level order: alpha, with "general" pushed to the end.
    const names = Array.from(groups.keys()).sort();
    const gi = names.indexOf('general');
    if (gi !== -1) { names.splice(gi, 1); names.push('general'); }

    // Stable sub-order inside each top group.
    const SUB_ORDER = [
      'integrator', 'post', 'camera', 'sky', 'stars', 'env', 'display', '',
    ];
    for (const top of names) {
      const g = groups.get(top);
      const ordered = new Map();
      for (const s of SUB_ORDER) if (g.subs.has(s)) ordered.set(s, g.subs.get(s));
      // Sweep up any leftover subs we didn't list in SUB_ORDER.
      for (const [s, payload] of g.subs) if (!ordered.has(s)) ordered.set(s, payload);
      g.subs = ordered;
    }

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

  // Look up a cvar in the cached sidePanelData by name. Used by the
  // date-picker widget to pull r_sky_month / r_sky_day when rendering
  // the combined picker for r_sky_year.
  function getCvarByName(name) {
    if (!sidePanelData) return null;
    for (const g of sidePanelData.names) {
      const grp = sidePanelData.groups.get(g);
      if (!grp || !grp.subs) continue;
      for (const [, items] of grp.subs) {
        for (const v of items.cvars) {
          if (v.name === name) return v;
        }
      }
    }
    return null;
  }

  // r_sky_year becomes a calendar date picker that combines year /
  // month / day. r_sky_month and r_sky_day are folded into it (we
  // hide their separate rows in renderCvarRow). When all three are
  // 0, the picker shows today's date and the underlying cvars stay
  // 0 = "use system date".
  function makeDatePicker(v, setCvar) {
    const widget = document.createElement('input');
    widget.type = 'date';
    widget.className = 'v v-date';
    const yr = parseInt(v.value, 10) || 0;
    const moV = getCvarByName('r_sky_month');
    const dyV = getCvarByName('r_sky_day');
    const mo = (moV && parseInt(moV.value, 10)) || 0;
    const dy = (dyV && parseInt(dyV.value, 10)) || 0;
    const today = new Date();
    const fy = yr || today.getUTCFullYear();
    const fm = mo || (today.getUTCMonth() + 1);
    const fd = dy || today.getUTCDate();
    widget.value = `${String(fy).padStart(4, '0')}-`
                 + `${String(fm).padStart(2, '0')}-`
                 + `${String(fd).padStart(2, '0')}`;
    widget.addEventListener('change', (e) => {
      e.stopPropagation();
      const m = widget.value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
      if (!m) return;
      const ny = parseInt(m[1], 10);
      const nm = parseInt(m[2], 10);
      const nd = parseInt(m[3], 10);
      // Send all three; refreshCvars debounce in setCvar coalesces.
      send({ type: 'exec', line: `r_sky_year ${ny}` });
      send({ type: 'exec', line: `r_sky_month ${nm}` });
      send({ type: 'exec', line: `r_sky_day ${nd}` });
      v.value = String(ny);
      if (moV) moV.value = String(nm);
      if (dyV) dyV.value = String(nd);
      clearTimeout(makeDatePicker._t);
      makeDatePicker._t = setTimeout(refreshCvars, 60);
    });
    widget.addEventListener('click', (e) => e.stopPropagation());
    return widget;
  }

  function renderCvarRow(v, fillInput) {
    // r_sky_month / r_sky_day are folded into r_sky_year's date
    // picker; hide their separate rows.
    if (v.name === 'r_sky_month' || v.name === 'r_sky_day') {
      return null;
    }

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
      v.value = String(newVal);   // optimistic; reconciled on the refresh below
      // Refetch the full cvar list so cascade updates (e.g. r_sky_city
      // mutating r_sky_lat / r_sky_lon via on_change) get reflected in
      // every panel column. Debounced so rapid drags coalesce into one
      // round-trip.
      clearTimeout(setCvar._t);
      setCvar._t = setTimeout(refreshCvars, 60);
    };

    const flags = (v.flags || 0) >>> 0;
    const isReadOnly = (flags & CVAR_READONLY) !== 0;
    const allowed = v.allowed_values || [];
    const isBoolean =
      allowed.length === 2 &&
      ((allowed[0] === '0' && allowed[1] === '1') ||
       (allowed[0] === '1' && allowed[1] === '0'));

    let widget;
    if (v.name === 'r_sky_year') {
      widget = makeDatePicker(v, setCvar);
    } else if (isReadOnly) {
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
    } else if (typeof v.slider_min === 'number' &&
               typeof v.slider_max === 'number' &&
               v.slider_max > v.slider_min) {
      // Slider: range input + a numeric readout. Drag commits live (each
      // 'input' event sends the cvar update); typing in the readout box
      // commits on Enter/blur. The renderSidePanel debounce in setCvar
      // keeps drag-while-rendering from thrashing.
      widget = document.createElement('div');
      widget.className = 'v v-slider';
      const range = document.createElement('input');
      range.type = 'range';
      range.min  = v.slider_min;
      range.max  = v.slider_max;
      range.step = v.slider_step || 0.01;
      const cur = parseFloat(v.value);
      range.value = Number.isFinite(cur) ? cur : v.slider_min;
      const readout = document.createElement('input');
      readout.type = 'text';
      readout.className = 'v-slider-readout';
      readout.value = v.value;
      readout.spellcheck = false;
      readout.autocomplete = 'off';
      range.addEventListener('input', (e) => {
        e.stopPropagation();
        readout.value = range.value;
        setCvar(range.value);
      });
      range.addEventListener('click', (e) => e.stopPropagation());
      const commitReadout = () => {
        if (readout.value === v.value) return;
        const n = parseFloat(readout.value);
        if (!Number.isFinite(n)) { readout.value = v.value; return; }
        range.value = n;
        setCvar(readout.value);
      };
      readout.addEventListener('keydown', (e) => {
        if (e.key === 'Enter')  { commitReadout(); readout.blur(); }
        if (e.key === 'Escape') { readout.value = v.value; readout.blur(); }
      });
      readout.addEventListener('blur',  commitReadout);
      readout.addEventListener('click', (e) => e.stopPropagation());
      // Explicit Apply button: a third commit path alongside Enter
      // and blur. Some users (and some browsers / focus-stealing
      // events) drop blur/change events; the button gives an
      // unambiguous "send this value now" trigger.
      const apply = document.createElement('button');
      apply.type = 'button';
      apply.className = 'v-apply';
      apply.textContent = 'Apply';
      apply.title = 'Send the typed value to the engine now';
      apply.addEventListener('click', (e) => {
        e.stopPropagation();
        commitReadout();
      });
      widget.appendChild(range);
      widget.appendChild(readout);
      widget.appendChild(apply);
    } else {
      // Free-form: text input that commits on Enter, blur, or
      // explicit Apply-button click. The button matters when blur
      // doesn't fire (focus-stealing modal, browser quirk, or the
      // user simply wanting to confirm without losing focus).
      widget = document.createElement('div');
      widget.className = 'v v-input-wrap';
      const inp = document.createElement('input');
      inp.type  = 'text';
      inp.className = 'v-input';
      inp.value = v.value;
      inp.spellcheck = false;
      inp.autocomplete = 'off';
      const commit = () => {
        if (inp.value !== v.value) setCvar(inp.value);
      };
      inp.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { commit(); inp.blur(); }
        if (e.key === 'Escape') { inp.value = v.value; inp.blur(); }
      });
      inp.addEventListener('blur',  commit);
      inp.addEventListener('click', (e) => e.stopPropagation());
      const apply = document.createElement('button');
      apply.type = 'button';
      apply.className = 'v-apply';
      apply.textContent = 'Apply';
      apply.title = 'Send the typed value to the engine now';
      apply.addEventListener('click', (e) => {
        e.stopPropagation();
        commit();
      });
      widget.appendChild(inp);
      widget.appendChild(apply);
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

  // Render the persistent "★ pinned" column on the left of the panel
  // body in modern mode. Always shown alongside whatever tab is active.
  // In classic mode this column is hidden via CSS, but we still keep
  // it in sync (cheap) so toggling modes doesn't lose state.
  function renderPinnedColumn(fillInput) {
    const list = document.getElementById('pinned-list');
    if (!list) return;
    list.innerHTML = '';
    if (!sidePanelData) return;
    const filter = (sideSearch && sideSearch.value || '').trim().toLowerCase();
    const ok = (n) => !filter || n.toLowerCase().includes(filter);

    // Walk groups in alphabetical order (matches main panel ordering),
    // emit only items that are in the pinned set. groups now have
    // a `subs` Map (top -> sub -> {cvars, commands}); iterate the
    // subs to find pinned items.
    for (const g of sidePanelData.names) {
      const grp = sidePanelData.groups.get(g);
      if (!grp || !grp.subs) continue;
      for (const [, items] of grp.subs) {
        for (const v of items.cvars) {
          if (pinned.has(v.name) && ok(v.name)) {
            const row = renderCvarRow(v, fillInput);
            if (row) list.appendChild(row);
          }
        }
        for (const c of items.commands) {
          if (pinned.has(c.name) && ok(c.name)) list.appendChild(renderCommandRow(c, fillInput));
        }
      }
    }
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

    // The persistent left column shows pinned items in modern mode.
    // Always re-render it -- cost is trivial and keeps it in sync.
    renderPinnedColumn(fillInput);

    if (mode === 'modern' && activeTab === 'stats') {
      renderStats();
      return;
    }

    // For modern cvars/commands tabs we suppress the other type.
    const showCvars    = (mode === 'classic') || activeTab === 'cvars';
    const showCommands = (mode === 'classic') || activeTab === 'commands';

    for (const top of sidePanelData.names) {
      const g = sidePanelData.groups.get(top);
      // First scan: are there any matching items under this top group?
      // We render the top heading lazily so a group with no surviving
      // children (after the search filter) doesn't leave a stranded
      // header.
      let topHeadEmitted = false;
      const emitTopHead = () => {
        if (topHeadEmitted) return;
        const head = document.createElement('div');
        head.className = 'grp-head';
        head.textContent = top;
        cvarsPanel.appendChild(head);
        topHeadEmitted = true;
      };

      for (const [sub, items] of g.subs) {
        const cvarsHere = showCvars    ? items.cvars.filter(v => matches(v.name))    : [];
        const cmdsHere  = showCommands ? items.commands.filter(v => matches(v.name)) : [];
        if (cvarsHere.length === 0 && cmdsHere.length === 0) continue;
        emitTopHead();
        if (sub) {
          const sh = document.createElement('div');
          sh.className = 'grp-subhead';
          sh.textContent = sub;
          cvarsPanel.appendChild(sh);
        }
        for (const v of cvarsHere) {
          const row = renderCvarRow(v, fillInput);
          if (row) cvarsPanel.appendChild(row);
        }
        for (const cmd of cmdsHere)  cvarsPanel.appendChild(renderCommandRow(cmd, fillInput));
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
        cvarMeta[v.name] = {
          allowed_values: v.allowed_values || [],
          slider_min:     v.slider_min,
          slider_max:     v.slider_max,
          slider_step:    v.slider_step,
          value:          v.value,
          default_value:  v.default,
          description:    v.description || '',
        };
      }
    }
    commandMeta = {};
    if (k && k.ok && k.commands) {
      for (const v of k.commands) {
        names.add(v.name);
        commandMeta[v.name] = {
          default_args: v.default_args || '',
          description:  v.description || '',
        };
      }
    }
    allNames = Array.from(names).sort();
  }

  // ---------- VS Code-style completion popup + inline ghost -----------------
  //
  // Two coupled UIs share state via `popupState`:
  //   1. The popup (#input-completions) -- a scrollable list of matches
  //      with name + kind chip + current value + truncated description.
  //   2. The inline ghost (#input-ghost) -- a dim preview of the
  //      currently-highlighted match's tail, rendered IN-PLACE in the
  //      input so the user sees what Tab will commit.
  //
  // Match scoring tries 3 modes in descending priority (the highest
  // scoring mode wins per candidate; all candidates with score > 0
  // get ranked in the popup):
  //   - PREFIX: candidate starts with the query. Tightest, top score.
  //   - SUBSTRING: query appears as a contiguous run inside the
  //     candidate. Bonus when the run starts at a word boundary
  //     (position 0 or right after `_`).
  //   - FUZZY: query chars appear in order, possibly with gaps. Used
  //     so typing "rbi" finds "r_bloom_intensity". Penalized vs the
  //     stricter modes so prefix/substring always sort above.
  //
  // popupState shape:
  //   { active: bool, items: [{name, kind, value, description, score, spans}],
  //     selected: int, token: {start, end, text, isToken0, firstTok} }
  let popupState = { active: false, items: [], selected: -1, token: null };
  const completionsEl = document.getElementById('input-completions');

  // Identify the word at the cursor (or just before it). Returns
  //   { start, end, text, isToken0, firstTok }
  // where text is the run of non-space chars containing the cursor.
  // `start`..`end` are indices into `input.value` ready for splice-
  // style replacement. isToken0 == true means we're completing the
  // first word of the line (a cvar/command name); firstTok is the
  // already-typed first token (used to pick value-position
  // candidates).
  function currentToken() {
    const v   = input.value;
    const pos = input.selectionStart ?? v.length;
    // Find the word boundaries around `pos`.
    let s = pos, e = pos;
    while (s > 0 && v[s - 1] !== ' ') --s;
    while (e < v.length && v[e] !== ' ') ++e;
    const text = v.slice(s, e);
    // Token-0 detection: scan from start of input through `s`; if
    // we see any non-space before `s`, we're past token 0.
    let isToken0 = true;
    let firstTok = '';
    {
      // Find the first non-space run -- that's "token 0" by definition.
      // The cursor's word is `isToken0` only when its start aligns
      // with the start of that run (`s === tStart`). A cursor anywhere
      // past the first run's trailing space lives in a later token.
      let i = 0;
      while (i < v.length && v[i] === ' ') ++i;
      const tStart = i;
      while (i < v.length && v[i] !== ' ') ++i;
      firstTok = v.slice(tStart, i);
      isToken0 = (s === tStart);
    }
    return { start: s, end: e, text, isToken0, firstTok };
  }

  // Score a candidate against the user's query. Returns
  //   { score, spans }
  // where score is a positive number (0 means no match) and spans is
  // an array of [from, to) char ranges within the candidate that
  // matched -- used by the popup to highlight the matching chars
  // visually. Higher score = ranked higher. Length-of-candidate tie-
  // breaker is applied by the caller after scoring.
  function scoreMatch(name, q) {
    if (!q) return { score: 1, spans: [] };
    const nLow = name.toLowerCase();
    const qLow = q.toLowerCase();
    // PREFIX -- tightest match, highest base score.
    if (nLow.startsWith(qLow)) {
      const tightness = qLow.length / nLow.length;   // 0..1
      return { score: 1000 + Math.round(tightness * 200), spans: [[0, qLow.length]] };
    }
    // SUBSTRING -- contiguous match anywhere.
    const idx = nLow.indexOf(qLow);
    if (idx !== -1) {
      const wordStart = (idx === 0) || nLow[idx - 1] === '_';
      const tightness = qLow.length / nLow.length;
      const wordBonus = wordStart ? 100 : 0;
      return {
        score: 500 + wordBonus + Math.round(tightness * 100) - idx,
        spans: [[idx, idx + qLow.length]],
      };
    }
    // FUZZY -- in-order subsequence match, char by char, with a
    // word-boundary bonus for each char that lands at the start of a
    // segment (after `_`). Pairs of consecutive matches get a small
    // streak bonus so "rbi" prefers "r_bloom_intensity" over
    // "r_blur_threshold_idiv" (hypothetical).
    let qi = 0, score = 0, lastMatch = -2;
    const spans = [];
    let runStart = -1;
    for (let i = 0; i < nLow.length && qi < qLow.length; ++i) {
      if (nLow[i] !== qLow[qi]) {
        if (runStart !== -1) { spans.push([runStart, i]); runStart = -1; }
        continue;
      }
      // Per-char scoring.
      const atWordStart = (i === 0) || nLow[i - 1] === '_';
      score += atWordStart ? 8 : 4;
      if (i === lastMatch + 1) score += 4;   // streak bonus
      lastMatch = i;
      if (runStart === -1) runStart = i;
      ++qi;
    }
    if (runStart !== -1) spans.push([runStart, lastMatch + 1]);
    if (qi < qLow.length) return { score: 0, spans: [] };   // didn't match all chars
    // Penalize fuzzy heavily so prefix/substring sort above. Density
    // bonus rewards tight matches (fewer skipped chars).
    const density = qLow.length / nLow.length;
    return { score: 100 + score + Math.round(density * 50), spans };
  }

  // Build the candidate pool for the current token context:
  //   - Token 0: every cvar + every command.
  //   - Token 1+ with `toggle` as first token: only cvars with
  //     allowed_values (those are the meaningful toggle targets).
  //   - Token 1+ otherwise: the named cvar's allowed_values (when it
  //     has any). Returns [] when there's nothing useful to suggest.
  function getCandidates(token) {
    const out = [];
    if (token.isToken0) {
      for (const n of allNames) {
        const cv  = cvarMeta[n];
        const cmd = commandMeta[n];
        if (cv) {
          out.push({
            name: n, kind: 'cvar',
            value: cv.value !== undefined ? String(cv.value) : '',
            description: cv.description || '',
          });
        } else if (cmd) {
          out.push({
            name: n, kind: 'cmd',
            value: cmd.default_args || '',
            description: cmd.description || '',
          });
        }
      }
      return out;
    }
    // Value position.
    if (token.firstTok === 'toggle') {
      // Reuse the already-sorted `allNames` and filter -- avoids
      // re-sorting on every keystroke (Object.keys(cvarMeta).sort()
      // is O(n log n); refreshCompletions runs from the `input`
      // event handler, so per-keystroke). `allNames` is sorted once
      // in refreshNames(); allNames -> filter is O(n).
      for (const n of allNames) {
        const cv = cvarMeta[n];
        if (cv && cv.allowed_values && cv.allowed_values.length > 0) {
          out.push({ name: n, kind: 'cvar',
                     value: cv.value !== undefined ? String(cv.value) : '',
                     description: cv.description || '' });
        }
      }
      return out;
    }
    const cv = cvarMeta[token.firstTok];
    if (cv && cv.allowed_values && cv.allowed_values.length > 0) {
      for (const v of cv.allowed_values) {
        out.push({ name: v, kind: 'value',
                   value: (String(v) === String(cv.value)) ? 'current'
                        : (String(v) === String(cv.default_value)) ? 'default' : '',
                   description: '' });
      }
    } else if (cv && cv.value !== undefined) {
      // Free-form cvar: offer the current value and (when different)
      // the default as one-shot suggestions.
      out.push({ name: String(cv.value), kind: 'value', value: 'current',
                 description: cv.description || '' });
      const dflt = cv.default_value;
      if (dflt !== undefined && String(dflt) !== String(cv.value)) {
        out.push({ name: String(dflt), kind: 'value', value: 'default',
                   description: cv.description || '' });
      }
    }
    return out;
  }

  // Recompute candidates + scores for the current cursor context and
  // (re)render the popup + inline ghost. Called from the `input`
  // event handler, from Tab, from cursor-movement keys, etc.
  function refreshCompletions(forceShow) {
    const token = currentToken();
    const pool  = getCandidates(token);
    if (pool.length === 0) { hideCompletions(); return; }

    // Empty-query gating:
    //   - Token 0 with empty text: don't auto-open. The popup would
    //     list every cvar + command in the engine every time the
    //     input becomes empty (e.g. fresh page load, post-Submit
    //     clear) -- that's noise. Token 0 needs at least one typed
    //     char to open.
    //   - Value position (token.isToken0 === false) with empty text:
    //     DO auto-open. This is the "user just typed `<cvar> ` and
    //     wants to see the value list including the current value"
    //     path.
    //   - forceShow=true (Tab from a hidden popup) always opens
    //     regardless of token text.
    if (!forceShow && token.text.length === 0 && token.isToken0) {
      hideCompletions();
      return;
    }

    // Score every candidate. Value-position rows tagged "current" /
    // "default" get a small score bonus so they sort to the top when
    // the popup opens at `<cvar> ` (empty query, value position) --
    // the user expressly asked for "show the current value". Bonuses
    // are small enough that any real prefix / substring / fuzzy match
    // on a typed query still beats them (prefix scores 1000+,
    // substring 500+, fuzzy 100+; +5 / +2 are noise at those scales).
    const scored = [];
    for (const c of pool) {
      const { score, spans } = scoreMatch(c.name, token.text);
      if (score <= 0) continue;
      let s = score;
      if      (c.value === 'current') s += 5;
      else if (c.value === 'default') s += 2;
      scored.push({ ...c, score: s, spans });
    }
    if (scored.length === 0) { hideCompletions(); return; }
    scored.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      if (a.name.length !== b.name.length) return a.name.length - b.name.length;
      return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
    });

    // Cap to a reasonable visible count -- the popup is scrollable
    // beyond that, but rendering 800 rows every keystroke is wasteful.
    const items = scored.slice(0, 60);
    popupState = { active: true, items, selected: 0, token };
    renderCompletions();
    renderGhostFromPopup();
  }

  function renderCompletions() {
    if (!completionsEl) return;
    if (!popupState.active || popupState.items.length === 0) {
      completionsEl.hidden = true;
      completionsEl.innerHTML = '';
      return;
    }
    const rows = popupState.items.map((it, i) => {
      // Build the name with match-span highlights.
      let name = '';
      let cursor = 0;
      const spans = it.spans || [];
      for (const [a, b] of spans) {
        if (a > cursor) name += escape(it.name.slice(cursor, a));
        name += '<span class="match">' + escape(it.name.slice(a, b)) + '</span>';
        cursor = b;
      }
      if (cursor < it.name.length) name += escape(it.name.slice(cursor));
      const valEl = it.value
          ? `<span class="completion-value">${escape(it.value)}</span>` : '';
      const kindLabel = it.kind === 'cmd' ? 'cmd'
                      : it.kind === 'value' ? '' : 'cvar';
      const kindEl = kindLabel
          ? `<span class="completion-kind">${escape(kindLabel)}</span>` : '';
      const descEl = it.description
          ? `<div class="completion-desc">${escape(it.description.slice(0, 120))}</div>` : '';
      const isSel = (i === popupState.selected);
      const sel   = isSel ? ' selected' : '';
      // aria-selected pairs with the row's `selected` CSS class. The
      // listbox container (#input-completions) has role="listbox";
      // each row is role="option" -- screen readers announce the
      // option marked aria-selected="true" so the highlight is
      // perceivable without sighted CSS.
      const aria  = ` aria-selected="${isSel ? 'true' : 'false'}"`;
      const opt_id = `completion-opt-${i}`;
      return `<div class="completion-row${sel}" role="option" id="${opt_id}"`
           + aria + ` data-idx="${i}">`
           + `<span class="completion-name">${name}</span>`
           + kindEl + valEl + descEl + `</div>`;
    }).join('');
    completionsEl.innerHTML = rows;
    completionsEl.hidden = false;
    // Scroll the selected row into view (popup is scrollable when N > visible).
    const sel = completionsEl.querySelector('.completion-row.selected');
    if (sel) sel.scrollIntoView({ block: 'nearest' });
    // ARIA wiring: input owns the listbox via aria-controls and points
    // at the currently-highlighted option via aria-activedescendant.
    // Screen readers reading the input then announce the selected
    // option's text on selection change (Up/Down) without focus
    // needing to move out of the input.
    input.setAttribute('aria-controls', 'input-completions');
    input.setAttribute('aria-expanded', 'true');
    if (popupState.selected >= 0) {
      input.setAttribute('aria-activedescendant',
        `completion-opt-${popupState.selected}`);
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  function renderGhostFromPopup() {
    if (!popupState.active || popupState.selected < 0) {
      ghostState = null;
      renderGhost();
      return;
    }
    const it = popupState.items[popupState.selected];
    const t  = popupState.token;
    // Two preconditions for the inline ghost:
    //   1. The candidate STARTS WITH the typed token (prefix fit).
    //      Substring / fuzzy hits don't concatenate cleanly with what
    //      the user already typed -- the popup row's match-span
    //      highlight is the affordance for those.
    //   2. The token ENDS at the input's EOL (and the cursor is also
    //      at EOL). For mid-line tokens (`r_bl|m` with cursor in the
    //      middle of a word), renderGhost() would otherwise paint the
    //      ghost tail past the entire input.value -- visually wrong,
    //      since the completion will splice into the middle.
    const fits = it.name.toLowerCase().startsWith(t.text.toLowerCase());
    const atEOL = (t.end === input.value.length) &&
                  (input.selectionStart === input.value.length);
    if (!fits || !atEOL) { ghostState = null; renderGhost(); return; }
    ghostState = {
      matches: popupState.items.map(x => x.name),
      index:   popupState.selected,
      before:  input.value.slice(0, t.start),
      prefix:  t.text,
      isToken0: t.isToken0,
      annotation: '',
    };
    renderGhost();
  }

  function hideCompletions() {
    popupState = { active: false, items: [], selected: -1, token: null };
    if (completionsEl) {
      completionsEl.hidden = true;
      completionsEl.innerHTML = '';
    }
    // Clear the ARIA wiring so the input no longer claims to own a
    // listbox / point at an option that's gone.
    input.removeAttribute('aria-controls');
    input.removeAttribute('aria-activedescendant');
    input.setAttribute('aria-expanded', 'false');
    ghostState = null;
    renderGhost();
  }

  function popupMoveSelection(dir) {
    if (!popupState.active || popupState.items.length === 0) return false;
    const n = popupState.items.length;
    popupState.selected = ((popupState.selected + dir) % n + n) % n;
    renderCompletions();
    renderGhostFromPopup();
    return true;
  }

  // Commit the currently-highlighted popup row by replacing the word
  // at the cursor with the candidate name (and a trailing space when
  // we're completing token 0 so the user lands at the value position
  // ready to keep typing). `chainNext` controls whether we
  // auto-reopen the popup at the new cursor position after commit:
  //   - Tab commits chain (Tab = "complete + show next suggestions")
  //   - Enter / click commits DON'T chain (those are "accept + stop"
  //     gestures -- otherwise Enter would loop the popup forever).
  // Mirrors VS Code's IntelliSense-vs-Tab semantics.
  function popupCommit(chainNext) {
    if (!popupState.active || popupState.selected < 0) return false;
    const it = popupState.items[popupState.selected];
    const t  = popupState.token;
    const v  = input.value;
    const tail = it.name + (t.isToken0 ? ' ' : '');
    input.value = v.slice(0, t.start) + tail + v.slice(t.end);
    const caret = t.start + tail.length;
    input.setSelectionRange(caret, caret);
    const wasToken0 = t.isToken0;
    hideCompletions();
    if (chainNext && wasToken0) refreshCompletions(/*forceShow=*/true);
    return true;
  }

  // Sync the absolutely-positioned ghost overlay with the input's
  // horizontal scroll position.  The input element scrolls its
  // content when the typed string exceeds the visible width; without
  // this translate, the ghost-typed/-tail spans stay anchored at
  // the left edge while the actual typed glyphs slide left,
  // misaligning the suggestion tail visually.  Called from
  // renderGhost() and from input/scroll listeners below.
  const ghostOverlay = document.getElementById('input-ghost');
  function syncGhostScroll() {
    if (!ghostOverlay) return;
    ghostOverlay.style.transform = 'translateX(' + (-input.scrollLeft) + 'px)';
  }
  function renderGhost() {
    if (!ghostState) {
      ghostTyped.textContent      = '';
      ghostTail.textContent       = '';
      ghostAnnotation.textContent = '';
      syncGhostScroll();
      return;
    }
    const match = ghostState.matches[ghostState.index];
    const fits  = match.length >= ghostState.prefix.length &&
                  match.startsWith(ghostState.prefix);
    if (fits) {
      ghostTyped.textContent      = input.value;
      ghostTail.textContent       = match.slice(ghostState.prefix.length);
      ghostAnnotation.textContent = ghostState.annotation || '';
    } else {
      ghostTyped.textContent      = '';
      ghostTail.textContent       = '';
      ghostAnnotation.textContent = '';
    }
    syncGhostScroll();
  }
  // Track input scroll directly too -- typing past the visible
  // width fires both 'input' and 'scroll' events; cursor moves
  // (arrow keys) only fire 'scroll'.
  input.addEventListener('scroll', syncGhostScroll, { passive: true });
  function dismissGhost() {
    if (!ghostState) return;
    ghostState = null;
    renderGhost();
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
    // Enter while the popup is up commits the highlighted match and
    // KEEPS the line in the input -- VS Code-style "accept the
    // suggestion, don't run yet". A second Enter (with the popup now
    // dismissed) actually submits the line. This avoids the
    // surprise-execute behaviour where Enter would both commit the
    // ghost AND fire the command in one keystroke.
    if (popupState.active) {
      popupCommit(/*chainNext=*/false);
      return;
    }
    const line = input.value;
    input.value = '';
    hideCompletions();
    exec(line);
  });

  input.addEventListener('keydown', (e) => {
    // Modifier keys alone are noise (Shift fires keydown before the
    // real key arrives) -- ignore them so a Shift-Up still routes to
    // the popup, Shift-Tab still cycles backward, etc.
    if (e.key === 'Shift' || e.key === 'Control' || e.key === 'Alt' ||
        e.key === 'Meta'  || e.key === 'CapsLock') {
      return;
    }

    // ---- Popup-active key handling --------------------------------------
    // Up/Down move the highlight inside the popup (NOT command
    // history); Tab commits the highlighted match; Enter is handled
    // by the submit listener above (commits + dismiss + keep line);
    // Esc dismisses; Right-arrow at end commits the prefix-fitting
    // ghost (the only case where the inline preview accurately
    // represents what gets inserted).
    if (popupState.active) {
      if (e.key === 'ArrowDown') { e.preventDefault(); popupMoveSelection(+1); return; }
      if (e.key === 'ArrowUp')   { e.preventDefault(); popupMoveSelection(-1); return; }
      if (e.key === 'Tab')       { e.preventDefault(); popupCommit(/*chainNext=*/true); return; }
      if (e.key === 'Escape')    { e.preventDefault(); hideCompletions(); return; }
      // End / Right-arrow-at-EOL commit the highlighted match without
      // chaining (matches the PR description: "Enter / click / Right-
      // arrow commit without chaining"). Previously these only fired
      // when ghostState existed (prefix fit), so substring / fuzzy
      // selections silently dropped these keys -- inconsistent UX.
      // The popup itself is the affordance regardless of match mode,
      // so commit unconditionally when the user reaches for End or
      // Right-arrow-at-EOL.
      if (e.key === 'End' ||
          (e.key === 'ArrowRight' && input.selectionStart === input.value.length)) {
        e.preventDefault();
        popupCommit(/*chainNext=*/false);
        return;
      }
      // Any other key dismisses the popup and falls through to default
      // input behaviour. The subsequent `input` event handler will
      // re-open the popup if the typed char still leaves us inside a
      // completable token.
      if (e.key !== 'Enter') {
        hideCompletions();
        // fall through
      }
    }

    // ---- Popup-inactive key handling ------------------------------------
    if (e.key === 'Tab') {
      // Open the popup -- the input handler that auto-shows only fires
      // on input events, not on bare Tab presses, so we route through
      // refreshCompletions(forceShow=true) which lists every candidate
      // for the current token (useful when the token is empty, e.g.
      // user typed "r_bloom " and wants to see the allowed values).
      e.preventDefault();
      refreshCompletions(/*forceShow=*/true);
      return;
    }
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

  // Mouse click on a popup row commits that row. We use mousedown
  // (not click) because click fires AFTER blur, and blur hides the
  // popup -- so a click on a row would otherwise hit empty space by
  // the time the click event actually arrived. Also: capture in
  // mousedown lets us preventDefault to keep the input focused so
  // the user can keep typing immediately after the commit.
  if (completionsEl) {
    completionsEl.addEventListener('mousedown', (e) => {
      const row = e.target.closest('.completion-row');
      if (!row) return;
      e.preventDefault();
      const idx = parseInt(row.dataset.idx, 10);
      if (!Number.isNaN(idx)) {
        popupState.selected = idx;
        popupCommit(/*chainNext=*/false);
        input.focus();
      }
    });
  }

  // Clicks INSIDE the input only re-anchor the cursor; refresh the
  // popup against the new cursor position so partial-cursor edits
  // still get a contextual suggestion. (Click outside the input
  // hides the popup via the blur path below.)
  input.addEventListener('mouseup', () => {
    // Microtask so input.selectionStart is the post-click position.
    setTimeout(() => refreshCompletions(/*forceShow=*/false), 0);
  });
  input.addEventListener('blur', () => {
    // Tiny delay so a popup-row mousedown still gets processed before
    // we hide the popup.
    setTimeout(() => { if (document.activeElement !== input) hideCompletions(); }, 80);
  });

  // Re-rank completions on every input mutation. This is the
  // "VS Code-like" affordance: as soon as the user types a single
  // char that matches anything in the candidate pool, the popup
  // appears -- no need to press Tab first. The popup is also kept
  // in sync as the user keeps typing (chars accepted -> matches
  // narrow, char deleted -> matches widen).
  input.addEventListener('input', () => {
    refreshCompletions(/*forceShow=*/false);
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

    dismissGhost();
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

  // -- Drag-to-resize: panel boundary + pinned/tabs split inside the panel.
  // Both store their last width in localStorage and apply it on load
  // via CSS custom properties. The drag handler updates the same
  // property so the layout reflows live.
  const setupResize = (handleId, cssVar, storageKey, defaultPx, minPx, maxPx, getStartFromEvent) => {
    const handle = document.getElementById(handleId);
    if (!handle) return;
    // maxPx may be a number OR a () => number so the outer panel can
    // track viewport width (re-evaluated each drag tick).
    const maxNow = () => (typeof maxPx === 'function') ? maxPx() : maxPx;
    const clamp  = (w) => Math.max(minPx, Math.min(maxNow(), w));
    // localStorage may hold a non-numeric / corrupted value -- parseInt
    // returns NaN and clamp(NaN) propagates NaN, ending up as `NaNpx`
    // on the CSS variable and breaking the layout. Validate and fall
    // back to defaultPx when the parse yields a non-finite number.
    const rawStored = parseInt(localStorage.getItem(storageKey) || defaultPx, 10);
    const stored = clamp(Number.isFinite(rawStored) ? rawStored : defaultPx);
    document.documentElement.style.setProperty(cssVar, stored + 'px');

    let dragging = false;
    let startX = 0, startW = 0;
    handle.addEventListener('mousedown', (e) => {
      dragging = true;
      startX = e.clientX;
      startW = parseInt(getComputedStyle(document.documentElement).getPropertyValue(cssVar), 10) || stored;
      handle.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = getStartFromEvent(e, startX);
      document.documentElement.style.setProperty(cssVar, clamp(startW + dx) + 'px');
    });
    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      const w = parseInt(getComputedStyle(document.documentElement).getPropertyValue(cssVar), 10);
      if (w) localStorage.setItem(storageKey, String(w));
    });
  };
  // Outer panel resize: dragging LEFT widens the panel. The cap is
  // viewport-relative so on wide monitors the side panel can stretch
  // well past 50%; on narrow viewports it collapses to the panel's
  // own minimum. 280px floor leaves the output column usable; the
  // extra 6px is the drag-handle grid column.
  const outerMaxPx = () => Math.max(280, window.innerWidth - 280 - 6);
  setupResize('panel-resize', '--side-panel-w', 'demont.panelW',
              480, 280, outerMaxPx, (e, sx) => sx - e.clientX);
  // Inner column resize: dragging RIGHT widens the pinned column.
  setupResize('inner-resize', '--pinned-col-w', 'demont.pinnedW',
              180, 100, 500, (e, sx) => e.clientX - sx);

  // Refresh names occasionally so newly-registered cvars show up.
  setInterval(refreshNames, 10_000);
  // Periodic value refresh so engine-driven cvar changes (sky-time
  // animation, auto-exposure, sun position auto-update under
  // r_sky_use_astronomical) are visible without a user click. 1 Hz is
  // enough for sliders to track without flooding the WS.
  //
  // Pause refresh during any pointer-down or focused-widget state so
  // the DOM rebuild doesn't tear out a slider being dragged or a
  // dropdown being picked. The pointer flag covers pointer drags
  // (range inputs may not retain focus during drag on every browser);
  // the activeElement check covers keyboard editing in <input>s and
  // open <select>s.
  let pointerActive = false;
  if (sidePanel) {
    sidePanel.addEventListener('pointerdown', () => { pointerActive = true; });
  }
  window.addEventListener('pointerup',     () => { pointerActive = false; });
  window.addEventListener('pointercancel', () => { pointerActive = false; });
  setInterval(() => {
    if (pointerActive) return;
    const ae = document.activeElement;
    if (ae && (ae.tagName === 'SELECT' || ae.tagName === 'INPUT')) return;
    refreshCvars();
  }, 1000);

  connect();
})();
