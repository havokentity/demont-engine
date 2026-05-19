# `web/editor/` -- React+Vite editor shell

This is the workspace for the DeMonT engine's optional editor mode. Each
panel runs in its own Chrome `--app` window spawned by the engine; all
panels share one WebSocket endpoint at `/ws`.

This document is for **panel authors** (agents 21+, future contributors).
It pins the contracts the shell provides, so a downstream panel only has
to think about its own content.

---

## Layout

```
web/editor/
├── package.json            # workspace root (engines: node >=20)
├── tsconfig.json           # path alias @demont/editor-shared -> shared/src
├── vite.config.ts          # multi-entry build, one html per panel
├── shared/                 # @demont/editor-shared
│   └── src/
│       ├── index.ts        # public barrel
│       ├── types.ts        # SceneObject, Selection, WS envelopes
│       ├── ws-client.ts    # WebSocketClient (reconnect, request id)
│       ├── store.ts        # SceneStore (zustand) + bindClientToStore
│       ├── endpoint.ts     # wsEndpoint() URL builder
│       ├── theme.css       # shared CSS variables + utility classes
│       └── components/     # Shell, TitleBar, Toolbar, StatusBar, Placeholder
└── panels/
    ├── scene-hierarchy/    # agent-21
    ├── inspector/          # agent-22
    ├── asset-browser/      # future
    └── toolbar/            # MVP demo (live)
```

Each panel has:
- `package.json` -- name only, no extra deps
- `index.html` -- Vite entry HTML, loads `./src/main.tsx`
- `src/main.tsx` -- ReactDOM root + `<Shell>` wrapper

## Build

```sh
cd web/editor
npm install
npm run build         # writes dist/, embedded by cmake
npm run dev           # Vite dev server on :5173 (HMR; needs r_editor_dev_mode 1)
npm run typecheck     # tsc --noEmit
```

The engine's `cmake/Editor.cmake` runs `npm install && npm run build`
automatically when any file in `web/editor/{shared,panels}/*` changes.
After the build, `cmake/EmbedEditorAssets.cmake` walks `dist/` and emits
a single `editor_assets.cpp` containing every file's bytes plus a
lookup table; this gets linked into `pt_editor` which `pt_console` and
`pt_engine` link against.

If `npm` is missing or `-DPT_BUILD_EDITOR=OFF` is passed, the embed
script emits a stub HTML that tells the user "run `npm run build`
first" -- the engine still links cleanly.

## Hot reload

```sh
cd web/editor && npm run dev   # starts Vite at http://localhost:5173
```

In the running engine, type `r_editor_dev_mode 1` then
`panel_open scene-hierarchy`. The engine opens
`http://localhost:5173/panels/scene-hierarchy/?engine_port=27960` --
Vite's HMR pipes React reloads on every save; the panel still
WebSockets to the engine's `:27960/ws` via the `engine_port` query
parameter (`shared/src/endpoint.ts`).

Flip the cvar back to `0` to return to the embedded `/editor/<name>`
route served by the engine.

## Engine `<->` panel WebSocket protocol

All panels speak the same protocol the existing console UI does
(`web/console.js`), with two editor-specific extras:

### Inbound (panel -> engine)

| `type` | Fields | Purpose |
| --- | --- | --- |
| `exec` | `line` | Run a console command line |
| `get_cvar` | `name` | Fetch a single cvar's full record |
| `set_cvar` | `name`, `value` | Set a cvar |
| `list_cvars` | `prefix?` | Enumerate cvars |
| `list_commands` | `prefix?` | Enumerate commands |
| `subscribe` / `unsubscribe` | `topics: string[]` | Topic filtering |
| `list_scene` | _none_ | **(agent-19)** Whole-scene JSON snapshot |
| `select` | `kind: 'prim'\|'light'\|...\|'none'`, `obj_id?` | **(agent-19)** Set the engine selection |

Every inbound message accepts an optional `id` string; the result
echoes it back so the client can match request <-> reply.

### Outbound (engine -> panel)

| `type` | Fields | When |
| --- | --- | --- |
| `result` | `ok`, `id?`, plus type-specific payload | Reply to an inbound message |
| `event` | `topic`, `ts`, `data` | Broadcast to topic subscribers |

Known event topics:
- `log` -- forwarded `LOG_INFO/WARN/ERROR` lines
- `selection_change` -- **(agent-19)** `data: { kind, id, selected }`
- `scene_dirty` -- **(agent-19)** scene mutated; client should refetch
- `frame_stats` -- (future, not in agent-20 scope)

## How to write a new panel

1. **Skeleton.** Copy `panels/scene-hierarchy/` to `panels/<your-name>/`,
   update `package.json` `name`, edit `index.html` `<title>`.

2. **Register the panel** in two places:
   - `web/editor/vite.config.ts` -- add to the `panels` array.
   - `src/editor/EditorRoutes.cpp` -- add to the `kPanels[]` table.

3. **Wrap your content in `<Shell>`.** The Shell takes care of:
   - opening the WebSocket
   - subscribing to `log` / `selection_change` / `scene_dirty`
   - calling `list_scene` once at mount + on every `scene_dirty`
   - mirroring those into `useSceneStore`
   - rendering the titlebar (with connection-state indicator)
   - rendering the statusbar (with scene + selection counters)

   ```tsx
   import { Shell, useSceneStore } from '@demont/editor-shared';

   export default function App() {
     const objects = useSceneStore(s => s.objects);
     const selection = useSceneStore(s => s.selection);
     return (
       <Shell title="My Panel">
         <ul>
           {objects.map(o => (
             <li key={`${o.kind}-${o.id}`}>
               {o.kind} #{o.id}
             </li>
           ))}
         </ul>
       </Shell>
     );
   }
   ```

4. **Read selection.** Use `useSceneStore(s => s.selection)`. The store
   reacts to `selection_change` events from the engine -- you don't
   need to subscribe yourself.

5. **Set selection.** Use the WebSocket client via the `withClient`
   render prop:

   ```tsx
   <Shell title="My Panel" withClient={(client) => (
     <button onClick={() => client.selectObject('prim', 1)}>
       Select prim 1
     </button>
   )} />
   ```

   This dispatches `{type: 'select', kind: 'prim', obj_id: 1}` over WS;
   the engine reports back via the `selection_change` topic, which the
   store picks up. Don't optimistically update the store -- let the
   engine be the source of truth.

6. **Dispatch property edits.** Use `client.exec()` with the matching
   console command:

   ```tsx
   client.exec(`prim_set_pos 1 1.0 2.0 3.0`);
   client.exec(`prim_set_albedo 1 0.8 0.4 0.4`);
   client.exec(`prim_set_emission 1 5.0 5.0 5.0`);
   client.exec(`prim_set_material 1 metal`);
   ```

   The engine emits `scene_dirty` afterward; the shell will refetch
   the scene snapshot for you.

7. **Theme** is opt-in. Import `@demont/editor-shared/theme.css` in
   your `main.tsx`. CSS variables (`--bg`, `--fg`, `--accent`, ...)
   are available to your component-level CSS.

## Types

`shared/src/types.ts` mirrors the C++ structs in `src/engine/Engine.h`
exactly. If the C++ side changes a field name, update the TS mirror in
the same commit. Discriminated unions on `kind`:

```ts
type SceneObject = AnalyticPrim | AnalyticLight | UnknownObject;

interface AnalyticPrim {
  kind: 'prim';
  id: number;
  type: 0 | 1;            // Sphere | Plane
  material: 0 | 1 | 2 | 3; // Lambert | Metal | Dielectric | Water
  pos: [number, number, number];
  radius: number;
  albedo: [number, number, number];
  roughness: number;
  ior: number;
  emission: [number, number, number];
}
```

The engine-side JSON shape may be either a flat `objects[]` array OR
per-kind buckets (`prims[]`, `lights[]`, ...). The store's
`flattenScene()` helper handles both -- you always read
`useSceneStore(s => s.objects)`.

## Engine commands and cvars

| Console command | Effect |
| --- | --- |
| `panel_open <name>` | Spawn Chrome `--app` window for the panel |
| `panel_close <name>` | Kill the tracked PID |
| `panel_open_all` | Open every known panel |
| `panel_close_all` | Close every tracked panel |
| `panels` | List panels + their open/closed state |

| Cvar | Default | Effect |
| --- | --- | --- |
| `r_editor_panels_autoopen` | _empty_ | CSV list of panels to open on engine init |
| `r_editor_dev_mode` | `0` | 0 = embedded `/editor/<name>`; 1 = Vite dev server |

Hotkeys (fired by the GLFW key handler in `Engine::Init`):

| Key | Action |
| --- | --- |
| F2 | `panel_open scene-hierarchy` |
| F3 | `panel_open inspector` |
| F4 | `panel_open asset-browser` |
| F5 | `panel_open toolbar` |

## Why zustand?

The store needs to be updated **from outside the React tree** (the
WebSocketClient lives in a `useRef`, not a hook). React Context can't
do that without re-renders gated by a provider. Zustand's
`useSceneStore.getState().setSelection(...)` works equally well inside
or outside a component. It also gives per-field selector subscriptions
out of the box -- a `useSceneStore(s => s.selection)` consumer only
re-renders when `selection` changes, not on every `scene_dirty`.

Jotai was a candidate; we picked zustand because the shape is small
and flat (one store, one set of fields) and the cross-tree update
pattern is the dominant concern.

## Tests

There are no automated panel tests yet. The shell currently has zero
runtime checks beyond TypeScript types -- panels are expected to:
1. Pass `npm run typecheck` (run in CI when we add it).
2. Render in a Chrome `--app` window without console errors.
3. Show a graceful "engine offline" state when the WebSocket can't
   connect (the titlebar status indicator does this for free).
