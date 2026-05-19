// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Derive the engine's WebSocket URL from the panel's window.location.
// All four panels share a single endpoint at /ws on the engine's
// HTTP port (net_port cvar, 27960 default).

export function wsEndpoint(): string {
  // In hosted mode the engine serves the HTML and we hit /ws on the
  // same Origin. In Vite dev mode (port 5173) we still target the
  // engine's HTTP port -- pull that from a URL query parameter
  // (?engine_port=27960) or fall back to the default.
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname || 'localhost';

  // If we're talking to the engine directly, the page's port IS the
  // engine port and we just reuse it. If we're on Vite (5173) the
  // user can override via ?engine_port=NNNN; otherwise we guess
  // 27960 (the demont default).
  let port = window.location.port;
  const inVite = port === '5173' || port === '5174';
  if (inVite) {
    const params = new URLSearchParams(window.location.search);
    port = params.get('engine_port') || '27960';
  }
  if (!port) port = '27960';

  return `${proto}//${host}:${port}/ws`;
}
