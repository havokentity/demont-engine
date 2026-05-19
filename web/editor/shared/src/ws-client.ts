// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Minimal-but-robust WebSocket client. Handles auto-reconnect with
// exponential backoff so a Chrome --app panel that outlives the
// engine process keeps trying to re-attach when the engine relaunches.
// Each command gets an `id` so the matching result fires the resolve
// callback; events on subscribed topics fan out to listeners.

import type {
  WsInbound,
  WsOutbound,
  WsResult,
  WsEvent,
} from './types';

export type ClientStatus =
  | 'connecting'
  | 'open'
  | 'closing'
  | 'closed'
  | 'error';

type EventListener = (event: WsEvent) => void;
type StatusListener = (status: ClientStatus) => void;

interface PendingRequest {
  resolve: (r: WsResult) => void;
  reject: (err: Error) => void;
  // ms-since-epoch
  sentAt: number;
}

export interface WebSocketClientOptions {
  url: string;
  // Topics to subscribe to on every (re)connect. Defaults match the
  // editor needs: log + selection_change + scene_dirty.
  topics?: string[];
  // Initial reconnect delay, ms. Doubles on each failure up to
  // maxReconnectMs.
  initialReconnectMs?: number;
  maxReconnectMs?: number;
  // Hard ceiling on outstanding requests. Anything beyond it gets
  // immediately rejected so a runaway producer doesn't grow the map
  // forever.
  maxPending?: number;
  // Per-request timeout, ms. The engine drains commands on the main
  // thread inside Console::Drain so most replies come within a frame
  // or two; 5s is a generous ceiling.
  requestTimeoutMs?: number;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private readonly url: string;
  private readonly topics: string[];
  private readonly initialReconnectMs: number;
  private readonly maxReconnectMs: number;
  private readonly maxPending: number;
  private readonly requestTimeoutMs: number;

  private reconnectMs: number;
  private reconnectTimer: number | null = null;
  private status: ClientStatus = 'connecting';
  private nextRequestId = 1;
  private pending = new Map<string, PendingRequest>();
  private eventListeners = new Map<string, Set<EventListener>>();
  private statusListeners = new Set<StatusListener>();
  private explicitClose = false;
  private timeoutTimer: number | null = null;

  constructor(opts: WebSocketClientOptions) {
    this.url = opts.url;
    this.topics = opts.topics ?? ['log', 'selection_change', 'scene_dirty'];
    this.initialReconnectMs = opts.initialReconnectMs ?? 500;
    this.maxReconnectMs = opts.maxReconnectMs ?? 8000;
    this.maxPending = opts.maxPending ?? 256;
    this.requestTimeoutMs = opts.requestTimeoutMs ?? 5000;
    this.reconnectMs = this.initialReconnectMs;
  }

  start(): void {
    this.explicitClose = false;
    this.openOnce();
    if (this.timeoutTimer == null) {
      this.timeoutTimer = window.setInterval(() => this.reapTimeouts(), 1000);
    }
  }

  close(): void {
    this.explicitClose = true;
    if (this.reconnectTimer != null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.timeoutTimer != null) {
      clearInterval(this.timeoutTimer);
      this.timeoutTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('closed');
  }

  getStatus(): ClientStatus {
    return this.status;
  }

  // ---- subscriptions ----
  onEvent(topic: string, listener: EventListener): () => void {
    let set = this.eventListeners.get(topic);
    if (!set) {
      set = new Set();
      this.eventListeners.set(topic, set);
    }
    set.add(listener);
    return () => set!.delete(listener);
  }

  onStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    // Fire current immediately so consumers can render initial state.
    queueMicrotask(() => listener(this.status));
    return () => this.statusListeners.delete(listener);
  }

  // ---- commands ----

  // Send a command and resolve on the matching {type:"result", id}.
  send<T extends WsResult = WsResult>(msg: WsInbound): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        reject(new Error('socket not open'));
        return;
      }
      if (this.pending.size >= this.maxPending) {
        reject(new Error('too many in-flight requests'));
        return;
      }
      const id = msg.id ?? `r${this.nextRequestId++}`;
      const out: WsInbound = { ...msg, id };
      this.pending.set(id, {
        resolve: resolve as (r: WsResult) => void,
        reject,
        sentAt: Date.now(),
      });
      try {
        this.ws.send(JSON.stringify(out));
      } catch (err) {
        this.pending.delete(id);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  // Convenience: run a raw console command line.
  exec(line: string): Promise<WsResult> {
    return this.send({ type: 'exec', line });
  }

  listScene(): Promise<WsResult> {
    return this.send({ type: 'list_scene' });
  }

  selectObject(kind: 'prim' | 'light' | 'csg' | 'sdf' | 'smoke' | 'none', objId?: number): Promise<WsResult> {
    if (kind === 'none') {
      return this.send({ type: 'select', kind: 'none' });
    }
    if (objId == null) {
      return Promise.reject(new Error('selectObject requires obj_id for non-none kind'));
    }
    return this.send({ type: 'select', kind, obj_id: objId });
  }

  // ---- internals ----

  private openOnce(): void {
    this.setStatus('connecting');
    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      this.setStatus('error');
      this.scheduleReconnect();
      return;
    }

    this.ws.addEventListener('open', () => {
      this.reconnectMs = this.initialReconnectMs;
      this.setStatus('open');
      // Re-subscribe to topics so a reconnect doesn't lose event
      // delivery for the panel.
      if (this.topics.length > 0) {
        try {
          this.ws!.send(JSON.stringify({
            type: 'subscribe',
            topics: this.topics,
          }));
        } catch {
          // Swallow -- if send throws here we'll hit onerror and
          // reconnect.
        }
      }
    });

    this.ws.addEventListener('message', (e) => {
      this.handleMessage(typeof e.data === 'string' ? e.data : '');
    });

    this.ws.addEventListener('error', () => {
      this.setStatus('error');
      // onclose will fire right after; reconnect there.
    });

    this.ws.addEventListener('close', () => {
      // Reject every in-flight request so callers don't hang.
      const err = new Error('socket closed');
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();

      this.setStatus('closed');
      if (!this.explicitClose) this.scheduleReconnect();
    });
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer != null) return;
    const delay = this.reconnectMs;
    this.reconnectMs = Math.min(this.reconnectMs * 2, this.maxReconnectMs);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.openOnce();
    }, delay);
  }

  private handleMessage(text: string): void {
    if (!text) return;
    let msg: WsOutbound;
    try {
      msg = JSON.parse(text) as WsOutbound;
    } catch {
      // Not JSON -- ignore.
      return;
    }
    if (msg.type === 'result') {
      const id = msg.id;
      if (id) {
        const p = this.pending.get(id);
        if (p) {
          this.pending.delete(id);
          p.resolve(msg);
        }
      }
      return;
    }
    if (msg.type === 'event') {
      const listeners = this.eventListeners.get(msg.topic);
      if (listeners) {
        for (const l of listeners) {
          try { l(msg); } catch { /* swallow listener errors */ }
        }
      }
      // Also fire wildcard listeners (topic = "*").
      const wild = this.eventListeners.get('*');
      if (wild) {
        for (const l of wild) {
          try { l(msg); } catch { /* swallow */ }
        }
      }
    }
  }

  private reapTimeouts(): void {
    if (this.pending.size === 0) return;
    const now = Date.now();
    const expired: string[] = [];
    for (const [id, p] of this.pending) {
      if (now - p.sentAt > this.requestTimeoutMs) {
        expired.push(id);
      }
    }
    for (const id of expired) {
      const p = this.pending.get(id);
      if (p) {
        this.pending.delete(id);
        p.reject(new Error('request timed out'));
      }
    }
  }

  private setStatus(s: ClientStatus): void {
    if (s === this.status) return;
    this.status = s;
    for (const l of this.statusListeners) {
      try { l(s); } catch { /* swallow */ }
    }
  }
}
