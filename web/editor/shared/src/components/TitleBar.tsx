// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import type { ClientStatus } from '../ws-client';

export interface TitleBarProps {
  title: string;
  status: ClientStatus;
}

function statusLabel(s: ClientStatus): string {
  switch (s) {
    case 'open':       return 'engine: connected';
    case 'connecting': return 'engine: connecting…';
    case 'error':      return 'engine: error';
    case 'closing':    return 'engine: closing…';
    case 'closed':     return 'engine: offline';
  }
}

function statusClass(s: ClientStatus): string {
  switch (s) {
    case 'open':       return 'is-open';
    case 'connecting': return 'is-connecting';
    case 'error':      return 'is-error';
    case 'closing':
    case 'closed':     return 'is-closed';
  }
}

export function TitleBar({ title, status }: TitleBarProps) {
  return (
    <header className="editor-titlebar">
      <span className="titlebar-brand">DeMonT</span>
      <span className="titlebar-title">{title}</span>
      <span className={`titlebar-status ${statusClass(status)}`}>
        {statusLabel(status)}
      </span>
    </header>
  );
}
