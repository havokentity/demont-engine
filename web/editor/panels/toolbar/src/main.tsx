// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Shell,
  Toolbar,
  ToolbarGroup,
  ToolbarButton,
  type WebSocketClient,
} from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';

interface ToolbarBodyProps {
  client: WebSocketClient;
}

function ToolbarBody({ client }: ToolbarBodyProps) {
  const [lastOutput, setLastOutput] = useState<string>('');
  const [lastError, setLastError] = useState<string>('');

  const run = async (line: string) => {
    setLastOutput('');
    setLastError('');
    try {
      const r = await client.exec(line);
      if (r.ok) {
        setLastOutput(r.output ?? `(ok) ${line}`);
      } else {
        setLastError(r.error ?? 'unknown error');
      }
    } catch (err) {
      setLastError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>
      <Toolbar>
        <ToolbarGroup>
          <ToolbarButton label="frame_info" onClick={() => void run('frame_info')} />
          <ToolbarButton label="sys_info" onClick={() => void run('sys_info')} />
          <ToolbarButton label="mem_report" onClick={() => void run('mem_report')} />
        </ToolbarGroup>
        <ToolbarGroup>
          <ToolbarButton label="cam_reset" onClick={() => void run('cam_reset')} />
          <ToolbarButton label="cam_list" onClick={() => void run('cam_list')} />
        </ToolbarGroup>
        <ToolbarGroup>
          <ToolbarButton label="list_favs" onClick={() => void run('list_favs')} />
        </ToolbarGroup>
      </Toolbar>
      <div style={{ flex: 1, overflow: 'auto', padding: '0 12px' }}>
        {lastOutput && (
          <pre style={{
            margin: 0,
            padding: '8px 10px',
            background: 'var(--bg-elev)',
            border: '1px solid var(--border)',
            color: 'var(--fg)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontSize: 11.5,
            lineHeight: 1.5,
            fontFamily: "'SF Mono', Menlo, Consolas, monospace",
            borderRadius: 4,
          }}>
            {lastOutput}
          </pre>
        )}
        {lastError && (
          <pre style={{
            margin: 0,
            padding: '8px 10px',
            background: 'var(--bg-elev)',
            border: '1px solid var(--error)',
            color: 'var(--error)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontSize: 11.5,
            lineHeight: 1.5,
            fontFamily: "'SF Mono', Menlo, Consolas, monospace",
            borderRadius: 4,
          }}>
            error: {lastError}
          </pre>
        )}
        {!lastOutput && !lastError && (
          <p className="dim" style={{ padding: '12px', margin: 0, fontSize: 12 }}>
            Click a button to dispatch the matching console command via
            WebSocket. The shell already speaks the existing exec / get_cvar /
            set_cvar / list_scene / select protocols.
          </p>
        )}
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell title="Toolbar" withClient={(client) => <ToolbarBody client={client} />} />
  </StrictMode>,
);
