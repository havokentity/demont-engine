// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import type { ReactNode } from 'react';

export interface PlaceholderProps {
  title: string;
  agent?: string;
  description?: ReactNode;
}

// Standard "agent-XX ships this" placeholder. Downstream agents
// replace the inner content while keeping the shell's titlebar /
// toolbar / statusbar in place.

export function Placeholder({ title, agent, description }: PlaceholderProps) {
  return (
    <div className="editor-placeholder">
      <h3>{title}</h3>
      {agent && (
        <p>
          <code>{agent}</code> ships this panel's content.
        </p>
      )}
      {description && <p>{description}</p>}
    </div>
  );
}
