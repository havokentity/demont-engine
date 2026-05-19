// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import type { ReactNode } from 'react';

export interface ToolbarProps {
  children?: ReactNode;
}

export interface ToolbarGroupProps {
  children?: ReactNode;
}

export interface ToolbarButtonProps {
  label: string;
  onClick?: () => void;
  active?: boolean;
  disabled?: boolean;
  title?: string;
}

export function Toolbar({ children }: ToolbarProps) {
  return <div className="editor-toolbar">{children}</div>;
}

export function ToolbarGroup({ children }: ToolbarGroupProps) {
  return <div className="editor-toolbar-group">{children}</div>;
}

export function ToolbarButton({
  label,
  onClick,
  active = false,
  disabled = false,
  title,
}: ToolbarButtonProps) {
  return (
    <button
      type="button"
      className={`editor-toolbar-button${active ? ' is-active' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={title ?? label}
    >
      {label}
    </button>
  );
}
