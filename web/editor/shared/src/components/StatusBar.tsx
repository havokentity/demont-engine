// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import type { ReactNode } from 'react';

export interface StatusBarProps {
  children?: ReactNode;
}

export interface StatusCellProps {
  label: string;
  value: ReactNode;
}

export function StatusBar({ children }: StatusBarProps) {
  return <footer className="editor-statusbar">{children}</footer>;
}

export function StatusCell({ label, value }: StatusCellProps) {
  return (
    <span className="status-cell">
      <span>{label}:</span>
      <strong>{value}</strong>
    </span>
  );
}
