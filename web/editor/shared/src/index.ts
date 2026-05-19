// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Public surface of @demont/editor-shared. Every editor panel under
// web/editor/panels/* imports its types, WebSocket client, store,
// and layout primitives through this barrel.

export type {
  AnalyticPrim,
  AnalyticLight,
  SceneObject,
  SelectionKind,
  Selection,
  SceneSnapshot,
  WsEvent,
  WsResult,
  WsInbound,
  WsOutbound,
  CVarValue,
  CommandReply,
} from './types';

export { MaterialName, PrimTypeName, LightTypeName } from './types';

export { WebSocketClient, type ClientStatus } from './ws-client';
export { useSceneStore, type SceneStore } from './store';

export { Shell } from './components/Shell';
export { TitleBar } from './components/TitleBar';
export { Toolbar, ToolbarGroup, ToolbarButton } from './components/Toolbar';
export { StatusBar } from './components/StatusBar';
export { Placeholder } from './components/Placeholder';

export { wsEndpoint } from './endpoint';
