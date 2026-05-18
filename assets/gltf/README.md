# glTF sample assets

Test fixtures for the glTF 2.0 importer (`src/renderer/GltfImporter.cpp`).
Each asset retains its original upstream license; the importer code
itself is MIT (project licence).

## Box.gltf, Box0.bin, Box.glb

A unit cube with normals and a single PBR material (red base color
factor, no texture). Useful as the smallest possible parse + render
smoke test. Shipped in both forms so the importer's text-glTF path
(`Box.gltf` + sidecar `Box0.bin`) and binary-glTF path (`Box.glb` with
embedded buffer) both have a fixture.

- Source: <https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/Box>
- Donated by: [Cesium](http://cesiumjs.org/) for glTF testing
- Licence: [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)

Console usage:

```
mesh_load_gltf assets/gltf/Box.gltf
mesh_load_gltf assets/gltf/Box.glb
```
