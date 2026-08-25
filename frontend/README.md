# AI Photo Enhancer Frontend

This directory contains the React and TypeScript frontend for AI Photo Enhancer. It provides the upload workspace, single-image comparison view, batch processing controls, and restoration settings that connect to the local FastAPI backend.

## Development

From this directory:

```bash
npm install
npm run dev
```

The Vite development server runs at `http://localhost:5173` and expects the backend at `http://localhost:8000`.

## Production build

```bash
npm run build
```

The generated `dist/` directory is a local build artifact and is excluded from version control.

## Linting

```bash
npm run lint
```

## Technical notes

The frontend uses React, TypeScript, Vite, Lucide icons, and Oxlint. The React Compiler is currently not enabled.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
