# Project Structure Guide

This repository now uses a clearer physical layout with a single canonical path set.

## Recommended Entry Points

```text
Smart_vision/
├── apps/
│   ├── api
│   ├── web
│   └── demo
├── packages/
│   └── model
├── data/
│   └── raw
└── docs/
    └── PROJECT_STRUCTURE.md
```

## Why This Layout

- `apps/`: deployable/runtime applications (API, Web, Demo)
- `packages/`: reusable code packages (model/search pipeline)
- `data/`: datasets and local data assets
- `docs/`: reports, release notes, architecture docs

## Canonical Paths

Use only these paths in scripts/docs/import guides:

- `apps/web`
- `apps/api`
- `apps/demo`
- `packages/model`
- `data/raw`
