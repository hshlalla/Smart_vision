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
    ├── architecture/
    ├── planning/
    ├── reports/
    └── release_notes/
```

## Why This Layout

- `apps/`: deployable/runtime applications (API, Web, Demo)
- `packages/`: reusable code packages (model/search pipeline)
- `data/`: datasets and local data assets
- `docs/architecture/`: architecture and structure guides
- `docs/planning/`: active plans and backlog
- `docs/reports/`: internal working notes for report writing
- `docs/release_notes/`: app/model/demo release notes

## Canonical Paths

Use only these paths in scripts/docs/import guides:

- `apps/web`
- `apps/api`
- `apps/demo`
- `packages/model`
- `data/raw`
