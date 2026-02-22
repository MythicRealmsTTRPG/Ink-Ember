# Ink & Ember - Product Requirements Document

## Original Problem Statement
Build a World Anvil alternative for Archivists, Dungeon Masters and World Builders. A worldbuilding and narrative management platform for writers and tabletop creators who work on long-term, complex projects.

## Philosophy
- Wiki-first presentation - readable, browsable, linkable
- Structured under the hood - typed, field-driven entries
- Creator ownership - exportability and long-term access
- Scales with complexity - single campaign to decades of history

## User Personas
1. **Tabletop Game Masters** - Running long campaigns, need session tracking
2. **Writers** - Working on series, sagas, shared universes
3. **Worldbuilders** - Maintaining large, interconnected settings
4. **Archivists** - Documenting complex lore systems

## Core Requirements
- No AI integration (user-created content)
- Offline-first with optional online collaboration
- 4 selectable design themes
- Responsive design (Desktop, Android, iOS)
- Export/import for data portability

## What's Been Implemented (Feb 2026)

### Backend (FastAPI + MongoDB)
- World management with full CRUD
- Articles system with 28 article types
- Timelines with events
- Custom calendars
- Chronicles with entries
- Maps with markers
- Family trees with members and connections
- Variables (world states, canon layers)
- Notebooks
- Todo tracking
- Diplomatic relations
- World statistics

### Frontend (React + Tailwind + Shadcn UI)
- 4 Themes: Ink & Ember, Archivist, Void State, Feywild
- Landing page with brand identity
- World selector with create/delete
- Dashboard with stats and quick actions
- Full articles system (create, edit, view, delete)
- Wiki-style article viewing with infoboxes
- Timeline management with visual events
- Calendar system
- Chronicles with entry management
- Maps with clickable markers
- Family trees with relationship management
- Variables management
- Notebooks with auto-save
- Todo tracking with priorities
- Settings page with theme switcher and export

## Prioritized Backlog

### P0 (Critical)
- [x] World creation and selection
- [x] Article CRUD with 28 types
- [x] Timelines and events
- [x] Theme system (4 themes)

### P1 (Important)
- [x] Chronicles and entries
- [x] Maps with markers
- [x] Family trees
- [x] Variables
- [x] Notebooks
- [x] Todos

### P2 (Nice to Have)
- [ ] Rich text editor for articles
- [ ] Image upload support
- [ ] Article linking UI (cross-references)
- [ ] Search across all content
- [ ] Whiteboard canvas feature
- [ ] RPG Character Sheet Designer
- [ ] Manuscripts module
- [ ] Calendar events integration

### P3 (Future)
- [ ] Offline-first with IndexedDB
- [ ] Sync mechanism for online collaboration
- [ ] Import from World Anvil
- [ ] PDF export
- [ ] Mobile app (React Native)

## Next Tasks
1. Add rich text editor (Tiptap or Slate) for article content
2. Implement image upload with S3/storage
3. Build article cross-linking UI
4. Add global search functionality
5. Implement whiteboard canvas
