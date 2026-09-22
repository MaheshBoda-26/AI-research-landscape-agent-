# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **monorepo** containing **ResolveX** — an autonomous customer resolution engine for action-heavy support workflows. The project handles scenarios like "I was charged twice and want to upgrade my plan" through a multi-agent system with deterministic autonomy gates.

### Repository Structure

```
Projects/
├── ResloveX/                    # Main project (autonomous support resolution)
│   ├── apps/
│   │   ├── api/                 # Fastify backend with Drizzle ORM
│   │   └── web/                 # React 19 + Vite 7 + Tailwind 4 frontend
│   ├── packages/
│   │   └── shared/              # Shared types, Zod schemas, constants
│   ├── tests/                   # Unit, integration, and evaluation tests
│   └── .claude/                 # Project-specific settings
├── Research-agent/              # Current working directory (empty)
└── [other projects]             # Various other projects in the workspace
```

The root `/Users/maheshboda/Projects/` is a git repository tracking multiple projects. `ResloveX` is the primary project in this workspace.

## Key Commands

### ResloveX Development Commands (run from `/Users/maheshboda/Projects/ResloveX/`)

```bash
# Install dependencies
npm install

# Development
npm run dev                    # Start both API (port 3001) and web (port 5173)
npm run dev:api                # API only
npm run dev:web                # Web only

# Database
npm run db:push                # Push schema changes (Drizzle)
npm run db:generate            # Generate migrations
npm run db:migrate             # Run migrations
npm run db:seed                # Seed demo data
npm run db:studio              # Open Drizzle Studio

# Build & Type-check
npm run build                  # Build all packages
npm run typecheck              # Type-check all workspaces

# Testing
npm run test                   # Run all tests (Vitest)
npm run test:unit              # Unit + integration tests
npm run test:e2e               # Playwright E2E tests
npm run eval:autonomy          # Autonomy gate evaluation (target: 90%+)

# Code Quality
npm run lint                   # ESLint all packages
npm run format                 # Prettier all packages
```

### Environment Setup

```bash
# API environment
cp apps/api/.env.example apps/api/.env
# Required: DATABASE_URL, FRESHWORKS_DOMAIN, FRESHWORKS_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID

# Web environment
cp apps/web/.env.example apps/web/.env
# Required: VITE_API_URL
```

## Architecture

### Monorepo Structure
- **Package Manager**: npm workspaces (also has pnpm-lock.yaml)
- **Shared Package**: `@resolvex/shared` - Single source of truth for types, schemas, constants
- **API**: `@resolvex/api` - Fastify 5, Drizzle ORM, PostgreSQL + pgvector
- **Web**: `@resolvex/web` - React 19, Vite 7, Tailwind 4, TanStack Query 5, shadcn/ui

### API Architecture (`apps/api/src/`)
```
src/
├── agents/           # Triage, Billing, Subscription agents
├── actions/          # Freshworks actions / MCP integrations
├── knowledge/        # Policy RAG with pgvector embeddings
├── verification/     # Autonomy gates, post-action verification
├── handoff/          # Human escalation with case briefs
├── traces/           # Agent run tracing & observability
├── evaluations/      # Autonomy gate evaluation suite
├── db/               # Drizzle schema, migrations, seed
├── routes/           # REST API + WebSocket endpoints
├── lib/              # Shared utilities, config
└── server/           # Fastify server entry point
```

### Key API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/conversations` | Create conversation |
| POST | `/api/triage` | Triage message → intents + tasks |
| POST | `/api/agent/process` | Full workflow (triage → specialists → verify) |
| GET | `/api/traces/:runId` | Get agent run trace |
| GET | `/api/handoffs` | List pending handoffs |

### Shared Package (`packages/shared/`)
Exports: types, schemas, constants, messaging
- `src/types` - TypeScript interfaces
- `src/schemas` - Zod validation schemas (all external boundaries)
- `src/constants` - Enums, constants
- `src/messaging` - Event bus for agent communication

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | TypeScript 5.x (strict mode) |
| Frontend | React 19, Vite 7, Tailwind CSS 4, TanStack Query 5, shadcn/ui |
| Backend | Fastify 5, Drizzle ORM, PostgreSQL 16+ with pgvector |
| Validation | Zod 4.x at all external boundaries |
| Testing | Vitest 3.x, Playwright 1.x |
| Observability | OpenTelemetry, custom agent traces |
| Voice | ElevenLabs Agents |
| CRM | Freshworks Agent Studio + AI Actions / MCP |
| Deployment | Vercel |

## Database Schema (PostgreSQL + pgvector)

Key tables:
- `customers` - Customer profiles & plans
- `transactions` - Billing records (duplicate detection)
- `subscriptions` - Plan subscriptions
- `conversations` - Chat/voice sessions
- `agent_runs` - Workflow execution records
- `tool_calls` - Freshworks/MCP action logs
- `verifications` - Post-action state checks
- `handoffs` - Human escalation queue
- `policy_embeddings` - pgvector for RAG

## Evaluation & Quality Gates

### Autonomy Gate Evaluation
```bash
npm run eval:autonomy
```
- Target: 90%+ accuracy
- Current: 100% (33/33 test cases)
- Tests deterministic rules controlling autonomous actions

### Test Structure
- `tests/unit/` - Unit tests for agents, verification, knowledge, handoff
- `tests/integration/` - Full workflow integration tests
- `tests/evaluation/` - Autonomy gate + E2E evaluation suites
- `apps/web/tests-e2e/` - Playwright E2E tests

## Development Patterns

### Adding Shared Types
1. Add types to `packages/shared/src/types/index.ts`
2. Add Zod schemas to `packages/shared/src/schemas/index.ts`
3. Export from `packages/shared/src/index.ts`
4. Run `npm run build` in shared package

### Database Changes
1. Modify `apps/api/src/db/schema.ts`
2. Run `npm run db:generate` to create migration
3. Run `npm run db:migrate` to apply

### Agent Development
- Agents live in `apps/api/src/agents/`
- Each agent: detects intent → retrieves context → checks autonomy gate → executes → verifies
- Autonomy gate in `apps/api/src/verification/autonomyGate.ts`

## Deployment

### Vercel (Recommended)
1. Push to GitHub
2. Import in Vercel
3. Configure environment variables
4. Deploy

### Production Environment Variables

**API** (`apps/api/.env`):
```
DATABASE_URL=postgresql://...
FRESHWORKS_DOMAIN=...
FRESHWORKS_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_AGENT_ID=...
NODE_ENV=production
```

**Web** (`apps/web/.env`):
```
VITE_API_URL=https://your-api.vercel.app
```

## Demo Flow
1. Open `/chat` in web app
2. Send: "I was charged twice and want to upgrade my plan"
3. Watch triage → billing + subscription tasks
4. View trace at `/trace?runId=...`
5. High-value refund ($600) → triggers handoff at `/handoffs`

## Important Notes

- **Node.js 20+** required (engines field in package.json)
- **PostgreSQL 16+ with pgvector** extension required
- **Freshworks & ElevenLabs credentials** never enter the browser (API-only)
- **Strict TypeScript** across all packages with path aliases configured
- **Zod validation** at all external boundaries (API input, web forms, DB)
- **Autonomy gates** are deterministic — no LLM judgment for action authorization
- **Agent traces** provide full observability for debugging and evaluation

## Current Git State
The root `/Users/maheshboda/Projects/` has an in-progress interactive rebase on main branch. The working directory has uncommitted changes in ResloveX (modified and deleted files per git status).