# AGENTS.md

IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning.

## Global Rules

All AI agents working in this repository MUST follow:

- .ai/postgres-sql-style-guide.md
- .ai/database-rls-policies.md

## Database Work

When:
- Writing SQL
- Creating migrations
- Designing tables
- Writing RLS
- Generating Supabase queries

You MUST:
1. Follow naming and formatting rules from `postgres-sql-style-guide.md`
2. Follow security and indexing requirements from `database-rls-policies.md`

## Priority

If generated code conflicts with these documents:
- Ask for clarification.