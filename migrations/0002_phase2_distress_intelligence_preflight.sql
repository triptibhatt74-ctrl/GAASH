-- Phase 2 preflight. Read-only; run before 0002_phase2_distress_intelligence.sql.
-- Requires successful Phase 1 migration 0001_phase1_foundation.sql.

SELECT to_regclass('public.users') AS users_table,
       to_regclass('public.case_contexts') AS case_contexts_table,
       to_regclass('public.conversations') AS conversations_table,
       to_regclass('public.screening_measurements') AS screening_measurements_table;

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('users', 'case_contexts', 'conversations', 'screening_measurements')
ORDER BY table_name, ordinal_position;
