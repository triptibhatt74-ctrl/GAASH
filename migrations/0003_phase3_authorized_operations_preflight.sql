-- GAASH Phase 3 preflight (read-only).
-- Run after Phase 1 and Phase 2 migrations, against staging first.

SELECT to_regclass('public.users') AS users_table,
       to_regclass('public.case_contexts') AS case_contexts_table,
       to_regclass('public.dynamic_distress_states') AS distress_states_table,
       to_regclass('public.monitoring_alerts') AS monitoring_alerts_table,
       to_regclass('public.screening_measurements') AS screening_measurements_table;

SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('users', 'case_contexts', 'dynamic_distress_states', 'monitoring_alerts', 'screening_measurements')
ORDER BY table_name, ordinal_position;

SELECT conname, pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid IN (
    'public.users'::regclass,
    'public.case_contexts'::regclass,
    'public.monitoring_alerts'::regclass
)
ORDER BY conname;

