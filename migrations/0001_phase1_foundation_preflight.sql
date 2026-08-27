-- Run against a non-production clone or staging database before 0001.
-- This file is read-only and intentionally contains no application content.

SELECT current_database() AS database_name, current_user AS migration_user;
SHOW server_version;

SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('users', 'user_profiles', 'conversations', 'conversation_summaries', 'user_memories', 'resources')
ORDER BY table_name;

SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'users'
ORDER BY ordinal_position;

SELECT conname, pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'public.users'::regclass
ORDER BY conname;

