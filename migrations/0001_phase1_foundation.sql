-- GAASH Phase 1: national geography, case context, durable memory provenance,
-- role foundation, and India-wide resource fields.
--
-- This migration is additive and transactional.  It does not recreate or wipe
-- existing tables.  Run it once with psql -v ON_ERROR_STOP=1 after preflight.

BEGIN;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'users_role_check') THEN
        ALTER TABLE users
            ADD CONSTRAINT users_role_check
            CHECK (role IN ('user', 'counsellor', 'authorized_official', 'admin'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_users_role ON users (role);

CREATE TABLE IF NOT EXISTS case_contexts (
    id TEXT PRIMARY KEY,
    public_case_reference TEXT NOT NULL UNIQUE,
    user_id BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    external_reference_id TEXT,
    state_ut TEXT NOT NULL,
    district TEXT,
    case_stage TEXT,
    monitoring_status TEXT NOT NULL DEFAULT 'not_enrolled'
        CHECK (monitoring_status IN ('not_enrolled', 'active', 'paused', 'closed')),
    preferred_language TEXT,
    consent_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (consent_status IN ('pending', 'granted', 'withdrawn')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT case_contexts_state_ut_check CHECK (state_ut ~ '^IN-[A-Z]{2}$')
);

CREATE INDEX IF NOT EXISTS idx_case_contexts_state_district
    ON case_contexts (state_ut, district);
CREATE INDEX IF NOT EXISTS idx_case_contexts_monitoring_status
    ON case_contexts (monitoring_status);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS preferred_language TEXT,
    ADD COLUMN IF NOT EXISTS last_message_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS metadata_updated_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_conversations_user_last_message
    ON conversations (user_id, last_message_at DESC NULLS LAST);

ALTER TABLE user_memories
    ADD COLUMN IF NOT EXISTS source_message_id BIGINT,
    ADD COLUMN IF NOT EXISTS source_kind TEXT NOT NULL DEFAULT 'conversation',
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'user_memories_source_kind_check') THEN
        ALTER TABLE user_memories
            ADD CONSTRAINT user_memories_source_kind_check
            CHECK (source_kind IN ('conversation', 'assessment', 'profile', 'user-confirmed'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'user_memories_status_check') THEN
        ALTER TABLE user_memories
            ADD CONSTRAINT user_memories_status_check
            CHECK (status IN ('active', 'superseded', 'withdrawn'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_user_memories_owner_status_updated
    ON user_memories (user_id, status, updated_at DESC);

ALTER TABLE resources
    ADD COLUMN IF NOT EXISTS category TEXT,
    ADD COLUMN IF NOT EXISTS state_ut TEXT,
    ADD COLUMN IF NOT EXISTS city TEXT,
    ADD COLUMN IF NOT EXISTS description TEXT,
    ADD COLUMN IF NOT EXISTS access_mode TEXT,
    ADD COLUMN IF NOT EXISTS eligibility TEXT,
    ADD COLUMN IF NOT EXISTS last_verified DATE,
    ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_resources_state_ut_district
    ON resources (state_ut, district);
CREATE INDEX IF NOT EXISTS idx_resources_category
    ON resources (category);

COMMIT;

