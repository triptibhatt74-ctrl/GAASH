-- GAASH Phase 2: additive, provenance-bearing longitudinal wellbeing events.
-- Run after Phase 1 and only after the preflight succeeds. This migration does
-- not alter or recreate existing user, conversation, assessment, or safety data.

BEGIN;

CREATE TABLE IF NOT EXISTS wellbeing_events (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    case_id TEXT REFERENCES case_contexts(id) ON DELETE SET NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT NOT NULL CHECK (source_type IN ('text', 'voice', 'assessment', 'engagement', 'check_in', 'safety')),
    structured_signals JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    source_reference TEXT,
    conversation_id TEXT,
    assessment_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_wellbeing_events_owner_time
    ON wellbeing_events (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_wellbeing_events_case_time
    ON wellbeing_events (case_id, timestamp DESC) WHERE case_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wellbeing_events_conversation_time
    ON wellbeing_events (user_id, conversation_id, timestamp DESC) WHERE conversation_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS dynamic_distress_states (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    case_id TEXT REFERENCES case_contexts(id) ON DELETE SET NULL,
    event_id BIGINT REFERENCES wellbeing_events(id) ON DELETE SET NULL,
    state TEXT NOT NULL CHECK (state IN ('unknown', 'stable', 'watch', 'elevated', 'high', 'critical')),
    trajectory TEXT NOT NULL CHECK (trajectory IN ('improving', 'stable', 'worsening', 'unknown')),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    contributing_indicators JSONB NOT NULL DEFAULT '[]'::jsonb,
    requires_human_review BOOLEAN NOT NULL DEFAULT FALSE,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dynamic_distress_owner_time
    ON dynamic_distress_states (user_id, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_dynamic_distress_human_review
    ON dynamic_distress_states (requires_human_review, computed_at DESC) WHERE requires_human_review;

CREATE TABLE IF NOT EXISTS monitoring_alerts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    case_id TEXT REFERENCES case_contexts(id) ON DELETE SET NULL,
    distress_state_id BIGINT REFERENCES dynamic_distress_states(id) ON DELETE SET NULL,
    state TEXT NOT NULL CHECK (state IN ('watch', 'elevated', 'high', 'critical')),
    reason TEXT NOT NULL,
    requires_human_review BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acknowledged', 'closed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_owner_status_time
    ON monitoring_alerts (user_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS monitoring_schedules (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    case_id TEXT REFERENCES case_contexts(id) ON DELETE SET NULL,
    cadence_days INTEGER NOT NULL DEFAULT 14 CHECK (cadence_days BETWEEN 7 AND 90),
    next_check_in_due TIMESTAMPTZ,
    missed_check_in BOOLEAN NOT NULL DEFAULT FALSE,
    last_evaluated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMIT;
