-- GAASH Phase 3: explicit support authorization, human alert review,
-- interventions, privacy-preserving operational aggregates and immutable audit
-- events.  Additive only; never recreates or alters private chat content.
--
-- Prerequisites: successful Phase 1 and Phase 2 migrations.

BEGIN;

CREATE TABLE IF NOT EXISTS case_authorizations (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES case_contexts(id) ON DELETE CASCADE,
    staff_user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    permission TEXT NOT NULL CHECK (permission IN ('case_summary', 'case_manage', 'intervention_update')),
    granted_by BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT case_authorizations_unique_grant UNIQUE (case_id, staff_user_id, permission)
);
CREATE INDEX IF NOT EXISTS idx_case_authorizations_staff_active
    ON case_authorizations (staff_user_id, case_id)
    WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS case_assignments (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES case_contexts(id) ON DELETE CASCADE,
    assigned_to_user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    assigned_by BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'reassigned', 'closed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_case_assignments_active_case
    ON case_assignments (case_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_case_assignments_assignee_active
    ON case_assignments (assigned_to_user_id, updated_at DESC) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS alert_reviews (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT NOT NULL REFERENCES monitoring_alerts(id) ON DELETE CASCADE,
    case_id TEXT REFERENCES case_contexts(id) ON DELETE SET NULL,
    reviewer_user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    decision TEXT NOT NULL CHECK (decision IN ('confirmed', 'not_concerning', 'needs_follow_up', 'false_positive')),
    note TEXT,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_alert_reviews_alert_time
    ON alert_reviews (alert_id, reviewed_at DESC);

CREATE TABLE IF NOT EXISTS intervention_records (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES case_contexts(id) ON DELETE CASCADE,
    intervention_type TEXT NOT NULL CHECK (intervention_type IN (
        'counselling', 'mental_health_referral', 'medical_referral', 'legal_aid',
        'witness_protection_support', 'rehabilitation', 'financial_compensation_support',
        'relocation_support_services', 'emergency_assistance'
    )),
    status TEXT NOT NULL DEFAULT 'planned' CHECK (status IN ('planned', 'in_progress', 'completed', 'cancelled')),
    created_by BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    assigned_to BIGINT REFERENCES users(id) ON DELETE SET NULL,
    notes TEXT,
    outcome TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_intervention_records_case_time
    ON intervention_records (case_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS case_notes (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES case_contexts(id) ON DELETE CASCADE,
    created_by BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    note TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_case_notes_case_time ON case_notes (case_id, created_at DESC);

-- Geography scopes authorize aggregates only.  They grant no access to
-- individual case details or private content.
CREATE TABLE IF NOT EXISTS official_geography_scopes (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scope_level TEXT NOT NULL CHECK (scope_level IN ('district', 'state_ut', 'national')),
    state_ut TEXT,
    district TEXT,
    granted_by BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT official_geography_scope_shape CHECK (
        (scope_level = 'district' AND state_ut IS NOT NULL AND district IS NOT NULL)
        OR (scope_level = 'state_ut' AND state_ut IS NOT NULL AND district IS NULL)
        OR (scope_level = 'national' AND state_ut IS NULL AND district IS NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_official_geography_scopes_active
    ON official_geography_scopes (user_id, scope_level, state_ut, district)
    WHERE revoked_at IS NULL;

-- Stores only linkage metadata for future verified external-case adapters.
-- It deliberately has no live government connector or credentials in this app.
CREATE TABLE IF NOT EXISTS external_case_links (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES case_contexts(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('nhaa', 'integrated_portal', 'manual_demo')),
    external_case_id TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    sync_status TEXT NOT NULL DEFAULT 'configuration_required' CHECK (sync_status IN ('configuration_required', 'pending', 'synced', 'failed', 'disabled')),
    last_synced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT external_case_links_source_unique UNIQUE (source, external_case_id)
);
CREATE INDEX IF NOT EXISTS idx_external_case_links_case ON external_case_links (case_id);

-- No sharing endpoint is exposed until verified recipient-possession delivery
-- is configured.  Tokens are stored as hashes; raw tokens never enter audit logs.
CREATE TABLE IF NOT EXISTS report_shares (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES case_contexts(id) ON DELETE CASCADE,
    created_by BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    recipient_email_hash TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL DEFAULT 'wellbeing_report' CHECK (scope IN ('wellbeing_report')),
    consent_recorded_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_report_shares_case_active
    ON report_shares (case_id, expires_at) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS audit_events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL UNIQUE,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actor_user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    case_id TEXT REFERENCES case_contexts(id) ON DELETE SET NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    previous_digest TEXT,
    event_digest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_events_case_time ON audit_events (case_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_actor_time ON audit_events (actor_user_id, occurred_at DESC);

CREATE OR REPLACE FUNCTION prevent_audit_event_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'audit_events are append-only';
END;
$$;

DROP TRIGGER IF EXISTS audit_events_no_update ON audit_events;
CREATE TRIGGER audit_events_no_update
    BEFORE UPDATE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_event_mutation();

DROP TRIGGER IF EXISTS audit_events_no_delete ON audit_events;
CREATE TRIGGER audit_events_no_delete
    BEFORE DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_event_mutation();

COMMIT;

