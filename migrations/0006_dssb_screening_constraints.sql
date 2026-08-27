BEGIN;

-- Non-destructively replace only the pre-DSS-B scale checks. Re-running this
-- migration is safe because the matching constraint is removed before add.
DO $$
DECLARE
    constraint_name TEXT;
BEGIN
    IF to_regclass('screening_sessions') IS NOT NULL THEN
        SELECT conname INTO constraint_name
        FROM pg_constraint
        WHERE conrelid = to_regclass('screening_sessions')
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%scale%'
          AND pg_get_constraintdef(oid) LIKE '%PHQ-9%'
          AND pg_get_constraintdef(oid) LIKE '%PSS-10%'
        LIMIT 1;
        IF constraint_name IS NOT NULL THEN
            EXECUTE format('ALTER TABLE screening_sessions DROP CONSTRAINT %I', constraint_name);
        END IF;
        ALTER TABLE screening_sessions
            ADD CONSTRAINT screening_sessions_scale_values_check
            CHECK (scale IN ('PHQ-9', 'GAD-7', 'PSS-10', 'DSS-B'));
    END IF;

    IF to_regclass('screening_measurements') IS NOT NULL THEN
        constraint_name := NULL;
        SELECT conname INTO constraint_name
        FROM pg_constraint
        WHERE conrelid = to_regclass('screening_measurements')
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%assessment_type%'
          AND pg_get_constraintdef(oid) LIKE '%PHQ-9%'
          AND pg_get_constraintdef(oid) LIKE '%PSS-10%'
        LIMIT 1;
        IF constraint_name IS NOT NULL THEN
            EXECUTE format('ALTER TABLE screening_measurements DROP CONSTRAINT %I', constraint_name);
        END IF;
        ALTER TABLE screening_measurements
            ADD CONSTRAINT screening_measurements_type_values_check
            CHECK (assessment_type IN ('PHQ-9', 'GAD-7', 'PSS-10', 'DSS-B'));
    END IF;
END $$;

COMMIT;
