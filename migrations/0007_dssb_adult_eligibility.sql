BEGIN;

-- Auth retains ownership of users; this profile extension is additive.
DO $$
BEGIN
    IF to_regclass('user_profiles') IS NOT NULL THEN
        ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS date_of_birth DATE;
    END IF;
END $$;

COMMIT;
