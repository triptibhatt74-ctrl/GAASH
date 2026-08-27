# Phase 3 rollback strategy

Do not drop Phase 3 tables as an automatic rollback. They contain human review
and audit evidence that must be retained according to the deployment's legal
and data-retention policy.

If the Phase 3 backend must be rolled back:

1. Stop routing staff users to the support portal and deploy the previous
   backend version.
2. Keep migration `0003` in place; all changes are additive and previous
   user-facing GAASH routes remain compatible.
3. Revoke active staff case authorizations and report shares through an
   authorized DBA procedure if a security incident requires immediate access
   removal. Do not delete audit events.
4. Only consider a separately approved, retention-aware data migration after a
   full backup and legal/security review. Never run `DROP TABLE` as incident
   response.

