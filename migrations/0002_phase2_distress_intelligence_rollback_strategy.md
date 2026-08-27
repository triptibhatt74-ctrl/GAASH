# Phase 2 rollback strategy

Do not run destructive rollback automatically.  If a deployment rollback is
needed, first stop the Phase 2 backend release and restore the Phase 1 backend
application files. The Phase 2 tables are additive and can be retained safely
while the data is reviewed. Any eventual table drop must be approved after a
backup and after confirming that no later release references the event,
distress, alert, or monitoring tables.
