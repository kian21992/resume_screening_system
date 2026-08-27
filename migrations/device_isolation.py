"""Idempotent migration for browser/device-isolated resume data."""

from sqlalchemy import inspect, text

from app.utils.device import LEGACY_DEVICE_ID


DEVICE_OWNED_TABLES = (
    'applicants',
    'resumes',
    'extracted_skills',
    'extracted_education',
    'extracted_experience',
    'extracted_certifications',
    'screening_results',
    'recommendation_logs',
)


def migrate_device_isolation(engine):
    """Add/backfill/index device_id on all resume-derived tables.

    Existing records cannot be attributed to a historical browser reliably.
    They are assigned a reserved identifier that the application will never
    issue, which quarantines them from every browser after the migration.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    existing_columns = {
        table: {column['name'] for column in inspector.get_columns(table)}
        for table in DEVICE_OWNED_TABLES
        if table in existing_tables
    }
    existing_indexes = {
        table: {index['name'] for index in inspector.get_indexes(table)}
        for table in DEVICE_OWNED_TABLES
        if table in existing_tables
    }
    changes = []

    with engine.begin() as connection:
        for table in DEVICE_OWNED_TABLES:
            if table not in existing_tables:
                continue

            if 'device_id' not in existing_columns.get(table, set()):
                connection.execute(text(
                    f'ALTER TABLE {table} ADD COLUMN device_id VARCHAR(64)'
                ))
                changes.append(f'added {table}.device_id')

            connection.execute(
                text(
                    f'UPDATE {table} SET device_id = :legacy_device_id '
                    "WHERE device_id IS NULL OR device_id = ''"
                ),
                {'legacy_device_id': LEGACY_DEVICE_ID},
            )

            index_name = f'ix_{table}_device_id'
            if index_name not in existing_indexes.get(table, set()):
                connection.execute(text(
                    f'CREATE INDEX {index_name} ON {table} (device_id)'
                ))
                changes.append(f'created {index_name}')

    return changes
