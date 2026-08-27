"""Idempotent migration for browser/device-isolated screening data."""

from sqlalchemy import inspect, text

from app.utils.device import LEGACY_DEVICE_ID


DEVICE_OWNED_TABLES = (
    'jobs',
    'screening_criteria',
    'applicants',
    'resumes',
    'extracted_skills',
    'extracted_education',
    'extracted_experience',
    'extracted_certifications',
    'screening_results',
    'recommendation_logs',
)


def _usable_device_for_job(connection, columns, job_id, created_by):
    """Infer a legacy job owner from its oldest device-owned candidate data."""
    direct_sources = (
        ('resumes', 'job_id'),
        ('screening_results', 'job_id'),
        ('applicants', 'applied_job_id'),
    )
    for table, job_column in direct_sources:
        table_columns = columns.get(table, set())
        if not {'id', 'device_id', job_column}.issubset(table_columns):
            continue
        device_id = connection.execute(text(
            f'SELECT device_id FROM {table} '
            f'WHERE {job_column} = :job_id '
            "AND device_id IS NOT NULL AND device_id != '' "
            'AND device_id != :legacy_device_id '
            'ORDER BY id LIMIT 1'
        ), {
            'job_id': job_id,
            'legacy_device_id': LEGACY_DEVICE_ID,
        }).scalar()
        if device_id:
            return device_id

    # A job without candidates can still be attributed when the creator has
    # uploaded a resume elsewhere. This preserves the usual single-browser
    # thesis database while refusing to guess when no evidence exists.
    resume_columns = columns.get('resumes', set())
    if created_by is not None and {'id', 'device_id', 'uploaded_by'}.issubset(resume_columns):
        return connection.execute(text(
            'SELECT device_id FROM resumes '
            'WHERE uploaded_by = :created_by '
            "AND device_id IS NOT NULL AND device_id != '' "
            'AND device_id != :legacy_device_id '
            'ORDER BY id LIMIT 1'
        ), {
            'created_by': created_by,
            'legacy_device_id': LEGACY_DEVICE_ID,
        }).scalar()
    return None


def migrate_device_isolation(engine):
    """Add, backfill, and index device ownership columns.

    Existing jobs are attributed from their oldest device-owned candidate data
    when possible. Records that cannot be attributed reliably are assigned a
    reserved identifier that the application never issues, quarantining them
    from every browser.
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
        # Add every ownership column and index before attempting relationship-
        # based backfills.
        for table in DEVICE_OWNED_TABLES:
            if table not in existing_tables:
                continue

            if 'device_id' not in existing_columns.get(table, set()):
                connection.execute(text(
                    f'ALTER TABLE {table} ADD COLUMN device_id VARCHAR(64)'
                ))
                changes.append(f'added {table}.device_id')

            index_name = f'ix_{table}_device_id'
            if index_name not in existing_indexes.get(table, set()):
                connection.execute(text(
                    f'CREATE INDEX {index_name} ON {table} (device_id)'
                ))
                changes.append(f'created {index_name}')

        refreshed = inspect(connection)
        current_columns = {
            table: {column['name'] for column in refreshed.get_columns(table)}
            for table in DEVICE_OWNED_TABLES
            if table in existing_tables
        }

        if 'jobs' in existing_tables and 'device_id' in current_columns.get('jobs', set()):
            job_columns = current_columns['jobs']
            selected_columns = 'id, created_by' if 'created_by' in job_columns else 'id, NULL'
            unowned_jobs = connection.execute(text(
                f'SELECT {selected_columns} FROM jobs '
                "WHERE device_id IS NULL OR device_id = ''"
            )).all()
            for job_id, created_by in unowned_jobs:
                inferred_device_id = _usable_device_for_job(
                    connection,
                    current_columns,
                    job_id,
                    created_by,
                )
                if inferred_device_id:
                    connection.execute(text(
                        'UPDATE jobs SET device_id = :device_id WHERE id = :job_id'
                    ), {'device_id': inferred_device_id, 'job_id': job_id})

        if (
            'screening_criteria' in existing_tables
            and {'device_id', 'job_id'}.issubset(current_columns.get('screening_criteria', set()))
            and {'id', 'device_id'}.issubset(current_columns.get('jobs', set()))
        ):
            connection.execute(text(
                'UPDATE screening_criteria SET device_id = ('
                'SELECT jobs.device_id FROM jobs '
                'WHERE jobs.id = screening_criteria.job_id'
                ') WHERE device_id IS NULL OR device_id = \'\''
            ))

        for table in DEVICE_OWNED_TABLES:
            if table not in existing_tables:
                continue
            connection.execute(text(
                f'UPDATE {table} SET device_id = :legacy_device_id '
                "WHERE device_id IS NULL OR device_id = ''"
            ), {'legacy_device_id': LEGACY_DEVICE_ID})

    return changes
