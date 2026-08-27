from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.models import Resume


def test_resume_table_uses_postgresql_compatible_text_type():
    ddl = str(CreateTable(Resume.__table__).compile(dialect=postgresql.dialect()))

    assert "original_text TEXT NOT NULL" in ddl
    assert "TEXT(" not in ddl
