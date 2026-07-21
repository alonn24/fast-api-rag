import re
from pathlib import Path

from app.config import get_settings
from app.db.models import EMBEDDING_DIM

SCHEMA_SQL_PATH = Path(__file__).parent.parent / "app" / "db" / "schema.sql"


def test_settings_embedding_dim_matches_models_constant():
    assert get_settings().embedding_dim == EMBEDDING_DIM


def test_schema_sql_vector_dimension_matches_models_constant():
    sql = SCHEMA_SQL_PATH.read_text()
    match = re.search(r"embedding\s+VECTOR\((\d+)\)", sql)
    assert match is not None, "schema.sql must declare 'embedding VECTOR(<dim>)'"
    assert int(match.group(1)) == EMBEDDING_DIM
