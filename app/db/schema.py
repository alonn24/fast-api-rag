from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


async def apply_schema(engine: AsyncEngine) -> None:
    sql = SCHEMA_SQL_PATH.read_text()
    async with engine.begin() as conn:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                await conn.execute(text(statement))
