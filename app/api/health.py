from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session

router = APIRouter()


@router.get("/health")
async def health(session: AsyncSession = Depends(get_db_session)) -> dict:
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}
