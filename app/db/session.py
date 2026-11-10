from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base


class Database:
    """Owns the SQLAlchemy async engine and session factory."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        is_sqlite = url.startswith("sqlite")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        self.engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            future=True,
            pool_pre_ping=not is_sqlite,
            connect_args=connect_args,
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session
