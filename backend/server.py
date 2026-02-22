from __future__ import annotations

import io
import json
import logging
import os
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query, UploadFile, File, Request  # noqa: F401 (kept for parity)
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse  # noqa: F401 (JSONResponse kept for parity)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Integer, String, Text, select, delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from starlette.middleware.cors import CORSMiddleware

# =========================
# Env / Logging
# =========================

ROOT_DIR = Path(__file__).resolve().parent
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ink-ember-backend")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def loads(s: str, default: Any):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default


def parse_cors(raw: str | None) -> tuple[list[str], bool]:
    """
    Starlette rule: allow_credentials cannot be True if allow_origins contains '*'.
    """
    if not raw:
        return (["*"], False)
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins or "*" in origins:
        return (["*"], False)
    return (origins, True)


# =========================
# SQLAlchemy (SQLite async)
# =========================

class Base(DeclarativeBase):
    pass


# ---- Tables ----

class WorldTable(Base):
    __tablename__ = "worlds"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    genre: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    cover_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    settings_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)


class ArticleTable(Base):
    __tablename__ = "articles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    article_type: Mapped[str] = mapped_column(String(64), nullable=False, default="generic")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cover_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    infobox_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    linked_articles_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    is_secret: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    custom_fields_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)


class TimelineTable(Base):
    __tablename__ = "timelines"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    color: Mapped[str] = mapped_column(String(20), nullable=False, default="#ff4500")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class TimelineEventTable(Base):
    __tablename__ = "timeline_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    timeline_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    date_label: Mapped[str] = mapped_column(String(120), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    linked_articles_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    color: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class CalendarTable(Base):
    __tablename__ = "calendars"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    months_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    days_per_week: Mapped[int] = mapped_column(Integer, nullable=False, default=7)
    day_names_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class ChronicleTable(Base):
    __tablename__ = "chronicles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chronicle_type: Mapped[str] = mapped_column(String(60), nullable=False, default="campaign_log")
    entries_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    linked_timeline_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)


class MapTable(Base):
    __tablename__ = "maps"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    markers_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class FamilyTreeTable(Base):
    __tablename__ = "family_trees"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    nodes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    connections_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class VariableTable(Base):
    __tablename__ = "variables"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    variable_type: Mapped[str] = mapped_column(String(60), nullable=False, default="world_state")
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    options_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class NotebookTable(Base):
    __tablename__ = "notebooks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    notebook_type: Mapped[str] = mapped_column(String(60), nullable=False, default="note")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)


class TodoTable(Base):
    __tablename__ = "todos"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    priority: Mapped[str] = mapped_column(String(30), nullable=False, default="medium")
    due_date: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class DiplomaticRelationTable(Base):
    __tablename__ = "diplomatic_relations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    entity1_id: Mapped[str] = mapped_column(String(36), nullable=False)
    entity1_name: Mapped[str] = mapped_column(String(200), nullable=False)
    entity2_id: Mapped[str] = mapped_column(String(36), nullable=False)
    entity2_name: Mapped[str] = mapped_column(String(200), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(60), nullable=False, default="neutral")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


engine: Optional[AsyncEngine] = None
SessionLocal: Optional[async_sessionmaker[AsyncSession]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, SessionLocal
    sqlite_url = os.getenv("SQLITE_URL")
    if not sqlite_url:
        # Preferred config: INK_EMBER_DATA_DIR (set by desktop app) or SQLITE_PATH
        data_dir = os.getenv("INK_EMBER_DATA_DIR")
        sqlite_path = os.getenv("SQLITE_PATH")
        if sqlite_path:
            db_path = Path(sqlite_path)
        else:
            base = Path(data_dir) if data_dir else (ROOT_DIR / ".." / "data")
            db_path = base / "ink_ember.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = f"sqlite+aiosqlite://{db_path.as_posix()}"
    engine = create_async_engine(sqlite_url, echo=False, future=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("SQLite ready (%s)", sqlite_url)
    yield

    if engine:
        await engine.dispose()
        logger.info("DB engine disposed")


async def get_session() -> AsyncSession:
    if SessionLocal is None:
        raise RuntimeError("DB not initialized")
    return SessionLocal()


# =========================
# App / Router
# =========================

app = FastAPI(title="Ink & Ember API", version="0.5.0-founders", lifespan=lifespan)
api_router = APIRouter(prefix="/api")
@api_router.get("/health")
async def health():
    return {"status": "ok", "time": now_iso()}



# =========================
# ENUMS
# =========================

class ArticleType(str, Enum):
    GENERIC = "generic"
    BUILDING = "building"
    CHARACTER = "character"
    COUNTRY = "country"
    MILITARY = "military"
    GODS_DEITIES = "gods_deities"
    GEOGRAPHY = "geography"
    ITEM = "item"
    ORGANIZATION = "organization"
    RELIGION = "religion"
    SPECIES = "species"
    VEHICLE = "vehicle"
    SETTLEMENT = "settlement"
    CONDITION = "condition"
    CONFLICT = "conflict"
    DOCUMENT = "document"
    CULTURE_ETHNICITY = "culture_ethnicity"
    LANGUAGE = "language"
    MATERIAL = "material"
    MILITARY_FORMATION = "military_formation"
    MYTH = "myth"
    NATURAL_LAW = "natural_law"
    PLOT = "plot"
    PROFESSION = "profession"
    PROSE = "prose"
    TITLE = "title"
    SPELL = "spell"
    TECHNOLOGY = "technology"
    TRADITION = "tradition"
    SESSION_REPORT = "session_report"


# =========================
# Pydantic Models
# =========================

class WorldBase(BaseModel):
    name: str
    description: Optional[str] = None
    genre: Optional[str] = None
    cover_image: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)


class World(WorldBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str
    updated_at: str


class WorldCreate(WorldBase):
    pass


class ArticleBase(BaseModel):
    world_id: str
    title: str
    article_type: ArticleType = ArticleType.GENERIC
    content: str = ""
    summary: Optional[str] = None
    cover_image: Optional[str] = None
    infobox: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    linked_articles: List[str] = Field(default_factory=list)
    is_secret: bool = False
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class Article(ArticleBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str
    updated_at: str


class ArticleCreate(ArticleBase):
    pass


class ArticleUpdate(BaseModel):
    title: Optional[str] = None
    article_type: Optional[ArticleType] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    cover_image: Optional[str] = None
    infobox: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    linked_articles: Optional[List[str]] = None
    is_secret: Optional[bool] = None
    custom_fields: Optional[Dict[str, Any]] = None


class TimelineBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    color: str = "#ff4500"


class Timeline(TimelineBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class TimelineCreate(TimelineBase):
    pass


class TimelineEventBase(BaseModel):
    world_id: str
    timeline_id: str
    title: str
    description: Optional[str] = None
    date_label: str
    sort_order: int = 0
    linked_articles: List[str] = Field(default_factory=list)
    color: Optional[str] = None


class TimelineEvent(TimelineEventBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class TimelineEventCreate(TimelineEventBase):
    pass


class CalendarBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    months: List[Dict[str, Any]] = Field(default_factory=list)
    days_per_week: int = 7
    day_names: List[str] = Field(default_factory=list)


class Calendar(CalendarBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class CalendarCreate(CalendarBase):
    pass


class ChronicleBase(BaseModel):
    world_id: str
    title: str
    description: Optional[str] = None
    chronicle_type: str = "campaign_log"
    entries: List[Dict[str, Any]] = Field(default_factory=list)
    linked_timeline_id: Optional[str] = None


class Chronicle(ChronicleBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str
    updated_at: str


class ChronicleCreate(ChronicleBase):
    pass


class MapBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    markers: List[Dict[str, Any]] = Field(default_factory=list)


class Map(MapBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class MapCreate(MapBase):
    pass


class FamilyTreeBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)


class FamilyTree(FamilyTreeBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class FamilyTreeCreate(FamilyTreeBase):
    pass


class VariableBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    variable_type: str = "world_state"
    value: Optional[str] = None
    options: List[str] = Field(default_factory=list)
    is_active: bool = True


class Variable(VariableBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class VariableCreate(VariableBase):
    pass


class NotebookBase(BaseModel):
    world_id: str
    title: str
    content: str = ""
    notebook_type: str = "note"


class Notebook(NotebookBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str
    updated_at: str


class NotebookCreate(NotebookBase):
    pass


class TodoBase(BaseModel):
    world_id: str
    title: str
    description: Optional[str] = None
    completed: bool = False
    priority: str = "medium"
    due_date: Optional[str] = None


class Todo(TodoBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class TodoCreate(TodoBase):
    pass


class DiplomaticRelationBase(BaseModel):
    world_id: str
    entity1_id: str
    entity1_name: str
    entity2_id: str
    entity2_name: str
    relation_type: str = "neutral"
    description: Optional[str] = None


class DiplomaticRelation(DiplomaticRelationBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str


class DiplomaticRelationCreate(DiplomaticRelationBase):
    pass


# =========================
# Row -> Model helpers
# =========================

def world_row_to_model(r: WorldTable) -> World:
    return World(
        id=r.id,
        name=r.name,
        description=r.description,
        genre=r.genre,
        cover_image=r.cover_image,
        settings=loads(r.settings_json, {}),
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


def article_row_to_model(r: ArticleTable) -> Article:
    return Article(
        id=r.id,
        world_id=r.world_id,
        title=r.title,
        article_type=ArticleType(r.article_type),
        content=r.content,
        summary=r.summary,
        cover_image=r.cover_image,
        infobox=loads(r.infobox_json, {}),
        tags=loads(r.tags_json, []),
        linked_articles=loads(r.linked_articles_json, []),
        is_secret=r.is_secret,
        custom_fields=loads(r.custom_fields_json, {}),
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


def timeline_row_to_model(r: TimelineTable) -> Timeline:
    return Timeline(
        id=r.id,
        world_id=r.world_id,
        name=r.name,
        description=r.description,
        color=r.color,
        created_at=r.created_at,
    )


def timeline_event_row_to_model(r: TimelineEventTable) -> TimelineEvent:
    return TimelineEvent(
        id=r.id,
        world_id=r.world_id,
        timeline_id=r.timeline_id,
        title=r.title,
        description=r.description,
        date_label=r.date_label,
        sort_order=r.sort_order,
        linked_articles=loads(r.linked_articles_json, []),
        color=r.color,
        created_at=r.created_at,
    )


def calendar_row_to_model(r: CalendarTable) -> Calendar:
    return Calendar(
        id=r.id,
        world_id=r.world_id,
        name=r.name,
        description=r.description,
        months=loads(r.months_json, []),
        days_per_week=r.days_per_week,
        day_names=loads(r.day_names_json, []),
        created_at=r.created_at,
    )


def chronicle_row_to_model(r: ChronicleTable) -> Chronicle:
    return Chronicle(
        id=r.id,
        world_id=r.world_id,
        title=r.title,
        description=r.description,
        chronicle_type=r.chronicle_type,
        entries=loads(r.entries_json, []),
        linked_timeline_id=r.linked_timeline_id,
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


def map_row_to_model(r: MapTable) -> Map:
    return Map(
        id=r.id,
        world_id=r.world_id,
        name=r.name,
        description=r.description,
        image_url=r.image_url,
        markers=loads(r.markers_json, []),
        created_at=r.created_at,
    )


def family_tree_row_to_model(r: FamilyTreeTable) -> FamilyTree:
    return FamilyTree(
        id=r.id,
        world_id=r.world_id,
        name=r.name,
        description=r.description,
        nodes=loads(r.nodes_json, []),
        connections=loads(r.connections_json, []),
        created_at=r.created_at,
    )


def variable_row_to_model(r: VariableTable) -> Variable:
    return Variable(
        id=r.id,
        world_id=r.world_id,
        name=r.name,
        description=r.description,
        variable_type=r.variable_type,
        value=r.value,
        options=loads(r.options_json, []),
        is_active=r.is_active,
        created_at=r.created_at,
    )


def notebook_row_to_model(r: NotebookTable) -> Notebook:
    return Notebook(
        id=r.id,
        world_id=r.world_id,
        title=r.title,
        content=r.content,
        notebook_type=r.notebook_type,
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


def todo_row_to_model(r: TodoTable) -> Todo:
    return Todo(
        id=r.id,
        world_id=r.world_id,
        title=r.title,
        description=r.description,
        completed=r.completed,
        priority=r.priority,
        due_date=r.due_date,
        created_at=r.created_at,
    )


def relation_row_to_model(r: DiplomaticRelationTable) -> DiplomaticRelation:
    return DiplomaticRelation(
        id=r.id,
        world_id=r.world_id,
        entity1_id=r.entity1_id,
        entity1_name=r.entity1_name,
        entity2_id=r.entity2_id,
        entity2_name=r.entity2_name,
        relation_type=r.relation_type,
        description=r.description,
        created_at=r.created_at,
    )


# =========================
# Routes: Root
# =========================

@api_router.get("/")
async def root():
    return {"message": "Ink & Ember API", "version": app.version}


# =========================
# Routes: Worlds
# =========================

@api_router.post("/worlds", response_model=World)
async def create_world(world: WorldCreate):
    async with await get_session() as session:
        w = WorldTable(
            id=str(uuid.uuid4()),
            name=world.name,
            description=world.description,
            genre=world.genre,
            cover_image=world.cover_image,
            settings_json=dumps(world.settings),
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        session.add(w)
        await session.commit()
        return world_row_to_model(w)


@api_router.get("/worlds", response_model=List[World])
async def get_worlds():
    async with await get_session() as session:
        rows = (await session.execute(select(WorldTable))).scalars().all()
        return [world_row_to_model(r) for r in rows]


@api_router.get("/worlds/{world_id}", response_model=World)
async def get_world(world_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(WorldTable).where(WorldTable.id == world_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="World not found")
        return world_row_to_model(r)


@api_router.put("/worlds/{world_id}", response_model=World)
async def update_world(world_id: str, world: WorldCreate):
    async with await get_session() as session:
        r = (await session.execute(select(WorldTable).where(WorldTable.id == world_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="World not found")

        r.name = world.name
        r.description = world.description
        r.genre = world.genre
        r.cover_image = world.cover_image
        r.settings_json = dumps(world.settings)
        r.updated_at = now_iso()

        await session.commit()
        return world_row_to_model(r)


@api_router.delete("/worlds/{world_id}")
async def delete_world(world_id: str):
    async with await get_session() as session:
        w = (await session.execute(select(WorldTable).where(WorldTable.id == world_id))).scalar_one_or_none()
        if not w:
            raise HTTPException(status_code=404, detail="World not found")

        # delete related data (matches your Mongo delete_many calls)
        await session.execute(delete(ArticleTable).where(ArticleTable.world_id == world_id))
        await session.execute(delete(TimelineTable).where(TimelineTable.world_id == world_id))
        await session.execute(delete(TimelineEventTable).where(TimelineEventTable.world_id == world_id))
        await session.execute(delete(CalendarTable).where(CalendarTable.world_id == world_id))
        await session.execute(delete(ChronicleTable).where(ChronicleTable.world_id == world_id))
        await session.execute(delete(MapTable).where(MapTable.world_id == world_id))
        await session.execute(delete(FamilyTreeTable).where(FamilyTreeTable.world_id == world_id))
        await session.execute(delete(VariableTable).where(VariableTable.world_id == world_id))
        await session.execute(delete(NotebookTable).where(NotebookTable.world_id == world_id))
        await session.execute(delete(TodoTable).where(TodoTable.world_id == world_id))
        await session.execute(delete(DiplomaticRelationTable).where(DiplomaticRelationTable.world_id == world_id))

        await session.delete(w)
        await session.commit()
        return {"message": "World deleted successfully"}


# =========================
# Routes: Articles
# =========================

@api_router.post("/articles", response_model=Article)
async def create_article(article: ArticleCreate):
    async with await get_session() as session:
        a = ArticleTable(
            id=str(uuid.uuid4()),
            world_id=article.world_id,
            title=article.title,
            article_type=article.article_type.value,
            content=article.content or "",
            summary=article.summary,
            cover_image=article.cover_image,
            infobox_json=dumps(article.infobox),
            tags_json=dumps(article.tags),
            linked_articles_json=dumps(article.linked_articles),
            is_secret=article.is_secret,
            custom_fields_json=dumps(article.custom_fields),
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        session.add(a)
        await session.commit()
        return article_row_to_model(a)


@api_router.get("/articles", response_model=List[Article])
async def get_articles(
    world_id: str,
    article_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
):
    async with await get_session() as session:
        stmt = select(ArticleTable).where(ArticleTable.world_id == world_id)
        if article_type:
            stmt = stmt.where(ArticleTable.article_type == article_type)
        if search:
            like = f"%{search}%"
            stmt = stmt.where((ArticleTable.title.like(like)) | (ArticleTable.content.like(like)))
        rows = (await session.execute(stmt.limit(limit))).scalars().all()

        # Keep Mongo-like behavior: tag exact match if search provided (optional)
        if search:
            out = []
            for r in rows:
                tags = loads(r.tags_json, [])
                if (search.lower() in (r.title or "").lower()) or (search.lower() in (r.content or "").lower()) or (search in tags):
                    out.append(r)
            rows = out

        return [article_row_to_model(r) for r in rows]


@api_router.get("/articles/{article_id}", response_model=Article)
async def get_article(article_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(ArticleTable).where(ArticleTable.id == article_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Article not found")
        return article_row_to_model(r)


@api_router.put("/articles/{article_id}", response_model=Article)
async def update_article(article_id: str, article: ArticleUpdate):
    async with await get_session() as session:
        r = (await session.execute(select(ArticleTable).where(ArticleTable.id == article_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Article not found")

        data = article.model_dump(exclude_none=True)
        if "title" in data:
            r.title = data["title"]
        if "article_type" in data:
            r.article_type = data["article_type"].value
        if "content" in data:
            r.content = data["content"] or ""
        if "summary" in data:
            r.summary = data["summary"]
        if "cover_image" in data:
            r.cover_image = data["cover_image"]
        if "infobox" in data:
            r.infobox_json = dumps(data["infobox"])
        if "tags" in data:
            r.tags_json = dumps(data["tags"])
        if "linked_articles" in data:
            r.linked_articles_json = dumps(data["linked_articles"])
        if "is_secret" in data:
            r.is_secret = data["is_secret"]
        if "custom_fields" in data:
            r.custom_fields_json = dumps(data["custom_fields"])

        r.updated_at = now_iso()
        await session.commit()
        return article_row_to_model(r)


@api_router.delete("/articles/{article_id}")
async def delete_article(article_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(ArticleTable).where(ArticleTable.id == article_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Article not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Article deleted successfully"}


# =========================
# Routes: Timelines
# =========================

@api_router.post("/timelines", response_model=Timeline)
async def create_timeline(timeline: TimelineCreate):
    async with await get_session() as session:
        t = TimelineTable(
            id=str(uuid.uuid4()),
            world_id=timeline.world_id,
            name=timeline.name,
            description=timeline.description,
            color=timeline.color,
            created_at=now_iso(),
        )
        session.add(t)
        await session.commit()
        return timeline_row_to_model(t)


@api_router.get("/timelines", response_model=List[Timeline])
async def get_timelines(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(TimelineTable).where(TimelineTable.world_id == world_id))).scalars().all()
        return [timeline_row_to_model(r) for r in rows]


@api_router.get("/timelines/{timeline_id}", response_model=Timeline)
async def get_timeline(timeline_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(TimelineTable).where(TimelineTable.id == timeline_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Timeline not found")
        return timeline_row_to_model(r)


@api_router.delete("/timelines/{timeline_id}")
async def delete_timeline(timeline_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(TimelineTable).where(TimelineTable.id == timeline_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Timeline not found")

        await session.execute(delete(TimelineEventTable).where(TimelineEventTable.timeline_id == timeline_id))
        await session.delete(r)
        await session.commit()
        return {"message": "Timeline deleted successfully"}


# =========================
# Routes: Timeline Events
# =========================

@api_router.post("/timeline-events", response_model=TimelineEvent)
async def create_timeline_event(event: TimelineEventCreate):
    async with await get_session() as session:
        e = TimelineEventTable(
            id=str(uuid.uuid4()),
            world_id=event.world_id,
            timeline_id=event.timeline_id,
            title=event.title,
            description=event.description,
            date_label=event.date_label,
            sort_order=event.sort_order,
            linked_articles_json=dumps(event.linked_articles),
            color=event.color,
            created_at=now_iso(),
        )
        session.add(e)
        await session.commit()
        return timeline_event_row_to_model(e)


@api_router.get("/timeline-events", response_model=List[TimelineEvent])
async def get_timeline_events(timeline_id: str):
    async with await get_session() as session:
        rows = (await session.execute(
            select(TimelineEventTable).where(TimelineEventTable.timeline_id == timeline_id).order_by(TimelineEventTable.sort_order.asc())
        )).scalars().all()
        return [timeline_event_row_to_model(r) for r in rows]


@api_router.delete("/timeline-events/{event_id}")
async def delete_timeline_event(event_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(TimelineEventTable).where(TimelineEventTable.id == event_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Event not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Event deleted successfully"}


# =========================
# Routes: Calendars
# =========================

@api_router.post("/calendars", response_model=Calendar)
async def create_calendar(calendar: CalendarCreate):
    async with await get_session() as session:
        c = CalendarTable(
            id=str(uuid.uuid4()),
            world_id=calendar.world_id,
            name=calendar.name,
            description=calendar.description,
            months_json=dumps(calendar.months),
            days_per_week=calendar.days_per_week,
            day_names_json=dumps(calendar.day_names),
            created_at=now_iso(),
        )
        session.add(c)
        await session.commit()
        return calendar_row_to_model(c)


@api_router.get("/calendars", response_model=List[Calendar])
async def get_calendars(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(CalendarTable).where(CalendarTable.world_id == world_id))).scalars().all()
        return [calendar_row_to_model(r) for r in rows]


@api_router.get("/calendars/{calendar_id}", response_model=Calendar)
async def get_calendar(calendar_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(CalendarTable).where(CalendarTable.id == calendar_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Calendar not found")
        return calendar_row_to_model(r)


@api_router.put("/calendars/{calendar_id}", response_model=Calendar)
async def update_calendar(calendar_id: str, calendar: CalendarCreate):
    async with await get_session() as session:
        r = (await session.execute(select(CalendarTable).where(CalendarTable.id == calendar_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Calendar not found")

        r.name = calendar.name
        r.description = calendar.description
        r.months_json = dumps(calendar.months)
        r.days_per_week = calendar.days_per_week
        r.day_names_json = dumps(calendar.day_names)

        await session.commit()
        return calendar_row_to_model(r)


@api_router.delete("/calendars/{calendar_id}")
async def delete_calendar(calendar_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(CalendarTable).where(CalendarTable.id == calendar_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Calendar not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Calendar deleted successfully"}


# =========================
# Routes: Chronicles
# =========================

@api_router.post("/chronicles", response_model=Chronicle)
async def create_chronicle(chronicle: ChronicleCreate):
    async with await get_session() as session:
        c = ChronicleTable(
            id=str(uuid.uuid4()),
            world_id=chronicle.world_id,
            title=chronicle.title,
            description=chronicle.description,
            chronicle_type=chronicle.chronicle_type,
            entries_json=dumps(chronicle.entries),
            linked_timeline_id=chronicle.linked_timeline_id,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        session.add(c)
        await session.commit()
        return chronicle_row_to_model(c)


@api_router.get("/chronicles", response_model=List[Chronicle])
async def get_chronicles(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(ChronicleTable).where(ChronicleTable.world_id == world_id))).scalars().all()
        return [chronicle_row_to_model(r) for r in rows]


@api_router.get("/chronicles/{chronicle_id}", response_model=Chronicle)
async def get_chronicle(chronicle_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(ChronicleTable).where(ChronicleTable.id == chronicle_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Chronicle not found")
        return chronicle_row_to_model(r)


@api_router.put("/chronicles/{chronicle_id}", response_model=Chronicle)
async def update_chronicle(chronicle_id: str, chronicle: ChronicleCreate):
    async with await get_session() as session:
        r = (await session.execute(select(ChronicleTable).where(ChronicleTable.id == chronicle_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Chronicle not found")

        r.title = chronicle.title
        r.description = chronicle.description
        r.chronicle_type = chronicle.chronicle_type
        r.entries_json = dumps(chronicle.entries)
        r.linked_timeline_id = chronicle.linked_timeline_id
        r.updated_at = now_iso()

        await session.commit()
        return chronicle_row_to_model(r)


@api_router.delete("/chronicles/{chronicle_id}")
async def delete_chronicle(chronicle_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(ChronicleTable).where(ChronicleTable.id == chronicle_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Chronicle not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Chronicle deleted successfully"}


# =========================
# Routes: Maps
# =========================

@api_router.post("/maps", response_model=Map)
async def create_map(map_data: MapCreate):
    async with await get_session() as session:
        m = MapTable(
            id=str(uuid.uuid4()),
            world_id=map_data.world_id,
            name=map_data.name,
            description=map_data.description,
            image_url=map_data.image_url,
            markers_json=dumps(map_data.markers),
            created_at=now_iso(),
        )
        session.add(m)
        await session.commit()
        return map_row_to_model(m)


@api_router.get("/maps", response_model=List[Map])
async def get_maps(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(MapTable).where(MapTable.world_id == world_id))).scalars().all()
        return [map_row_to_model(r) for r in rows]


@api_router.get("/maps/{map_id}", response_model=Map)
async def get_map(map_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(MapTable).where(MapTable.id == map_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Map not found")
        return map_row_to_model(r)


@api_router.put("/maps/{map_id}", response_model=Map)
async def update_map(map_id: str, map_data: MapCreate):
    async with await get_session() as session:
        r = (await session.execute(select(MapTable).where(MapTable.id == map_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Map not found")

        r.name = map_data.name
        r.description = map_data.description
        r.image_url = map_data.image_url
        r.markers_json = dumps(map_data.markers)

        await session.commit()
        return map_row_to_model(r)


@api_router.delete("/maps/{map_id}")
async def delete_map(map_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(MapTable).where(MapTable.id == map_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Map not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Map deleted successfully"}


# =========================
# Routes: Family Trees
# =========================

@api_router.post("/family-trees", response_model=FamilyTree)
async def create_family_tree(tree: FamilyTreeCreate):
    async with await get_session() as session:
        t = FamilyTreeTable(
            id=str(uuid.uuid4()),
            world_id=tree.world_id,
            name=tree.name,
            description=tree.description,
            nodes_json=dumps(tree.nodes),
            connections_json=dumps(tree.connections),
            created_at=now_iso(),
        )
        session.add(t)
        await session.commit()
        return family_tree_row_to_model(t)


@api_router.get("/family-trees", response_model=List[FamilyTree])
async def get_family_trees(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(FamilyTreeTable).where(FamilyTreeTable.world_id == world_id))).scalars().all()
        return [family_tree_row_to_model(r) for r in rows]


@api_router.get("/family-trees/{tree_id}", response_model=FamilyTree)
async def get_family_tree(tree_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(FamilyTreeTable).where(FamilyTreeTable.id == tree_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Family tree not found")
        return family_tree_row_to_model(r)


@api_router.put("/family-trees/{tree_id}", response_model=FamilyTree)
async def update_family_tree(tree_id: str, tree: FamilyTreeCreate):
    async with await get_session() as session:
        r = (await session.execute(select(FamilyTreeTable).where(FamilyTreeTable.id == tree_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Family tree not found")

        r.name = tree.name
        r.description = tree.description
        r.nodes_json = dumps(tree.nodes)
        r.connections_json = dumps(tree.connections)

        await session.commit()
        return family_tree_row_to_model(r)


@api_router.delete("/family-trees/{tree_id}")
async def delete_family_tree(tree_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(FamilyTreeTable).where(FamilyTreeTable.id == tree_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Family tree not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Family tree deleted successfully"}


# =========================
# Routes: Variables
# =========================

@api_router.post("/variables", response_model=Variable)
async def create_variable(variable: VariableCreate):
    async with await get_session() as session:
        v = VariableTable(
            id=str(uuid.uuid4()),
            world_id=variable.world_id,
            name=variable.name,
            description=variable.description,
            variable_type=variable.variable_type,
            value=variable.value,
            options_json=dumps(variable.options),
            is_active=variable.is_active,
            created_at=now_iso(),
        )
        session.add(v)
        await session.commit()
        return variable_row_to_model(v)


@api_router.get("/variables", response_model=List[Variable])
async def get_variables(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(VariableTable).where(VariableTable.world_id == world_id))).scalars().all()
        return [variable_row_to_model(r) for r in rows]


@api_router.put("/variables/{variable_id}", response_model=Variable)
async def update_variable(variable_id: str, variable: VariableCreate):
    async with await get_session() as session:
        r = (await session.execute(select(VariableTable).where(VariableTable.id == variable_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Variable not found")

        r.world_id = variable.world_id
        r.name = variable.name
        r.description = variable.description
        r.variable_type = variable.variable_type
        r.value = variable.value
        r.options_json = dumps(variable.options)
        r.is_active = variable.is_active

        await session.commit()
        return variable_row_to_model(r)


@api_router.delete("/variables/{variable_id}")
async def delete_variable(variable_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(VariableTable).where(VariableTable.id == variable_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Variable not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Variable deleted successfully"}


# =========================
# Routes: Notebooks
# =========================

@api_router.post("/notebooks", response_model=Notebook)
async def create_notebook(notebook: NotebookCreate):
    async with await get_session() as session:
        n = NotebookTable(
            id=str(uuid.uuid4()),
            world_id=notebook.world_id,
            title=notebook.title,
            content=notebook.content or "",
            notebook_type=notebook.notebook_type,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        session.add(n)
        await session.commit()
        return notebook_row_to_model(n)


@api_router.get("/notebooks", response_model=List[Notebook])
async def get_notebooks(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(NotebookTable).where(NotebookTable.world_id == world_id))).scalars().all()
        return [notebook_row_to_model(r) for r in rows]


@api_router.get("/notebooks/{notebook_id}", response_model=Notebook)
async def get_notebook(notebook_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(NotebookTable).where(NotebookTable.id == notebook_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Notebook not found")
        return notebook_row_to_model(r)


@api_router.put("/notebooks/{notebook_id}", response_model=Notebook)
async def update_notebook(notebook_id: str, notebook: NotebookCreate):
    async with await get_session() as session:
        r = (await session.execute(select(NotebookTable).where(NotebookTable.id == notebook_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Notebook not found")

        r.world_id = notebook.world_id
        r.title = notebook.title
        r.content = notebook.content or ""
        r.notebook_type = notebook.notebook_type
        r.updated_at = now_iso()

        await session.commit()
        return notebook_row_to_model(r)


@api_router.delete("/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(NotebookTable).where(NotebookTable.id == notebook_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Notebook not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Notebook deleted successfully"}


# =========================
# Routes: Todos
# =========================

@api_router.post("/todos", response_model=Todo)
async def create_todo(todo: TodoCreate):
    async with await get_session() as session:
        t = TodoTable(
            id=str(uuid.uuid4()),
            world_id=todo.world_id,
            title=todo.title,
            description=todo.description,
            completed=todo.completed,
            priority=todo.priority,
            due_date=todo.due_date,
            created_at=now_iso(),
        )
        session.add(t)
        await session.commit()
        return todo_row_to_model(t)


@api_router.get("/todos", response_model=List[Todo])
async def get_todos(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(select(TodoTable).where(TodoTable.world_id == world_id))).scalars().all()
        return [todo_row_to_model(r) for r in rows]


@api_router.put("/todos/{todo_id}", response_model=Todo)
async def update_todo(todo_id: str, todo: TodoCreate):
    async with await get_session() as session:
        r = (await session.execute(select(TodoTable).where(TodoTable.id == todo_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Todo not found")

        r.world_id = todo.world_id
        r.title = todo.title
        r.description = todo.description
        r.completed = todo.completed
        r.priority = todo.priority
        r.due_date = todo.due_date

        await session.commit()
        return todo_row_to_model(r)


@api_router.delete("/todos/{todo_id}")
async def delete_todo(todo_id: str):
    async with await get_session() as session:
        r = (await session.execute(select(TodoTable).where(TodoTable.id == todo_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Todo not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Todo deleted successfully"}


# =========================
# Routes: Diplomatic Relations
# =========================

@api_router.post("/diplomatic-relations", response_model=DiplomaticRelation)
async def create_diplomatic_relation(relation: DiplomaticRelationCreate):
    async with await get_session() as session:
        d = DiplomaticRelationTable(
            id=str(uuid.uuid4()),
            world_id=relation.world_id,
            entity1_id=relation.entity1_id,
            entity1_name=relation.entity1_name,
            entity2_id=relation.entity2_id,
            entity2_name=relation.entity2_name,
            relation_type=relation.relation_type,
            description=relation.description,
            created_at=now_iso(),
        )
        session.add(d)
        await session.commit()
        return relation_row_to_model(d)


@api_router.get("/diplomatic-relations", response_model=List[DiplomaticRelation])
async def get_diplomatic_relations(world_id: str):
    async with await get_session() as session:
        rows = (await session.execute(
            select(DiplomaticRelationTable).where(DiplomaticRelationTable.world_id == world_id)
        )).scalars().all()
        return [relation_row_to_model(r) for r in rows]


@api_router.delete("/diplomatic-relations/{relation_id}")
async def delete_diplomatic_relation(relation_id: str):
    async with await get_session() as session:
        r = (await session.execute(
            select(DiplomaticRelationTable).where(DiplomaticRelationTable.id == relation_id)
        )).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Relation not found")
        await session.delete(r)
        await session.commit()
        return {"message": "Relation deleted successfully"}


# =========================
# Routes: Stats
# =========================

@api_router.get("/stats/{world_id}")
async def get_world_stats(world_id: str):
    async with await get_session() as session:
        # counts (SQLite)
        articles_count = len((await session.execute(select(ArticleTable.id).where(ArticleTable.world_id == world_id))).all())
        timelines_count = len((await session.execute(select(TimelineTable.id).where(TimelineTable.world_id == world_id))).all())
        chronicles_count = len((await session.execute(select(ChronicleTable.id).where(ChronicleTable.world_id == world_id))).all())
        maps_count = len((await session.execute(select(MapTable.id).where(MapTable.world_id == world_id))).all())

        characters_count = len((
            await session.execute(
                select(ArticleTable.id).where(ArticleTable.world_id == world_id, ArticleTable.article_type == "character")
            )
        ).all())

        locations_count = len((
            await session.execute(
                select(ArticleTable.id).where(
                    ArticleTable.world_id == world_id,
                    ArticleTable.article_type.in_(["settlement", "geography", "building", "country"]),
                )
            )
        ).all())

        return {
            "articles": articles_count,
            "timelines": timelines_count,
            "characters": characters_count,
            "locations": locations_count,
            "chronicles": chronicles_count,
            "maps": maps_count,
        }


# =========================
# Foundry VTT Integration
# =========================

def article_to_foundry_journal(article: dict) -> dict:
    content_html = ""
    if article.get("summary"):
        content_html += f"<p><em>{article['summary']}</em></p>"
    if article.get("content"):
        content_html += f"<div>{str(article['content']).replace(chr(10), '<br/>')}</div>"

    if article.get("infobox") and isinstance(article["infobox"], dict) and len(article["infobox"]) > 0:
        content_html += "<hr/><h3>Details</h3><table>"
        for key, value in article["infobox"].items():
            content_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
        content_html += "</table>"

    return {
        "_id": article["id"].replace("-", "")[:16],
        "name": article["title"],
        "pages": [
            {
                "_id": f"page{article['id'].replace('-', '')[:12]}",
                "name": article["title"],
                "type": "text",
                "title": {"show": True, "level": 1},
                "text": {"content": content_html, "format": 1},
                "sort": 0,
                "ownership": {"default": -1},
            }
        ],
        "folder": None,
        "sort": 0,
        "ownership": {"default": 0},
        "flags": {
            "ink-ember": {
                "articleId": article["id"],
                "articleType": article.get("article_type", "generic"),
                "tags": article.get("tags", []),
                "isSecret": article.get("is_secret", False),
            }
        },
    }


def article_to_foundry_actor(article: dict) -> dict:
    infobox = article.get("infobox", {})
    return {
        "_id": article["id"].replace("-", "")[:16],
        "name": article["title"],
        "type": "character",
        "img": article.get("cover_image", "icons/svg/mystery-man.svg"),
        "system": {
            "biography": {
                "value": article.get("content", ""),
                "public": article.get("summary", ""),
            },
            "details": infobox,
        },
        "prototypeToken": {
            "name": article["title"],
            "displayName": 20,
            "actorLink": True,
            "disposition": 0,
        },
        "items": [],
        "effects": [],
        "folder": None,
        "sort": 0,
        "ownership": {"default": 0},
        "flags": {"ink-ember": {"articleId": article["id"], "articleType": "character"}},
    }


def article_to_foundry_item(article: dict) -> dict:
    infobox = article.get("infobox", {})
    item_type = "equipment"
    if article.get("article_type") == "spell":
        item_type = "spell"
    elif article.get("article_type") == "vehicle":
        item_type = "equipment"

    return {
        "_id": article["id"].replace("-", "")[:16],
        "name": article["title"],
        "type": item_type,
        "img": article.get("cover_image", "icons/svg/item-bag.svg"),
        "system": {
            "description": {"value": article.get("content", ""), "chat": article.get("summary", "")},
            "details": infobox,
        },
        "effects": [],
        "folder": None,
        "sort": 0,
        "ownership": {"default": 0},
        "flags": {
            "ink-ember": {
                "articleId": article["id"],
                "articleType": article.get("article_type", "item"),
            }
        },
    }


def foundry_journal_to_article(journal: dict, world_id: str) -> dict:
    content = ""
    for page in journal.get("pages", []):
        if page.get("text", {}).get("content"):
            content += page["text"]["content"]

    import re
    plain_content = re.sub(r"<[^>]+>", "", content)

    flags = journal.get("flags", {}).get("ink-ember", {})

    return {
        "world_id": world_id,
        "title": journal.get("name", "Imported Journal"),
        "article_type": flags.get("articleType", "document"),
        "content": plain_content,
        "summary": None,
        "cover_image": None,
        "infobox": {},
        "tags": flags.get("tags", ["foundry-import"]),
        "is_secret": flags.get("isSecret", False),
        "custom_fields": {},
    }


def foundry_actor_to_article(actor: dict, world_id: str) -> dict:
    system = actor.get("system", {})
    biography = system.get("biography", {})

    content = ""
    if isinstance(biography, dict):
        content = biography.get("value", "") or biography.get("public", "")
    elif isinstance(biography, str):
        content = biography

    import re
    plain_content = re.sub(r"<[^>]+>", "", str(content))

    return {
        "world_id": world_id,
        "title": actor.get("name", "Imported Character"),
        "article_type": "character",
        "content": plain_content,
        "summary": None,
        "cover_image": actor.get("img"),
        "infobox": system.get("details", {}),
        "tags": ["foundry-import"],
        "is_secret": False,
        "custom_fields": {},
    }


class FoundryExportRequest(BaseModel):
    article_ids: Optional[List[str]] = None
    export_type: str = "all"  # all, journals, actors, items


class FoundryImportData(BaseModel):
    journals: List[dict] = Field(default_factory=list)
    actors: List[dict] = Field(default_factory=list)
    items: List[dict] = Field(default_factory=list)


async def fetch_world_name(session: AsyncSession, world_id: str) -> str:
    w = (await session.execute(select(WorldTable).where(WorldTable.id == world_id))).scalar_one_or_none()
    if not w:
        raise HTTPException(status_code=404, detail="World not found")
    return w.name


async def fetch_articles_as_dicts(session: AsyncSession, world_id: str, limit: int = 1000) -> List[dict]:
    rows = (await session.execute(select(ArticleTable).where(ArticleTable.world_id == world_id).limit(limit))).scalars().all()
    out: List[dict] = []
    for r in rows:
        out.append({
            "id": r.id,
            "world_id": r.world_id,
            "title": r.title,
            "article_type": r.article_type,
            "content": r.content,
            "summary": r.summary,
            "cover_image": r.cover_image,
            "infobox": loads(r.infobox_json, {}),
            "tags": loads(r.tags_json, []),
            "linked_articles": loads(r.linked_articles_json, []),
            "is_secret": r.is_secret,
            "custom_fields": loads(r.custom_fields_json, {}),
            "created_at": r.created_at,
            "updated_at": r.updated_at,
        })
    return out


@api_router.post("/foundry/export/{world_id}")
async def export_to_foundry(world_id: str, request: FoundryExportRequest):
    async with await get_session() as session:
        articles = await fetch_articles_as_dicts(session, world_id)
        if request.article_ids:
            wanted = set(request.article_ids)
            articles = [a for a in articles if a["id"] in wanted]

        journals: List[dict] = []
        actors: List[dict] = []
        items: List[dict] = []

        for article in articles:
            at = article.get("article_type", "generic")
            if at == "character":
                if request.export_type in ["all", "actors"]:
                    actors.append(article_to_foundry_actor(article))
                if request.export_type in ["all", "journals"]:
                    journals.append(article_to_foundry_journal(article))
            elif at in ["item", "spell", "vehicle"]:
                if request.export_type in ["all", "items"]:
                    items.append(article_to_foundry_item(article))
                if request.export_type in ["all", "journals"]:
                    journals.append(article_to_foundry_journal(article))
            else:
                if request.export_type in ["all", "journals"]:
                    journals.append(article_to_foundry_journal(article))

        world_name = await fetch_world_name(session, world_id)
        module_id = world_name.lower().replace(" ", "-").replace("'", "")[:20]

        return {
            "module_id": module_id,
            "world_name": world_name,
            "export_date": now_iso(),
            "journals": journals,
            "actors": actors,
            "items": items,
            "stats": {"journals": len(journals), "actors": len(actors), "items": len(items)},
        }


@api_router.get("/foundry/export/{world_id}/download")
async def download_foundry_module(world_id: str):
    async with await get_session() as session:
        world_name = await fetch_world_name(session, world_id)
        module_id = world_name.lower().replace(" ", "-").replace("'", "")[:20]

        articles = await fetch_articles_as_dicts(session, world_id)

        journals: List[dict] = []
        actors: List[dict] = []
        items: List[dict] = []

        for a in articles:
            journals.append(article_to_foundry_journal(a))
            if a.get("article_type") == "character":
                actors.append(article_to_foundry_actor(a))
            elif a.get("article_type") in ["item", "spell", "vehicle"]:
                items.append(article_to_foundry_item(a))

        module_json = {
            "id": module_id,
            "title": f"{world_name} - Ink & Ember Export",
            "description": f"Worldbuilding content exported from Ink & Ember for {world_name}",
            "version": "1.0.0",
            "compatibility": {"minimum": "10", "verified": "12"},
            "authors": [{"name": "Ink & Ember", "url": "https://ink-ember.app"}],
            "packs": [
                {
                    "name": "journals",
                    "label": f"{world_name} - Lore",
                    "path": "packs/journals",
                    "type": "JournalEntry",
                    "ownership": {"PLAYER": "OBSERVER", "ASSISTANT": "OWNER"},
                }
            ],
            "flags": {"ink-ember": {"worldId": world_id, "exportDate": now_iso()}},
        }

        if actors:
            module_json["packs"].append({
                "name": "actors",
                "label": f"{world_name} - Characters",
                "path": "packs/actors",
                "type": "Actor",
                "ownership": {"PLAYER": "OBSERVER", "ASSISTANT": "OWNER"},
            })
        if items:
            module_json["packs"].append({
                "name": "items",
                "label": f"{world_name} - Items",
                "path": "packs/items",
                "type": "Item",
                "ownership": {"PLAYER": "OBSERVER", "ASSISTANT": "OWNER"},
            })

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{module_id}/module.json", json.dumps(module_json, indent=2))
            zf.writestr(f"{module_id}/packs/journals.db", "\n".join(json.dumps(j) for j in journals))
            if actors:
                zf.writestr(f"{module_id}/packs/actors.db", "\n".join(json.dumps(a) for a in actors))
            if items:
                zf.writestr(f"{module_id}/packs/items.db", "\n".join(json.dumps(i) for i in items))

            readme = f"""# {world_name} - Ink & Ember Export

This module was exported from Ink & Ember worldbuilding platform.

## Contents
- {len(journals)} Journal Entries (Lore articles)
- {len(actors)} Actors (Characters)
- {len(items)} Items (Equipment, Spells)

## Installation
1. Extract this ZIP to your Foundry VTT `Data/modules/` folder
2. Restart Foundry VTT
3. Enable the module in your world's Module Management
4. Import content from the Compendium packs

## Note
This export preserves Ink & Ember article IDs in flags for future sync capabilities.

Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            zf.writestr(f"{module_id}/README.md", readme)

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={module_id}-foundry-module.zip"},
        )


@api_router.post("/foundry/import/{world_id}")
async def import_from_foundry(world_id: str, data: FoundryImportData):
    async with await get_session() as session:
        # verify world
        _ = await fetch_world_name(session, world_id)

        imported = {
            "journals": 0,
            "actors": 0,
            "items": 0,
            "articles_created": [],
        }

        # journals -> articles
        for journal in data.journals or []:
            a = foundry_journal_to_article(journal, world_id)
            row = ArticleTable(
                id=str(uuid.uuid4()),
                world_id=world_id,
                title=a["title"],
                article_type=a["article_type"],
                content=a.get("content", "") or "",
                summary=a.get("summary"),
                cover_image=a.get("cover_image"),
                infobox_json=dumps(a.get("infobox", {})),
                tags_json=dumps(a.get("tags", [])),
                linked_articles_json=dumps([]),
                is_secret=bool(a.get("is_secret", False)),
                custom_fields_json=dumps(a.get("custom_fields", {})),
                created_at=now_iso(),
                updated_at=now_iso(),
            )
            session.add(row)
            imported["journals"] += 1
            imported["articles_created"].append({"id": row.id, "title": row.title, "source": "journal"})

        # actors -> character articles
        for actor in data.actors or []:
            a = foundry_actor_to_article(actor, world_id)
            row = ArticleTable(
                id=str(uuid.uuid4()),
                world_id=world_id,
                title=a["title"],
                article_type="character",
                content=a.get("content", "") or "",
                summary=a.get("summary"),
                cover_image=a.get("cover_image"),
                infobox_json=dumps(a.get("infobox", {})),
                tags_json=dumps(a.get("tags", ["foundry-import"])),
                linked_articles_json=dumps([]),
                is_secret=False,
                custom_fields_json=dumps(a.get("custom_fields", {})),
                created_at=now_iso(),
                updated_at=now_iso(),
            )
            session.add(row)
            imported["actors"] += 1
            imported["articles_created"].append({"id": row.id, "title": row.title, "source": "actor"})

        # items -> item/spell articles
        for item in data.items or []:
            article_type = "item"
            if item.get("type") == "spell":
                article_type = "spell"

            row = ArticleTable(
                id=str(uuid.uuid4()),
                world_id=world_id,
                title=item.get("name", "Imported Item"),
                article_type=article_type,
                content=item.get("system", {}).get("description", {}).get("value", "") or "",
                summary=item.get("system", {}).get("description", {}).get("chat", ""),
                cover_image=item.get("img"),
                infobox_json=dumps(item.get("system", {}).get("details", {})),
                tags_json=dumps(["foundry-import"]),
                linked_articles_json=dumps([]),
                is_secret=False,
                custom_fields_json=dumps({}),
                created_at=now_iso(),
                updated_at=now_iso(),
            )
            session.add(row)
            imported["items"] += 1
            imported["articles_created"].append({"id": row.id, "title": row.title, "source": "item"})

        await session.commit()
        return {
            "success": True,
            "imported": imported,
            "message": f"Imported {imported['journals']} journals, {imported['actors']} actors, {imported['items']} items",
        }


@api_router.get("/foundry/module-template/{world_id}")
async def get_foundry_module_template(world_id: str):
    async with await get_session() as session:
        world_name = await fetch_world_name(session, world_id)

        module_id = f"ink-ember-{world_name.lower().replace(' ', '-')[:15]}"

        module_js = f'''/**
 * Ink & Ember Integration Module for Foundry VTT
 * World: {world_name}
 *
 * This module syncs content from your Ink & Ember worldbuilding platform.
 */

const INK_EMBER_CONFIG = {{
    apiUrl: "YOUR_INK_EMBER_API_URL",  // Replace with your Ink & Ember API URL
    worldId: "{world_id}",
    worldName: "{world_name}"
}};

Hooks.once('init', () => {{
    console.log('Ink & Ember | Initializing module for {world_name}');

    game.settings.register('{module_id}', 'apiUrl', {{
        name: 'Ink & Ember API URL',
        hint: 'The URL to your Ink & Ember instance API',
        scope: 'world',
        config: true,
        type: String,
        default: INK_EMBER_CONFIG.apiUrl
    }});
}});

Hooks.once('ready', () => {{
    console.log('Ink & Ember | Module ready');

    if (game.user.isGM) {{
        Hooks.on('renderJournalDirectory', (app, html, data) => {{
            const button = $(`<button class="ink-ember-sync"><i class="fas fa-sync"></i> Sync from Ink & Ember</button>`);
            button.click(() => syncFromInkEmber());
            html.find('.directory-footer').append(button);
        }});
    }}
}});

async function syncFromInkEmber() {{
    const apiUrl = game.settings.get('{module_id}', 'apiUrl');

    if (!apiUrl || apiUrl === 'YOUR_INK_EMBER_API_URL') {{
        ui.notifications.error('Please configure your Ink & Ember API URL in module settings');
        return;
    }}

    ui.notifications.info('Syncing from Ink & Ember...');

    try {{
        const response = await fetch(`${{apiUrl}}/api/foundry/export/{world_id}`, {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ export_type: 'journals' }})
        }});

        if (!response.ok) throw new Error('Failed to fetch from Ink & Ember');

        const data = await response.json();

        for (const journal of data.journals) {{
            const existing = game.journal.find(j =>
                j.getFlag('ink-ember', 'articleId') === journal.flags['ink-ember'].articleId
            );

            if (existing) {{
                await existing.update({{
                    name: journal.name,
                    pages: journal.pages
                }});
            }} else {{
                await JournalEntry.create(journal);
            }}
        }}

        ui.notifications.info(`Synced ${{data.journals.length}} journal entries from Ink & Ember`);

    }} catch (error) {{
        console.error('Ink & Ember sync error:', error);
        ui.notifications.error('Failed to sync from Ink & Ember');
    }}
}}

window.InkEmber = {{
    sync: syncFromInkEmber,
    config: INK_EMBER_CONFIG
}};
'''

        module_json = {
            "id": module_id,
            "title": f"Ink & Ember - {world_name}",
            "description": f"Live integration with Ink & Ember worldbuilding platform for {world_name}",
            "version": "1.0.0",
            "compatibility": {"minimum": "10", "verified": "12"},
            "authors": [{"name": "Ink & Ember"}],
            "esmodules": ["scripts/ink-ember.js"],
            "styles": ["styles/ink-ember.css"],
            "flags": {"ink-ember": {"worldId": world_id}},
        }

        module_css = """
.ink-ember-sync {
    margin-top: 5px;
    background: linear-gradient(135deg, #ff4500, #ff6b35);
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
    width: 100%;
}

.ink-ember-sync:hover {
    background: linear-gradient(135deg, #ff6b35, #ff4500);
}

.ink-ember-sync i {
    margin-right: 5px;
}
"""

        return {
            "module_id": module_id,
            "files": {
                "module.json": module_json,
                "scripts/ink-ember.js": module_js,
                "styles/ink-ember.css": module_css,
            },
            "instructions": [
                f"1. Create a folder named '{module_id}' in your Foundry VTT Data/modules/ directory",
                "2. Save module.json to the root of that folder",
                "3. Create a 'scripts' subfolder and save ink-ember.js there",
                "4. Create a 'styles' subfolder and save ink-ember.css there",
                "5. Restart Foundry VTT and enable the module",
                "6. Configure the API URL in module settings",
                "7. Use the 'Sync from Ink & Ember' button in the Journal tab",
            ],
        }


# =========================
# Wire router + CORS
# =========================

app.include_router(api_router)

cors_origins, cors_allow_credentials = parse_cors(os.environ.get("CORS_ORIGINS"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],

# =========================
# Frontend (production)
# =========================

FRONTEND_BUILD_DIR = (ROOT_DIR.parent / "frontend" / "build").resolve()

if FRONTEND_BUILD_DIR.exists():

    @app.get("/", include_in_schema=False)
    async def serve_root():
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Serve real files (static assets) if present; otherwise return index.html
        candidate = (FRONTEND_BUILD_DIR / full_path).resolve()
        try:
            # Prevent path traversal
            candidate.relative_to(FRONTEND_BUILD_DIR)
        except Exception:
            return FileResponse(FRONTEND_BUILD_DIR / "index.html")

        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")
