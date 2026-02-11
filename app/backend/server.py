# server.py
from __future__ import annotations

import os
import uuid
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.cors import CORSMiddleware

# ================== BOOTSTRAP ==================

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ink_ember_api")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


mongo_url = require_env("MONGO_URL")
db_name = require_env("DB_NAME")

client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

app = FastAPI(title="Ink & Ember API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# ================== ENUMS ==================


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


# ================== MIXINS ==================


class TimestampMixin(BaseModel):
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class IdMixin(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ================== MODELS ==================


class WorldBase(BaseModel):
    name: str
    description: Optional[str] = None
    genre: Optional[str] = None
    cover_image: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)


class World(WorldBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class WorldCreate(WorldBase):
    pass


class WorldUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    genre: Optional[str] = None
    cover_image: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


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


class Article(ArticleBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


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


class Timeline(TimelineBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class TimelineCreate(TimelineBase):
    pass


class TimelineUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None


class TimelineEventBase(BaseModel):
    world_id: str
    timeline_id: str
    title: str
    description: Optional[str] = None
    date_label: str
    sort_order: int = 0
    linked_articles: List[str] = Field(default_factory=list)
    color: Optional[str] = None


class TimelineEvent(TimelineEventBase, IdMixin):
    model_config = ConfigDict(extra="ignore")
    created_at: str = Field(default_factory=utc_now_iso)


class TimelineEventCreate(TimelineEventBase):
    pass


class TimelineEventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    date_label: Optional[str] = None
    sort_order: Optional[int] = None
    linked_articles: Optional[List[str]] = None
    color: Optional[str] = None


class CalendarBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    months: List[Dict[str, Any]] = Field(default_factory=list)
    days_per_week: int = 7
    day_names: List[str] = Field(default_factory=list)


class Calendar(CalendarBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class CalendarCreate(CalendarBase):
    pass


class CalendarUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    months: Optional[List[Dict[str, Any]]] = None
    days_per_week: Optional[int] = None
    day_names: Optional[List[str]] = None


class ChronicleBase(BaseModel):
    world_id: str
    title: str
    description: Optional[str] = None
    chronicle_type: str = "campaign_log"
    entries: List[Dict[str, Any]] = Field(default_factory=list)
    linked_timeline_id: Optional[str] = None


class Chronicle(ChronicleBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class ChronicleCreate(ChronicleBase):
    pass


class ChronicleUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    chronicle_type: Optional[str] = None
    entries: Optional[List[Dict[str, Any]]] = None
    linked_timeline_id: Optional[str] = None


class MapBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    markers: List[Dict[str, Any]] = Field(default_factory=list)


class Map(MapBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class MapCreate(MapBase):
    pass


class MapUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    markers: Optional[List[Dict[str, Any]]] = None


class FamilyTreeBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)


class FamilyTree(FamilyTreeBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class FamilyTreeCreate(FamilyTreeBase):
    pass


class FamilyTreeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    connections: Optional[List[Dict[str, Any]]] = None


class VariableBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    variable_type: str = "world_state"
    value: Optional[str] = None
    options: List[str] = Field(default_factory=list)
    is_active: bool = True


class Variable(VariableBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class VariableCreate(VariableBase):
    pass


class VariableUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    variable_type: Optional[str] = None
    value: Optional[str] = None
    options: Optional[List[str]] = None
    is_active: Optional[bool] = None


class NotebookBase(BaseModel):
    world_id: str
    title: str
    content: str = ""
    notebook_type: str = "note"


class Notebook(NotebookBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class NotebookCreate(NotebookBase):
    pass


class NotebookUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    notebook_type: Optional[str] = None


class TodoBase(BaseModel):
    world_id: str
    title: str
    description: Optional[str] = None
    completed: bool = False
    priority: str = "medium"
    due_date: Optional[str] = None


class Todo(TodoBase, IdMixin, TimestampMixin):
    model_config = ConfigDict(extra="ignore")


class TodoCreate(TodoBase):
    pass


class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    completed: Optional[bool] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None


class DiplomaticRelationBase(BaseModel):
    world_id: str
    entity1_id: str
    entity1_name: str
    entity2_id: str
    entity2_name: str
    relation_type: str = "neutral"
    description: Optional[str] = None


class DiplomaticRelation(DiplomaticRelationBase, IdMixin):
    model_config = ConfigDict(extra="ignore")
    created_at: str = Field(default_factory=utc_now_iso)


class DiplomaticRelationCreate(DiplomaticRelationBase):
    pass


class DiplomaticRelationUpdate(BaseModel):
    entity1_id: Optional[str] = None
    entity1_name: Optional[str] = None
    entity2_id: Optional[str] = None
    entity2_name: Optional[str] = None
    relation_type: Optional[str] = None
    description: Optional[str] = None


# ================== HELPERS ==================


async def assert_world_exists(world_id: str) -> None:
    exists = await db.worlds.find_one({"id": world_id}, {"_id": 1})
    if not exists:
        raise HTTPException(status_code=404, detail="World not found")


async def assert_timeline_exists(timeline_id: str) -> None:
    exists = await db.timelines.find_one({"id": timeline_id}, {"_id": 1})
    if not exists:
        raise HTTPException(status_code=404, detail="Timeline not found")


def non_null_update(payload: BaseModel) -> Dict[str, Any]:
    return {k: v for k, v in payload.model_dump().items() if v is not None}


# ================== STARTUP / SHUTDOWN ==================


@app.on_event("startup")
async def ensure_indexes():
    # Unique IDs per collection
    await db.worlds.create_index("id", unique=True)
    await db.articles.create_index("id", unique=True)
    await db.timelines.create_index("id", unique=True)
    await db.timeline_events.create_index("id", unique=True)
    await db.calendars.create_index("id", unique=True)
    await db.chronicles.create_index("id", unique=True)
    await db.maps.create_index("id", unique=True)
    await db.family_trees.create_index("id", unique=True)
    await db.variables.create_index("id", unique=True)
    await db.notebooks.create_index("id", unique=True)
    await db.todos.create_index("id", unique=True)
    await db.diplomatic_relations.create_index("id", unique=True)

    # Performance indexes
    await db.articles.create_index([("world_id", 1), ("article_type", 1)])
    await db.articles.create_index([("world_id", 1), ("title", 1)])
    await db.timelines.create_index([("world_id", 1)])
    await db.timeline_events.create_index([("timeline_id", 1), ("sort_order", 1)])
    await db.calendars.create_index([("world_id", 1)])
    await db.chronicles.create_index([("world_id", 1)])
    await db.maps.create_index([("world_id", 1)])
    await db.family_trees.create_index([("world_id", 1)])
    await db.variables.create_index([("world_id", 1)])
    await db.notebooks.create_index([("world_id", 1)])
    await db.todos.create_index([("world_id", 1), ("completed", 1)])
    await db.diplomatic_relations.create_index([("world_id", 1)])


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


# ================== ROOT ==================


@api_router.get("/")
async def root():
    return {"message": "Ink & Ember API", "version": app.version}


# ================== WORLD ROUTES ==================


@api_router.post("/worlds", response_model=World)
async def create_world(world: WorldCreate):
    world_obj = World(**world.model_dump())
    await db.worlds.insert_one(world_obj.model_dump())
    return world_obj


@api_router.get("/worlds", response_model=List[World])
async def get_worlds():
    return await db.worlds.find({}, {"_id": 0}).to_list(1000)


@api_router.get("/worlds/{world_id}", response_model=World)
async def get_world(world_id: str):
    world = await db.worlds.find_one({"id": world_id}, {"_id": 0})
    if not world:
        raise HTTPException(status_code=404, detail="World not found")
    return world


@api_router.patch("/worlds/{world_id}", response_model=World)
async def patch_world(world_id: str, payload: WorldUpdate):
    existing = await db.worlds.find_one({"id": world_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="World not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.worlds.update_one({"id": world_id}, {"$set": update_data})
    return await db.worlds.find_one({"id": world_id}, {"_id": 0)


# Optional compatibility PUT (same behavior as PATCH but expects a "full-ish" payload)
@api_router.put("/worlds/{world_id}", response_model=World)
async def put_world(world_id: str, payload: WorldCreate):
    update_payload = WorldUpdate(**payload.model_dump())
    return await patch_world(world_id, update_payload)


@api_router.delete("/worlds/{world_id}")
async def delete_world(world_id: str):
    result = await db.worlds.delete_one({"id": world_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="World not found")

    # Cascade delete all related data by world_id
    await db.articles.delete_many({"world_id": world_id})
    await db.timelines.delete_many({"world_id": world_id})
    await db.timeline_events.delete_many({"world_id": world_id})
    await db.calendars.delete_many({"world_id": world_id})
    await db.chronicles.delete_many({"world_id": world_id})
    await db.maps.delete_many({"world_id": world_id})
    await db.family_trees.delete_many({"world_id": world_id})
    await db.variables.delete_many({"world_id": world_id})
    await db.notebooks.delete_many({"world_id": world_id})
    await db.todos.delete_many({"world_id": world_id})
    await db.diplomatic_relations.delete_many({"world_id": world_id})

    return {"message": "World deleted successfully"}


# ================== ARTICLE ROUTES ==================


@api_router.post("/articles", response_model=Article)
async def create_article(article: ArticleCreate):
    await assert_world_exists(article.world_id)
    article_obj = Article(**article.model_dump())
    await db.articles.insert_one(article_obj.model_dump())
    return article_obj


@api_router.get("/articles", response_model=List[Article])
async def get_articles(
    world_id: str,
    article_type: Optional[ArticleType] = None,
    search: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
):
    await assert_world_exists(world_id)

    query: Dict[str, Any] = {"world_id": world_id}

    if article_type:
        query["article_type"] = article_type.value

    if search:
        query["$or"] = [
            {"title": {"$regex": search, "$options": "i"}},
            {"content": {"$regex": search, "$options": "i"}},
            {"tags": {"$in": [search]}},
        ]

    return await db.articles.find(query, {"_id": 0}).to_list(limit)


@api_router.get("/articles/{article_id}", response_model=Article)
async def get_article(article_id: str):
    article = await db.articles.find_one({"id": article_id}, {"_id": 0})
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@api_router.patch("/articles/{article_id}", response_model=Article)
async def patch_article(article_id: str, payload: ArticleUpdate):
    existing = await db.articles.find_one({"id": article_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Article not found")

    update_data = non_null_update(payload)
    if "article_type" in update_data and isinstance(update_data["article_type"], ArticleType):
        update_data["article_type"] = update_data["article_type"].value
    update_data["updated_at"] = utc_now_iso()

    await db.articles.update_one({"id": article_id}, {"$set": update_data})
    return await db.articles.find_one({"id": article_id}, {"_id": 0)


@api_router.put("/articles/{article_id}", response_model=Article)
async def put_article(article_id: str, payload: ArticleCreate):
    update_payload = ArticleUpdate(**payload.model_dump())
    return await patch_article(article_id, update_payload)


@api_router.delete("/articles/{article_id}")
async def delete_article(article_id: str):
    result = await db.articles.delete_one({"id": article_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"message": "Article deleted successfully"}


# ================== TIMELINE ROUTES ==================


@api_router.post("/timelines", response_model=Timeline)
async def create_timeline(timeline: TimelineCreate):
    await assert_world_exists(timeline.world_id)
    timeline_obj = Timeline(**timeline.model_dump())
    await db.timelines.insert_one(timeline_obj.model_dump())
    return timeline_obj


@api_router.get("/timelines", response_model=List[Timeline])
async def get_timelines(world_id: str):
    await assert_world_exists(world_id)
    return await db.timelines.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/timelines/{timeline_id}", response_model=Timeline)
async def get_timeline(timeline_id: str):
    timeline = await db.timelines.find_one({"id": timeline_id}, {"_id": 0})
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not found")
    return timeline


@api_router.patch("/timelines/{timeline_id}", response_model=Timeline)
async def patch_timeline(timeline_id: str, payload: TimelineUpdate):
    existing = await db.timelines.find_one({"id": timeline_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Timeline not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.timelines.update_one({"id": timeline_id}, {"$set": update_data})
    return await db.timelines.find_one({"id": timeline_id}, {"_id": 0)


@api_router.put("/timelines/{timeline_id}", response_model=Timeline)
async def put_timeline(timeline_id: str, payload: TimelineCreate):
    update_payload = TimelineUpdate(**payload.model_dump())
    return await patch_timeline(timeline_id, update_payload)


@api_router.delete("/timelines/{timeline_id}")
async def delete_timeline(timeline_id: str):
    result = await db.timelines.delete_one({"id": timeline_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Timeline not found")
    await db.timeline_events.delete_many({"timeline_id": timeline_id})
    return {"message": "Timeline deleted successfully"}


# ================== TIMELINE EVENT ROUTES ==================


@api_router.post("/timeline-events", response_model=TimelineEvent)
async def create_timeline_event(event: TimelineEventCreate):
    await assert_world_exists(event.world_id)
    await assert_timeline_exists(event.timeline_id)

    event_obj = TimelineEvent(**event.model_dump())
    await db.timeline_events.insert_one(event_obj.model_dump())
    return event_obj


@api_router.get("/timeline-events", response_model=List[TimelineEvent])
async def get_timeline_events(timeline_id: str):
    return (
        await db.timeline_events.find({"timeline_id": timeline_id}, {"_id": 0})
        .sort("sort_order", 1)
        .to_list(500)
    )


@api_router.get("/timeline-events/{event_id}", response_model=TimelineEvent)
async def get_timeline_event(event_id: str):
    event = await db.timeline_events.find_one({"id": event_id}, {"_id": 0})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@api_router.patch("/timeline-events/{event_id}", response_model=TimelineEvent)
async def patch_timeline_event(event_id: str, payload: TimelineEventUpdate):
    existing = await db.timeline_events.find_one({"id": event_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Event not found")

    update_data = non_null_update(payload)
    await db.timeline_events.update_one({"id": event_id}, {"$set": update_data})
    return await db.timeline_events.find_one({"id": event_id}, {"_id": 0})


@api_router.delete("/timeline-events/{event_id}")
async def delete_timeline_event(event_id: str):
    result = await db.timeline_events.delete_one({"id": event_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted successfully"}


# ================== CALENDAR ROUTES ==================


@api_router.post("/calendars", response_model=Calendar)
async def create_calendar(calendar: CalendarCreate):
    await assert_world_exists(calendar.world_id)
    calendar_obj = Calendar(**calendar.model_dump())
    await db.calendars.insert_one(calendar_obj.model_dump())
    return calendar_obj


@api_router.get("/calendars", response_model=List[Calendar])
async def get_calendars(world_id: str):
    await assert_world_exists(world_id)
    return await db.calendars.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/calendars/{calendar_id}", response_model=Calendar)
async def get_calendar(calendar_id: str):
    calendar = await db.calendars.find_one({"id": calendar_id}, {"_id": 0})
    if not calendar:
        raise HTTPException(status_code=404, detail="Calendar not found")
    return calendar


@api_router.patch("/calendars/{calendar_id}", response_model=Calendar)
async def patch_calendar(calendar_id: str, payload: CalendarUpdate):
    existing = await db.calendars.find_one({"id": calendar_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Calendar not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.calendars.update_one({"id": calendar_id}, {"$set": update_data})
    return await db.calendars.find_one({"id": calendar_id}, {"_id": 0})


@api_router.put("/calendars/{calendar_id}", response_model=Calendar)
async def put_calendar(calendar_id: str, payload: CalendarCreate):
    update_payload = CalendarUpdate(**payload.model_dump())
    return await patch_calendar(calendar_id, update_payload)


@api_router.delete("/calendars/{calendar_id}")
async def delete_calendar(calendar_id: str):
    result = await db.calendars.delete_one({"id": calendar_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Calendar not found")
    return {"message": "Calendar deleted successfully"}


# ================== CHRONICLE ROUTES ==================


@api_router.post("/chronicles", response_model=Chronicle)
async def create_chronicle(chronicle: ChronicleCreate):
    await assert_world_exists(chronicle.world_id)
    chronicle_obj = Chronicle(**chronicle.model_dump())
    await db.chronicles.insert_one(chronicle_obj.model_dump())
    return chronicle_obj


@api_router.get("/chronicles", response_model=List[Chronicle])
async def get_chronicles(world_id: str):
    await assert_world_exists(world_id)
    return await db.chronicles.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/chronicles/{chronicle_id}", response_model=Chronicle)
async def get_chronicle(chronicle_id: str):
    chronicle = await db.chronicles.find_one({"id": chronicle_id}, {"_id": 0})
    if not chronicle:
        raise HTTPException(status_code=404, detail="Chronicle not found")
    return chronicle


@api_router.patch("/chronicles/{chronicle_id}", response_model=Chronicle)
async def patch_chronicle(chronicle_id: str, payload: ChronicleUpdate):
    existing = await db.chronicles.find_one({"id": chronicle_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Chronicle not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.chronicles.update_one({"id": chronicle_id}, {"$set": update_data})
    return await db.chronicles.find_one({"id": chronicle_id}, {"_id": 0})


@api_router.put("/chronicles/{chronicle_id}", response_model=Chronicle)
async def put_chronicle(chronicle_id: str, payload: ChronicleCreate):
    update_payload = ChronicleUpdate(**payload.model_dump())
    return await patch_chronicle(chronicle_id, update_payload)


@api_router.delete("/chronicles/{chronicle_id}")
async def delete_chronicle(chronicle_id: str):
    result = await db.chronicles.delete_one({"id": chronicle_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chronicle not found")
    return {"message": "Chronicle deleted successfully"}


# ================== MAP ROUTES ==================


@api_router.post("/maps", response_model=Map)
async def create_map(map_data: MapCreate):
    await assert_world_exists(map_data.world_id)
    map_obj = Map(**map_data.model_dump())
    await db.maps.insert_one(map_obj.model_dump())
    return map_obj


@api_router.get("/maps", response_model=List[Map])
async def get_maps(world_id: str):
    await assert_world_exists(world_id)
    return await db.maps.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/maps/{map_id}", response_model=Map)
async def get_map(map_id: str):
    map_doc = await db.maps.find_one({"id": map_id}, {"_id": 0})
    if not map_doc:
        raise HTTPException(status_code=404, detail="Map not found")
    return map_doc


@api_router.patch("/maps/{map_id}", response_model=Map)
async def patch_map(map_id: str, payload: MapUpdate):
    existing = await db.maps.find_one({"id": map_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Map not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.maps.update_one({"id": map_id}, {"$set": update_data})
    return await db.maps.find_one({"id": map_id}, {"_id": 0})


@api_router.put("/maps/{map_id}", response_model=Map)
async def put_map(map_id: str, payload: MapCreate):
    update_payload = MapUpdate(**payload.model_dump())
    return await patch_map(map_id, update_payload)


@api_router.delete("/maps/{map_id}")
async def delete_map(map_id: str):
    result = await db.maps.delete_one({"id": map_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Map not found")
    return {"message": "Map deleted successfully"}


# ================== FAMILY TREE ROUTES ==================


@api_router.post("/family-trees", response_model=FamilyTree)
async def create_family_tree(tree: FamilyTreeCreate):
    await assert_world_exists(tree.world_id)
    tree_obj = FamilyTree(**tree.model_dump())
    await db.family_trees.insert_one(tree_obj.model_dump())
    return tree_obj


@api_router.get("/family-trees", response_model=List[FamilyTree])
async def get_family_trees(world_id: str):
    await assert_world_exists(world_id)
    return await db.family_trees.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/family-trees/{tree_id}", response_model=FamilyTree)
async def get_family_tree(tree_id: str):
    tree = await db.family_trees.find_one({"id": tree_id}, {"_id": 0})
    if not tree:
        raise HTTPException(status_code=404, detail="Family tree not found")
    return tree


@api_router.patch("/family-trees/{tree_id}", response_model=FamilyTree)
async def patch_family_tree(tree_id: str, payload: FamilyTreeUpdate):
    existing = await db.family_trees.find_one({"id": tree_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Family tree not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.family_trees.update_one({"id": tree_id}, {"$set": update_data})
    return await db.family_trees.find_one({"id": tree_id}, {"_id": 0})


@api_router.put("/family-trees/{tree_id}", response_model=FamilyTree)
async def put_family_tree(tree_id: str, payload: FamilyTreeCreate):
    update_payload = FamilyTreeUpdate(**payload.model_dump())
    return await patch_family_tree(tree_id, update_payload)


@api_router.delete("/family-trees/{tree_id}")
async def delete_family_tree(tree_id: str):
    result = await db.family_trees.delete_one({"id": tree_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Family tree not found")
    return {"message": "Family tree deleted successfully"}


# ================== VARIABLE ROUTES ==================


@api_router.post("/variables", response_model=Variable)
async def create_variable(variable: VariableCreate):
    await assert_world_exists(variable.world_id)
    variable_obj = Variable(**variable.model_dump())
    await db.variables.insert_one(variable_obj.model_dump())
    return variable_obj


@api_router.get("/variables", response_model=List[Variable])
async def get_variables(world_id: str):
    await assert_world_exists(world_id)
    return await db.variables.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/variables/{variable_id}", response_model=Variable)
async def get_variable(variable_id: str):
    variable = await db.variables.find_one({"id": variable_id}, {"_id": 0})
    if not variable:
        raise HTTPException(status_code=404, detail="Variable not found")
    return variable


@api_router.patch("/variables/{variable_id}", response_model=Variable)
async def patch_variable(variable_id: str, payload: VariableUpdate):
    existing = await db.variables.find_one({"id": variable_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Variable not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.variables.update_one({"id": variable_id}, {"$set": update_data})
    return await db.variables.find_one({"id": variable_id}, {"_id": 0})


@api_router.put("/variables/{variable_id}", response_model=Variable)
async def put_variable(variable_id: str, payload: VariableCreate):
    update_payload = VariableUpdate(**payload.model_dump())
    return await patch_variable(variable_id, update_payload)


@api_router.delete("/variables/{variable_id}")
async def delete_variable(variable_id: str):
    result = await db.variables.delete_one({"id": variable_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Variable not found")
    return {"message": "Variable deleted successfully"}


# ================== NOTEBOOK ROUTES ==================


@api_router.post("/notebooks", response_model=Notebook)
async def create_notebook(notebook: NotebookCreate):
    await assert_world_exists(notebook.world_id)
    notebook_obj = Notebook(**notebook.model_dump())
    await db.notebooks.insert_one(notebook_obj.model_dump())
    return notebook_obj


@api_router.get("/notebooks", response_model=List[Notebook])
async def get_notebooks(world_id: str):
    await assert_world_exists(world_id)
    return await db.notebooks.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/notebooks/{notebook_id}", response_model=Notebook)
async def get_notebook(notebook_id: str):
    notebook = await db.notebooks.find_one({"id": notebook_id}, {"_id": 0})
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook


@api_router.patch("/notebooks/{notebook_id}", response_model=Notebook)
async def patch_notebook(notebook_id: str, payload: NotebookUpdate):
    existing = await db.notebooks.find_one({"id": notebook_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Notebook not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.notebooks.update_one({"id": notebook_id}, {"$set": update_data})
    return await db.notebooks.find_one({"id": notebook_id}, {"_id": 0})


@api_router.put("/notebooks/{notebook_id}", response_model=Notebook)
async def put_notebook(notebook_id: str, payload: NotebookCreate):
    update_payload = NotebookUpdate(**payload.model_dump())
    return await patch_notebook(notebook_id, update_payload)


@api_router.delete("/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    result = await db.notebooks.delete_one({"id": notebook_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"message": "Notebook deleted successfully"}


# ================== TODO ROUTES ==================


@api_router.post("/todos", response_model=Todo)
async def create_todo(todo: TodoCreate):
    await assert_world_exists(todo.world_id)
    todo_obj = Todo(**todo.model_dump())
    await db.todos.insert_one(todo_obj.model_dump())
    return todo_obj


@api_router.get("/todos", response_model=List[Todo])
async def get_todos(world_id: str):
    await assert_world_exists(world_id)
    return await db.todos.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/todos/{todo_id}", response_model=Todo)
async def get_todo(todo_id: str):
    todo = await db.todos.find_one({"id": todo_id}, {"_id": 0})
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo


@api_router.patch("/todos/{todo_id}", response_model=Todo)
async def patch_todo(todo_id: str, payload: TodoUpdate):
    existing = await db.todos.find_one({"id": todo_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Todo not found")

    update_data = non_null_update(payload)
    update_data["updated_at"] = utc_now_iso()

    await db.todos.update_one({"id": todo_id}, {"$set": update_data})
    return await db.todos.find_one({"id": todo_id}, {"_id": 0})


@api_router.put("/todos/{todo_id}", response_model=Todo)
async def put_todo(todo_id: str, payload: TodoCreate):
    update_payload = TodoUpdate(**payload.model_dump())
    return await patch_todo(todo_id, update_payload)


@api_router.delete("/todos/{todo_id}")
async def delete_todo(todo_id: str):
    result = await db.todos.delete_one({"id": todo_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"message": "Todo deleted successfully"}


# ================== DIPLOMATIC RELATION ROUTES ==================


@api_router.post("/diplomatic-relations", response_model=DiplomaticRelation)
async def create_diplomatic_relation(relation: DiplomaticRelationCreate):
    await assert_world_exists(relation.world_id)
    relation_obj = DiplomaticRelation(**relation.model_dump())
    await db.diplomatic_relations.insert_one(relation_obj.model_dump())
    return relation_obj


@api_router.get("/diplomatic-relations", response_model=List[DiplomaticRelation])
async def get_diplomatic_relations(world_id: str):
    await assert_world_exists(world_id)
    return await db.diplomatic_relations.find({"world_id": world_id}, {"_id": 0}).to_list(100)


@api_router.get("/diplomatic-relations/{relation_id}", response_model=DiplomaticRelation)
async def get_diplomatic_relation(relation_id: str):
    rel = await db.diplomatic_relations.find_one({"id": relation_id}, {"_id": 0})
    if not rel:
        raise HTTPException(status_code=404, detail="Relation not found")
    return rel


@api_router.patch("/diplomatic-relations/{relation_id}", response_model=DiplomaticRelation)
async def patch_diplomatic_relation(relation_id: str, payload: DiplomaticRelationUpdate):
    existing = await db.diplomatic_relations.find_one({"id": relation_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Relation not found")

    update_data = non_null_update(payload)
    await db.diplomatic_relations.update_one({"id": relation_id}, {"$set": update_data})
    return await db.diplomatic_relations.find_one({"id": relation_id}, {"_id": 0)


@api_router.delete("/diplomatic-relations/{relation_id}")
async def delete_diplomatic_relation(relation_id: str):
    result = await db.diplomatic_relations.delete_one({"id": relation_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Relation not found")
    return {"message": "Relation deleted successfully"}


# ================== STATS ROUTES ==================


@api_router.get("/stats/{world_id}")
async def get_world_stats(world_id: str):
    await assert_world_exists(world_id)

    articles_count = await db.articles.count_documents({"world_id": world_id})
    timelines_count = await db.timelines.count_documents({"world_id": world_id})
    characters_count = await db.articles.count_documents(
        {"world_id": world_id, "article_type": ArticleType.CHARACTER.value}
    )
    locations_count = await db.articles.count_documents(
        {
            "world_id": world_id,
            "article_type": {
                "$in": [
                    ArticleType.SETTLEMENT.value,
                    ArticleType.GEOGRAPHY.value,
                    ArticleType.BUILDING.value,
                    ArticleType.COUNTRY.value,
                ]
            },
        }
    )
    chronicles_count = await db.chronicles.count_documents({"world_id": world_id})
    maps_count = await db.maps.count_documents({"world_id": world_id})

    return {
        "articles": articles_count,
        "timelines": timelines_count,
        "characters": characters_count,
        "locations": locations_count,
        "chronicles": chronicles_count,
        "maps": maps_count,
    }


# ================== APP WIRING ==================

app.include_router(api_router)

origins_raw = os.environ.get("CORS_ORIGINS", "*").strip()
allow_origins = ["*"] if origins_raw == "*" else [o.strip() for o in origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
