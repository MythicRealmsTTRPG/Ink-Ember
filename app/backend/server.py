"from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
from enum import Enum

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title=\"Ink & Ember API\", version=\"1.0.0\")

# Create a router with the /api prefix
api_router = APIRouter(prefix=\"/api\")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== ENUMS ==============

class ArticleType(str, Enum):
    GENERIC = \"generic\"
    BUILDING = \"building\"
    CHARACTER = \"character\"
    COUNTRY = \"country\"
    MILITARY = \"military\"
    GODS_DEITIES = \"gods_deities\"
    GEOGRAPHY = \"geography\"
    ITEM = \"item\"
    ORGANIZATION = \"organization\"
    RELIGION = \"religion\"
    SPECIES = \"species\"
    VEHICLE = \"vehicle\"
    SETTLEMENT = \"settlement\"
    CONDITION = \"condition\"
    CONFLICT = \"conflict\"
    DOCUMENT = \"document\"
    CULTURE_ETHNICITY = \"culture_ethnicity\"
    LANGUAGE = \"language\"
    MATERIAL = \"material\"
    MILITARY_FORMATION = \"military_formation\"
    MYTH = \"myth\"
    NATURAL_LAW = \"natural_law\"
    PLOT = \"plot\"
    PROFESSION = \"profession\"
    PROSE = \"prose\"
    TITLE = \"title\"
    SPELL = \"spell\"
    TECHNOLOGY = \"technology\"
    TRADITION = \"tradition\"
    SESSION_REPORT = \"session_report\"

# ============== MODELS ==============

class WorldBase(BaseModel):
    name: str
    description: Optional[str] = None
    genre: Optional[str] = None
    cover_image: Optional[str] = None
    settings: Optional[Dict[str, Any]] = {}

class World(WorldBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class WorldCreate(WorldBase):
    pass

class ArticleBase(BaseModel):
    world_id: str
    title: str
    article_type: ArticleType = ArticleType.GENERIC
    content: Optional[str] = \"\"
    summary: Optional[str] = None
    cover_image: Optional[str] = None
    infobox: Optional[Dict[str, Any]] = {}
    tags: Optional[List[str]] = []
    linked_articles: Optional[List[str]] = []
    is_secret: Optional[bool] = False
    custom_fields: Optional[Dict[str, Any]] = {}

class Article(ArticleBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
    color: Optional[str] = \"#ff4500\"

class Timeline(TimelineBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class TimelineCreate(TimelineBase):
    pass

class TimelineEventBase(BaseModel):
    world_id: str
    timeline_id: str
    title: str
    description: Optional[str] = None
    date_label: str
    sort_order: int = 0
    linked_articles: Optional[List[str]] = []
    color: Optional[str] = None

class TimelineEvent(TimelineEventBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class TimelineEventCreate(TimelineEventBase):
    pass

class CalendarBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    months: Optional[List[Dict[str, Any]]] = []
    days_per_week: Optional[int] = 7
    day_names: Optional[List[str]] = []

class Calendar(CalendarBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CalendarCreate(CalendarBase):
    pass

class ChronicleBase(BaseModel):
    world_id: str
    title: str
    description: Optional[str] = None
    chronicle_type: Optional[str] = \"campaign_log\"
    entries: Optional[List[Dict[str, Any]]] = []
    linked_timeline_id: Optional[str] = None

class Chronicle(ChronicleBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ChronicleCreate(ChronicleBase):
    pass

class MapBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    markers: Optional[List[Dict[str, Any]]] = []

class Map(MapBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class MapCreate(MapBase):
    pass

class FamilyTreeBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = []
    connections: Optional[List[Dict[str, Any]]] = []

class FamilyTree(FamilyTreeBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class FamilyTreeCreate(FamilyTreeBase):
    pass

class VariableBase(BaseModel):
    world_id: str
    name: str
    description: Optional[str] = None
    variable_type: Optional[str] = \"world_state\"
    value: Optional[str] = None
    options: Optional[List[str]] = []
    is_active: Optional[bool] = True

class Variable(VariableBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class VariableCreate(VariableBase):
    pass

class NotebookBase(BaseModel):
    world_id: str
    title: str
    content: Optional[str] = \"\"
    notebook_type: Optional[str] = \"note\"

class Notebook(NotebookBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class NotebookCreate(NotebookBase):
    pass

class TodoBase(BaseModel):
    world_id: str
    title: str
    description: Optional[str] = None
    completed: Optional[bool] = False
    priority: Optional[str] = \"medium\"
    due_date: Optional[str] = None

class Todo(TodoBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class TodoCreate(TodoBase):
    pass

class DiplomaticRelationBase(BaseModel):
    world_id: str
    entity1_id: str
    entity1_name: str
    entity2_id: str
    entity2_name: str
    relation_type: Optional[str] = \"neutral\"
    description: Optional[str] = None

class DiplomaticRelation(DiplomaticRelationBase):
    model_config = ConfigDict(extra=\"ignore\")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DiplomaticRelationCreate(DiplomaticRelationBase):
    pass

# ============== WORLD ROUTES ==============

@api_router.get(\"/\")
async def root():
    return {\"message\": \"Ink & Ember API\", \"version\": \"1.0.0\"}

@api_router.post(\"/worlds\", response_model=World)
async def create_world(world: WorldCreate):
    world_obj = World(**world.model_dump())
    doc = world_obj.model_dump()
    await db.worlds.insert_one(doc)
    return world_obj

@api_router.get(\"/worlds\", response_model=List[World])
async def get_worlds():
    worlds = await db.worlds.find({}, {\"_id\": 0}).to_list(1000)
    return worlds

@api_router.get(\"/worlds/{world_id}\", response_model=World)
async def get_world(world_id: str):
    world = await db.worlds.find_one({\"id\": world_id}, {\"_id\": 0})
    if not world:
        raise HTTPException(status_code=404, detail=\"World not found\")
    return world

@api_router.put(\"/worlds/{world_id}\", response_model=World)
async def update_world(world_id: str, world: WorldCreate):
    existing = await db.worlds.find_one({\"id\": world_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"World not found\")
    update_data = world.model_dump()
    update_data[\"updated_at\"] = datetime.now(timezone.utc).isoformat()
    await db.worlds.update_one({\"id\": world_id}, {\"$set\": update_data})
    updated = await db.worlds.find_one({\"id\": world_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/worlds/{world_id}\")
async def delete_world(world_id: str):
    result = await db.worlds.delete_one({\"id\": world_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"World not found\")
    # Also delete all related data
    await db.articles.delete_many({\"world_id\": world_id})
    await db.timelines.delete_many({\"world_id\": world_id})
    await db.timeline_events.delete_many({\"world_id\": world_id})
    await db.calendars.delete_many({\"world_id\": world_id})
    await db.chronicles.delete_many({\"world_id\": world_id})
    await db.maps.delete_many({\"world_id\": world_id})
    await db.family_trees.delete_many({\"world_id\": world_id})
    await db.variables.delete_many({\"world_id\": world_id})
    await db.notebooks.delete_many({\"world_id\": world_id})
    await db.todos.delete_many({\"world_id\": world_id})
    await db.diplomatic_relations.delete_many({\"world_id\": world_id})
    return {\"message\": \"World deleted successfully\"}

# ============== ARTICLE ROUTES ==============

@api_router.post(\"/articles\", response_model=Article)
async def create_article(article: ArticleCreate):
    article_obj = Article(**article.model_dump())
    doc = article_obj.model_dump()
    await db.articles.insert_one(doc)
    return article_obj

@api_router.get(\"/articles\", response_model=List[Article])
async def get_articles(
    world_id: str,
    article_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=100, le=1000)
):
    query = {\"world_id\": world_id}
    if article_type:
        query[\"article_type\"] = article_type
    if search:
        query[\"$or\"] = [
            {\"title\": {\"$regex\": search, \"$options\": \"i\"}},
            {\"content\": {\"$regex\": search, \"$options\": \"i\"}},
            {\"tags\": {\"$in\": [search]}}
        ]
    articles = await db.articles.find(query, {\"_id\": 0}).to_list(limit)
    return articles

@api_router.get(\"/articles/{article_id}\", response_model=Article)
async def get_article(article_id: str):
    article = await db.articles.find_one({\"id\": article_id}, {\"_id\": 0})
    if not article:
        raise HTTPException(status_code=404, detail=\"Article not found\")
    return article

@api_router.put(\"/articles/{article_id}\", response_model=Article)
async def update_article(article_id: str, article: ArticleUpdate):
    existing = await db.articles.find_one({\"id\": article_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Article not found\")
    update_data = {k: v for k, v in article.model_dump().items() if v is not None}
    update_data[\"updated_at\"] = datetime.now(timezone.utc).isoformat()
    await db.articles.update_one({\"id\": article_id}, {\"$set\": update_data})
    updated = await db.articles.find_one({\"id\": article_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/articles/{article_id}\")
async def delete_article(article_id: str):
    result = await db.articles.delete_one({\"id\": article_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Article not found\")
    return {\"message\": \"Article deleted successfully\"}

# ============== TIMELINE ROUTES ==============

@api_router.post(\"/timelines\", response_model=Timeline)
async def create_timeline(timeline: TimelineCreate):
    timeline_obj = Timeline(**timeline.model_dump())
    doc = timeline_obj.model_dump()
    await db.timelines.insert_one(doc)
    return timeline_obj

@api_router.get(\"/timelines\", response_model=List[Timeline])
async def get_timelines(world_id: str):
    timelines = await db.timelines.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return timelines

@api_router.get(\"/timelines/{timeline_id}\", response_model=Timeline)
async def get_timeline(timeline_id: str):
    timeline = await db.timelines.find_one({\"id\": timeline_id}, {\"_id\": 0})
    if not timeline:
        raise HTTPException(status_code=404, detail=\"Timeline not found\")
    return timeline

@api_router.delete(\"/timelines/{timeline_id}\")
async def delete_timeline(timeline_id: str):
    result = await db.timelines.delete_one({\"id\": timeline_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Timeline not found\")
    await db.timeline_events.delete_many({\"timeline_id\": timeline_id})
    return {\"message\": \"Timeline deleted successfully\"}

# ============== TIMELINE EVENT ROUTES ==============

@api_router.post(\"/timeline-events\", response_model=TimelineEvent)
async def create_timeline_event(event: TimelineEventCreate):
    event_obj = TimelineEvent(**event.model_dump())
    doc = event_obj.model_dump()
    await db.timeline_events.insert_one(doc)
    return event_obj

@api_router.get(\"/timeline-events\", response_model=List[TimelineEvent])
async def get_timeline_events(timeline_id: str):
    events = await db.timeline_events.find({\"timeline_id\": timeline_id}, {\"_id\": 0}).sort(\"sort_order\", 1).to_list(500)
    return events

@api_router.delete(\"/timeline-events/{event_id}\")
async def delete_timeline_event(event_id: str):
    result = await db.timeline_events.delete_one({\"id\": event_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Event not found\")
    return {\"message\": \"Event deleted successfully\"}

# ============== CALENDAR ROUTES ==============

@api_router.post(\"/calendars\", response_model=Calendar)
async def create_calendar(calendar: CalendarCreate):
    calendar_obj = Calendar(**calendar.model_dump())
    doc = calendar_obj.model_dump()
    await db.calendars.insert_one(doc)
    return calendar_obj

@api_router.get(\"/calendars\", response_model=List[Calendar])
async def get_calendars(world_id: str):
    calendars = await db.calendars.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return calendars

@api_router.get(\"/calendars/{calendar_id}\", response_model=Calendar)
async def get_calendar(calendar_id: str):
    calendar = await db.calendars.find_one({\"id\": calendar_id}, {\"_id\": 0})
    if not calendar:
        raise HTTPException(status_code=404, detail=\"Calendar not found\")
    return calendar

@api_router.put(\"/calendars/{calendar_id}\", response_model=Calendar)
async def update_calendar(calendar_id: str, calendar: CalendarCreate):
    existing = await db.calendars.find_one({\"id\": calendar_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Calendar not found\")
    await db.calendars.update_one({\"id\": calendar_id}, {\"$set\": calendar.model_dump()})
    updated = await db.calendars.find_one({\"id\": calendar_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/calendars/{calendar_id}\")
async def delete_calendar(calendar_id: str):
    result = await db.calendars.delete_one({\"id\": calendar_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Calendar not found\")
    return {\"message\": \"Calendar deleted successfully\"}

# ============== CHRONICLE ROUTES ==============

@api_router.post(\"/chronicles\", response_model=Chronicle)
async def create_chronicle(chronicle: ChronicleCreate):
    chronicle_obj = Chronicle(**chronicle.model_dump())
    doc = chronicle_obj.model_dump()
    await db.chronicles.insert_one(doc)
    return chronicle_obj

@api_router.get(\"/chronicles\", response_model=List[Chronicle])
async def get_chronicles(world_id: str):
    chronicles = await db.chronicles.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return chronicles

@api_router.get(\"/chronicles/{chronicle_id}\", response_model=Chronicle)
async def get_chronicle(chronicle_id: str):
    chronicle = await db.chronicles.find_one({\"id\": chronicle_id}, {\"_id\": 0})
    if not chronicle:
        raise HTTPException(status_code=404, detail=\"Chronicle not found\")
    return chronicle

@api_router.put(\"/chronicles/{chronicle_id}\", response_model=Chronicle)
async def update_chronicle(chronicle_id: str, chronicle: ChronicleCreate):
    existing = await db.chronicles.find_one({\"id\": chronicle_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Chronicle not found\")
    update_data = chronicle.model_dump()
    update_data[\"updated_at\"] = datetime.now(timezone.utc).isoformat()
    await db.chronicles.update_one({\"id\": chronicle_id}, {\"$set\": update_data})
    updated = await db.chronicles.find_one({\"id\": chronicle_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/chronicles/{chronicle_id}\")
async def delete_chronicle(chronicle_id: str):
    result = await db.chronicles.delete_one({\"id\": chronicle_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Chronicle not found\")
    return {\"message\": \"Chronicle deleted successfully\"}

# ============== MAP ROUTES ==============

@api_router.post(\"/maps\", response_model=Map)
async def create_map(map_data: MapCreate):
    map_obj = Map(**map_data.model_dump())
    doc = map_obj.model_dump()
    await db.maps.insert_one(doc)
    return map_obj

@api_router.get(\"/maps\", response_model=List[Map])
async def get_maps(world_id: str):
    maps = await db.maps.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return maps

@api_router.get(\"/maps/{map_id}\", response_model=Map)
async def get_map(map_id: str):
    map_doc = await db.maps.find_one({\"id\": map_id}, {\"_id\": 0})
    if not map_doc:
        raise HTTPException(status_code=404, detail=\"Map not found\")
    return map_doc

@api_router.put(\"/maps/{map_id}\", response_model=Map)
async def update_map(map_id: str, map_data: MapCreate):
    existing = await db.maps.find_one({\"id\": map_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Map not found\")
    await db.maps.update_one({\"id\": map_id}, {\"$set\": map_data.model_dump()})
    updated = await db.maps.find_one({\"id\": map_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/maps/{map_id}\")
async def delete_map(map_id: str):
    result = await db.maps.delete_one({\"id\": map_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Map not found\")
    return {\"message\": \"Map deleted successfully\"}

# ============== FAMILY TREE ROUTES ==============

@api_router.post(\"/family-trees\", response_model=FamilyTree)
async def create_family_tree(tree: FamilyTreeCreate):
    tree_obj = FamilyTree(**tree.model_dump())
    doc = tree_obj.model_dump()
    await db.family_trees.insert_one(doc)
    return tree_obj

@api_router.get(\"/family-trees\", response_model=List[FamilyTree])
async def get_family_trees(world_id: str):
    trees = await db.family_trees.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return trees

@api_router.get(\"/family-trees/{tree_id}\", response_model=FamilyTree)
async def get_family_tree(tree_id: str):
    tree = await db.family_trees.find_one({\"id\": tree_id}, {\"_id\": 0})
    if not tree:
        raise HTTPException(status_code=404, detail=\"Family tree not found\")
    return tree

@api_router.put(\"/family-trees/{tree_id}\", response_model=FamilyTree)
async def update_family_tree(tree_id: str, tree: FamilyTreeCreate):
    existing = await db.family_trees.find_one({\"id\": tree_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Family tree not found\")
    await db.family_trees.update_one({\"id\": tree_id}, {\"$set\": tree.model_dump()})
    updated = await db.family_trees.find_one({\"id\": tree_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/family-trees/{tree_id}\")
async def delete_family_tree(tree_id: str):
    result = await db.family_trees.delete_one({\"id\": tree_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Family tree not found\")
    return {\"message\": \"Family tree deleted successfully\"}

# ============== VARIABLE ROUTES ==============

@api_router.post(\"/variables\", response_model=Variable)
async def create_variable(variable: VariableCreate):
    variable_obj = Variable(**variable.model_dump())
    doc = variable_obj.model_dump()
    await db.variables.insert_one(doc)
    return variable_obj

@api_router.get(\"/variables\", response_model=List[Variable])
async def get_variables(world_id: str):
    variables = await db.variables.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return variables

@api_router.put(\"/variables/{variable_id}\", response_model=Variable)
async def update_variable(variable_id: str, variable: VariableCreate):
    existing = await db.variables.find_one({\"id\": variable_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Variable not found\")
    await db.variables.update_one({\"id\": variable_id}, {\"$set\": variable.model_dump()})
    updated = await db.variables.find_one({\"id\": variable_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/variables/{variable_id}\")
async def delete_variable(variable_id: str):
    result = await db.variables.delete_one({\"id\": variable_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Variable not found\")
    return {\"message\": \"Variable deleted successfully\"}

# ============== NOTEBOOK ROUTES ==============

@api_router.post(\"/notebooks\", response_model=Notebook)
async def create_notebook(notebook: NotebookCreate):
    notebook_obj = Notebook(**notebook.model_dump())
    doc = notebook_obj.model_dump()
    await db.notebooks.insert_one(doc)
    return notebook_obj

@api_router.get(\"/notebooks\", response_model=List[Notebook])
async def get_notebooks(world_id: str):
    notebooks = await db.notebooks.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return notebooks

@api_router.get(\"/notebooks/{notebook_id}\", response_model=Notebook)
async def get_notebook(notebook_id: str):
    notebook = await db.notebooks.find_one({\"id\": notebook_id}, {\"_id\": 0})
    if not notebook:
        raise HTTPException(status_code=404, detail=\"Notebook not found\")
    return notebook

@api_router.put(\"/notebooks/{notebook_id}\", response_model=Notebook)
async def update_notebook(notebook_id: str, notebook: NotebookCreate):
    existing = await db.notebooks.find_one({\"id\": notebook_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Notebook not found\")
    update_data = notebook.model_dump()
    update_data[\"updated_at\"] = datetime.now(timezone.utc).isoformat()
    await db.notebooks.update_one({\"id\": notebook_id}, {\"$set\": update_data})
    updated = await db.notebooks.find_one({\"id\": notebook_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/notebooks/{notebook_id}\")
async def delete_notebook(notebook_id: str):
    result = await db.notebooks.delete_one({\"id\": notebook_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Notebook not found\")
    return {\"message\": \"Notebook deleted successfully\"}

# ============== TODO ROUTES ==============

@api_router.post(\"/todos\", response_model=Todo)
async def create_todo(todo: TodoCreate):
    todo_obj = Todo(**todo.model_dump())
    doc = todo_obj.model_dump()
    await db.todos.insert_one(doc)
    return todo_obj

@api_router.get(\"/todos\", response_model=List[Todo])
async def get_todos(world_id: str):
    todos = await db.todos.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return todos

@api_router.put(\"/todos/{todo_id}\", response_model=Todo)
async def update_todo(todo_id: str, todo: TodoCreate):
    existing = await db.todos.find_one({\"id\": todo_id})
    if not existing:
        raise HTTPException(status_code=404, detail=\"Todo not found\")
    await db.todos.update_one({\"id\": todo_id}, {\"$set\": todo.model_dump()})
    updated = await db.todos.find_one({\"id\": todo_id}, {\"_id\": 0})
    return updated

@api_router.delete(\"/todos/{todo_id}\")
async def delete_todo(todo_id: str):
    result = await db.todos.delete_one({\"id\": todo_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Todo not found\")
    return {\"message\": \"Todo deleted successfully\"}

# ============== DIPLOMATIC RELATION ROUTES ==============

@api_router.post(\"/diplomatic-relations\", response_model=DiplomaticRelation)
async def create_diplomatic_relation(relation: DiplomaticRelationCreate):
    relation_obj = DiplomaticRelation(**relation.model_dump())
    doc = relation_obj.model_dump()
    await db.diplomatic_relations.insert_one(doc)
    return relation_obj

@api_router.get(\"/diplomatic-relations\", response_model=List[DiplomaticRelation])
async def get_diplomatic_relations(world_id: str):
    relations = await db.diplomatic_relations.find({\"world_id\": world_id}, {\"_id\": 0}).to_list(100)
    return relations

@api_router.delete(\"/diplomatic-relations/{relation_id}\")
async def delete_diplomatic_relation(relation_id: str):
    result = await db.diplomatic_relations.delete_one({\"id\": relation_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Relation not found\")
    return {\"message\": \"Relation deleted successfully\"}

# ============== STATS ROUTES ==============

@api_router.get(\"/stats/{world_id}\")
async def get_world_stats(world_id: str):
    articles_count = await db.articles.count_documents({\"world_id\": world_id})
    timelines_count = await db.timelines.count_documents({\"world_id\": world_id})
    characters_count = await db.articles.count_documents({\"world_id\": world_id, \"article_type\": \"character\"})
    locations_count = await db.articles.count_documents({\"world_id\": world_id, \"article_type\": {\"$in\": [\"settlement\", \"geography\", \"building\", \"country\"]}})
    chronicles_count = await db.chronicles.count_documents({\"world_id\": world_id})
    maps_count = await db.maps.count_documents({\"world_id\": world_id})
    
    return {
        \"articles\": articles_count,
        \"timelines\": timelines_count,
        \"characters\": characters_count,
        \"locations\": locations_count,
        \"chronicles\": chronicles_count,
        \"maps\": maps_count
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

@app.on_event(\"shutdown\")
async def shutdown_db_client():
    client.close()
"