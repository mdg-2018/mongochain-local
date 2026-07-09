"""Memory storage using MongoDB Atlas for mongochain.

Uses Atlas Vector Search Automated Embedding (`autoEmbed`) — no client-side
embedding generation. Atlas manages embeddings for the indexed text field.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure

from .config import MemoryConfig


# ponytail: autoEmbed indexes the raw text field directly; no client-side vectors.
class MemoryStore:
    """Handles all memory operations with MongoDB.

    Manages three types of memory:
    - conversation_history: Raw chat log (90-day TTL)
    - short_term_memory: Session summaries per user (7-day TTL)
    - long_term_memory: Persistent user-specific facts

    Vector search is powered by Atlas Automated Embedding (`autoEmbed`) —
    embeddings are generated and managed by Atlas at index-time and query-time.
    """

    CONVERSATION = "conversation_history"
    SHORT_TERM = "short_term_memory"
    LONG_TERM = "long_term_memory"
    AGENT_METADATA = "agent_metadata"
    AGENT_TOOLS = "agent_tools"

    # Field on each collection that gets auto-embedded.
    CONVERSATION_TEXT_FIELD = "content"
    SHORT_TERM_TEXT_FIELD = "summary"
    LONG_TERM_TEXT_FIELD = "content"

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        config: Optional[MemoryConfig] = None
    ):
        self.db_name = db_name
        self.config = config or MemoryConfig()

        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]

        self._conversation = self._db[self.CONVERSATION]
        self._short_term = self._db[self.SHORT_TERM]
        self._long_term = self._db[self.LONG_TERM]
        self._agent_metadata = self._db[self.AGENT_METADATA]
        self._agent_tools = self._db[self.AGENT_TOOLS]

    def setup_collections(self) -> dict:
        """Create collections and indexes."""
        status = {
            "collections_created": [],
            "indexes_created": [],
            "vector_index_status": []
        }

        for coll_name in [self.CONVERSATION, self.SHORT_TERM, self.LONG_TERM, self.AGENT_METADATA, self.AGENT_TOOLS]:
            if coll_name not in self._db.list_collection_names():
                self._db.create_collection(coll_name)
                status["collections_created"].append(coll_name)

        self._conversation.create_index("expires_at", expireAfterSeconds=0)
        status["indexes_created"].append(f"{self.CONVERSATION}.expires_at (TTL)")
        self._conversation.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
        status["indexes_created"].append(f"{self.CONVERSATION}.user_id+timestamp")
        status["vector_index_status"].append({
            "collection": self.CONVERSATION,
            **self._create_auto_embed_index(self._conversation, "conversation_vector_index", self.CONVERSATION_TEXT_FIELD),
        })

        self._short_term.create_index("expires_at", expireAfterSeconds=0)
        status["indexes_created"].append(f"{self.SHORT_TERM}.expires_at (TTL)")
        self._short_term.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
        status["indexes_created"].append(f"{self.SHORT_TERM}.user_id+created_at")
        status["vector_index_status"].append({
            "collection": self.SHORT_TERM,
            **self._create_auto_embed_index(self._short_term, "short_term_vector_index", self.SHORT_TERM_TEXT_FIELD),
        })

        self._long_term.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
        status["indexes_created"].append(f"{self.LONG_TERM}.user_id+timestamp")
        status["vector_index_status"].append({
            "collection": self.LONG_TERM,
            **self._create_auto_embed_index(self._long_term, "long_term_vector_index", self.LONG_TERM_TEXT_FIELD),
        })

        return status

    def _create_auto_embed_index(self, collection, index_name: str, path: str) -> dict:
        """Create an Atlas Vector Search index using autoEmbed on the text field.

        Atlas generates and manages embeddings for `path` at index- and query-time
        using the configured Voyage AI model. No API key handling on our side.

        Returns a dict: {"name": ..., "status": "created"|"exists"|"failed", "error": str|None}.
        """
        try:
            for idx in collection.list_search_indexes():
                if idx.get("name") == index_name:
                    return {"name": index_name, "status": "exists", "error": None}
        except OperationFailure:
            pass

        definition = {
            "fields": [
                {
                    "type": "autoEmbed",
                    "modality": "text",
                    "path": path,
                    "model": self.config.embedding_model,
                },
                {"type": "filter", "path": "user_id"},
            ]
        }
        if collection.name == self.LONG_TERM:
            definition["fields"].append({"type": "filter", "path": "category"})

        try:
            collection.create_search_index({
                "name": index_name,
                "type": "vectorSearch",
                "definition": definition,
            })
            return {"name": index_name, "status": "created", "error": None}
        except OperationFailure as e:
            msg = getattr(e, "details", {}).get("errmsg") if hasattr(e, "details") else None
            msg = msg or str(e)
            if "already exists" in msg.lower():
                return {"name": index_name, "status": "exists", "error": None}
            return {"name": index_name, "status": "failed", "error": msg}

    # ==================== Conversation History ====================

    def store_conversation(self, user_id: str, role: str, content: str) -> str:
        """Store a message in conversation history. Atlas auto-embeds `content`."""
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.config.conversation_ttl_days)
        doc = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
            "expires_at": expires_at,
        }
        return str(self._conversation.insert_one(doc).inserted_id)

    def get_conversation_history(self, user_id: str, limit: Optional[int] = None) -> list[dict]:
        limit = limit or self.config.conversation_limit
        cursor = self._conversation.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        messages = list(cursor)
        messages.reverse()
        return messages

    def search_conversations(self, user_id: str, query_text: str, limit: int = 5) -> list[dict]:
        """Vector search over conversations via autoEmbed."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "conversation_vector_index",
                    "path": self.CONVERSATION_TEXT_FIELD,
                    "query": query_text,
                    "numCandidates": limit * 20,
                    "limit": limit * 2,
                    "filter": {"user_id": user_id},
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "role": 1,
                    "content": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
            {"$limit": limit},
        ]
        try:
            return list(self._conversation.aggregate(pipeline))
        except OperationFailure:
            return []

    # ==================== Short-Term Memory ====================

    def store_short_term(
        self,
        user_id: str,
        summary: str,
        topics_discussed: list[str],
        actions_taken: list[str],
        session_id: Optional[str] = None,
    ) -> str:
        """Store a session summary. Atlas auto-embeds `summary`."""
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.config.short_term_ttl_days)
        doc = {
            "user_id": user_id,
            "summary": summary,
            "topics_discussed": topics_discussed,
            "actions_taken": actions_taken,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "expires_at": expires_at,
        }
        return str(self._short_term.insert_one(doc).inserted_id)

    def get_recent_short_term(self, user_id: str, limit: Optional[int] = None) -> list[dict]:
        limit = limit or self.config.short_term_limit
        cursor = self._short_term.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        return list(cursor)

    def search_short_term(self, user_id: str, query_text: str, limit: Optional[int] = None) -> list[dict]:
        """Vector search over short-term memory via autoEmbed."""
        limit = limit or self.config.short_term_limit
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "short_term_vector_index",
                    "path": self.SHORT_TERM_TEXT_FIELD,
                    "query": query_text,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {"user_id": user_id},
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "summary": 1,
                    "topics_discussed": 1,
                    "actions_taken": 1,
                    "created_at": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        try:
            return list(self._short_term.aggregate(pipeline))
        except OperationFailure:
            return self.get_recent_short_term(user_id, limit)

    # ==================== Long-Term Memory ====================

    def store_long_term(
        self,
        user_id: str,
        content: str,
        category: str = "general",
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a user-specific fact. Atlas auto-embeds `content`."""
        doc = {
            "user_id": user_id,
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc),
        }
        return str(self._long_term.insert_one(doc).inserted_id)

    def search_long_term(
        self,
        user_id: str,
        query_text: str,
        limit: Optional[int] = None,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Vector search over long-term memory via autoEmbed."""
        limit = limit or self.config.long_term_limit
        filter_doc: dict = {"user_id": user_id}
        if category:
            filter_doc["category"] = category

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "long_term_vector_index",
                    "path": self.LONG_TERM_TEXT_FIELD,
                    "query": query_text,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": filter_doc,
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "content": 1,
                    "category": 1,
                    "metadata": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        try:
            return list(self._long_term.aggregate(pipeline))
        except OperationFailure:
            return self.get_user_long_term(user_id, limit)

    def get_user_long_term(
        self,
        user_id: str,
        limit: int = 10,
        category: Optional[str] = None,
    ) -> list[dict]:
        query = {"user_id": user_id}
        if category:
            query["category"] = category
        cursor = self._long_term.find(query).sort("timestamp", -1).limit(limit)
        return list(cursor)

    # ==================== Agent Metadata ====================

    def save_agent_metadata(
        self,
        name: str,
        persona: str,
        llm_provider: str,
        llm_model: str,
        collaborators: list[str],
        description: Optional[str] = None,
    ) -> dict:
        now = datetime.now(timezone.utc)
        existing = self._agent_metadata.find_one({"_id": "agent_config"})

        doc = {
            "_id": "agent_config",
            "name": name,
            "persona": persona,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "collaborators": collaborators,
            "description": description,
            "updated_at": now,
            "created_at": existing.get("created_at", now) if existing else now,
        }

        self._agent_metadata.replace_one({"_id": "agent_config"}, doc, upsert=True)
        return doc

    def get_agent_metadata(self) -> Optional[dict]:
        return self._agent_metadata.find_one({"_id": "agent_config"})

    def update_agent_persona(self, persona: str) -> bool:
        result = self._agent_metadata.update_one(
            {"_id": "agent_config"},
            {"$set": {"persona": persona, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0

    def update_agent_collaborators(self, collaborators: list[str]) -> bool:
        result = self._agent_metadata.update_one(
            {"_id": "agent_config"},
            {"$set": {"collaborators": collaborators, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0

    # ==================== Agent Tools ====================

    def register_tool(self, name: str, description: str, parameters: dict) -> str:
        now = datetime.now(timezone.utc)
        doc = {
            "_id": name,
            "name": name,
            "description": description,
            "parameters": parameters,
            "created_at": now,
            "updated_at": now,
        }
        self._agent_tools.replace_one({"_id": name}, doc, upsert=True)
        self._agent_metadata.update_one(
            {"_id": "agent_config"},
            {"$addToSet": {"tools": name}, "$set": {"updated_at": now}},
        )
        return name

    def get_tools(self) -> list[dict]:
        return list(self._agent_tools.find({}))

    def get_tool(self, name: str) -> Optional[dict]:
        return self._agent_tools.find_one({"_id": name})

    def remove_tool(self, name: str) -> bool:
        result = self._agent_tools.delete_one({"_id": name})
        if result.deleted_count > 0:
            self._agent_metadata.update_one(
                {"_id": "agent_config"},
                {"$pull": {"tools": name}, "$set": {"updated_at": datetime.now(timezone.utc)}},
            )
            return True
        return False

    # ==================== Utility Methods ====================

    def clear_user_conversation(self, user_id: str):
        self._conversation.delete_many({"user_id": user_id})

    def clear_user_short_term(self, user_id: str):
        self._short_term.delete_many({"user_id": user_id})

    def clear_user_long_term(self, user_id: str):
        self._long_term.delete_many({"user_id": user_id})

    def clear_user_all(self, user_id: str):
        self.clear_user_conversation(user_id)
        self.clear_user_short_term(user_id)
        self.clear_user_long_term(user_id)

    def clear_all(self):
        self._conversation.delete_many({})
        self._short_term.delete_many({})
        self._long_term.delete_many({})

    def get_user_stats(self, user_id: str) -> dict:
        return {
            "conversation_count": self._conversation.count_documents({"user_id": user_id}),
            "short_term_count": self._short_term.count_documents({"user_id": user_id}),
            "long_term_count": self._long_term.count_documents({"user_id": user_id}),
        }

    def get_stats(self) -> dict:
        return {
            "total_conversations": self._conversation.count_documents({}),
            "total_short_term": self._short_term.count_documents({}),
            "total_long_term": self._long_term.count_documents({}),
            "unique_users": len(self._conversation.distinct("user_id")),
        }

    def close(self):
        self._client.close()
