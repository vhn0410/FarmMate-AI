from typing import Any, Dict, Iterable, Optional, List
from pymongo.collection import Collection
from langgraph.store.base import BaseStore

class MongoLongTermStore(BaseStore):
    """
    Custom KV Store for long-term memory in MongoDB.
    Compatible with LangGraph 1.x KV Store API.
    """
    def __init__(self, collection: Collection):
        self.collection = collection
        self.collection.create_index("key", unique=True)

    # ----------------------------
    # Single operations
    # ----------------------------
    async def aget(self, key: str) -> Optional[Any]:
        doc = self.collection.find_one({"key": key})
        return doc["value"] if doc else None

    async def aput(self, key: str, value: Any) -> None:
        self.collection.update_one(
            {"key": key},
            {"$set": {"key": key, "value": value}},
            upsert=True
        )

    async def adelete(self, key: str) -> None:
        self.collection.delete_one({"key": key})

    # ----------------------------
    # Batch operations
    # ----------------------------
    def batch(self, items: Iterable[Dict[str, Any]]) -> None:
        """Sync batch insert/update"""
        for item in items:
            self.collection.update_one(
                {"key": item["key"]},
                {"$set": {"key": item["key"], "value": item["value"]}},
                upsert=True
            )

    async def abatch(self, items: Iterable[Dict[str, Any]]) -> None:
        """Async batch insert/update"""
        # You can just run sync batch inside asyncio if MongoDB driver is sync
        self.batch(items)
