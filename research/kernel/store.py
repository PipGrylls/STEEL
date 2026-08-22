"""MongoDB result store and its gates.

Imported only by `research/mcp_server/server.py`. Agents reach the store
through MCP tools, never through this module -- a library can be bypassed
by an agent writing its own script, which is the behaviour the apparatus
exists to prevent.
"""
from typing import Any

from pymongo import MongoClient

from .definitions import FIELDS as DEFINITION_FIELDS

EXTRACTION_METHODS = {"table", "figure", "text", "abstract"}
VERIFICATION_METHODS = {"arxiv-api-resolved", "doi-resolved", "manual-pdf"}

# `_check` used to return early for any doc whose `kind` wasn't exactly the
# string "measurement" -- which meant a typo ("measurment"), a wrong case
# ("Measurement"), or a missing `kind` field bypassed every gate below and
# was written unchecked. That is allow-by-default for anything malformed,
# which defeats the point of a gate. This allowlist makes "not a kind we
# recognise" a loud refusal instead of a silent pass. Do not remove it as a
# "simplification" -- the early return it replaces was the bug.
KNOWN_KINDS = {"source", "measurement", "model_run", "derivation_run", "claim", "question"}


class GateViolation(Exception):
    """A write was refused because it would break a spec gate."""


class Store:
    def __init__(self, uri: str, db: str = "steel_research"):
        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._db = self._client[db]

    def drop(self) -> None:
        self._client.drop_database(self._db.name)

    def ensure_schema(self) -> None:
        """Indexes for the pre-check query path."""
        self._db.artifacts.create_index("kind")
        self._db.artifacts.create_index("definition.quantity")
        self._db.artifacts.create_index("definition.mass_def")
        self._db.sources.create_index("source_id", unique=True)

    def verify_source(self, source_id: str, method: str) -> dict:
        """Register a source as verified. Recollection is not verification."""
        if method not in VERIFICATION_METHODS:
            raise GateViolation(
                f"verification_method must be one of {sorted(VERIFICATION_METHODS)}")
        doc = {"source_id": source_id, "verification_method": method}
        self._db.sources.update_one({"source_id": source_id},
                                    {"$set": doc}, upsert=True)
        return doc

    def _check(self, doc: dict[str, Any]) -> None:
        kind = doc.get("kind")
        if kind not in KNOWN_KINDS:
            raise GateViolation(
                f"kind must be one of {sorted(KNOWN_KINDS)}, got {kind!r}")
        if kind != "measurement":
            return
        source_id = doc.get("source_id")
        if not source_id or not self._db.sources.find_one({"source_id": source_id}):
            raise GateViolation(
                f"measurement requires a verified source; {source_id!r} is not registered")
        definition = doc.get("definition", {})
        if not isinstance(definition, dict):
            raise GateViolation(
                f"definition must be a dict, got {type(definition).__name__}")
        missing = [f for f in DEFINITION_FIELDS if f not in definition]
        if missing:
            raise GateViolation(
                f"definition missing required field(s): {', '.join(missing)}")
        source_snapshot = doc.get("source_snapshot", {})
        if not isinstance(source_snapshot, dict):
            raise GateViolation(
                f"source_snapshot must be a dict, got {type(source_snapshot).__name__}")
        extraction = source_snapshot.get("extraction")
        if extraction not in EXTRACTION_METHODS:
            raise GateViolation(
                f"extraction must be one of {sorted(EXTRACTION_METHODS)}, got {extraction!r}")

    def put(self, doc: dict[str, Any]) -> str:
        self._check(doc)
        self._db.artifacts.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        return doc["_id"]

    def query(self, spec: dict[str, Any]) -> list[dict]:
        return list(self._db.artifacts.find(spec))
