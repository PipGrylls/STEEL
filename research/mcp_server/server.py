"""The research-store MCP server: the only write path to the store.

Gates are enforced here rather than in a Python library because an agent
holding a library can write its own script around it; an agent holding an
MCP tool cannot.

Note on the package name: this package is `mcp_server`, not `mcp` -- a
local package named `mcp` would shadow the installed `mcp` SDK that this
module imports below, causing it to import itself.

Note on the SDK API: the installed SDK is `mcp==2.0.0`, in which
`mcp.server.fastmcp.FastMCP` does not exist. The public server class is
`mcp.server.MCPServer`, which exposes `add_tool(fn, name=...)` and
`run(transport=...)`. Verified with `inspect.signature(MCPServer.add_tool)`
against the installed package before writing this.
"""
from mcp.server import MCPServer

from kernel.store import GateViolation, Store


def build_server(uri: str, db: str = "steel_research") -> dict:
    """Return the tool callables, so tests can exercise them directly."""
    store = Store(uri, db=db)
    store.ensure_schema()

    def store_verify_source(source_id: str, method: str) -> dict:
        try:
            return {"ok": True, "source": store.verify_source(source_id, method)}
        except GateViolation as e:
            return {"ok": False, "error": str(e)}

    def store_put(doc: dict) -> dict:
        try:
            return {"ok": True, "id": store.put(doc)}
        except GateViolation as e:
            return {"ok": False, "error": str(e)}

    def store_query(spec: dict) -> dict:
        results = store.query(spec)
        for r in results:
            # Mongo returns `_id` as whatever was stored there. Every write
            # in this store goes through `put()`, which requires the caller
            # to supply a string `_id`, so in practice this is always
            # already a string. But `query` has no such guarantee for
            # documents that predate that invariant or that were written
            # some other way, and a raw `bson.ObjectId` is not
            # JSON-serialisable -- it would blow up crossing the MCP tool
            # boundary. Drop `_id` in that case rather than crash or emit
            # something the caller can't decode.
            if not isinstance(r.get("_id"), str):
                r.pop("_id", None)
        return {"ok": True, "results": results}

    return {"store_verify_source": store_verify_source,
            "store_put": store_put, "store_query": store_query}


def main() -> None:
    uri = "mongodb://localhost:27017"
    tools = build_server(uri)
    server = MCPServer("research-store")
    for name, fn in tools.items():
        server.add_tool(fn, name=name)
    server.run()


if __name__ == "__main__":
    main()
