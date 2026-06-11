"""
mcp_server.py
-------------
The MCP Server — defines tools, resources, and prompts.

This runs as a SEPARATE PROCESS. Your app never imports this directly.
Communication happens via stdio (stdin/stdout).

Three things this server provides:
  Tools     → things the LLM can DO
  Resources → data the app can READ directly
  Prompts   → pre-built prompt templates
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Create the server — give it a name
mcp = FastMCP("DocumentServer", log_level="ERROR")


# =============================================================================
# IN-MEMORY DOCUMENTS
# These are our fake documents — stored in a simple dict.
# In a real app this would be a database, file system, or external API.
# =============================================================================

docs = {
    "report.md":    "The quarterly report shows revenue of $4.2M, up 12% from last quarter.",
    "plan.md":      "The project plan outlines three phases: design, build, and deploy.",
    "notes.md":     "Meeting notes: team agreed to use Python for the backend.",
}


# =============================================================================
# TOOLS — things the LLM can call
# Defined with @mcp.tool decorator
# The LLM sees the name and description, uses them to decide when to call
# =============================================================================

@mcp.tool(
    name="read_document",
    description="Read the contents of a document by its ID.",
)
def read_document(
    doc_id: str = Field(description="The document ID e.g. 'report.md'"),
) -> str:
    if doc_id not in docs:
        return f"Error: document '{doc_id}' not found."
    return docs[doc_id]


@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing a specific string with new text.",
)
def edit_document(
    doc_id:  str = Field(description="The document ID to edit"),
    old_str: str = Field(description="The exact text to replace"),
    new_str: str = Field(description="The new text to insert"),
) -> str:
    if doc_id not in docs:
        return f"Error: document '{doc_id}' not found."
    if old_str not in docs[doc_id]:
        return f"Error: text '{old_str}' not found in document."
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)
    return f"Successfully updated '{doc_id}'."


# =============================================================================
# RESOURCES — data the app reads directly (not via LLM)
# Defined with @mcp.resource decorator
# Your app calls these directly to get data — LLM not involved
# =============================================================================

@mcp.resource("docs://list", mime_type="application/json")
def list_documents() -> list[str]:
    """Return a list of all document IDs."""
    return list(docs.keys())


@mcp.resource("docs://content/{doc_id}", mime_type="text/plain")
def get_document(doc_id: str) -> str:
    """Return the content of a specific document."""
    if doc_id not in docs:
        return f"Error: document '{doc_id}' not found."
    return docs[doc_id]


# =============================================================================
# PROMPTS — pre-built prompt templates
# Defined with @mcp.prompt decorator
# Your app fetches these and sends them to the LLM as ready-made instructions
# =============================================================================

@mcp.prompt(
    name="summarize",
    description="Summarize a document in one sentence.",
)
def summarize_prompt(
    doc_id: str = Field(description="The document ID to summarize"),
) -> str:
    return (
        f"Read the document with ID '{doc_id}' using the read_document tool, "
        f"then summarize its contents in exactly one sentence."
    )


@mcp.prompt(
    name="rewrite",
    description="Rewrite a document to be more formal.",
)
def rewrite_prompt(
    doc_id: str = Field(description="The document ID to rewrite"),
) -> str:
    return (
        f"Read the document with ID '{doc_id}' using the read_document tool. "
        f"Rewrite it in a more formal professional tone. "
        f"Then use edit_document to save the rewritten version."
    )


# =============================================================================
# ENTRY POINT
# Run via stdio transport — communicates through stdin/stdout
# This is how the client launches and talks to this server
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
