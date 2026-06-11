"""
mcp_client.py
-------------
The MCP Client — bridge between your app and the MCP server.

Your app never talks to the server directly.
It always goes through the client.

The client:
  - Launches the server as a subprocess
  - Communicates via stdio (stdin/stdout)
  - Exposes simple methods: list_tools, call_tool, list_resources, read_resource, get_prompt
"""

import asyncio
import json
from typing import Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool
from pydantic import AnyUrl


class MCPClient:
    """
    Connects to an MCP server and exposes its tools, resources, and prompts.

    Transcript parallel:
      ListToolsRequest  → list_tools()
      ListToolsResult   → return value of list_tools()
      CallToolRequest   → call_tool()
      CallToolResult    → return value of call_tool()
    """

    def __init__(self, server_script: str = "mcp_server.py"):
        self._server_script = server_script
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        """
        Launch the server as a subprocess and establish a stdio connection.
        Transcript: "client and server communicate over standard input output"
        """
        server_params = StdioServerParameters(
            command = "python3",
            args    = [self._server_script],
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self._session.initialize()
        print(f"  [MCP] Connected to server: {self._server_script}")

    def _session_or_error(self) -> ClientSession:
        if not self._session:
            raise ConnectionError("Not connected. Call connect() first.")
        return self._session

    # ── Tools ─────────────────────────────────────────────────────────────

    async def list_tools(self) -> list[dict]:
        """
        Ask the server for all available tools.
        Returns tool dicts in the format the LLM expects.

        Transcript: "ListToolsRequest → ListToolsResult"
        """
        result = await self._session_or_error().list_tools()
        return [
            {
                "name":         t.name,
                "description":  t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        Ask the server to execute a tool.
        Returns the result as a string.

        Transcript: "CallToolRequest → CallToolResult"
        """
        result = await self._session_or_error().call_tool(tool_name, tool_args)
        # Extract text content from result
        texts = [
            item.text
            for item in result.content
            if hasattr(item, "text")
        ]
        return "\n".join(texts)

    # ── Resources ─────────────────────────────────────────────────────────

    async def list_resources(self) -> list[str]:
        """Ask the server for all available resource URIs."""
        result = await self._session_or_error().list_resources()
        return [str(r.uri) for r in result.resources]

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource directly — no LLM involved.
        Transcript: "resources are data the app reads directly"
        """
        result = await self._session_or_error().read_resource(AnyUrl(uri))
        resource = result.contents[0]
        text = resource.text if hasattr(resource, "text") else ""

        # Parse JSON if applicable
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    # ── Prompts ───────────────────────────────────────────────────────────

    async def list_prompts(self) -> list[str]:
        """Ask the server for all available prompt names."""
        result = await self._session_or_error().list_prompts()
        return [p.name for p in result.prompts]

    async def get_prompt(self, prompt_name: str, args: dict) -> str:
        """
        Fetch a pre-built prompt template from the server.
        Returns the prompt text ready to send to the LLM.
        Transcript: "prompts are pre-built instructions your app can grab"
        """
        result = await self._session_or_error().get_prompt(prompt_name, args)
        texts = [
            msg.content.text
            for msg in result.messages
            if hasattr(msg.content, "text")
        ]
        return "\n".join(texts)

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def cleanup(self):
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.cleanup()
