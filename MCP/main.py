"""
main.py
-------
Wires the MCP client to Ollama/Llama.

This is the coordinator — it:
  1. Connects to the MCP server via the client
  2. Gets the tool list from the server
  3. Sends user messages + tools to the LLM
  4. Executes tool calls via the client
  5. Feeds results back to the LLM
  6. Shows the final answer

Claude API version:  core/claude.py uses anthropic.Anthropic()
Ollama version:      we use requests.post() to localhost:11434
Everything else is identical.
"""

import asyncio
import json
import sys
import requests
from mcp_client import MCPClient

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5:7b"


# =============================================================================
# 1. LLM CALL — Ollama instead of Claude API
#    This is the ONLY part that differs from the original project.
#    Same inputs, same outputs, different endpoint.
# =============================================================================

def llm_chat(messages: list, tools: list = None) -> dict:
    """Call Ollama. Returns the raw message dict."""
    payload = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        # Ollama uses OpenAI-compatible format: type+function wrapper, "parameters" key
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t["description"],
                    "parameters":  t["input_schema"],
                },
            }
            for t in tools
        ]
    res = requests.post(OLLAMA_URL, json=payload)
    res.raise_for_status()
    return res.json()["message"]


# =============================================================================
# 2. CHAT LOOP
#    Mirrors the transcript's full flow:
#      get tools → send to LLM → execute tool calls → final answer
# =============================================================================

async def chat(client: MCPClient, user_input: str) -> str:
    """
    Full conversation turn:
      1. Get tools from MCP server
      2. Send user message + tools to LLM
      3. If LLM calls a tool → execute via MCP client → loop
      4. Return final answer
    """
    # Step 1: get tools from MCP server
    # Transcript: "app asks MCP client to get a list of tools"
    tools = await client.list_tools()

    messages = [
        {
            "role":    "system",
            "content": "You are a helpful assistant with access to documents. "
                       "Use the available tools to read and edit documents when needed.",
        },
        {"role": "user", "content": user_input},
    ]

    # Step 2: agentic loop
    max_iterations = 5
    for _ in range(max_iterations):

        response = llm_chat(messages, tools=tools)
        messages.append(response)

        tool_calls = response.get("tool_calls")

        # No tool calls → LLM has final answer
        if not tool_calls:
            return response.get("content", "")

        # Step 3: execute each tool call via MCP client
        # Transcript: "app asks MCP client to run a tool"
        for call in tool_calls:
            fn   = call["function"]
            name = fn["name"]
            args = fn.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            print(f"  [tool] {name}({args})")

            # Client sends CallToolRequest to server
            # Server executes the tool, returns CallToolResult
            result = await client.call_tool(name, args)
            print(f"  [result] {result[:100]}")

            messages.append({"role": "tool", "content": result})

    return "Max iterations reached."


# =============================================================================
# 3. DEMO — shows all three server features: tools, resources, prompts
# =============================================================================

async def main():
    async with MCPClient("mcp_server.py") as client:

        # ── Show what the server provides ─────────────────────────────────
        print("\n" + "="*55)
        print("SERVER CAPABILITIES")
        print("="*55)

        tools     = await client.list_tools()
        resources = await client.list_resources()
        prompts   = await client.list_prompts()

        print(f"\nTools ({len(tools)}):")
        for t in tools:
            print(f"  - {t['name']}: {t['description']}")

        print(f"\nResources ({len(resources)}):")
        for r in resources:
            print(f"  - {r}")

        print(f"\nPrompts ({len(prompts)}):")
        for p in prompts:
            print(f"  - {p}")

        # ── Demo 1: Use a resource directly (no LLM needed) ───────────────
        # Transcript: "resources are data your app reads directly"
        print("\n" + "="*55)
        print("DEMO 1 — Resource: list documents")
        print("="*55)
        doc_list = await client.read_resource("docs://list")
        print(f"Available documents: {doc_list}")

        # ── Demo 2: Use a prompt template ─────────────────────────────────
        # Transcript: "prompts are pre-built instructions your app can grab"
        print("\n" + "="*55)
        print("DEMO 2 — Prompt: fetch summarize template")
        print("="*55)
        prompt_text = await client.get_prompt("summarize", {"doc_id": "report.md"})
        print(f"Prompt template:\n  {prompt_text}")

        # ── Demo 3: Tool use via LLM ───────────────────────────────────────
        # Transcript: full flow — user → app → LLM → tool → result → LLM → answer
        print("\n" + "="*55)
        print("DEMO 3 — Tool use: read a document")
        print("="*55)
        answer = await chat(client, "What does the report.md document say?")
        print(f"\nAnswer: {answer}")

        # ── Demo 4: Multi-step tool use ────────────────────────────────────
        print("\n" + "="*55)
        print("DEMO 4 — Tool use: edit a document")
        print("="*55)
        answer = await chat(
            client,
            "In report.md, replace '$4.2M' with '$5.1M' to reflect the updated figures."
        )
        print(f"\nAnswer: {answer}")

        # Verify the edit by reading resource directly
        content = await client.read_resource("docs://content/report.md")
        print(f"\nVerified content: {content}")

        # ── Demo 5: CLI chat loop ──────────────────────────────────────────
        print("\n" + "="*55)
        print("DEMO 5 — Interactive chat (type 'quit' to exit)")
        print("="*55)
        print("Available docs:", doc_list)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit"):
                break
            answer = await chat(client, user_input)
            print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
