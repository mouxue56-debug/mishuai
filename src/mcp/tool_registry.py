"""MCP Tool Registry - discovers, registers, and executes tools.

Inspired by py-xiaozhi's MCP tool framework.
Tools are auto-discovered from the tools/ directory.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Optional

from src.utils.config_loader import get_main_config
from src.utils.logger import get_logger

logger = get_logger("mcp")


class MCPTool:
    """Base class for all MCP tools.

    Subclass this to create a new tool. Must implement:
    - name: str - unique tool identifier
    - description: str - what the tool does (shown to LLM)
    - parameters: dict - JSON Schema for parameters
    - execute(**kwargs) - the tool logic
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters.

        Returns:
            Tool result (will be converted to string for LLM).
        """
        raise NotImplementedError

    def get_schema(self) -> dict:
        """Get the tool schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": [
                    k for k, v in self.parameters.items()
                    if v.get("required", False)
                ],
            },
        }


class ToolRegistry:
    """Registry for MCP tools. Auto-discovers and manages tools."""

    def __init__(self):
        self._tools: dict[str, MCPTool] = {}
        self._enabled_tools: list[str] = []

    async def initialize(self):
        """Discover and register all tools from the tools/ directory."""
        config = get_main_config()
        self._enabled_tools = config.get("mcp", {}).get("enabled_tools", [])

        # Auto-discover tools
        tools_package = "src.mcp.tools"
        tools_path = Path(__file__).parent / "tools"

        if not tools_path.exists():
            logger.warning(f"Tools directory not found: {tools_path}")
            return

        for importer, modname, ispkg in pkgutil.iter_modules([str(tools_path)]):
            if modname.startswith("_"):
                continue

            try:
                module = importlib.import_module(f"{tools_package}.{modname}")

                # Find MCPTool subclasses in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        inspect.isclass(attr)
                        and issubclass(attr, MCPTool)
                        and attr is not MCPTool
                        and attr.name  # Must have a name
                    ):
                        tool = attr()
                        if not self._enabled_tools or tool.name in self._enabled_tools:
                            self._tools[tool.name] = tool
                            logger.info(f"Registered tool: {tool.name}")

            except Exception as e:
                logger.error(f"Failed to load tool module {modname}: {e}")

        logger.info(f"Tool registry: {len(self._tools)} tools loaded")

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool parameters.

        Returns:
            Tool execution result.

        Raises:
            ValueError: If tool is not found.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        logger.info(f"Executing tool: {tool_name}")
        try:
            result = await tool.execute(**arguments)
            logger.info(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

    def get_all_schemas(self) -> list[dict]:
        """Get schemas for all registered tools (for LLM function calling)."""
        return [tool.get_schema() for tool in self._tools.values()]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
