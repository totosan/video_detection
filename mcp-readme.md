# Using the Video Detection System with MCP and MCPO

This document outlines how to set up the environment, test the Model Context Protocol (MCP) server directly, and run it behind the `mcpo` proxy for enhanced usability and security.

## 1. Prepare Environment

This assumes you are using `uv` for environment and package management.

-   **Initialize Project (if not done):**
    ```bash
    # Only needed once per project
    uv init
    ```
-   **Create/Activate Virtual Environment:**
    ```bash
    # Creates a .venv directory if it doesn't exist
    uv venv
    # Activate the environment (syntax depends on your shell)
    # Example for bash/zsh:
    source .venv/bin/activate
    # Example for fish:
    source .venv/bin/activate.fish
    # Example for Powershell:
    .venv\Scripts\Activate.ps1
    ```
-   **Install Dependencies:**
    Install all dependencies defined in `pyproject.toml`, including `mcp[cli]` and `mcpo`.
    ```bash
    uv sync
    ```
    *   `mcp[cli]`: Installs the core MCP library and its command-line tools.
    *   `mcpo`: Installs the MCP-to-OpenAPI proxy server.

## 2. Test MCP Server Directly (Development/Debugging)

You can run the MCP server directly using `mcp` commands for quick testing. This uses raw stdio for communication.

-   **Using `mcp dev` (for development):**
    This command watches for file changes and restarts the server automatically.
    ```bash
    # Ensure your virtual environment is active
    mcp dev mcp_server.py
    ```
    *   `mcp dev`: Starts the MCP server in development mode.
    *   `mcp_server.py`: Your script containing the MCP tool definitions.

-   **Using `mcp run` (standard execution):**
    Runs the server once without auto-reloading.
    ```bash
    # Ensure your virtual environment is active
    mcp run mcp_server.py
    ```

## 3. Run with MCPO (Recommended for Integration)

`mcpo` acts as a proxy, exposing your MCP server over standard HTTP with an OpenAPI interface. This makes it secure, stable, and compatible with tools like Open WebUI.

-   **Command Structure:**
    ```bash
    uvx mcpo [mcpo options] -- [your mcp server command]
    ```
    *   `uvx`: A `uv` command to run an application installed in an ephemeral (temporary) environment, ensuring you use the installed `mcpo` version without activating the main venv explicitly if preferred. You can also use `uv run mcpo ...` if your main venv is active.
    *   `[mcpo options]`: Flags to configure `mcpo`, like `--host`, `--port`, `--api-key`.
    *   `--`: **Crucial separator**. Tells `mcpo` that the following arguments are the command to start your actual MCP server.
    *   `[your mcp server command]`: The command to run your MCP server, typically using `uv run mcp run ...`.

-   **Example Call:**
    Run `mcpo` on port 8081, proxying the `mcp_server.py` script.
    ```bash
    # Ensure your virtual environment is active OR use uvx as shown
    uvx mcpo --host localhost --port 8081 -- uv run --with mcp mcp run mcp_server.py
    ```
    *   `--host localhost`: Makes the server accessible only locally. Use `0.0.0.0` to allow external access (use with caution).
    *   `--port 8081`: The port where `mcpo` will listen for HTTP requests.
    *   `--`: Separator.
    *   `uv run --with mcp mcp run mcp_server.py`: The command `mcpo` will execute internally to start your MCP server via stdio.

-   **Accessing the Server:**
    *   **API Endpoint:** Your MCP tools are now available via HTTP POST requests to `http://localhost:8081/invoke/{tool_name}`.
    *   **OpenAPI Docs:** `mcpo` automatically generates interactive documentation. Access it at `http://localhost:8081/docs`.

## 4. Integrate with Open WebUI

Once `mcpo` is running and exposing your MCP server over HTTP:

1.  **Start `mcpo`:** Use the command from Step 3. Make note of the host, port, and API key.
2.  **Configure Open WebUI:**
    *   Navigate to the Open WebUI settings/admin panel where you configure tools or API endpoints.
    *   Add a new OpenAPI server endpoint.
    *   **URL:** Enter the `mcpo` base URL (e.g., `http://localhost:8081` or the externally accessible URL if configured).
    *   **Authentication:** If you used `--api-key`, configure Bearer token authentication in Open WebUI, providing the API key you set.
    *   Refer to the official Open WebUI documentation for the specific steps on adding OpenAPI tools: [Open WebUI Docs - OpenAPI Servers](https://docs.openwebui.com/openapi-servers/mcp) (This link is based on the `mcpo` README, verify it's the correct section).

Your video detection tools (`start_detection_system`, `get_current_detections`, etc.) should now be available for use within Open WebUI.