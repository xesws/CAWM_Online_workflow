# OpenHands Project Context

## Overview
OpenHands (formerly OpenDevin) is an AI-powered software development agent capable of performing engineering tasks, running commands, and browsing the web. This repository contains the source code for the agent, its runtime environment, and the user interface.

## Directory Structure
The repository root (`.`) contains a wrapper structure. The core project resides in the `OpenHands/` subdirectory.

*   `OpenHands/`: **Project Root**. Contains `Makefile`, `pyproject.toml`, and source code.
    *   `openhands/`: Backend Python source code (FastAPI, agent logic, runtime).
    *   `frontend/`: Frontend React application (Vite, Remix/React Router).
    *   `microagents/`: Definitions for specialized agent behaviors.
    *   `containers/`: Docker build contexts for the sandbox environment.
    *   `tests/`: Unit and integration tests.

## Development Setup

**Important:** Most commands should be executed from the `OpenHands/` subdirectory.

### Prerequisites
*   **Python:** 3.12
*   **Node.js:** >= 22.x
*   **Poetry:** >= 1.8
*   **Docker:** Required for the sandbox runtime.

### Initial Setup
1.  Navigate to the project root:
    ```bash
    cd OpenHands
    ```
2.  Build the project (installs dependencies, sets up hooks, builds frontend):
    ```bash
    make build
    ```
3.  Configure the environment (LLM keys, workspace):
    ```bash
    make setup-config
    ```

## Running the Application

From the `OpenHands/` directory:

*   **Full Stack (Backend + Frontend):**
    ```bash
    make run
    ```
    *   Frontend: http://localhost:3001
    *   Backend: http://localhost:3000

*   **Backend Only:**
    ```bash
    make start-backend
    ```

*   **Frontend Only:**
    ```bash
    make start-frontend
    ```

*   **Docker Mode:**
    ```bash
    make docker-run
    ```

## Testing & Quality

### Backend (Python)
*   **Run Unit Tests:**
    ```bash
    poetry run pytest ./tests/unit/test_*.py
    ```
*   **Linting:**
    ```bash
    make lint-backend
    ```
    (Uses `ruff`, `mypy` via pre-commit hooks)

### Frontend (TypeScript/React)
*   **Run Tests:**
    ```bash
    cd frontend && npm run test
    ```
*   **Linting:**
    ```bash
    make lint-frontend
    ```

## Dependency Management

*   **Python:** Managed via Poetry.
    *   Add dependency: `poetry add <package_name>`
    *   Update lockfile: `poetry lock --no-update`
*   **Frontend:** Managed via npm.
    *   Add dependency: `cd frontend && npm install <package_name>`

## Key Configuration Files
*   `OpenHands/config.toml`: Main configuration (LLM providers, agent settings).
*   `OpenHands/pyproject.toml`: Python dependencies and tool config.
*   `OpenHands/frontend/package.json`: Frontend dependencies and scripts.
*   `OpenHands/Makefile`: Central entry point for build and run tasks.
