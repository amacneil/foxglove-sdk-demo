# SDK Game

An interactive 3D world built using the Foxglove SDK.

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Run the server:
```bash
poetry run python main.py
```

The server will start on http://localhost:8000 and the Foxglove WebSocket server will be available at ws://localhost:8765.

## Development

This project uses:
- Foxglove SDK for 3D visualization
- FastAPI for the web server
- Uvicorn as the ASGI server 