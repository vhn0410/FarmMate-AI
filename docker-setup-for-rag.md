# Docker Setup Guide for RAG Agent

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for the container

## Quick Start

### 1. Prepare Your Environment

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Google AI
GOOGLE_API_KEY=your_google_key

# Cohere
COHERE_API_KEY=your_cohere_key

# Groq (if using)
GROQ_API_KEY=your_groq_key

# Other configuration
PERSIST_DIR=db_agriculture3/chroma_db
CHUNKS_DB=chunks.duckdb
```

### 2. Build and Run

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f rag-agent
```

### 3. Verify Deployment

Check health status:
```bash
curl http://localhost:8000/health
```

Test the chat endpoint:
```bash
curl -X POST http://localhost:8000/chat_stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how can you help with agriculture?"}'
```

## Common Commands

```bash
# Stop the service
docker-compose down

# Restart the service
docker-compose restart

# View logs
docker-compose logs -f

# Rebuild after code changes
docker-compose up -d --build

# Remove everything including volumes
docker-compose down -v
```

## Production Deployment

For production, consider:

1. **Multi-stage build** for smaller image:
```dockerfile
FROM python:3.11-slim as builder
# Build dependencies
FROM python:3.11-slim
# Copy only necessary files
```

2. **Environment-specific compose files**:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

3. **Resource limits** in docker-compose.yml:
```yaml
services:
  rag-agent:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

4. **Reverse proxy** (nginx/traefik) for SSL and load balancing

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs rag-agent

# Check if databases exist
ls -la db_agriculture3/chroma_db
ls -la chunks.duckdb
```

### Permission issues
```bash
# Fix permissions on mounted volumes
sudo chown -R $USER:$USER db_agriculture3 chunks.duckdb
```

### Out of memory
```bash
# Increase Docker memory limit in Docker Desktop settings
# or add to docker-compose.yml:
    mem_limit: 4g
    mem_reservation: 2g
```

## Development Mode

Uncomment volume mounts in docker-compose.yml to enable hot-reload:

```yaml
volumes:
  - ./src:/app/src
  - ./main.py:/app/main.py
```

Then add `--reload` to the CMD:
```bash
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```