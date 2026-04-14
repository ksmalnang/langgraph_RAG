# RFC 5424 Structured Logging

This application implements **RFC 5424** compliant structured logging using JSON format for production environments while maintaining human-readable text logs for development.

## What is RFC 5424?

RFC 5424 is "The Syslog Protocol" - an IETF standard that defines a structured log message format with the following fields:

- **TIMESTAMP**: When the event occurred (ISO 8601 / RFC 3339 format)
- **HOSTNAME**: The machine where the event occurred
- **APP-NAME**: The application that generated the event
- **PROCID**: Process ID
- **MSGID**: Message identifier
- **STRUCTURED-DATA**: Additional contextual metadata
- **MESSAGE**: Human-readable description of the event

## Configuration

The logging behavior is controlled via environment variables in your `.env` file:

```env
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                   # "json" for RFC 5424, "text" for human-readable
APP_NAME=langgraph-agent-ai      # Application identifier in logs
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Minimum log level to output |
| `LOG_FORMAT` | `json` | `json` for RFC 5424 structured logs, `text` for colored console output |
| `APP_NAME` | `langgraph-agent-ai` | Application name included in log entries |

## JSON Log Format (Production)

When `LOG_FORMAT=json`, each log entry is a single-line JSON object containing all RFC 5424 fields:

```json
{
  "message": "Upserted 42 chunks for doc 'user_guide.pdf'",
  "timestamp": "2026-04-14T00:51:06.536741+00:00",
  "hostname": "prod-server-01",
  "app_name": "langgraph-agent-ai",
  "procid": 12345,
  "msgid": "app.ingestion.upserter",
  "severity": "info",
  "logger": "app.ingestion.upserter",
  "module": "upserter",
  "function": "upsert_chunks",
  "line": 87,
  "thread": "MainThread",
  "thread_id": 140234567890,
  "process_name": "MainProcess",
  "document_id": "doc_123",
  "chunk_count": 42
}
```

### Benefits of JSON Format

- **Machine-parseable**: Easy to ingest into log aggregation systems (ELK, Datadog, Splunk, etc.)
- **Structured metadata**: Custom fields are preserved for filtering and analysis
- **Standardized**: RFC 5424 compliance ensures compatibility with syslog infrastructure
- **Query-friendly**: Fields can be indexed and searched efficiently

## Text Log Format (Development)

When `LOG_FORMAT=text`, logs are formatted for human readability with ANSI colors:

```
2026-04-14 07:51:06 | INFO     | app.ingestion.upserter:upsert_chunks:87 — Upserted 42 chunks for doc 'user_guide.pdf'
```

Color coding:
- **DEBUG**: Cyan
- **INFO**: Green
- **WARNING**: Yellow
- **ERROR**: Red
- **CRITICAL**: Magenta + Bold

## Usage

### Basic Logging

```python
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Simple log messages
logger.info("Starting ingestion pipeline")
logger.warning("Rate limit approaching: 80% capacity")
logger.error("Failed to connect to Qdrant")

# With structured metadata (appears as JSON fields)
logger.info(
    "Upserted %d chunks for doc '%s'", 
    chunk_count, 
    filename,
    extra={"document_id": doc_id, "operation": "upsert"}
)
```

### Exception Logging

```python
try:
    result = await process_document(doc_id)
except Exception as e:
    logger.exception("Failed to process document %s", doc_id)
    # Automatically includes stack trace in the "message" field
```

### Custom Structured Fields

Any keyword arguments or `extra` dict items become structured fields in the JSON output:

```python
# Using f-string (values embedded in message)
logger.info(f"Processing user_id={user_id}")

# Using structured fields (better for filtering)
logger.info("Processing user request", extra={"user_id": user_id, "action": "classify"})
```

## Log Aggregation Integration

### Elasticsearch / Kibana

JSON logs can be directly ingested by Filebeat, Fluentd, or Logstash:

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  paths:
    - /var/log/app/*.log
  json.keys_under_root: true
  json.add_error_key: true
  json.message_key: message
```

### Datadog

The structured fields map to Datadog's standard attributes:

```json
{
  "timestamp": "2026-04-14T00:51:06.536741+00:00",
  "severity": "info",
  "message": "...",
  "hostname": "prod-server-01",
  "app_name": "langgraph-agent-ai"
}
```

### Splunk

Splunk automatically extracts JSON fields at search time when using the `kv` or `json` sourcetype.

## Third-Party Logger Suppression

The following third-party loggers are set to WARNING level to reduce noise:

- `httpx`
- `httpcore`
- `qdrant_client`
- `urllib3`

## Implementation Details

### Files Modified

- `app/utils/logger.py`: Main logging configuration with `RFC5424JsonFormatter`
- `app/config.py`: Added `LOG_FORMAT` and `APP_NAME` settings
- `pyproject.toml`: Added `python-json-logger` dependency

### Dependencies

- **python-json-logger** (>=3.2.1): Provides the base JSON formatter with extensible field handling

### RFC 5424 Field Mapping

| RFC 5424 Field | Python Logger Field | Description |
|----------------|---------------------|-------------|
| TIMESTAMP | `record.created` | ISO 8601 timestamp with UTC timezone |
| HOSTNAME | `socket.gethostname()` | Machine hostname |
| APP-NAME | `settings.app_name` | Application identifier |
| PROCID | `os.getpid()` | Process ID |
| MSGID | `record.name` | Logger name (module path) |
| SEVERITY | `record.levelname` | Log level (lowercase) |
| MESSAGE | `record.getMessage()` | Formatted log message |
| STRUCTURED-DATA | `extra={}` | Additional contextual fields |

## Best Practices

1. **Use parameterized logging**: `logger.info("User %s logged in", user_id)` instead of f-strings
2. **Add structured metadata**: Use `extra={}` for fields you want to filter/search on
3. **Include context**: Add request IDs, user IDs, session IDs to trace operations
4. **Log at appropriate levels**: DEBUG for details, INFO for operations, WARNING for issues, ERROR for failures
5. **Don't log sensitive data**: Avoid logging passwords, tokens, or PII

## Migration Guide

### From Previous Implementation

The refactoring maintains backward compatibility:

- ✅ All existing `logger.info()`, `logger.warning()`, etc. calls work unchanged
- ✅ `get_logger(__name__)` pattern still works
- ✅ `setup_logging()` is still called at application startup
- ✅ Default behavior is now JSON instead of text (change with `LOG_FORMAT=text`)

### Switching Between Formats

**Production (JSON):**
```env
LOG_FORMAT=json
```

**Development (Text with colors):**
```env
LOG_FORMAT=text
```

No code changes required - just update the environment variable.
