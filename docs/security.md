## Security Policy (Public vs Student Flows)

This document defines trust boundaries and enforcement for API routes.

### Route Policy

- `POST /chat`
  - public route for generic/admin chat
  - rate limited (`CHAT_RATE_LIMIT`)
  - if a session is student-bound, request must include `X-Student-Access-Token`
- `GET /chat/{session_id}/history`
  - public for non-student sessions
  - if session is student-bound, requires `X-Student-Access-Token`
- `POST /auth/login`
  - public route (credential exchange with SIAKAD)
  - rate limited (`AUTH_LOGIN_RATE_LIMIT`)
  - returns:
    - `session_id`
    - `student_access_token`
- `POST /ingest`
  - operator route
  - in non-local env (`APP_ENV` not local/dev/test), `INGEST_API_KEY` must be configured
  - request must send `X-Ingest-Token` matching `INGEST_API_KEY`
  - rate limited (`INGEST_RATE_LIMIT`)

### Session Trust Semantics

`session_id` is not sufficient by itself for student-data access.

For authenticated student sessions:
- login stores a server-side auth binding (`siakad_auth:{session_id}`) in Redis
- client receives a one-time generated `student_access_token`
- chat/history endpoints verify token hash using `session_id` binding
- without valid token, student-bound sessions are rejected with `401`

### Logging Rules

Sensitive values are not logged in raw form:
- session IDs are hashed/truncated for log correlation
- email addresses are masked
- chat logs use message length, not message content

### CORS Rules

- wildcard CORS is blocked outside local/dev/test environments
- local default allows localhost origins when `CORS_ALLOW_ORIGINS` is empty
- non-local environments must explicitly set allowed origins

