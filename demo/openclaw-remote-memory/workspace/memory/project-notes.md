# Project Architecture

## Overview
Our application uses a microservices architecture with three main services:
- **Gateway Service**: Handles incoming HTTP requests, rate limiting, and authentication. Runs on port 8080.
- **Processing Service**: Core business logic for data transformation and validation. Uses RabbitMQ for async jobs.
- **Storage Service**: Manages PostgreSQL database and S3 file uploads.

## Deployment
We deploy on Kubernetes using Helm charts. The CI/CD pipeline runs on GitHub Actions.
- Staging: auto-deploys from `develop` branch
- Production: manual approval from `main` branch
- Rollback: use `helm rollback <release> <revision>`

## API Conventions
All endpoints follow REST conventions:
- Authentication via Bearer token in `Authorization` header
- Pagination with `?page=1&limit=20`
- Error responses use RFC 7807 Problem Details format
- Rate limit: 100 requests per minute per API key

## Database Schema
Key tables:
- `users`: id, email, name, created_at, updated_at
- `projects`: id, user_id, title, description, status (draft|active|archived)
- `tasks`: id, project_id, assignee_id, title, priority (low|medium|high|critical), due_date

## Team Contacts
- Backend lead: alice@example.com
- Frontend lead: bob@example.com
- DevOps: charlie@example.com
- On-call rotation: PagerDuty schedule "eng-primary"
