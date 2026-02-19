# Incident Runbook

## High CPU on Processing Service
1. Check Grafana dashboard: https://grafana.internal/d/processing-overview
2. Look for stuck jobs: `kubectl exec -it processing-0 -- rabbitmqctl list_queues`
3. If queue depth > 10000, scale up: `kubectl scale deployment processing --replicas=5`
4. If single pod is stuck, restart: `kubectl delete pod processing-<id>`

## Database Connection Pool Exhaustion
1. Check current connections: `SELECT count(*) FROM pg_stat_activity;`
2. Kill idle connections older than 10 min:
   ```sql
   SELECT pg_terminate_backend(pid) FROM pg_stat_activity
   WHERE state = 'idle' AND query_start < now() - interval '10 minutes';
   ```
3. If persistent, increase `max_connections` in PostgreSQL config and restart

## S3 Upload Failures
1. Check AWS status page: https://status.aws.amazon.com/
2. Verify IAM role permissions: `aws sts get-caller-identity`
3. Check bucket policy allows PutObject from our VPC endpoint
4. Temporary workaround: enable local file fallback with env `STORAGE_FALLBACK=local`

## Gateway 502 Errors
1. Check if processing service is healthy: `curl -s http://processing:8081/health`
2. Verify Envoy sidecar is running: `kubectl describe pod gateway-<id>`
3. Check for certificate expiry: `openssl s_client -connect gateway:443 -servername api.example.com`
4. Escalate to DevOps if networking issue (charlie@example.com)
