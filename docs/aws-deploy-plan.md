# AWS Deployment Plan (Option B — ECS Fargate + RDS)

Status: **in progress** — IAM user created, provisioning not yet started.

## Goal

Host the FastAPI RAG backend on AWS, and add a client-side snippet to the
GH Pages portfolio (`https://alonn24.github.io/`) that calls the deployed
API's `/query` endpoint.

## Decisions made

- **Region**: `eu-west-1` (matches existing AWS CLI default config).
- **Compute**: ECS Fargate (not EC2 + docker-compose) — chosen over the
  simpler EC2 option per user preference for a "properly cloud-native" setup.
- **Database**: RDS Postgres (single-AZ, `db.t4g.micro`) with the `vector`
  extension enabled, replacing the local `db` docker-compose service.
- **No domain available.** Browsers block `fetch()` from an HTTPS page
  (GH Pages) to a plain HTTP endpoint (mixed-content blocking), and ACM
  won't issue a free cert without a domain to validate. Workaround: put a
  **CloudFront distribution** in front of the ALB. CloudFront provides a
  free `*.cloudfront.net` HTTPS endpoint with an AWS-issued cert, no domain
  purchase needed. Negligible added cost at this traffic level.
- **VPC**: use the account's default VPC (not a custom VPC) to save setup
  time — acceptable for a project this size.

## Resources to provision (in order)

1. ~~IAM user for deployment~~ **DONE** — see below.
2. RDS Postgres instance (`db.t4g.micro`, single-AZ), `vector` extension
   enabled via parameter group / `CREATE EXTENSION`.
3. ECR repository for the FastAPI app image.
4. Build and push the app's Docker image to ECR.
5. Secrets Manager entries: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`,
   `DATABASE_URL` (pointing at the RDS instance).
6. ECS cluster + Fargate task definition (app container, secrets injected
   from Secrets Manager, CloudWatch log group) + service (1 task).
7. Application Load Balancer, HTTP listener (port 80) → target group →
   Fargate service. (No HTTPS on the ALB itself — CloudFront handles TLS.)
8. CloudFront distribution in front of the ALB, using the default
   `*.cloudfront.net` domain for HTTPS.
9. Code change: add CORS middleware to the FastAPI app
   (`allow_origins=["https://alonn24.github.io"]`), restricted to the
   `/query` (and any other needed) endpoints.
10. Client snippet for the GH Pages page: a small JS block that `fetch()`s
    `https://<cloudfront-domain>/query` with the user's question and
    renders the answer. (Page repo is not local — user edits it directly
    once the API URL is known.)

## Already done

- Created a scoped IAM user `fastapi-rag-deployer` (not root) with an
  inline policy covering: `ecr:*`, `ecs:*`, `ec2:*`, `elasticloadbalancing:*`,
  `rds:*`, `logs:*`, `secretsmanager:*`, `acm:*`, narrow `iam:*Role*`
  actions needed for ECS task roles, and `sts:GetCallerIdentity`. No admin
  or billing access.
- Configured local AWS CLI profile `fastapi-rag-deployer` (region
  `eu-west-1`) with that user's access key. Root credentials are no longer
  used for this work.

## Cost estimate

Roughly **$35-40/mo** always-on at this scale:
- RDS `db.t4g.micro`: ~$13/mo
- Fargate (0.25 vCPU / 0.5 GB, always-on): ~$9/mo
- ALB: ~$16/mo + minor data transfer
- CloudFront: ~$1-2/mo at low traffic
- Secrets Manager: ~$0.40/secret/mo (3 secrets ≈ $1.2/mo)

## Next steps (not started)

Provision in this order: RDS → ECR (build+push image) → Secrets Manager →
ECS Fargate → ALB → CloudFront → CORS code change → GH Pages client
snippet. Use the `fastapi-rag-deployer` AWS CLI profile for every `aws`
command (`--profile fastapi-rag-deployer`), not the default/root profile.

## Open question for user before resuming

None currently blocking — the domain/HTTPS decision (CloudFront route) and
IAM setup are resolved. Resume by starting RDS provisioning.
