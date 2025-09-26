############################################
# Variables
############################################
variable "name"            { type = string  default = "hyper-os" }
variable "region"          { type = string  default = "us-east-1" }

# Toggle/choose KMS for customer-managed encryption
variable "use_kms"         { type = bool    default = true }
variable "kms_key_arn"     { type = string  default = null }  # if null and use_kms=true, create a key

# Principals allowed to read secrets (ARNS of IAM roles, e.g., IRSA roles from iam_policies.tf)
variable "reader_role_arns" {
  type    = list(string)
  default = []  # e.g. ["arn:aws:iam::123456789012:role/hyper-os-sa-api-gw"]
}

# Optional rotation lambda ARN (if you already have a rotation function)
variable "rotation_lambda_arn" { type = string default = null }

############################################
# KMS (optional)
############################################
resource "aws_kms_key" "secrets" {
  count                   = var.use_kms && var.kms_key_arn == null ? 1 : 0
  description             = "${var.name} secrets encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  tags = { Project = var.name, Role = "secrets-kms" }
}
locals {
  secrets_kms_arn = var.use_kms
    ? (var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.secrets[0].arn)
    : null
}

############################################
# Secrets (add more as needed)
############################################
# 1) Third-party feeds / API keys
resource "aws_secretsmanager_secret" "api_news" {
  name       = "${var.name}/api/news"
  kms_key_id = local.secrets_kms_arn
  tags       = { Project = var.name, Type = "api" }
}

resource "aws_secretsmanager_secret" "api_marketdata" {
  name       = "${var.name}/api/marketdata"
  kms_key_id = local.secrets_kms_arn
  tags       = { Project = var.name, Type = "api" }
}

# 2) Internal service creds (DB, ClickHouse, etc.)
resource "aws_secretsmanager_secret" "db_clickhouse" {
  name       = "${var.name}/db/clickhouse"
  kms_key_id = local.secrets_kms_arn
  tags       = { Project = var.name, Type = "db" }
}

# 3) Trading / execution keys (keep separately namespaced)
resource "aws_secretsmanager_secret" "exec_broker" {
  name       = "${var.name}/exec/broker"
  kms_key_id = local.secrets_kms_arn
  tags       = { Project = var.name, Type = "exec" }
}

# Initial values (create versions). Use `terraform apply -var='...'` or follow-up `secretsmanager put-secret-value`
resource "aws_secretsmanager_secret_version" "api_news_v" {
  secret_id     = aws_secretsmanager_secret.api_news.id
  secret_string = jsonencode({ NEWS_API_KEY = "REPLACE_ME" })
}
resource "aws_secretsmanager_secret_version" "api_marketdata_v" {
  secret_id     = aws_secretsmanager_secret.api_marketdata.id
  secret_string = jsonencode({ PROVIDER = "POLYGON", API_KEY = "REPLACE_ME" })
}
resource "aws_secretsmanager_secret_version" "db_clickhouse_v" {
  secret_id     = aws_secretsmanager_secret.db_clickhouse.id
  secret_string = jsonencode({ HOST = "clickhouse.analytics.svc.cluster.local", USER = "ch_user", PASSWORD = "REPLACE_ME" })
}
resource "aws_secretsmanager_secret_version" "exec_broker_v" {
  secret_id     = aws_secretsmanager_secret.exec_broker.id
  secret_string = jsonencode({ BROKER = "REPLACE", API_KEY = "REPLACE_ME", API_SECRET = "REPLACE_ME" })
}

############################################
# Resource policies (limit who can read)
############################################
data "aws_iam_policy_document" "secrets_resource" {
  # Deny cross-account by default; allow specific roles below
  statement {
    sid     = "DenyRootAccountWideRead"
    effect  = "Deny"
    principals { type = "*", identifiers = ["*"] }
    actions = ["secretsmanager:GetSecretValue","secretsmanager:DescribeSecret"]
    resources = ["*"]
    condition {
      test     = "StringNotEquals"
      variable = "aws:PrincipalArn"
      values   = var.reader_role_arns
    }
  }
}

# Attach same tight policy to each secret
resource "aws_secretsmanager_secret_policy" "api_news" {
  secret_arn = aws_secretsmanager_secret.api_news.arn
  policy     = data.aws_iam_policy_document.secrets_resource.json
}
resource "aws_secretsmanager_secret_policy" "api_marketdata" {
  secret_arn = aws_secretsmanager_secret.api_marketdata.arn
  policy     = data.aws_iam_policy_document.secrets_resource.json
}
resource "aws_secretsmanager_secret_policy" "db_clickhouse" {
  secret_arn = aws_secretsmanager_secret.db_clickhouse.arn
  policy     = data.aws_iam_policy_document.secrets_resource.json
}
resource "aws_secretsmanager_secret_policy" "exec_broker" {
  secret_arn = aws_secretsmanager_secret.exec_broker.arn
  policy     = data.aws_iam_policy_document.secrets_resource.json
}

############################################
# Rotation (optional)
############################################
resource "aws_secretsmanager_secret_rotation" "api_news" {
  count                 = var.rotation_lambda_arn == null ? 0 : 1
  secret_id             = aws_secretsmanager_secret.api_news.id
  rotation_lambda_arn   = var.rotation_lambda_arn
  rotation_rules { automatically_after_days = 30 }
}

############################################
# Outputs
############################################
output "secrets_arns" {
  value = {
    api_news      = aws_secretsmanager_secret.api_news.arn
    api_marketdata= aws_secretsmanager_secret.api_marketdata.arn
    db_clickhouse = aws_secretsmanager_secret.db_clickhouse.arn
    exec_broker   = aws_secretsmanager_secret.exec_broker.arn
  }
}