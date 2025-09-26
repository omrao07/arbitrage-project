############################################
# AWS IAM for EKS IRSA (Argo, Flink, Feast, ClickHouse)
############################################

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

############################################
# Variables
############################################
variable "name"                  { type = string -  default - "hyper-os" }
variable "region"                { type = string - default - "us-east-1" }

# From your EKS stack/module (module.eks.oidc_provider_arn and url)
variable "eks_oidc_provider_arn" { type = string }
variable "eks_oidc_provider_url" { type = string } # e.g. "oidc.eks.us-east-1.amazonaws.com/id/XXXX"

# Data lake / artifacts
variable "s3_bucket_curated"     { type = string } # e.g. "hf-curated"
variable "s3_artifacts_prefix"   { type = string -  default - "" } # optional prefix scope

# Optional KMS for envelope encryption (S3, Secrets)
variable "kms_key_arn"           { type = string - default - null }

# Limit access to specific SQS queues if you use them
variable "sqs_queue_arns"        { type = list(string) - default - [] }

# K8s service accounts (namespace/name) -> role purpose
# Keep in sync with your K8s manifests / Helm values
variable "service_accounts" {
  description = "Map of components to K8s service accounts"
  type = map(object({
    namespace = string
    name      = string
  }))
  default = {
    argo_controller  = { namespace = "analytics", name = "argo-workflow-controller" }
    argo_server      = { namespace = "analytics", name = "argo-server" }
    flink_operator   = { namespace = "analytics", name = "flink-operator" }
    feast_operator   = { namespace = "data",      name = "feast-operator" }
    clickhouse_sa    = { namespace = "analytics", name = "ch-backup" }      # backups to S3
    workflows        = { namespace = "analytics", name = "argo-workflow" }  # pods spawned by Argo
  }
}

############################################
# OIDC provider ref
############################################
data "aws_iam_openid_connect_provider" "eks" {
  arn = var.eks_oidc_provider_arn
}

locals {
  oidc_url       = var.eks_oidc_provider_url
  bucket_arn     = "arn:aws:s3:::${var.s3_bucket_curated}"
  bucket_prefix  = length(var.s3_artifacts_prefix) > 0 ? "${var.s3_artifacts_prefix}" : ""
  bucket_arn_all = length(local.bucket_prefix) > 0 ? "${local.bucket_arn}/${local.bucket_prefix}/*" : "${local.bucket_arn}/*"

  # Convenience: compose subjects for SA trust
  sa_subjects = {
    for k, v in var.service_accounts :
    k => "system:serviceaccount:${v.namespace}:${v.name}"
  }
}

############################################
# Trust policy (IRSA) generator
############################################
# Creates a role trust policy for a specific K8s service account
data "aws_iam_policy_document" "irsa_trust" {
  for_each = var.service_accounts

  statement {
    effect = "Allow"
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.eks.arn]
    }
    actions = ["sts:AssumeRoleWithWebIdentity"]
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_url}:sub"
      values   = [local.sa_subjects[each.key]]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

############################################
# Base policies (S3, ECR, Logs, Secrets, KMS, SQS)
############################################
# S3 read/write limited to curated bucket/prefix
data "aws_iam_policy_document" "s3_rw" {
  statement {
    sid     = "ListBucket"
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [local.bucket_arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = [local.bucket_prefix, "${local.bucket_prefix}/*"]
    }
  }
  statement {
    sid     = "ObjectRW"
    effect  = "Allow"
    actions = ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload","s3:ListBucketMultipartUploads"]
    resources = [local.bucket_arn_all]
  }
}

# ECR read/pull
data "aws_iam_policy_document" "ecr_read" {
  statement {
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages"
    ]
    resources = ["*"]
  }
}

# CloudWatch Logs write
data "aws_iam_policy_document" "logs_write" {
  statement {
    effect = "Allow"
    actions = ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"]
    resources = ["*"]
  }
}

# Secrets Manager read
data "aws_iam_policy_document" "secrets_read" {
  statement {
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue","secretsmanager:DescribeSecret","secretsmanager:ListSecrets"]
    resources = ["*"]
  }
}

# Optional KMS decrypt for S3/Secrets (scoped to key)
data "aws_iam_policy_document" "kms_decrypt" {
  count = var.kms_key_arn == null ? 0 : 1
  statement {
    effect    = "Allow"
    actions   = ["kms:Decrypt","kms:DescribeKey"]
    resources = [var.kms_key_arn]
  }
}

# Optional SQS access (publisher/subscriber)
data "aws_iam_policy_document" "sqs_rw" {
  count = length(var.sqs_queue_arns) == 0 ? 0 : 1
  statement {
    effect  = "Allow"
    actions = ["sqs:SendMessage","sqs:ReceiveMessage","sqs:DeleteMessage","sqs:GetQueueAttributes","sqs:GetQueueUrl","sqs:ChangeMessageVisibility","sqs:ListQueues"]
    resources = var.sqs_queue_arns
  }
}

############################################
# Managed inline policies
############################################
resource "aws_iam_policy" "s3_rw" {
  name   = "${var.name}-s3-rw"
  policy = data.aws_iam_policy_document.s3_rw.json
}

resource "aws_iam_policy" "ecr_read" {
  name   = "${var.name}-ecr-read"
  policy = data.aws_iam_policy_document.ecr_read.json
}

resource "aws_iam_policy" "logs_write" {
  name   = "${var.name}-logs-write"
  policy = data.aws_iam_policy_document.logs_write.json
}

resource "aws_iam_policy" "secrets_read" {
  name   = "${var.name}-secrets-read"
  policy = data.aws_iam_policy_document.secrets_read.json
}

resource "aws_iam_policy" "kms_decrypt" {
  count  = var.kms_key_arn == null ? 0 : 1
  name   = "${var.name}-kms-decrypt"
  policy = data.aws_iam_policy_document.kms_decrypt[0].json
}

resource "aws_iam_policy" "sqs_rw" {
  count  = length(var.sqs_queue_arns) == 0 ? 0 : 1
  name   = "${var.name}-sqs-rw"
  policy = data.aws_iam_policy_document.sqs_rw[0].json
}

############################################
# Roles per component (least-privileged)
############################################
# Argo (controller/server) – needs S3 (artifacts), Logs, ECR, Secrets, (KMS if used)
resource "aws_iam_role" "argo" {
  name               = "${var.name}-sa-argo"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust["argo_controller"].json
}
resource "aws_iam_role_policy_attachment" "argo_attach" {
  for_each = {
    s3    = aws_iam_policy.s3_rw.arn
    ecr   = aws_iam_policy.ecr_read.arn
    logs  = aws_iam_policy.logs_write.arn
    sec   = aws_iam_policy.secrets_read.arn
    kms   = var.kms_key_arn == null ? null : aws_iam_policy.kms_decrypt[0].arn
  }
  role       = aws_iam_role.argo.name
  policy_arn = each.value == null ? aws_iam_policy.s3_rw.arn : each.value
  lifecycle  { ignore_changes = [policy_arn] } # harmless when kms=null evaluated
}

# Flink Operator / Jobs – S3 checkpoints, Logs, ECR, (SQS optional)
resource "aws_iam_role" "flink" {
  name               = "${var.name}-sa-flink"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust["flink_operator"].json
}
resource "aws_iam_role_policy_attachment" "flink_attach" {
  for_each = {
    s3   = aws_iam_policy.s3_rw.arn
    ecr  = aws_iam_policy.ecr_read.arn
    logs = aws_iam_policy.logs_write.arn
    sqs  = length(var.sqs_queue_arns) == 0 ? null : aws_iam_policy.sqs_rw[0].arn
  }
  role       = aws_iam_role.flink.name
  policy_arn = each.value == null ? aws_iam_policy.s3_rw.arn : each.value
  lifecycle  { ignore_changes = [policy_arn] }
}

# Feast Operator – S3 registry, Secrets
resource "aws_iam_role" "feast" {
  name               = "${var.name}-sa-feast"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust["feast_operator"].json
}
resource "aws_iam_role_policy_attachment" "feast_attach" {
  for_each = {
    s3   = aws_iam_policy.s3_rw.arn
    sec  = aws_iam_policy.secrets_read.arn
    logs = aws_iam_policy.logs_write.arn
  }
  role       = aws_iam_role.feast.name
  policy_arn = each.value
}

# ClickHouse backup – S3 backups only
resource "aws_iam_role" "clickhouse" {
  name               = "${var.name}-sa-clickhouse"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust["clickhouse_sa"].json
}
resource "aws_iam_role_policy_attachment" "clickhouse_attach" {
  for_each = {
    s3   = aws_iam_policy.s3_rw.arn
    logs = aws_iam_policy.logs_write.arn
  }
  role       = aws_iam_role.clickhouse.name
  policy_arn = each.value
}

# Argo workflow pods (templates) – same as controller but scoped to pod SA
resource "aws_iam_role" "workflows" {
  name               = "${var.name}-sa-workflows"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust["workflows"].json
}
resource "aws_iam_role_policy_attachment" "workflows_attach" {
  for_each = {
    s3   = aws_iam_policy.s3_rw.arn
    ecr  = aws_iam_policy.ecr_read.arn
    sec  = aws_iam_policy.secrets_read.arn
    logs = aws_iam_policy.logs_write.arn
  }
  role       = aws_iam_role.workflows.name
  policy_arn = each.value
}

############################################
# Outputs
############################################
output "irsa_roles" {
  value = {
    argo        = aws_iam_role.argo.arn
    flink       = aws_iam_role.flink.arn
    feast       = aws_iam_role.feast.arn
    clickhouse  = aws_iam_role.clickhouse.arn
    workflows   = aws_iam_role.workflows.arn
  }
}