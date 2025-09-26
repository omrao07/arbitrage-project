############################################
# Module: s3_data_bucket
############################################
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "name"            { type = string }
variable "logs_bucket_arn" { type = string }
variable "use_kms"         { type = bool }
variable "kms_key_arn"     { type = string - default - null }
variable "lifecycle_rules" {
  type = list(object({
    id      = string
    enabled = bool
    noncurrent_version_expiration = optional(object({
      noncurrent_days = number
    }))
    transition  = optional(object({
      days          = number
      storage_class = string
    }))
    transitions = optional(list(object({
      days          = number
      storage_class = string
    })))
  }))
}
variable "tags"        { type = map(string)- default - {} }
variable "policy_json" { type = string }

resource "aws_s3_bucket" "this" {
  bucket = var.name
  tags   = var.tags
}

resource "aws_s3_bucket_public_access_block" "this" {
  bucket                  = aws_s3_bucket.this.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "this" {
  bucket = aws_s3_bucket.this.id
  rule { object_ownership = "BucketOwnerEnforced" }
}

resource "aws_s3_bucket_versioning" "this" {
  bucket = aws_s3_bucket.this.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_logging" "this" {
  bucket        = aws_s3_bucket.this.id
  target_bucket = var.logs_bucket_arn
  target_prefix = "${var.name}/"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  bucket = aws_s3_bucket.this.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.use_kms ? var.kms_key_arn : null
      sse_algorithm     = var.use_kms ? "aws:kms" : "AES256"
    }
    bucket_key_enabled = !var.use_kms
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "this" {
  bucket = aws_s3_bucket.this.id

  dynamic "rule" {
    for_each = var.lifecycle_rules
    content {
      id     = rule.value.id
      status = rule.value.enabled ? "Enabled" : "Disabled"

      dynamic "noncurrent_version_expiration" {
        for_each = try([rule.value.noncurrent_version_expiration], [])
        content {
          noncurrent_days = noncurrent_version_expiration.value.noncurrent_days
        }
      }

      dynamic "transition" {
        for_each = try([rule.value.transition], [])
        content {
          days          = transition.value.days
          storage_class = transition.value.storage_class
        }
      }

      dynamic "transition" {
        for_each = try(rule.value.transitions, [])
        content {
          days          = transition.value.days
          storage_class = transition.value.storage_class
        }
      }
    }
  }
}