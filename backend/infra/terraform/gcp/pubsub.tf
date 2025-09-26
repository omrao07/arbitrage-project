############################################
# Google Pub/Sub — production configuration
############################################

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

#############################################################
# Variables
#############################################################

variable "project_id" { type = string }
variable "region"     { type = string - "us-central1" }

# Top-level IAM convenience (optional)
variable "topic_publishers" {
  description = "Principals that can publish to ALL topics (e.g., serviceAccount:ingestor@proj.iam.gserviceaccount.com)"
  type        = list(string)
  default     = []
}

variable "subscription_pullers" {
  description = "Principals that can pull from ALL subscriptions"
  type        = list(string)
  default     = []
}

# Topics config
# - schema: either reference an existing schema name OR create_inline_json to create google_pubsub_schema
# - kms_key: optional CMEK (projects/.../locations/.../keyRings/.../cryptoKeys/...)
variable "topics" {
  description = "Map of topics"
  type = map(object({
    labels                = map(string)
    message_retention     = string           # e.g. "604800s" (7 days)
    kms_key               = optional(string)
    schema = optional(object({
      name               = optional(string)  # existing schema name
      type               = optional(string)  # "AVRO" or "PROTOCOL_BUFFER" (used if creating)
      create_inline_json = optional(string)  # JSON schema string (if provided, a schema will be created)
      encoding           = optional(string)  # "JSON" or "BINARY" (defaults to JSON)
    }))
  }))
  default = {
    "market-ticks" = {
      labels            = { domain = "market", stream = "ticks" }
      message_retention = "604800s"
      kms_key           = null
      schema = {
        create_inline_json = <<EOF
{ "type":"record","name":"Tick","namespace":"market",
  "fields":[
    {"name":"ts","type":"long"},
    {"name":"symbol","type":"string"},
    {"name":"px","type":"double"},
    {"name":"sz","type":"double"}
  ]}
EOF
        type     = "AVRO"
        encoding = "JSON"
      }
    }
    "news-signals" = {
      labels            = { domain = "news", stream = "signals" }
      message_retention = "259200s"
      kms_key           = null
      schema = null
    }
  }
}

# Subscriptions config (one flat map, each entry references a topic key)
# delivery_type = "PULL" | "PUSH"
# dlq.create = true -> auto-create a -dlq topic
variable "subscriptions" {
  description = "Map of subscriptions"
  type = map(object({
    topic_key               = string
    ack_deadline_seconds    = number
    retain_acked_messages   = bool
    message_retention       = string          # e.g. "1209600s" (14 days)
    filter                  = optional(string)
    delivery_type           = string          # "PULL" or "PUSH"
    push_endpoint           = optional(string)
    oidc_service_account    = optional(string) # for push auth (service account email)
    bigquery = optional(object({
      table          = string                 # project.dataset.table
      write_metadata = optional(bool)         # default false
      use_topic_schema = optional(bool)       # default false
    }))
    dlq = optional(object({
      create                 = bool
      max_delivery_attempts  = number         # e.g. 10
      topic_name_override    = optional(string)
    }))
    labels = optional(map(string))
  }))
  default = {
    "ticks-live-pull" = {
      topic_key             = "market-ticks"
      ack_deadline_seconds  = 20
      retain_acked_messages = true
      message_retention     = "1209600s"
      delivery_type         = "PULL"
      filter                = null
      bigquery              = null
      dlq = {
        create                = true
        max_delivery_attempts = 10
      }
      labels = { team = "ingest" }
    }
    "signals-to-bq" = {
      topic_key             = "news-signals"
      ack_deadline_seconds  = 20
      retain_acked_messages = false
      message_retention     = "604800s"
      delivery_type         = "PULL"
      bigquery = {
        table            = "YOUR_PROJECT.feeds.news_signals"
        write_metadata   = true
        use_topic_schema = false
      }
      dlq = null
      labels = { team = "ml" }
    }
  }
}

#############################################################
# Schemas (created only when inline JSON provided)
#############################################################

# Build a subset of topics that request inline schema creation
locals {
  topics_requiring_schema = {
    for k, v in var.topics :
    k => v
    if try(v.schema.create_inline_json, null) != null
  }
}

resource "google_pubsub_schema" "schema" {
  for_each = local.topics_requiring_schema

  name       = "${each.key}-schema"
  type       = upper(try(each.value.schema.type, "AVRO"))
  definition = trimspace(each.value.schema.create_inline_json)
}

#############################################################
# Topics
#############################################################

resource "google_pubsub_topic" "topic" {
  for_each = var.topics

  name    = each.key
  labels  = each.value.labels

  message_retention_duration = each.value.message_retention

  dynamic "kms_key_name" {
    for_each = each.value.kms_key == null ? [] : [each.value.kms_key]
    content {
      # nothing – using dynamic here would not work; kms_key_name is a scalar
    }
  }

  # Use plain attribute when provided
  kms_key_name = try(each.value.kms_key, null)

  # Attach schema if requested (either created above or referenced by name)
  dynamic "schema_settings" {
    for_each = try(each.value.schema, null) == null ? [] : [each.value.schema]
    content {
      schema = (
        try(schema_settings.value.name, null) != null
          ? schema_settings.value.name
          : google_pubsub_schema.schema[each.key].id
      )
      encoding = upper(try(schema_settings.value.encoding, "JSON"))
    }
  }
}

#############################################################
# Dead-letter Topics (auto-create when requested)
#############################################################

locals {
  subs_with_dlq = {
    for k, v in var.subscriptions :
    k => v if try(v.dlq.create, false)
  }
}

resource "google_pubsub_topic" "dlq" {
  for_each = local.subs_with_dlq

  name   = coalesce(try(each.value.dlq.topic_name_override, null), "${each.key}-dlq")
  labels = merge(lookup(var.topics[each.value.topic_key].labels, {}), { purpose = "dlq" })
}

#############################################################
# Subscriptions
#############################################################

resource "google_pubsub_subscription" "sub" {
  for_each = var.subscriptions

  name   = each.key
  topic  = google_pubsub_topic.topic[each.value.topic_key].name
  labels = try(each.value.labels, {})

  ack_deadline_seconds    = each.value.ack_deadline_seconds
  retain_acked_messages   = each.value.retain_acked_messages
  message_retention_duration = each.value.message_retention

  filter = try(each.value.filter, null)

  dynamic "dead_letter_policy" {
    for_each = try(each.value.dlq.create, false) ? [1] : []
    content {
      dead_letter_topic     = google_pubsub_topic.dlq[each.key].name
      max_delivery_attempts = each.value.dlq.max_delivery_attempts
    }
  }

  dynamic "push_config" {
    for_each = upper(each.value.delivery_type) == "PUSH" ? [1] : []
    content {
      push_endpoint = each.value.push_endpoint
      dynamic "oidc_token" {
        for_each = try(each.value.oidc_service_account, null) == null ? [] : [1]
        content {
          service_account_email = each.value.oidc_service_account
        }
      }
    }
  }

  # BigQuery export (optional)
  dynamic "bigquery_config" {
    for_each = try(each.value.bigquery, null) == null ? [] : [each.value.bigquery]
    content {
      table               = bigquery_config.value.table
      use_topic_schema    = try(bigquery_config.value.use_topic_schema, false)
      write_metadata      = try(bigquery_config.value.write_metadata, false)
    }
  }
}

#############################################################
# IAM (project-wide convenience + per-resource hooks)
#############################################################

# Allow global publisher/puller sets
resource "google_pubsub_topic_iam_binding" "all_publishers" {
  for_each = length(var.topic_publishers) > 0 ? var.topics : {}
  topic    = google_pubsub_topic.topic[each.key].name
  role     = "roles/pubsub.publisher"
  members  = var.topic_publishers
}

resource "google_pubsub_subscription_iam_binding" "all_pullers" {
  for_each     = length(var.subscription_pullers) > 0 ? var.subscriptions : {}
  subscription = google_pubsub_subscription.sub[each.key].name
  role         = "roles/pubsub.subscriber"
  members      = var.subscription_pullers
}

#############################################################
# Outputs
#############################################################

output "topics" {
  value = { for k, v in google_pubsub_topic.topic : k => v.name }
}

output "subscriptions" {
  value = { for k, v in google_pubsub_subscription.sub : k => v.name }
}

output "dlq_topics" {
  value = { for k, v in google_pubsub_topic.dlq : k => v.name }
}