############################################
# Variables
############################################

variable "project_id"              { type = string }
variable "region"                  { type = string  default = "us-central1" }
variable "name"                    { type = string  default = "hyper-os" }     # prefix
variable "labels"                  { type = map(string) default = { project = "hyper-os" } }

# CMEK
variable "use_cmek"                { type = bool    default = false }
variable "kms_key"                 { type = string  default = null }           # projects/.../locations/.../keyRings/.../cryptoKeys/...
variable "create_kms"              { type = bool    default = false }
variable "kms_location"            { type = string  default = "us" }           # must match bucket location/dual-region

# Access logging bucket
variable "enable_access_logs"      { type = bool    default = true }

# Lifecycle & retention
variable "lifecycle_nearline_days" { type = number  default = 30 }
variable "lifecycle_coldline_days" { type = number  default = 180 }
variable "lifecycle_archive_days"  { type = number  default = 365 }
variable "object_retention_days"   { type = number  default = 0 }              # 0 = no retention policy
variable "enable_versioning"       { type = bool    default = true }

# Pub/Sub notifications (optional)
variable "enable_notifications"    { type = bool    default = false }
variable "notify_events"           { type = list(string) default = ["OBJECT_FINALIZE"] } # OBJECT_* events

############################################
# Providers & helpers
############################################

data "google_project" "this" { project_id = var.project_id }

locals {
  suffix        = data.google_project.this.number
  raw_bucket    = "${var.name}-lake-raw-${local.suffix}"
  proc_bucket   = "${var.name}-lake-processed-${local.suffix}"
  cur_bucket    = "${var.name}-lake-curated-${local.suffix}"
  logs_bucket   = "${var.name}-lake-logs-${local.suffix}"

  # Decide CMEK key to use
  use_cmek_key  = var.use_cmek ? (var.create_kms ? google_kms_crypto_key.lake[0].id : var.kms_key) : null
  uniform_access = true
}

############################################
# (Optional) Create CMEK key
############################################

resource "google_kms_key_ring" "lake" {
  count    = var.use_cmek && var.create_kms ? 1 : 0
  project  = var.project_id
  name     = "${var.name}-lake-ring"
  location = var.kms_location
}

resource "google_kms_crypto_key" "lake" {
  count           = var.use_cmek && var.create_kms ? 1 : 0
  name            = "${var.name}-lake-key"
  key_ring        = google_kms_key_ring.lake[0].id
  rotation_period = "7776000s" # 90 days
  lifecycle { prevent_destroy = false }
}

############################################
# Logs bucket (for access logs)
############################################

resource "google_storage_bucket" "logs" {
  count                        = var.enable_access_logs ? 1 : 0
  project                      = var.project_id
  name                         = local.logs_bucket
  location                     = var.region
  storage_class                = "STANDARD"
  uniform_bucket_level_access  = local.uniform_access
  public_access_prevention     = "enforced"
  force_destroy                = false
  labels                       = merge(var.labels, { tier = "logs" })

  dynamic "encryption" {
    for_each = var.use_cmek ? [1] : []
    content { default_kms_key_name = local.use_cmek_key }
  }

  versioning { enabled = var.enable_versioning }

  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "COLDLINE" }
    condition { age = 180 }
  }
  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 730 } # 2y
  }
}

############################################
# Lake buckets: raw, processed, curated
############################################

resource "google_storage_bucket" "raw" {
  project                      = var.project_id
  name                         = local.raw_bucket
  location                     = var.region
  storage_class                = "STANDARD"
  uniform_bucket_level_access  = local.uniform_access
  public_access_prevention     = "enforced"
  force_destroy                = false
  labels                       = merge(var.labels, { tier = "raw" })

  dynamic "encryption" {
    for_each = var.use_cmek ? [1] : []
    content { default_kms_key_name = local.use_cmek_key }
  }

  versioning { enabled = var.enable_versioning }

  dynamic "retention_policy" {
    for_each = var.object_retention_days > 0 ? [1] : []
    content {
      retention_period = var.object_retention_days * 24 * 60 * 60
      is_locked        = false  # set true only when you are 200% sure
    }
  }

  logging {
    log_bucket = var.enable_access_logs ? google_storage_bucket.logs[0].name : null
    log_object_prefix = "raw"
  }

  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "NEARLINE" }
    condition { age = var.lifecycle_nearline_days }
  }
  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "COLDLINE" }
    condition { age = var.lifecycle_coldline_days }
  }
  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "ARCHIVE" }
    condition { age = var.lifecycle_archive_days }
  }
}

resource "google_storage_bucket" "processed" {
  project                      = var.project_id
  name                         = local.proc_bucket
  location                     = var.region
  storage_class                = "STANDARD"
  uniform_bucket_level_access  = local.uniform_access
  public_access_prevention     = "enforced"
  force_destroy                = false
  labels                       = merge(var.labels, { tier = "processed" })

  dynamic "encryption" {
    for_each = var.use_cmek ? [1] : []
    content { default_kms_key_name = local.use_cmek_key }
  }

  versioning { enabled = var.enable_versioning }

  logging {
    log_bucket = var.enable_access_logs ? google_storage_bucket.logs[0].name : null
    log_object_prefix = "processed"
  }

  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "NEARLINE" }
    condition { age = var.lifecycle_nearline_days }
  }
  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "COLDLINE" }
    condition { age = var.lifecycle_coldline_days }
  }
  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "ARCHIVE" }
    condition { age = var.lifecycle_archive_days }
  }
}

resource "google_storage_bucket" "curated" {
  project                      = var.project_id
  name                         = local.cur_bucket
  location                     = var.region
  storage_class                = "STANDARD"
  uniform_bucket_level_access  = local.uniform_access
  public_access_prevention     = "enforced"
  force_destroy                = false
  labels                       = merge(var.labels, { tier = "curated" })

  dynamic "encryption" {
    for_each = var.use_cmek ? [1] : []
    content { default_kms_key_name = local.use_cmek_key }
  }

  versioning { enabled = var.enable_versioning }

  logging {
    log_bucket = var.enable_access_logs ? google_storage_bucket.logs[0].name : null
    log_object_prefix = "curated"
  }

  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "NEARLINE" }
    condition { age = var.lifecycle_nearline_days }
  }
  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "COLDLINE" }
    condition { age = var.lifecycle_coldline_days }
  }
  lifecycle_rule {
    action { type = "SetStorageClass", storage_class = "ARCHIVE" }
    condition { age = var.lifecycle_archive_days }
  }
}

############################################
# Optional Pub/Sub notifications on OBJECT_* events
############################################

resource "google_pubsub_topic" "lake_events" {
  count   = var.enable_notifications ? 1 : 0
  name    = "${var.name}-lake-events"
  project = var.project_id
  labels  = var.labels
}

# Allow buckets to publish to topic
resource "google_pubsub_topic_iam_member" "allow_bucket_raw" {
  count   = var.enable_notifications ? 1 : 0
  topic   = google_pubsub_topic.lake_events[0].name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.this.number}@gs-project-accounts.iam.gserviceaccount.com"
}
resource "google_pubsub_topic_iam_member" "allow_bucket_processed" {
  count   = var.enable_notifications ? 1 : 0
  topic   = google_pubsub_topic.lake_events[0].name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.this.number}@gs-project-accounts.iam.gserviceaccount.com"
}
resource "google_pubsub_topic_iam_member" "allow_bucket_curated" {
  count   = var.enable_notifications ? 1 : 0
  topic   = google_pubsub_topic.lake_events[0].name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.this.number}@gs-project-accounts.iam.gserviceaccount.com"
}

resource "google_storage_notification" "raw_notify" {
  count        = var.enable_notifications ? 1 : 0
  bucket       = google_storage_bucket.raw.name
  payload_format = "JSON_API_V1"
  topic        = google_pubsub_topic.lake_events[0].id
  event_types  = var.notify_events
}
resource "google_storage_notification" "processed_notify" {
  count        = var.enable_notifications ? 1 : 0
  bucket       = google_storage_bucket.processed.name
  payload_format = "JSON_API_V1"
  topic        = google_pubsub_topic.lake_events[0].id
  event_types  = var.notify_events
}
resource "google_storage_notification" "curated_notify" {
  count        = var.enable_notifications ? 1 : 0
  bucket       = google_storage_bucket.curated.name
  payload_format = "JSON_API_V1"
  topic        = google_pubsub_topic.lake_events[0].id
  event_types  = var.notify_events
}

############################################
# Outputs
############################################

output "gcs_lake_buckets" {
  value = {
    raw       = google_storage_bucket.raw.name
    processed = google_storage_bucket.processed.name
    curated   = google_storage_bucket.curated.name
    logs      = var.enable_access_logs ? google_storage_bucket.logs[0].name : null
  }
}

output "gcs_cmek_key" {
  value       = local.use_cmek_key
  description = "CMEK key resource ID used for default encryption (null if Google-managed)"
}