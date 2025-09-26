############################################
# Variables
############################################
variable "project_id"          { type = string }
variable "region"              { type = string  default = "us-central1" }
variable "name"                { type = string  default = "hyper-os" }

# Buckets from gcs_lake.tf (pass outputs/known names)
variable "bucket_raw"          { type = string }
variable "bucket_processed"    { type = string }
variable "bucket_curated"      { type = string }
variable "use_cmek"            { type = bool    default = false }
variable "kms_key"             { type = string  default = null } # projects/.../cryptoKeys/...

# BigQuery dataset (from bigquery.tf)
variable "bq_dataset_id"       { type = string  default = "hyper_os_analytics" }

# KSA (Kubernetes ServiceAccount) identities that need cloud perms
# Format: namespace/name
variable "ksa_storage_gateway" { type = string  default = "data/storage-gateway" }
variable "ksa_clickhouse"      { type = string  default = "analytics/clickhouse" }
variable "ksa_api_gateway"     { type = string  default = "os/api-gateway" }
variable "ksa_streamer"        { type = string  default = "data/streamer" }         # Pub/Sub consumer/producer
variable "ksa_bq_loader"       { type = string  default = "analytics/bq-loader" }

############################################
# Providers & helpers
############################################
provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "p" { project_id = var.project_id }

locals {
  wi_pool = "${var.project_id}.svc.id.goog"

  # helper to split "ns/name"
  ksa = {
    storage_gateway = { ns = split("/", var.ksa_storage_gateway)[0], name = split("/", var.ksa_storage_gateway)[1] }
    clickhouse      = { ns = split("/", var.ksa_clickhouse)[0],      name = split("/", var.ksa_clickhouse)[1] }
    api_gateway     = { ns = split("/", var.ksa_api_gateway)[0],     name = split("/", var.ksa_api_gateway)[1] }
    streamer        = { ns = split("/", var.ksa_streamer)[0],        name = split("/", var.ksa_streamer)[1] }
    bq_loader       = { ns = split("/", var.ksa_bq_loader)[0],       name = split("/", var.ksa_bq_loader)[1] }
  }

  # Workload Identity member strings
  wi = {
    storage_gateway = "serviceAccount:${local.wi_pool}[${local.ksa.storage_gateway.ns}/${local.ksa.storage_gateway.name}]"
    clickhouse      = "serviceAccount:${local.wi_pool}[${local.ksa.clickhouse.ns}/${local.ksa.clickhouse.name}]"
    api_gateway     = "serviceAccount:${local.wi_pool}[${local.ksa.api_gateway.ns}/${local.ksa.api_gateway.name}]"
    streamer        = "serviceAccount:${local.wi_pool}[${local.ksa.streamer.ns}/${local.ksa.streamer.name}]"
    bq_loader       = "serviceAccount:${local.wi_pool}[${local.ksa.bq_loader.ns}/${local.ksa.bq_loader.name}]"
  }
}

############################################
# Cloud Service Accounts (1 per workload)
############################################
resource "google_service_account" "sa_storage_gateway" {
  account_id   = "${var.name}-sa-storage-gw"
  display_name = "Storage Gateway SA"
}
resource "google_service_account" "sa_clickhouse" {
  account_id   = "${var.name}-sa-clickhouse"
  display_name = "ClickHouse SA"
}
resource "google_service_account" "sa_api_gateway" {
  account_id   = "${var.name}-sa-api-gw"
  display_name = "API Gateway SA"
}
resource "google_service_account" "sa_streamer" {
  account_id   = "${var.name}-sa-streamer"
  display_name = "Streaming/Bus SA"
}
resource "google_service_account" "sa_bq_loader" {
  account_id   = "${var.name}-sa-bq-loader"
  display_name = "BQ Loader SA"
}

############################################
# Bind KSAs to Cloud SAs (Workload Identity)
############################################
resource "google_service_account_iam_member" "wi_storage_gateway" {
  service_account_id = google_service_account.sa_storage_gateway.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.wi.storage_gateway
}
resource "google_service_account_iam_member" "wi_clickhouse" {
  service_account_id = google_service_account.sa_clickhouse.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.wi.clickhouse
}
resource "google_service_account_iam_member" "wi_api_gateway" {
  service_account_id = google_service_account.sa_api_gateway.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.wi.api_gateway
}
resource "google_service_account_iam_member" "wi_streamer" {
  service_account_id = google_service_account.sa_streamer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.wi.streamer
}
resource "google_service_account_iam_member" "wi_bq_loader" {
  service_account_id = google_service_account.sa_bq_loader.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.wi.bq_loader
}

############################################
# PERMISSIONS (least privilege)
############################################

# ---- GCS lake access
# storage-gateway: read raw/processed, write curated (adjust as you like)
resource "google_storage_bucket_iam_member" "raw_ro" {
  bucket = var.bucket_raw
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.sa_storage_gateway.email}"
}
resource "google_storage_bucket_iam_member" "processed_ro" {
  bucket = var.bucket_processed
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.sa_storage_gateway.email}"
}
resource "google_storage_bucket_iam_member" "curated_rw" {
  bucket = var.bucket_curated
  role   = "roles/storage.objectAdmin" # put/delete in curated
  member = "serviceAccount:${google_service_account.sa_storage_gateway.email}"
}

# clickhouse: backup/restore to curated
resource "google_storage_bucket_iam_member" "ch_curated_rw" {
  bucket = var.bucket_curated
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.sa_clickhouse.email}"
}

# api-gateway: optional read-only curated (to serve files)
resource "google_storage_bucket_iam_member" "api_curated_ro" {
  bucket = var.bucket_curated
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.sa_api_gateway.email}"
}

# ---- BigQuery access
# bq-loader: write tables into dataset
resource "google_bigquery_dataset_iam_member" "bq_loader_editor" {
  project    = var.project_id
  dataset_id = var.bq_dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.sa_bq_loader.email}"
}

# api-gateway: read-only dataset
resource "google_bigquery_dataset_iam_member" "api_viewer" {
  project    = var.project_id
  dataset_id = var.bq_dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.sa_api_gateway.email}"
}

# ---- Pub/Sub (if you enabled bucket notifications or bus streaming)
# streamer SA: can publish & subscribe
resource "google_project_iam_member" "streamer_pubsub_pub" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.sa_streamer.email}"
}
resource "google_project_iam_member" "streamer_pubsub_sub" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.sa_streamer.email}"
}

# ---- Secret Manager (API keys, creds)
resource "google_project_iam_member" "api_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.sa_api_gateway.email}"
}

# ---- KMS (CMEK) optional
resource "google_kms_crypto_key_iam_member" "kms_encrypt_decrypt_storage_gw" {
  count  = var.use_cmek ? 1 : 0
  crypto_key_id = var.kms_key
  role   = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member = "serviceAccount:${google_service_account.sa_storage_gateway.email}"
}
resource "google_kms_crypto_key_iam_member" "kms_encrypt_decrypt_clickhouse" {
  count  = var.use_cmek ? 1 : 0
  crypto_key_id = var.kms_key
  role   = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member = "serviceAccount:${google_service_account.sa_clickhouse.email}"
}

############################################
# Outputs
############################################
output "service_accounts" {
  value = {
    storage_gateway = google_service_account.sa_storage_gateway.email
    clickhouse      = google_service_account.sa_clickhouse.email
    api_gateway     = google_service_account.sa_api_gateway.email
    streamer        = google_service_account.sa_streamer.email
    bq_loader       = google_service_account.sa_bq_loader.email
  }
}

output "workload_identity_members" {
  value = {
    storage_gateway = local.wi.storage_gateway
    clickhouse      = local.wi.clickhouse
    api_gateway     = local.wi.api_gateway
    streamer        = local.wi.streamer
    bq_loader       = local.wi.bq_loader
  }
}