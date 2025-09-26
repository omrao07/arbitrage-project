#########################################################
# Variables (set via terraform.tfvars or parent module)
#########################################################

variable "project_id"            { type = string }
variable "region"                { type = string  default = "US" }          # BQ location (US/EU/region)
variable "dataset_id"            { type = string  default = "hyper_os_analytics" }
variable "dataset_friendly"      { type = string  default = "Hyper OS Analytics" }
variable "default_table_exp_days"{ type = number  default = 0 }             # 0 = never expire
variable "use_cmek"              { type = bool    default = false }
variable "kms_key_id"            { type = string  default = null }          # projects/.../locations/.../keyRings/.../cryptoKeys/...
variable "labels"                { type = map(string) default = { project = "hyper-os" } }

# Optional: principals
variable "analyst_members"       { type = list(string) default = [] }       # e.g. ["group:analysts@yourco.com"]
variable "engineer_members"      { type = list(string) default = [] }       # e.g. ["group:data-eng@yourco.com"]
variable "reader_members"        { type = list(string) default = [] }

# Optional: GCS transfer
variable "enable_gcs_transfer"   { type = bool   default = false }
variable "gcs_source_bucket"     { type = string default = null }           # gs://bucket/path/…
variable "gcs_schedule"          { type = string default = "every 24 hours" }
variable "gcs_file_pattern"      { type = string default = "*.parquet" }    # or *.csv
variable "transfer_display_name" { type = string default = "lake-to-bq" }

#########################################################
# Dataset (CMEK optional)
#########################################################

resource "google_bigquery_dataset" "dataset" {
  project                    = var.project_id
  dataset_id                 = var.dataset_id
  friendly_name              = var.dataset_friendly
  location                   = var.region
  default_table_expiration_ms= var.default_table_exp_days == 0 ? null : var.default_table_exp_days * 24 * 60 * 60 * 1000
  labels                     = var.labels

  dynamic "default_encryption_configuration" {
    for_each = var.use_cmek ? [1] : []
    content {
      kms_key_name = var.kms_key_id
    }
  }
}

#########################################################
# Tables (partitioned + clustered) — examples you can extend
#########################################################

# 1) market_bars_1m: ingestion-time partition + cluster on symbol
resource "google_bigquery_table" "market_bars_1m" {
  project   = var.project_id
  dataset_id= google_bigquery_dataset.dataset.dataset_id
  table_id  = "market_bars_1m"

  time_partitioning {
    type                     = "DAY"
    require_partition_filter = true
  }
  clustering = ["symbol"]

  schema = jsonencode([
    { name="ts",        type="TIMESTAMP", mode="REQUIRED", description="bar end timestamp (UTC)" },
    { name="symbol",    type="STRING",    mode="REQUIRED" },
    { name="open",      type="FLOAT",     mode="NULLABLE" },
    { name="high",      type="FLOAT",     mode="NULLABLE" },
    { name="low",       type="FLOAT",     mode="NULLABLE" },
    { name="close",     type="FLOAT",     mode="NULLABLE" },
    { name="volume",    type="FLOAT",     mode="NULLABLE" },
    { name="venue",     type="STRING",    mode="NULLABLE" },
    { name="source",    type="STRING",    mode="NULLABLE" }
  ])

  dynamic "encryption_configuration" {
    for_each = var.use_cmek ? [1] : []
    content { kms_key_name = var.kms_key_id }
  }

  labels = merge(var.labels, { table = "market_bars_1m" })
}

# 2) news_index: partition on published_date, cluster on ticker/topic
resource "google_bigquery_table" "news_index" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "news_index"

  time_partitioning {
    type                     = "DAY"
    field                    = "published_date" # partition by column
    require_partition_filter = true
  }
  clustering = ["ticker", "topic"]

  schema = jsonencode([
    { name="published_date", type="DATE",      mode="REQUIRED" },
    { name="ts",             type="TIMESTAMP", mode="REQUIRED" },
    { name="source",         type="STRING",    mode="REQUIRED" },
    { name="headline",       type="STRING",    mode="REQUIRED" },
    { name="body",           type="STRING",    mode="NULLABLE" },
    { name="ticker",         type="STRING",    mode="NULLABLE" },
    { name="actor",          type="STRING",    mode="NULLABLE" },
    { name="topic",          type="STRING",    mode="NULLABLE" },
    { name="sentiment",      type="FLOAT",     mode="NULLABLE" },
    { name="relevance",      type="FLOAT",     mode="NULLABLE" },
    { name="novelty",        type="FLOAT",     mode="NULLABLE" }
  ])

  dynamic "encryption_configuration" {
    for_each = var.use_cmek ? [1] : []
    content { kms_key_name = var.kms_key_id }
  }

  labels = merge(var.labels, { table = "news_index" })
}

# 3) strategy_pnl_daily: DATE partition, cluster on strategy_id
resource "google_bigquery_table" "strategy_pnl_daily" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "strategy_pnl_daily"

  time_partitioning {
    type                     = "DAY"
    field                    = "date"
    require_partition_filter = true
  }
  clustering = ["strategy_id"]

  schema = jsonencode([
    { name="date",          type="DATE",   mode="REQUIRED" },
    { name="strategy_id",   type="STRING", mode="REQUIRED" },
    { name="gross_pnl",     type="FLOAT",  mode="NULLABLE" },
    { name="net_pnl",       type="FLOAT",  mode="NULLABLE" },
    { name="fees",          type="FLOAT",  mode="NULLABLE" },
    { name="slippage",      type="FLOAT",  mode="NULLABLE" },
    { name="risk_bucket",   type="STRING", mode="NULLABLE" },
    { name="leverage",      type="FLOAT",  mode="NULLABLE" }
  ])

  dynamic "encryption_configuration" {
    for_each = var.use_cmek ? [1] : []
    content { kms_key_name = var.kms_key_id }
  }

  labels = merge(var.labels, { table = "strategy_pnl_daily" })
}

#########################################################
# Row-Level Security (RLS) — policy on strategy_pnl_daily
#########################################################

# Example: only show rows where strategy_id in an allowed list (via session param)
# You can switch to FILTER USING with auth_user() in authorized UDFs if needed.
resource "google_bigquery_row_access_policy" "pnl_rls" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = google_bigquery_table.strategy_pnl_daily.table_id
  row_access_policy_id = "strategy_scope"

  # Simple expression: assumes a session parameter @allowed_ids (array<string>)
  # You can also tie this to a mapping table and use an authorized view instead.
  predicate_expression = "strategy_id IN UNNEST(@allowed_ids)"

  depends_on = [google_bigquery_table.strategy_pnl_daily]
}

#########################################################
# Authorized View (exposes only selected columns/rows)
#########################################################

# Create a restricted view over news_index (no body text, only aggregates)
resource "google_bigquery_table" "vw_news_sentiment_daily" {
  project     = var.project_id
  dataset_id  = google_bigquery_dataset.dataset.dataset_id
  table_id    = "vw_news_sentiment_daily"
  view {
    query = <<-SQL
      SELECT
        published_date,
        ticker,
        topic,
        COUNT(*) AS news_count,
        AVG(sentiment) AS avg_sentiment,
        AVG(relevance) AS avg_relevance
      FROM `${var.project_id}.${var.dataset_id}.news_index`
      WHERE published_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
      GROUP BY published_date, ticker, topic
    SQL
    use_legacy_sql = false
  }

  labels = merge(var.labels, { view = "news_sentiment_daily" })
}

# Authorize a consumer dataset (e.g., for a downstream project) to query this view
# (If you need cross-project access, set consumer_project_id accordingly.)
variable "consumer_project_id" { type = string default = null }
variable "consumer_dataset_id" { type = string default = null }

resource "google_bigquery_dataset_access" "authorize_view" {
  count     = var.consumer_project_id != null && var.consumer_dataset_id != null ? 1 : 0
  dataset_id= google_bigquery_dataset.dataset.dataset_id
  project   = var.project_id

  view {
    project_id = var.project_id
    dataset_id = google_bigquery_dataset.dataset.dataset_id
    table_id   = google_bigquery_table.vw_news_sentiment_daily.table_id
  }

  dataset {
    project_id = var.consumer_project_id
    dataset_id = var.consumer_dataset_id
  }
}

#########################################################
# IAM — dataset-level roles
#########################################################

resource "google_bigquery_dataset_iam_binding" "analyst" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  role       = "roles/bigquery.dataViewer"
  members    = var.reader_members
}

resource "google_bigquery_dataset_iam_binding" "engineer" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  members    = var.analyst_members
}

resource "google_bigquery_dataset_iam_binding" "admin" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  role       = "roles/bigquery.admin"
  members    = var.engineer_members
}

#########################################################
# Optional: GCS → BigQuery Transfer (parquet/csv autoload)
#########################################################

# Service account for transfers
resource "google_service_account" "transfer_sa" {
  count        = var.enable_gcs_transfer ? 1 : 0
  project      = var.project_id
  account_id   = "bq-transfer-${replace(var.dataset_id, "_", "-")}"
  display_name = "BQ Transfer for ${var.dataset_id}"
}

# Allow SA to read GCS & write to BQ
resource "google_project_iam_member" "transfer_bq_user" {
  count   = var.enable_gcs_transfer ? 1 : 0
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.transfer_sa[0].email}"
}
resource "google_project_iam_member" "transfer_gcs_reader" {
  count   = var.enable_gcs_transfer ? 1 : 0
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.transfer_sa[0].email}"
}

# Transfer config (loads matching objects into market_bars_1m)
resource "google_bigquery_data_transfer_config" "gcs_to_bq" {
  count        = var.enable_gcs_transfer ? 1 : 0
  project      = var.project_id
  display_name = var.transfer_display_name
  location     = var.region
  data_source_id = "google_cloud_storage"
  schedule     = var.gcs_schedule

  params = {
    data_path_template = "gs://${var.gcs_source_bucket}/${var.gcs_file_pattern}"
    destination_table_name_template = "market_bars_1m"
    file_format = "PARQUET" # or CSV
    write_disposition = "WRITE_APPEND"
    partition_spec = jsonencode({ partition_type = "DAY", field = "ts" })
  }

  service_account_name = google_service_account.transfer_sa[0].email
  dataset_region       = var.region

  depends_on = [google_bigquery_table.market_bars_1m]
}

#########################################################
# Outputs
#########################################################

output "bigquery_dataset_fqn" {
  value = "${var.project_id}.${google_bigquery_dataset.dataset.dataset_id}"
}

output "market_bars_1m_table" {
  value = "${var.project_id}.${var.dataset_id}.${google_bigquery_table.market_bars_1m.table_id}"
}

output "news_index_table" {
  value = "${var.project_id}.${var.dataset_id}.${google_bigquery_table.news_index.table_id}"
}

output "strategy_pnl_daily_table" {
  value = "${var.project_id}.${var.dataset_id}.${google_bigquery_table.strategy_pnl_daily.table_id}"
}