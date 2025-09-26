############################################
# Variables
############################################
variable "project_id"       { type = string }
variable "region"           { type = string  default = "us-central1" }
variable "name"             { type = string  default = "hyper-os" }

# Instance config: choose "regional" or "multi-region"
# Examples: "regional-us-central1", "nam-eur-asia1"
variable "instance_config"  { type = string  default = "regional-us-central1" }
variable "instance_nodes"   { type = number  default = 1 }   # scale up (2, 3, …) or use processing_units
variable "processing_units" { type = number  default = null } # alt to nodes (1000 = 1 node)

# CMEK optional
variable "use_cmek"         { type = bool    default = false }
variable "kms_key"          { type = string  default = null } # projects/.../cryptoKeys/...

# Databases
variable "databases" {
  type = map(object({
    ddl = list(string) # schema statements
  }))
  default = {
    core = {
      ddl = [
        # --- Example trading schema ---
        "CREATE TABLE Strategies (strategy_id STRING(64) NOT NULL, name STRING(256), created_ts TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)) PRIMARY KEY(strategy_id)",
        "CREATE TABLE Positions (strategy_id STRING(64) NOT NULL, symbol STRING(32) NOT NULL, qty FLOAT64, avg_price FLOAT64, updated_ts TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)) PRIMARY KEY(strategy_id, symbol)",
        "CREATE TABLE Orders (order_id STRING(64) NOT NULL, strategy_id STRING(64) NOT NULL, symbol STRING(32), side STRING(8), qty FLOAT64, price FLOAT64, status STRING(32), created_ts TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)) PRIMARY KEY(order_id)"
      ]
    }
    audit = {
      ddl = [
        "CREATE TABLE AuditEvents (event_id STRING(64) NOT NULL, ts TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true), actor STRING(64), action STRING(MAX), details STRING(MAX)) PRIMARY KEY(event_id)"
      ]
    }
  }
}

# IAM bindings
variable "db_admins"  { type = list(string) default = [] } # ["user:alice@example.com"]
variable "db_users"   { type = list(string) default = [] } # read/write
variable "db_readers" { type = list(string) default = [] } # read-only

############################################
# Provider
############################################
provider "google" {
  project = var.project_id
  region  = var.region
}

############################################
# Spanner Instance
############################################
resource "google_spanner_instance" "instance" {
  name         = "${var.name}-spanner"
  config       = var.instance_config
  display_name = "${var.name} Spanner"
  project      = var.project_id

  # Either nodes or processing_units
  num_nodes        = var.processing_units == null ? var.instance_nodes : null
  processing_units = var.processing_units

  labels = { project = var.name }
}

############################################
# Databases + schema
############################################
resource "google_spanner_database" "db" {
  for_each     = var.databases
  instance     = google_spanner_instance.instance.name
  name         = each.key
  project      = var.project_id

  ddl          = each.value.ddl

  dynamic "encryption_config" {
    for_each = var.use_cmek ? [1] : []
    content {
      kms_key_name = var.kms_key
    }
  }

  depends_on = [google_spanner_instance.instance]
}

############################################
# IAM Bindings
############################################
resource "google_spanner_instance_iam_binding" "admin" {
  instance = google_spanner_instance.instance.name
  role     = "roles/spanner.admin"
  members  = var.db_admins
}

resource "google_spanner_database_iam_binding" "users" {
  for_each = google_spanner_database.db
  database = each.value.name
  instance = google_spanner_instance.instance.name
  role     = "roles/spanner.databaseUser"
  members  = var.db_users
}

resource "google_spanner_database_iam_binding" "readers" {
  for_each = google_spanner_database.db
  database = each.value.name
  instance = google_spanner_instance.instance.name
  role     = "roles/spanner.databaseReader"
  members  = var.db_readers
}

############################################
# Outputs
############################################
output "spanner_instance" {
  value = google_spanner_instance.instance.name
}
output "spanner_databases" {
  value = { for k, v in google_spanner_database.db : k => v.name }
}