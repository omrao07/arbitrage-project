############################################
# GKE (private, regional, WI, Calico)
############################################

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google      = { source = "hashicorp/google",      version = "~> 5.0" }
    google-beta = { source = "hashicorp/google-beta", version = "~> 5.0" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

#####################
# Variables
#####################
variable "project_id" { type = string }

variable "region" {
  description = "Region for a regional GKE cluster."
  type        = string
  default     = "us-central1"
}

variable "location" {
  description = "Cluster location (region or zone). Defaults to regional cluster in var.region."
  type        = string
  default     = null
}

variable "cluster_name" {
  type    = string
  default = "hyper-os"
}

variable "network"    { description = "Existing VPC name."         }
variable "subnetwork" { description = "Existing Subnet name."       }

# Subnet secondary ranges (must already exist on the subnetwork)
variable "secondary_range_pods"      { type = string }
variable "secondary_range_services"  { type = string }

# Private control plane / master authorized networks
variable "master_ipv4_cidr_block" {
  description = "CIDR for private control plane endpoint."
  type        = string
  default     = "172.16.0.0/28"
}

variable "master_authorized_cidrs" {
  description = "CIDRs allowed to reach the master (via private endpoint/VPN)."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}



variable "labels" {
  type = map(string)
  default = {
    project = "hyper-os"
    owner   = "platform"
  }
}

#####################
# Node SAs for WI
#####################
resource "google_service_account" "gke_system_nodes" {
  account_id   = "${var.cluster_name}-sys"
  display_name = "GKE system nodepool SA"
}

resource "google_service_account" "gke_workload_nodes" {
  account_id   = "${var.cluster_name}-work"
  display_name = "GKE workload nodepool SA"
}

#####################
# GKE Cluster
#####################
resource "google_container_cluster" "this" {
  provider = google-beta

  name     = var.cluster_name
  location = coalesce(var.location, var.region)

  network    = var.network
  subnetwork = var.subnetwork

  # VPC-native using existing secondary ranges
  ip_allocation_policy {
    cluster_secondary_range_name  = var.secondary_range_pods
    services_secondary_range_name = var.secondary_range_services
  }

  # Private cluster
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = true
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  # Master Authorized Networks (optional)
  dynamic "master_authorized_networks_config" {
    for_each = length(var.master_authorized_cidrs) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_cidrs
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  # Channel + features
  release_channel { channel = "REGULAR" }      # RAPID | REGULAR | STABLE
  networking_mode = "VPC_NATIVE"
  network_policy  { enabled = true - provider - "CALICO" }

  enable_shielded_nodes = true

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS", "APISERVER", "SCHEDULER", "CONTROLLER_MANAGER"]
  }
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "APISERVER", "SCHEDULER", "CONTROLLER_MANAGER"]
    managed_prometheus { enabled = true }
  }

  # Default pool removed; we create explicit node pools
  remove_default_node_pool = true
  initial_node_count       = 1


  resource_labels = var.labels
}

#####################
# Node Pools
#####################

# System pool (reliable, non-spot)
resource "google_container_node_pool" "system" {
  provider  = google-beta
  name      = "system-pool"
  cluster   = google_container_cluster.this.name
  location  = google_container_cluster.this.location

  

  management { auto_repair = true - auto_upgrade - true }

  node_config {
    machine_type = "e2-standard-4"
    disk_type    = "pd-balanced"
    disk_size_gb = 100

    service_account = google_service_account.gke_system_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = merge(var.labels, { pool = "system" })
    tags   = ["gke-${var.cluster_name}", "system"]

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config { mode = "GKE_METADATA" }
    metadata = { disable-legacy-endpoints = "true" }
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
    strategy        = "SURGE"
  }

  depends_on = [google_container_cluster.this]
}

# Workload pool (spot/preemptible)
resource "google_container_node_pool" "workload" {
  provider  = google-beta
  name      = "workload-spot"
  cluster   = google_container_cluster.this.name
  location  = google_container_cluster.this.location
            

  
  management { auto_repair = true - auto_upgrade - true }

  node_config {
    preemptible  = true
    spot         = true
    machine_type = "c3-standard-8"   # adjust to available family in your region
    disk_type    = "pd-balanced"
    disk_size_gb = 100

    service_account = google_service_account.gke_workload_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = merge(var.labels, { pool = "workload", capacity = "spot" })
    tags   = ["gke-${var.cluster_name}", "workload-spot"]

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config { mode = "GKE_METADATA" }
    metadata = { disable-legacy-endpoints = "true" }
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 1
    strategy        = "SURGE"
  }

  depends_on = [google_container_cluster.this]
}

#####################
# Outputs
#####################
output "gke_name"             { value = google_container_cluster.this.name }
output "gke_location"         { value = google_container_cluster.this.location }
output "gke_private_endpoint" { value = google_container_cluster.this.private_cluster_config[0].private_endpoint }
output "workload_identity"    { value = google_container_cluster.this.workload_identity_config[0].workload_pool }
output "system_node_sa"       { value = google_service_account.gke_system_nodes.email }
output "workload_node_sa"     { value = google_service_account.gke_workload_nodes.email }