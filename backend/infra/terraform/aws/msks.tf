############################################
# Amazon MSK (Provisioned, TLS-only)
############################################

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

############################################
# Variables
############################################
variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "hyper-os-msk"
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  description = "At least 2–3 private subnets in different AZs"
  type        = list(string)
}

variable "client_sg_ids" {
  description = "Security groups of clients allowed to connect (e.g., EKS worker SG)"
  type        = list(string)
  default     = []
}

variable "broker_instance_type" {
  type    = string
  default = "kafka.m7g.large"
}

variable "number_of_broker_nodes" {
  description = "Must be a multiple of the number of subnets/AZs"
  type        = number
  default     = 3
}

variable "volume_size_gb" {
  type    = number
  default = 1000
}

variable "kafka_version" {
  type    = string
  default = "3.6.0"
}

variable "tags" {
  type    = map(string)
  default = { Project = "hyper-os", Stack = "msk" }
}

############################################
# Security Group
############################################
resource "aws_security_group" "msk_brokers" {
  name        = "${var.name}-brokers-sg"
  description = "MSK broker access"
  vpc_id      = var.vpc_id
  tags        = var.tags

  # Egress anywhere (for logs, metrics)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Allow clients (TLS 9094) from approved SGs
resource "aws_security_group_rule" "from_clients_kafka" {
  for_each                 = toset(var.client_sg_ids)
  type                     = "ingress"
  security_group_id        = aws_security_group.msk_brokers.id
  from_port                = 9094
  to_port                  = 9094
  protocol                 = "tcp"
  source_security_group_id = each.value
  description              = "Kafka TLS"
}

############################################
# MSK Configuration
############################################
resource "aws_msk_configuration" "config" {
  name           = "${var.name}-cfg"
  kafka_versions = [var.kafka_version]

  server_properties = <<-EOT
    auto.create.topics.enable=false
    default.replication.factor=3
    min.insync.replicas=2
    num.partitions=12
    log.retention.hours=168
    log.segment.bytes=1073741824
    log.retention.check.interval.ms=300000
    inter.broker.listener.name=TLS
    ssl.client.auth=required
  EOT
}

############################################
# MSK Cluster
############################################
resource "aws_msk_cluster" "this" {
  cluster_name           = var.name
  kafka_version          = var.kafka_version
  number_of_broker_nodes = var.number_of_broker_nodes

  broker_node_group_info {
    instance_type   = var.broker_instance_type
    client_subnets  = var.private_subnet_ids
    security_groups = [aws_security_group.msk_brokers.id]

    storage_info {
      ebs_storage_info { volume_size = var.volume_size_gb }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.config.arn
    revision = aws_msk_configuration.config.latest_revision
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  client_authentication {
    tls {
      certificate_authority_arns = [] # Use ACM PCA or import your own CA here
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = "/msk/${var.name}"
      }
    }
  }

  open_monitoring {
    prometheus {
      jmx_exporter  { enabled_in_broker = true }
      node_exporter { enabled_in_broker = true }
    }
  }

  tags = var.tags
}

############################################
# Outputs
############################################
output "bootstrap_brokers_tls" {
  value       = aws_msk_cluster.this.bootstrap_brokers_tls
  description = "TLS bootstrap string"
}

output "msk_security_group_id" {
  value       = aws_security_group.msk_brokers.id
  description = "Security Group applied to MSK brokers"
}