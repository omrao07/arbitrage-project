############################################
# EKS (private, IRSA, addons, spot pool)
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
variable "region"           { type = string - default - "us-east-1" }
variable "name"             { type = string - default - "hyper-os" }
variable "kubernetes_version" { type = string - default - "1.29" }

# From your VPC module/stack
variable "vpc_id"           { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "public_subnet_ids"  { type = list(string) }

# Toggle public API if you need it (default private only)
variable "cluster_endpoint_public_access"  { type = bool - default - false }
variable "cluster_endpoint_private_access" { type = bool - default - true  }

variable "tags" {
  type = map(string)
  default = { Project = "hyper-os", Stack = "eks" }
}

############################################
# EKS Cluster
############################################
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.8"   # v20+ (AWS provider v5+ compatible)

  cluster_name    = var.name
  cluster_version = var.kubernetes_version

  vpc_id     = var.vpc_id
  subnet_ids = var.private_subnet_ids

  enable_irsa = true

  cluster_endpoint_public_access  = var.cluster_endpoint_public_access
  cluster_endpoint_private_access = var.cluster_endpoint_private_access

  # Control-plane logs
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # Encryption at rest for Secrets (KMS created by module)
  create_kms_key = true

  # Addons (leave version = null to let AWS pick latest compatible)
  cluster_addons = {
    vpc-cni = {
      most_recent = true
      preserve    = true
      configuration_values = jsonencode({
        env = { ENABLE_PREFIX_DELEGATION = "true" } # more pods per node
      })
    }
    kube-proxy = { most_recent = true, preserve = true }
    coredns    = { most_recent = true, preserve = true }
    aws-ebs-csi-driver = {
      most_recent = true
      preserve    = true
    }
  }

  # Security group rules (allow node->cluster and egress)
  cluster_security_group_additional_rules = {
    egress_all = {
      description      = "Cluster egress"
      type             = "egress"
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = ["0.0.0.0/0"]
    }
  }

  # Managed node groups
  eks_managed_node_groups = {
    system = {
      name                = "${var.name}-system"
      subnet_ids          = var.private_subnet_ids
      instance_types      = ["m6i.large"]
      capacity_type       = "ON_DEMAND"
      desired_size        = 2
      min_size            = 1
      max_size            = 3
      ami_type            = "AL2_x86_64"
      disk_size           = 80
      labels              = { pool = "system" }
      taints              = []  # add system taints if you want to pin system workloads
      update_config       = { max_unavailable = 1 }
    }

    spot = {
      name           = "${var.name}-spot"
      subnet_ids     = var.private_subnet_ids
      instance_types = ["c6i.large","c6a.large","m6i.large"]  # diversified
      capacity_type  = "SPOT"
      desired_size   = 2
      min_size       = 0
      max_size       = 10
      ami_type       = "AL2_x86_64"
      disk_size      = 100
      labels         = { pool = "workload", capacity = "spot" }
      update_config  = { max_unavailable = 1 }
      # Prefer capacity-optimized spot allocation
      launch_template_tags = { "karpenter.sh/discovery" = var.name }
    }
  }

  # Allow the caller (your IAM principal) to be cluster admin
  enable_cluster_creator_admin_permissions = true

  tags = var.tags
}

############################################
# (Optional) Public subnets for load balancers
############################################
# If you’ll use public NLB/ALB, EKS just needs the subnets tagged correctly.
# Tag public subnets outside if not already:
# resource "aws_ec2_tag" "alb_public" {
#   for_each = toset(var.public_subnet_ids)
#   resource_id = each.value
#   key   = "kubernetes.io/role/elb"
#   value = "1"
# }

############################################
# Outputs
############################################
output "cluster_name"      { value = module.eks.cluster_name }
output "cluster_endpoint"  { value = module.eks.cluster_endpoint }
output "cluster_version"   { value = module.eks.cluster_version }
output "oidc_provider_arn" { value = module.eks.oidc_provider_arn }
output "cluster_security_group_id" { value = module.eks.cluster_security_group_id }
output "managed_node_group_arns"   { value = module.eks.eks_managed_node_groups_arns }