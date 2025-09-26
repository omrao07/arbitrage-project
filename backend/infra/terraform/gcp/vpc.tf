#############################################
# VPC (AWS) – production-ready configuration
#############################################

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

#############################################
# Input Variables
#############################################

variable "region" {
  description = "AWS region for the VPC"
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Project/environment name"
  type        = string
  default     = "hyper-os"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.42.0.0/16"
}

variable "azs" {
  description = "Availability Zones to use"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.42.1.0/24", "10.42.2.0/24"]
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.42.101.0/24", "10.42.102.0/24"]
}

variable "nat_gateways" {
  description = "Number of NAT gateways"
  type        = number
  default     = 1
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {
    Project = "hyper-os"
    Stack   = "network"
  }
}

#############################################
# VPC Module (terraform-aws-modules/vpc/aws)
#############################################

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = var.name
  cidr = var.vpc_cidr

  azs             = var.azs
  public_subnets  = var.public_subnets
  private_subnets = var.private_subnets

  enable_nat_gateway     = true
  single_nat_gateway     = var.nat_gateways == 1 ? true : false
  enable_dns_support     = true
  enable_dns_hostnames   = true
  enable_flow_log        = var.enable_flow_logs
  create_flow_log_cloudwatch_log_group = var.enable_flow_logs
  create_flow_log_cloudwatch_iam_role  = var.enable_flow_logs

  tags = merge(
    var.tags,
    { Name = "${var.name}-vpc" }
  )
}

#############################################
# Outputs
#############################################

output "vpc_id" {
  value = module.vpc.vpc_id
}

output "public_subnets" {
  value = module.vpc.public_subnets
}

output "private_subnets" {
  value = module.vpc.private_subnets
}