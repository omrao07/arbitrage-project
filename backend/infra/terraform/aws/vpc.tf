terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

########################
# Variables
########################
variable "region"   { type = string - default - "us-east-1" }
variable "name"     { type = string - default - "hyper-os" }
variable "vpc_cidr" { type = string - default - "10.50.0.0/16" }

variable "public_subnet_cidr"  { type = string - default - "10.50.1.0/24" }
variable "private_subnet_cidr" { type = string - default - "10.50.101.0/24" }

variable "az" { type = string - default - "us-east-1a" }

########################
# VPC + IGW
########################
resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.name}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = "${var.name}-igw" }
}

########################
# Subnets
########################
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = var.az
  map_public_ip_on_launch = true
  tags = { Name = "${var.name}-public" }
}

resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.this.id
  cidr_block        = var.private_subnet_cidr
  availability_zone = var.az
  tags = { Name = "${var.name}-private" }
}

########################
# NAT Gateway
########################
resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.name}-nat-eip" }
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id
  tags          = { Name = "${var.name}-nat" }

  depends_on = [aws_internet_gateway.igw]
}

########################
# Route Tables
########################
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.name}-rt-public" }
}

resource "aws_route_table_association" "public_assoc" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
  tags = { Name = "${var.name}-rt-private" }
}

resource "aws_route_table_association" "private_assoc" {
  subnet_id      = aws_subnet.private.id
  route_table_id = aws_route_table.private.id
}

########################
# Outputs
########################
output "vpc_id"          { value = aws_vpc.this.id }
output "public_subnet"   { value = aws_subnet.public.id }
output "private_subnet"  { value = aws_subnet.private.id }
output "nat_gateway_id"  { value = aws_nat_gateway.nat.id }
output "igw_id"          { value = aws_internet_gateway.igw.id }