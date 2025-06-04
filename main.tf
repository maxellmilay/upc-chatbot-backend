provider "aws" {
  region = "us-east-1"
  profile = "gala"
}

resource "aws_vpc" "default" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support = true
  tags = {
    Name = "upc-chatbot-backend-ec2-vpc"
  }
}

resource "aws_internet_gateway" "default" {
  vpc_id = aws_vpc.default.id
  tags = {
    Name = "upc-chatbot-backend-ec2-internet-gateway"
  }
}

resource "aws_route_table" "default" {
  vpc_id = aws_vpc.default.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.default.id
  }
  tags = {
    Name = "upc-chatbot-backend-ec2-route-table"
  }
}

resource "aws_subnet" "subnet1" {
  vpc_id = aws_vpc.default.id
  cidr_block = "10.0.1.0/24"
  map_public_ip_on_launch = false
  availability_zone = "us-east-1a"
  tags = {
    Name = "upc-chatbot-backend-ec2-subnet-1"
  }
}

resource "aws_subnet" "subnet2" {
  vpc_id = aws_vpc.default.id
  cidr_block = "10.0.2.0/24"
  map_public_ip_on_launch = false
  availability_zone = "us-east-1b"
  tags = {
    Name = "upc-chatbot-backend-ec2-subnet-2"
  }
}

resource "aws_route_table_association" "a" {
  subnet_id = aws_subnet.subnet1.id
  route_table_id = aws_route_table.default.id
}

resource "aws_route_table_association" "b" {
  subnet_id = aws_subnet.subnet2.id
  route_table_id = aws_route_table.default.id
}

resource "aws_security_group" "ec2_sg" {
  vpc_id = aws_vpc.default.id

  ingress {
    from_port = 22
    to_port = 22
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port = 80
    to_port = 80
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "upc-chatbot-backend-ec2-security-group"
  }
}

variable "django_secret_key" {
  description = "Secret key for Django application"
  type        = string
  sensitive   = true
}

resource "aws_key_pair" "deployer" {
  key_name   = "upc-chatbot-key" # Choose a name
  public_key = file("${path.module}/upc-chatbot-key.pub") # Path to the public key file
}
 
resource "aws_instance" "web" {
  ami                    = "ami-0c101f26f147fa7fd" # Amazon Linux 2023 AMI
  instance_type          = "t3.micro"
  subnet_id              = aws_subnet.subnet1.id 
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]

  key_name = "upc-chatbot-key"

  associate_public_ip_address = true
  user_data_replace_on_change = true

  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  user_data = file("${path.module}/cloud-init.sh")

  tags = {
    Name = "upc-chatbot-backend-ec2"
  }
}

resource "aws_iam_role" "ec2_role" {
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Principal = {
        Service = "ec2.amazonaws.com",
      },
      Effect = "Allow",
    }],
  })
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "upc-chatbot-backend-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.web.public_ip
}
