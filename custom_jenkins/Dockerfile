# Use Jenkins LTS as the base image
FROM jenkins/jenkins:lts

# Switch to root to install Docker
USER root

# Install Docker
RUN apt-get update -y && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

# Add Jenkins user to the Docker group
RUN groupadd -f docker && usermod -aG docker jenkins

# Ensure Docker directory exists
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

# Switch back to Jenkins user
USER jenkins
