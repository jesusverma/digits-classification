docker build -t digits_classification_dependecy:v1 -f DockerfileDependecy .
# Build the Docker image
docker build -t digits_classification:v1 -f Dockerfile .

# Run the Docker container with the volume
docker run digits_classification:v1