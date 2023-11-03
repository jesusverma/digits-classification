
# Build the Docker image
docker build -t digits_classification:v1 -f docker/Dockerfile .

# Run the Docker container with the volume
docker run digits_classification:v1