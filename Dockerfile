# Use the DependencyDockerfile as the base image
FROM digits_classification_dependecy:v1

# Set the working directory to /digits
WORKDIR /digits

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set environment variable for Flask app
ENV FLASK_APP=api/api.py

# Define the default command to run when the container starts
CMD ["flask", "run", "--host=0.0.0.0"]

# Run pytest for unit testing
CMD ["pytest"]
