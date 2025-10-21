# Use Python base image
FROM python:3.10

# Set the working directory in container
WORKDIR /app

# Copy all project files to /app
COPY . /app

RUN apt update -y && apt install awscli -y

# Install dependencies
RUN pip install -r requirements.txt


# Run the application
CMD ["python", "app.py"]
