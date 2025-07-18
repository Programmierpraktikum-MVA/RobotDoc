# syntax=docker/dockerfile:1.4
FROM robotdoc-backend-base as base

# Set working directory
WORKDIR /app

# Copy the application code


# Expose Flask port
EXPOSE 5000

# Run Flask
CMD ["flask", "--app", "app.py", "run", "-h", "0.0.0.0"]
