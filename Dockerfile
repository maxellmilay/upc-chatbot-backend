# Use an official Python runtime as a base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=main.settings
ENV DEBUG=False
ENV IN_DOCKER=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Create static files directory
RUN mkdir -p staticfiles

# Create static files directory and set permissions
RUN mkdir -p /app/staticfiles && chmod -R 755 /app/staticfiles

# Collect static files
RUN python manage.py collectstatic --noinput

# Set the Python path to include the directory
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8080

# Start Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main.wsgi:application", "--timeout", "120"]
