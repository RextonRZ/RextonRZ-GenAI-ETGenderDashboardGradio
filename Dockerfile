# Start from a basic Python environment
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install all the Python libraries listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files (notebook, data) into the container
COPY . .

# Expose the port that Voila will run on
EXPOSE 7860

# This is the command that will run when the container starts.
# It launches Voila and tells it which notebook to serve.
CMD ["voila", "dashboard.ipynb", "--port=7860", "--no-browser", "--Voila.server_url=/", "--Voila.base_url=/"]