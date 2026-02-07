# Step 1: Base image (Python)
FROM python:3.10-slim

# Step 2: Working directory inside container
WORKDIR /app

# Step 3: Copy requirements
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all project files
COPY . .

# Step 6: Expose Flask port
EXPOSE 8080

# Step 7: Start the app
CMD ["python", "app.py"]