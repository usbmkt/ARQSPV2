@echo off
echo "Starting backend server..."
start "Backend" cmd /k "python main.py"

echo "Starting frontend server..."
start "Frontend" cmd /k "npm run dev"
