#!/bin/bash
# Smart City Planning — Start Script
# Runs the Flask API backend and the React+Vite frontend simultaneously.

# Kill any existing processes on ports 5000 and 5173 to avoid conflicts
lsof -ti:5001 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

echo "=========================================="
echo "🏙️  Starting EarthVQA Smart City Platform"
echo "=========================================="

echo "[1/2] Starting Flask Backend UI API on port 5001..."
.venv/bin/python backend/app.py --port 5001 > backend.log 2>&1 &
BACKEND_PID=$!

echo "[2/2] Starting React + Vite Frontend on port 5173..."
cd frontend
npm run dev -- --port 5173 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both services are running in the background!"
echo "   - Backend Logs: tail -f backend.log"
echo "   - Frontend Logs: tail -f frontend.log"
echo ""
echo "🌐 Access your beautifully designed Dashboard here:"
echo "   http://localhost:5173"
echo ""
echo "Type 'kill $BACKEND_PID $FRONTEND_PID' to stop the servers."
