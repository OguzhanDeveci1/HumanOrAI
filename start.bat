@echo off
echo ========================================
echo Human or AI Text Classifier
echo ========================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Starting web application...
echo Server will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py
