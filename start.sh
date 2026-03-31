#!/bin/bash
# ============================================
# Iris 3.0 - Automatic Startup Script (Linux/Mac)
# ============================================

set -e

echo "🔮 Iris 3.0 - Иридологичен Анализатор"
echo "======================================"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Грешка: Python не е инсталиран!"
        echo "   Моля инсталирайте Python 3.8+ от https://www.python.org/downloads/"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo "✓ Използва се: $($PYTHON_CMD --version)"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Създаване на виртуална среда..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "🔄 Активиране на виртуална среда..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Обновяване на pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📥 Инсталиране на зависимости..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Всички зависимости са инсталирани!"
echo ""

# Check if IRIS_WORKER_URL is set
if [ -n "$IRIS_WORKER_URL" ]; then
    echo "🤖 AI анализ: ВКЛЮЧЕН"
    echo "   Worker URL: $IRIS_WORKER_URL"
else
    echo "ℹ️  AI анализ: ИЗКЛЮЧЕН (само обработка на изображения)"
    echo "   За да включите AI анализ, стартирайте с:"
    echo "   IRIS_WORKER_URL=\"https://your-worker.workers.dev\" ./start.sh"
fi

echo ""
echo "🚀 Стартиране на сървъра..."
echo "   Отворете браузъра на: http://localhost:5000"
echo ""
echo "   Натиснете Ctrl+C за спиране на сървъра"
echo "======================================"
echo ""

# Start the Flask app
$PYTHON_CMD app.py
