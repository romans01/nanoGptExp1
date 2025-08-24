#!/bin/bash
# Скрипт для быстрого анализа логов обучения nanoGPT

# Активируем виртуальную среду
source venv/bin/activate

# Проверяем аргументы
if [ $# -eq 0 ]; then
    echo "🔍 Анализирую текущий лог training.log..."
    python plot_training_advanced.py training.log
elif [ $# -eq 1 ]; then
    echo "🔍 Анализирую лог: $1"
    python plot_training_advanced.py "$1"
else
    echo "❌ Использование: $0 [путь_к_логу]"
    echo "   Без аргументов анализирует training.log"
    exit 1
fi

echo ""
echo "📁 Результаты сохранены в папке training_plots/"
echo "🖼️  Графики: training_metrics_*.png"
echo "📄 Отчет: training_summary_*.txt"
echo "💾 Данные: training_data_*.json"
