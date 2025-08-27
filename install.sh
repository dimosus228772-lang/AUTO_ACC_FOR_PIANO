#!/bin/bash
# Автоустановщик и запускатель для авто-аккомпанимента

echo "[INFO] Установка виртуального окружения..."
python3 -m venv venv

echo "[INFO] Активация окружения..."
source venv/bin/activate

echo "[INFO] Установка зависимостей..."
pip install --upgrade pip
pip install mido python-rtmidi music21 numpy

echo "[INFO] Выдаём права на запуск .py файлов..."
chmod +x run_safe.py auto_accomp_pro.py

echo "[INFO] Установка завершена!"
echo "Запустить аккомпанемент можно командой:"
echo "    ./run_safe.py"
