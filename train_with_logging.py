#!/usr/bin/env python3
"""
Обертка для train.py с улучшенным логированием
Сохраняет все выводы в файл и создает графики в реальном времени
"""

import subprocess
import sys
import threading
import time
import os
from pathlib import Path
import re

def parse_and_plot_logs(log_file):
    """Парсит логи и создает графики каждые 100 итераций"""
    try:
        from plot_training_advanced import parse_training_log, plot_training_metrics, create_plots_directory, get_timestamp
        
        if not Path(log_file).exists():
            return
            
        data = parse_training_log(log_file)
        
        if data['training']['iters']:
            latest_iter = data['training']['iters'][-1]
            
            # Создаем графики каждые 100 итераций
            if latest_iter % 100 == 0 and latest_iter > 0:
                plots_dir = create_plots_directory()
                timestamp = get_timestamp()
                plot_file = plots_dir / f'iter_{latest_iter}_{timestamp}.png'
                
                # Определяем суффикс для заголовка
                model_info = data.get('model_info', {})
                title_suffix = f" (iter {latest_iter})"
                if 'dataset' in model_info:
                    title_suffix += f" - {model_info['dataset']}"
                
                plot_training_metrics(data, plot_file, title_suffix)
                print(f"📊 Графики обновлены: {plot_file}")
                
    except Exception as e:
        print(f"⚠️  Ошибка при создании графиков: {e}")

def monitor_training(log_file):
    """Мониторинг логов в реальном времени"""
    print(f"🔄 Начинаю мониторинг логов: {log_file}")
    
    last_size = 0
    while True:
        try:
            if Path(log_file).exists():
                current_size = Path(log_file).stat().st_size
                if current_size > last_size:
                    # Файл обновился, парсим новые данные
                    parse_and_plot_logs(log_file)
                    last_size = current_size
            
            time.sleep(10)  # Проверяем каждые 10 секунд
            
        except KeyboardInterrupt:
            print("\n🛑 Мониторинг остановлен")
            break
        except Exception as e:
            print(f"⚠️  Ошибка мониторинга: {e}")
            time.sleep(5)

def run_training_with_logging(args):
    """Запускает обучение с логированием"""
    
    log_file = 'training.log'
    
    # Определяем какой train.py использовать
    train_script = 'train.py'
    
    # Проверяем конфиг на наличие параметров центрирования
    if args:
        config_file = args[0] if not args[0].startswith('--') else None
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_content = f.read()
                # Проверяем новые расширенные параметры центрирования
                if any(param in config_content for param in ['center_v', 'center_mlp', 'center_residual', 'center_embeddings']):
                    train_script = 'train_advanced_centered.py'
                    print(f"🎯 Обнаружены расширенные параметры центрирования, используем {train_script}")
                # Проверяем старые параметры центрирования
                elif any(param in config_content for param in ['use_centered_attention', 'center_qk', 'center_final_output']):
                    train_script = 'train_centered.py'
                    print(f"🎯 Обнаружены параметры центрирования, используем {train_script}")
    
    # Команда для запуска обучения
    cmd = ['python', train_script] + args
    
    print(f"🚀 Запускаю обучение: {' '.join(cmd)}")
    print(f"📝 Логи сохраняются в: {log_file}")
    print("🔄 Мониторинг графиков включен")
    print("🛑 Нажмите Ctrl+C для остановки\n")
    
    # Запускаем мониторинг в отдельном потоке
    monitor_thread = threading.Thread(target=monitor_training, args=(log_file,), daemon=True)
    monitor_thread.start()
    
    try:
        # Запускаем процесс обучения
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Читаем вывод построчно и дублируем в консоль и файл
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()
            
            process.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Обучение прервано пользователем")
        if 'process' in locals():
            process.terminate()
    
    # Создаем финальные графики и отчет
    print("\n📊 Создаю финальный анализ...")
    
    try:
        from plot_training_advanced import analyze_training_log
        analyze_training_log(log_file)
        print("✅ Финальный анализ завершен!")
    except Exception as e:
        print(f"⚠️  Ошибка при создании финального анализа: {e}")

if __name__ == "__main__":
    # Передаем все аргументы командной строки в train.py
    training_args = sys.argv[1:] if len(sys.argv) > 1 else [
        'config/train_shakespeare_char.py', 
        '--wandb_log=False', 
        '--max_iters=500'
    ]
    
    run_training_with_logging(training_args)
