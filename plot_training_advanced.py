#!/usr/bin/env python3
"""
Улучшенный скрипт для визуализации логов обучения nanoGPT
- Сохраняет графики в отдельную папку
- Добавляет временные метки к именам файлов
- Записывает параметры модели под графиками
"""

import re
import os
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def create_plots_directory():
    """Создает папку для графиков если её нет"""
    plots_dir = Path("training_plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def get_timestamp():
    """Возвращает текущую временную метку для имени файла"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_training_log(log_file_or_text):
    """Парсит логи обучения и извлекает метрики"""
    
    if isinstance(log_file_or_text, (str, Path)) and Path(log_file_or_text).exists():
        with open(log_file_or_text, 'r') as f:
            text = f.read()
    else:
        text = str(log_file_or_text)
    
    # Регулярные выражения для извлечения метрик
    step_pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
    iter_pattern = r'iter (\d+): loss ([\d.]+), time ([\d.]+)ms, mfu ([\d.-]+)%'
    
    # Извлекаем параметры модели
    params_pattern = r'number of parameters: ([\d.]+)([MK]?)'
    vocab_pattern = r'vocab_size[=\s]+(\d+)'
    config_patterns = {
        'n_layer': r'n_layer[=\s]+(\d+)',
        'n_head': r'n_head[=\s]+(\d+)', 
        'n_embd': r'n_embd[=\s]+(\d+)',
        'block_size': r'block_size[=\s]+(\d+)',
        'batch_size': r'batch_size[=\s]+(\d+)',
        'learning_rate': r'learning_rate[=\s]+([\d.e-]+)',
        'dropout': r'dropout[=\s]+([\d.]+)',
        'max_iters': r'max_iters[=\s]+(\d+)',
        'dataset': r'dataset[=\s]+[\'"]?([^\'"\s]+)[\'"]?'
    }
    
    steps, train_losses, val_losses = [], [], []
    iters, losses, times, mfus = [], [], [], []
    
    # Извлекаем данные о validation
    for match in re.finditer(step_pattern, text):
        steps.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))
        val_losses.append(float(match.group(3)))
    
    # Извлекаем данные о training iterations
    for match in re.finditer(iter_pattern, text):
        iters.append(int(match.group(1)))
        losses.append(float(match.group(2)))
        times.append(float(match.group(3)))
        mfu_val = float(match.group(4))
        # Игнорируем отрицательные MFU (начальные значения)
        mfus.append(max(0, mfu_val))
    
    # Извлекаем параметры модели
    model_info = {}
    
    # Количество параметров
    params_match = re.search(params_pattern, text)
    if params_match:
        num = float(params_match.group(1))
        unit = params_match.group(2)
        if unit == 'M':
            num *= 1e6
        elif unit == 'K':
            num *= 1e3
        model_info['parameters'] = f"{num/1e6:.2f}M"
    
    # Размер словаря
    vocab_match = re.search(vocab_pattern, text)
    if vocab_match:
        model_info['vocab_size'] = int(vocab_match.group(1))
    
    # Другие параметры
    for param, pattern in config_patterns.items():
        match = re.search(pattern, text)
        if match:
            try:
                if param in ['learning_rate', 'dropout']:
                    model_info[param] = float(match.group(1))
                else:
                    model_info[param] = match.group(1)
            except:
                model_info[param] = match.group(1)
    
    return {
        'validation': {'steps': steps, 'train_loss': train_losses, 'val_loss': val_losses},
        'training': {'iters': iters, 'loss': losses, 'time': times, 'mfu': mfus},
        'model_info': model_info
    }

def format_model_info(model_info):
    """Форматирует информацию о модели для отображения"""
    if not model_info:
        return "Model parameters: Not available"
    
    lines = []
    
    # Основные параметры
    if 'parameters' in model_info:
        lines.append(f"Parameters: {model_info['parameters']}")
    if 'vocab_size' in model_info:
        lines.append(f"Vocab size: {model_info['vocab_size']:,}")
    if 'dataset' in model_info:
        lines.append(f"Dataset: {model_info['dataset']}")
    
    # Архитектура
    arch_params = []
    for param in ['n_layer', 'n_head', 'n_embd', 'block_size']:
        if param in model_info:
            arch_params.append(f"{param}={model_info[param]}")
    if arch_params:
        lines.append(f"Architecture: {', '.join(arch_params)}")
    
    # Обучение
    train_params = []
    for param in ['batch_size', 'learning_rate', 'dropout', 'max_iters']:
        if param in model_info:
            if param == 'learning_rate':
                train_params.append(f"lr={model_info[param]}")
            else:
                train_params.append(f"{param}={model_info[param]}")
    if train_params:
        lines.append(f"Training: {', '.join(train_params)}")
    
    return '\n'.join(lines)

def plot_training_metrics(data, save_path=None, title_suffix=""):
    """Строит графики метрик обучения с параметрами модели"""
    
    if save_path is None:
        plots_dir = create_plots_directory()
        timestamp = get_timestamp()
        save_path = plots_dir / f"training_metrics_{timestamp}.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Заголовок с временной меткой
    main_title = f'nanoGPT Training Metrics{title_suffix}'
    if title_suffix:
        main_title += f' - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # График 1: Training и Validation Loss
    if data['validation']['steps']:
        axes[0, 0].plot(data['validation']['steps'], data['validation']['train_loss'], 
                       label='Train Loss', color='blue', marker='o', markersize=4, linewidth=2)
        axes[0, 0].plot(data['validation']['steps'], data['validation']['val_loss'], 
                       label='Validation Loss', color='red', marker='s', markersize=4, linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Train vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')  # Логарифмическая шкала для лучшей видимости
    
    # График 2: Training Loss (детальный)
    if data['training']['iters']:
        axes[0, 1].plot(data['training']['iters'], data['training']['loss'], 
                       color='green', alpha=0.7, linewidth=1)
        # Добавляем скользящее среднее
        if len(data['training']['loss']) > 10:
            window = min(50, len(data['training']['loss']) // 10)
            smoothed = np.convolve(data['training']['loss'], np.ones(window)/window, mode='valid')
            smoothed_iters = data['training']['iters'][window-1:]
            axes[0, 1].plot(smoothed_iters, smoothed, color='darkgreen', linewidth=2, 
                           label=f'Smoothed (window={window})')
            axes[0, 1].legend()
        
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss (Detailed)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # График 3: Training Time per Iteration
    if data['training']['time']:
        axes[1, 0].plot(data['training']['iters'], data['training']['time'], 
                       color='orange', alpha=0.6, linewidth=1)
        # Скользящее среднее для времени
        if len(data['training']['time']) > 10:
            window = min(50, len(data['training']['time']) // 10)
            smoothed_time = np.convolve(data['training']['time'], np.ones(window)/window, mode='valid')
            smoothed_iters = data['training']['iters'][window-1:]
            axes[1, 0].plot(smoothed_iters, smoothed_time, color='darkorange', linewidth=2,
                           label=f'Smoothed (window={window})')
            axes[1, 0].legend()
        
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Training Time per Iteration')
        axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Model FLOPs Utilization (MFU)
    if data['training']['mfu']:
        # Фильтруем положительные значения MFU
        valid_mfu = [(i, m) for i, m in zip(data['training']['iters'], data['training']['mfu']) if m > 0]
        if valid_mfu:
            valid_iters, valid_mfu_vals = zip(*valid_mfu)
            axes[1, 1].plot(valid_iters, valid_mfu_vals, 
                           color='purple', alpha=0.7, linewidth=1)
            
            # Скользящее среднее для MFU
            if len(valid_mfu_vals) > 10:
                window = min(50, len(valid_mfu_vals) // 10)
                smoothed_mfu = np.convolve(valid_mfu_vals, np.ones(window)/window, mode='valid')
                smoothed_iters = valid_iters[window-1:]
                axes[1, 1].plot(smoothed_iters, smoothed_mfu, color='darkviolet', linewidth=2,
                               label=f'Smoothed (window={window})')
                axes[1, 1].legend()
        
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('MFU (%)')
        axes[1, 1].set_title('Model FLOPs Utilization')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Добавляем информацию о модели внизу
    model_info_text = format_model_info(data.get('model_info', {}))
    fig.text(0.02, 0.02, model_info_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Оставляем место для информации о модели
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Графики сохранены в: {save_path}")
    plt.close()
    
    return save_path

def create_summary_report(data, save_path=None):
    """Создает текстовый отчет с метриками"""
    
    if save_path is None:
        plots_dir = create_plots_directory()
        timestamp = get_timestamp()
        save_path = plots_dir / f"training_summary_{timestamp}.txt"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЕТ ПО ОБУЧЕНИЮ МОДЕЛИ nanoGPT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Информация о модели
        f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
        f.write("-" * 20 + "\n")
        model_info = data.get('model_info', {})
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Статистика обучения
        if data['training']['iters']:
            f.write("СТАТИСТИКА ОБУЧЕНИЯ:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Общее количество итераций: {max(data['training']['iters'])}\n")
            f.write(f"Начальный training loss: {data['training']['loss'][0]:.4f}\n")
            f.write(f"Финальный training loss: {data['training']['loss'][-1]:.4f}\n")
            f.write(f"Улучшение: {data['training']['loss'][0] / data['training']['loss'][-1]:.1f}x\n")
            
            if data['training']['time']:
                avg_time = np.mean(data['training']['time'])
                f.write(f"Среднее время на итерацию: {avg_time:.1f} мс\n")
            
            if data['training']['mfu']:
                valid_mfu = [m for m in data['training']['mfu'] if m > 0]
                if valid_mfu:
                    avg_mfu = np.mean(valid_mfu)
                    f.write(f"Средняя утилизация GPU (MFU): {avg_mfu:.1f}%\n")
            f.write("\n")
        
        # Статистика валидации
        if data['validation']['steps']:
            f.write("СТАТИСТИКА ВАЛИДАЦИИ:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Начальный train loss: {data['validation']['train_loss'][0]:.4f}\n")
            f.write(f"Финальный train loss: {data['validation']['train_loss'][-1]:.4f}\n")
            f.write(f"Начальный val loss: {data['validation']['val_loss'][0]:.4f}\n")
            f.write(f"Финальный val loss: {data['validation']['val_loss'][-1]:.4f}\n")
            
            # Проверка на переобучение
            final_train = data['validation']['train_loss'][-1]
            final_val = data['validation']['val_loss'][-1]
            overfitting = final_val / final_train
            f.write(f"Соотношение val/train loss: {overfitting:.2f}")
            if overfitting > 2.0:
                f.write(" (возможно переобучение)")
            f.write("\n\n")
    
    print(f"📄 Отчет сохранен в: {save_path}")
    return save_path

def analyze_training_log(log_file):
    """Полный анализ лога обучения с созданием графиков и отчета"""
    
    print(f"🔍 Анализирую лог: {log_file}")
    data = parse_training_log(log_file)
    
    if not data['training']['iters'] and not data['validation']['steps']:
        print("❌ Не найдено данных для анализа")
        return
    
    plots_dir = create_plots_directory()
    timestamp = get_timestamp()
    
    # Определяем суффикс для заголовка
    model_info = data.get('model_info', {})
    title_suffix = ""
    if 'dataset' in model_info:
        title_suffix = f" ({model_info['dataset']})"
    
    # Создаем графики
    plot_path = plot_training_metrics(data, 
                                    plots_dir / f"training_metrics_{timestamp}.png",
                                    title_suffix)
    
    # Создаем отчет
    report_path = create_summary_report(data, 
                                      plots_dir / f"training_summary_{timestamp}.txt")
    
    # Сохраняем данные в JSON для дальнейшего анализа
    json_path = plots_dir / f"training_data_{timestamp}.json"
    with open(json_path, 'w') as f:
        # Конвертируем numpy arrays в списки для JSON
        json_data = {
            'validation': {
                'steps': data['validation']['steps'],
                'train_loss': data['validation']['train_loss'],
                'val_loss': data['validation']['val_loss']
            },
            'training': {
                'iters': data['training']['iters'],
                'loss': data['training']['loss'],
                'time': data['training']['time'],
                'mfu': data['training']['mfu']
            },
            'model_info': data['model_info']
        }
        json.dump(json_data, f, indent=2)
    
    print(f"💾 Данные сохранены в: {json_path}")
    print(f"✅ Анализ завершен! Файлы в папке: {plots_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        if Path(log_file).exists():
            analyze_training_log(log_file)
        else:
            print(f"❌ Файл {log_file} не найден")
    else:
        # Анализируем текущий лог если он есть
        if Path("training.log").exists():
            analyze_training_log("training.log")
        else:
            print("❌ Не найден файл training.log")
            print("Использование: python plot_training_advanced.py [путь_к_логу]")
