#!/usr/bin/env python3
"""
Улучшенная система построения графиков обучения с:
- Четкими подписями моделей
- Сравнением нескольких моделей на одном графике
- Понятными подписями осей (без научной нотации)
- Лучшим форматированием
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import datetime
import json
import os

def get_timestamp():
    """Возвращает текущую временную метку"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_plots_directory():
    """Создает директорию для графиков если её нет"""
    plots_dir = Path("training_plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def parse_training_log(log_file_or_text):
    """Парсит лог обучения и извлекает метрики"""
    
    if isinstance(log_file_or_text, (str, Path)) and Path(log_file_or_text).exists():
        with open(log_file_or_text, 'r') as f:
            text = f.read()
    else:
        text = str(log_file_or_text)
    
    data = {
        'training': {'iters': [], 'loss': [], 'time': [], 'mfu': []},
        'validation': {'steps': [], 'train_loss': [], 'val_loss': []},
        'model_info': {},
        'config_info': {}
    }
    
    # Извлекаем информацию о модели
    model_patterns = {
        'parameters': r'number of parameters: ([\d.]+)M',
        'n_layer': r'n_layer = (\d+)',
        'n_head': r'n_head = (\d+)', 
        'n_embd': r'n_embd = (\d+)',
        'batch_size': r'batch_size = (\d+)',
        'learning_rate': r'learning_rate = ([\d.e-]+)',
        'max_iters': r'max_iters = (\d+)',
        'dataset': r'dataset = (\w+)',
        'block_size': r'block_size = (\d+)',
        'dropout': r'dropout = ([\d.]+)'
    }
    
    for key, pattern in model_patterns.items():
        match = re.search(pattern, text)
        if match:
            try:
                if key in ['n_layer', 'n_head', 'n_embd', 'batch_size', 'max_iters', 'block_size']:
                    data['model_info'][key] = int(match.group(1))
                elif key in ['learning_rate', 'dropout']:
                    data['model_info'][key] = float(match.group(1))
                else:
                    data['model_info'][key] = match.group(1)
            except ValueError:
                data['model_info'][key] = match.group(1)
    
    # Извлекаем информацию о центрировании
    centering_patterns = {
        'use_centered_attention': r'use_centered_attention = (True|False)',
        'center_qk': r'center_qk = (True|False)',
        'center_final_output': r'center_final_output = (True|False)',
        'center_block_output': r'center_block_output = (True|False)',
        'centering_mode': r'centering_mode = [\'"](\w+)[\'"]'
    }
    
    for key, pattern in centering_patterns.items():
        match = re.search(pattern, text)
        if match:
            if key in ['use_centered_attention', 'center_qk', 'center_final_output', 'center_block_output']:
                data['config_info'][key] = match.group(1) == 'True'
            else:
                data['config_info'][key] = match.group(1)
    
    # Парсим метрики обучения
    iter_pattern = r'iter (\d+): loss ([\d.]+), time ([\d.]+)ms, mfu ([-\d.]+)%'
    for match in re.finditer(iter_pattern, text):
        data['training']['iters'].append(int(match.group(1)))
        data['training']['loss'].append(float(match.group(2)))
        data['training']['time'].append(float(match.group(3)))
        data['training']['mfu'].append(float(match.group(4)))
    
    # Парсим метрики валидации
    step_pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
    for match in re.finditer(step_pattern, text):
        data['validation']['steps'].append(int(match.group(1)))
        data['validation']['train_loss'].append(float(match.group(2)))
        data['validation']['val_loss'].append(float(match.group(3)))
    
    return data

def determine_model_type(data):
    """Определяет тип модели на основе конфигурации"""
    config = data.get('config_info', {})
    
    if config.get('use_centered_attention') and config.get('center_qk'):
        if config.get('center_final_output'):
            return "Full Centered (QK + Final)"
        else:
            return "QK Centered"
    elif config.get('center_final_output'):
        return "Final Centered"
    elif config.get('center_block_output'):
        return "Block Centered"
    else:
        return "Baseline"

def format_model_info(data):
    """Форматирует информацию о модели для отображения"""
    model_info = data.get('model_info', {})
    config_info = data.get('config_info', {})
    
    lines = []
    
    # Тип модели
    model_type = determine_model_type(data)
    lines.append(f"Model: {model_type}")
    
    # Архитектура
    arch_params = []
    for param in ['n_layer', 'n_head', 'n_embd']:
        if param in model_info:
            arch_params.append(f"{param.split('_')[1]}={model_info[param]}")
    if arch_params:
        lines.append(f"Architecture: {', '.join(arch_params)}")
    
    # Параметры
    if 'parameters' in model_info:
        lines.append(f"Parameters: {model_info['parameters']}")
    
    # Обучение
    train_params = []
    for param in ['batch_size', 'learning_rate', 'max_iters', 'dataset']:
        if param in model_info:
            if param == 'learning_rate':
                train_params.append(f"lr={model_info[param]}")
            else:
                train_params.append(f"{param}={model_info[param]}")
    if train_params:
        lines.append(f"Training: {', '.join(train_params)}")
    
    return '\n'.join(lines)

def format_axis_labels(ax, axis='y'):
    """Форматирует подписи осей без научной нотации"""
    if axis == 'y':
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3f}' if x < 1 else f'{x:.1f}'))
    elif axis == 'x':
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))

def plot_single_model(data, save_path=None, title_suffix=""):
    """Строит графики для одной модели"""
    
    if save_path is None:
        plots_dir = create_plots_directory()
        timestamp = get_timestamp()
        model_type = determine_model_type(data).replace(' ', '_').lower()
        save_path = plots_dir / f"training_metrics_{model_type}_{timestamp}.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Заголовок с типом модели
    model_type = determine_model_type(data)
    main_title = f'nanoGPT Training: {model_type}{title_suffix}'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # График 1: Training и Validation Loss
    if data['validation']['steps']:
        axes[0, 0].plot(data['validation']['steps'], data['validation']['train_loss'], 
                       label='Train Loss', color='blue', marker='o', markersize=4, linewidth=2)
        axes[0, 0].plot(data['validation']['steps'], data['validation']['val_loss'], 
                       label='Validation Loss', color='red', marker='s', markersize=4, linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Train vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        format_axis_labels(axes[0, 0], 'y')
        format_axis_labels(axes[0, 0], 'x')
    
    # График 2: Training Loss (детальный)
    if data['training']['iters']:
        axes[0, 1].plot(data['training']['iters'], data['training']['loss'], 
                       color='green', alpha=0.7, linewidth=1, label='Training Loss')
        # Добавляем скользящее среднее
        if len(data['training']['loss']) > 10:
            window = min(50, len(data['training']['loss']) // 10)
            smoothed = np.convolve(data['training']['loss'], np.ones(window)/window, mode='valid')
            smoothed_iters = data['training']['iters'][window-1:]
            axes[0, 1].plot(smoothed_iters, smoothed, color='darkgreen', linewidth=2, 
                           label=f'Smoothed (window={window})')
        
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Detailed Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        format_axis_labels(axes[0, 1], 'y')
        format_axis_labels(axes[0, 1], 'x')
    
    # График 3: Время на итерацию
    if data['training']['time']:
        axes[1, 0].plot(data['training']['iters'], data['training']['time'], 
                       color='orange', alpha=0.7, linewidth=1)
        # Скользящее среднее для времени
        if len(data['training']['time']) > 10:
            window = min(20, len(data['training']['time']) // 5)
            smoothed_time = np.convolve(data['training']['time'], np.ones(window)/window, mode='valid')
            smoothed_iters = data['training']['iters'][window-1:]
            axes[1, 0].plot(smoothed_iters, smoothed_time, color='darkorange', linewidth=2,
                           label=f'Smoothed (window={window})')
            axes[1, 0].legend()
        
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Training Time per Iteration')
        axes[1, 0].grid(True, alpha=0.3)
        format_axis_labels(axes[1, 0], 'y')
        format_axis_labels(axes[1, 0], 'x')
    
    # График 4: MFU (Model FLOPs Utilization)
    if data['training']['mfu'] and any(mfu > 0 for mfu in data['training']['mfu']):
        valid_mfu = [(iter_num, mfu) for iter_num, mfu in zip(data['training']['iters'], data['training']['mfu']) if mfu > 0]
        if valid_mfu:
            valid_iters, valid_mfu_vals = zip(*valid_mfu)
            axes[1, 1].plot(valid_iters, valid_mfu_vals, color='purple', alpha=0.7, linewidth=1)
            # Скользящее среднее для MFU
            if len(valid_mfu_vals) > 10:
                window = min(20, len(valid_mfu_vals) // 5)
                smoothed_mfu = np.convolve(valid_mfu_vals, np.ones(window)/window, mode='valid')
                smoothed_iters = valid_iters[window-1:]
                axes[1, 1].plot(smoothed_iters, smoothed_mfu, color='darkviolet', linewidth=2,
                               label=f'Smoothed (window={window})')
                axes[1, 1].legend()
        
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('MFU (%)')
        axes[1, 1].set_title('Model FLOPs Utilization')
        axes[1, 1].grid(True, alpha=0.3)
        format_axis_labels(axes[1, 1], 'y')
        format_axis_labels(axes[1, 1], 'x')
    
    # Добавляем информацию о модели
    model_info_text = format_model_info(data)
    fig.text(0.02, 0.02, model_info_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Место для информации о модели
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_comparison(data_list, model_names, save_path=None, title="Model Comparison"):
    """Сравнивает несколько моделей на одном графике"""
    
    if save_path is None:
        plots_dir = create_plots_directory()
        timestamp = get_timestamp()
        save_path = plots_dir / f"comparison_{timestamp}.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Заголовок
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Цвета для разных моделей
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # График 1: Validation Loss сравнение
    for i, (data, name) in enumerate(zip(data_list, model_names)):
        if data['validation']['steps']:
            color = colors[i % len(colors)]
            axes[0, 0].plot(data['validation']['steps'], data['validation']['val_loss'], 
                           label=name, color=color, marker='o', markersize=3, linewidth=2)
    
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Validation Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    format_axis_labels(axes[0, 0], 'y')
    format_axis_labels(axes[0, 0], 'x')
    
    # График 2: Training Loss сравнение (сглаженный)
    for i, (data, name) in enumerate(zip(data_list, model_names)):
        if data['training']['iters'] and len(data['training']['loss']) > 10:
            color = colors[i % len(colors)]
            window = min(50, len(data['training']['loss']) // 10)
            smoothed = np.convolve(data['training']['loss'], np.ones(window)/window, mode='valid')
            smoothed_iters = data['training']['iters'][window-1:]
            axes[0, 1].plot(smoothed_iters, smoothed, color=color, linewidth=2, label=name)
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Training Loss (Smoothed)')
    axes[0, 1].set_title('Training Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    format_axis_labels(axes[0, 1], 'y')
    format_axis_labels(axes[0, 1], 'x')
    
    # График 3: Время обучения сравнение
    for i, (data, name) in enumerate(zip(data_list, model_names)):
        if data['training']['time'] and len(data['training']['time']) > 10:
            color = colors[i % len(colors)]
            window = min(20, len(data['training']['time']) // 5)
            smoothed_time = np.convolve(data['training']['time'], np.ones(window)/window, mode='valid')
            smoothed_iters = data['training']['iters'][window-1:]
            axes[1, 0].plot(smoothed_iters, smoothed_time, color=color, linewidth=2, label=name)
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Time per Iteration (ms)')
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    format_axis_labels(axes[1, 0], 'y')
    format_axis_labels(axes[1, 0], 'x')
    
    # График 4: Финальные результаты (барный график)
    final_losses = []
    final_times = []
    
    for data in data_list:
        if data['validation']['val_loss']:
            final_losses.append(data['validation']['val_loss'][-1])
        else:
            final_losses.append(None)
        
        if data['training']['time']:
            final_times.append(np.mean(data['training']['time'][-50:]))  # Среднее за последние 50 итераций
        else:
            final_times.append(None)
    
    # Барный график финальных validation loss
    valid_losses = [(name, loss) for name, loss in zip(model_names, final_losses) if loss is not None]
    if valid_losses:
        names, losses = zip(*valid_losses)
        bars = axes[1, 1].bar(range(len(names)), losses, color=colors[:len(names)])
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Final Validation Loss')
        axes[1, 1].set_title('Final Results Comparison')
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        format_axis_labels(axes[1, 1], 'y')
        
        # Добавляем значения на столбцы
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def compare_best_models():
    """Сравнивает лучшие модели из экспериментов"""
    
    # Ищем логи лучших моделей
    model_dirs = [
        ('out-centering-bpe-baseline_bpe', 'Baseline'),
        ('out-centering-bpe-qk_centered_bpe', 'QK Centered')
    ]
    
    data_list = []
    model_names = []
    
    for model_dir, model_name in model_dirs:
        # Ищем соответствующий лог в training_plots
        log_files = list(Path('training_plots').glob('training_data_*.json'))
        
        # Пытаемся найти лог для этой модели
        model_data = None
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    
                # Проверяем, соответствует ли этот лог нашей модели
                if model_name == 'Baseline':
                    if not data.get('config_info', {}).get('use_centered_attention', True):
                        model_data = data
                        break
                elif model_name == 'QK Centered':
                    config = data.get('config_info', {})
                    if (config.get('use_centered_attention') and 
                        config.get('center_qk') and 
                        not config.get('center_final_output')):
                        model_data = data
                        break
            except:
                continue
        
        if model_data:
            data_list.append(model_data)
            model_names.append(model_name)
    
    if len(data_list) >= 2:
        save_path = plot_comparison(data_list, model_names, 
                                   title="Best Models Comparison: Baseline vs QK Centered")
        print(f"📊 Сравнительный график сохранен: {save_path}")
        return save_path
    else:
        print("❌ Не найдено достаточно данных для сравнения")
        return None

def main():
    """Основная функция для тестирования"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            compare_best_models()
        else:
            # Парсим указанный лог файл
            log_file = sys.argv[1]
            if os.path.exists(log_file):
                data = parse_training_log(log_file)
                save_path = plot_single_model(data)
                print(f"📊 График сохранен: {save_path}")
            else:
                print(f"❌ Файл не найден: {log_file}")
    else:
        # По умолчанию сравниваем лучшие модели
        compare_best_models()

if __name__ == "__main__":
    main()
