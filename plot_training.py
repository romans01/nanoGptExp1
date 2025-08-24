#!/usr/bin/env python3
"""
Простой скрипт для визуализации логов обучения nanoGPT
Использует matplotlib для построения графиков loss
"""

import re
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file_or_text):
    """Парсит логи обучения и извлекает метрики"""
    
    if isinstance(log_file_or_text, (str, Path)) and Path(log_file_or_text).exists():
        with open(log_file_or_text, 'r') as f:
            text = f.read()
    else:
        text = str(log_file_or_text)
    
    # Регулярные выражения для извлечения метрик
    step_pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
    iter_pattern = r'iter (\d+): loss ([\d.]+), time ([\d.]+)ms, mfu ([\d.]+)%'
    
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
        mfus.append(float(match.group(4)))
    
    return {
        'validation': {'steps': steps, 'train_loss': train_losses, 'val_loss': val_losses},
        'training': {'iters': iters, 'loss': losses, 'time': times, 'mfu': mfus}
    }

def plot_training_metrics(data, save_path='training_plots.png'):
    """Строит графики метрик обучения"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('nanoGPT Training Metrics', fontsize=16)
    
    # График 1: Training и Validation Loss
    if data['validation']['steps']:
        axes[0, 0].plot(data['validation']['steps'], data['validation']['train_loss'], 
                       label='Train Loss', color='blue', marker='o')
        axes[0, 0].plot(data['validation']['steps'], data['validation']['val_loss'], 
                       label='Validation Loss', color='red', marker='s')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Train vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Training Loss (детальный)
    if data['training']['iters']:
        axes[0, 1].plot(data['training']['iters'], data['training']['loss'], 
                       color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss (Detailed)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: Training Time per Iteration
    if data['training']['time']:
        axes[1, 0].plot(data['training']['iters'], data['training']['time'], 
                       color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Training Time per Iteration')
        axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Model FLOPs Utilization (MFU)
    if data['training']['mfu']:
        axes[1, 1].plot(data['training']['iters'], data['training']['mfu'], 
                       color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('MFU (%)')
        axes[1, 1].set_title('Model FLOPs Utilization')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Графики сохранены в: {save_path}")
    plt.close()  # Закрываем фигуру вместо показа

def monitor_training_live(log_file='training.log', update_interval=10):
    """Мониторинг обучения в реальном времени"""
    import time
    
    print("🔄 Начинаю мониторинг обучения...")
    print("📝 Логи сохраняются в:", log_file)
    print("⏱️  Обновление каждые", update_interval, "секунд")
    print("🛑 Нажмите Ctrl+C для остановки")
    
    try:
        while True:
            if Path(log_file).exists():
                data = parse_training_log(log_file)
                if data['training']['iters']:
                    latest_iter = data['training']['iters'][-1]
                    latest_loss = data['training']['loss'][-1]
                    latest_mfu = data['training']['mfu'][-1] if data['training']['mfu'] else 0
                    
                    print(f"📈 Iter {latest_iter}: Loss {latest_loss:.4f}, MFU {latest_mfu:.1f}%")
                    
                    # Обновляем графики каждые несколько итераций
                    if latest_iter % 50 == 0:
                        plot_training_metrics(data, f'training_plots_iter_{latest_iter}.png')
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Мониторинг остановлен")
        if Path(log_file).exists():
            data = parse_training_log(log_file)
            plot_training_metrics(data, 'final_training_plots.png')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Если передан файл лога
        log_file = sys.argv[1]
        if Path(log_file).exists():
            data = parse_training_log(log_file)
            plot_training_metrics(data)
        else:
            print(f"❌ Файл {log_file} не найден")
    else:
        # Пример использования с тестовыми данными
        print("📊 Создаю пример графиков...")
        
        # Тестовые данные
        test_data = {
            'validation': {
                'steps': [0, 250, 500, 750, 1000],
                'train_loss': [4.2, 3.1, 2.8, 2.5, 2.3],
                'val_loss': [4.3, 3.2, 2.9, 2.6, 2.4]
            },
            'training': {
                'iters': list(range(0, 100, 10)),
                'loss': [4.2 - i*0.02 + np.random.normal(0, 0.1) for i in range(10)],
                'time': [15 + np.random.normal(0, 2) for _ in range(10)],
                'mfu': [25 + i*0.5 + np.random.normal(0, 1) for i in range(10)]
            }
        }
        
        plot_training_metrics(test_data, 'example_training_plots.png')
