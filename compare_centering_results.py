#!/usr/bin/env python3
"""
Скрипт для сравнения результатов экспериментов с центрированием
"""

import os
import re
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def extract_final_loss_from_checkpoint(checkpoint_path):
    """Извлекает финальный loss из checkpoint"""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint.get('best_val_loss', None)
    except:
        return None

def parse_experiment_results():
    """Парсит результаты всех экспериментов"""
    experiments = [
        'baseline', 'qk_centered', 'final_centered', 
        'full_centered', 'momentum_centered'
    ]
    
    results = {}
    
    for exp_name in experiments:
        out_dir = f"out-centering-{exp_name}"
        checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
        
        if os.path.exists(checkpoint_path):
            final_loss = extract_final_loss_from_checkpoint(checkpoint_path)
            
            # Также попробуем найти информацию в логах
            log_info = {}
            
            results[exp_name] = {
                'final_val_loss': final_loss,
                'checkpoint_exists': True,
                'log_info': log_info
            }
        else:
            results[exp_name] = {
                'final_val_loss': None,
                'checkpoint_exists': False,
                'log_info': {}
            }
    
    return results

def create_comparison_plot(results):
    """Создает график сравнения результатов"""
    
    # Фильтруем эксперименты с валидными результатами
    valid_results = {k: v for k, v in results.items() 
                    if v['final_val_loss'] is not None}
    
    if not valid_results:
        print("❌ Нет валидных результатов для сравнения")
        return
    
    # Подготавливаем данные для графика
    exp_names = list(valid_results.keys())
    final_losses = [valid_results[name]['final_val_loss'] for name in exp_names]
    
    # Красивые названия для экспериментов
    pretty_names = {
        'baseline': 'Базовая модель',
        'qk_centered': 'Центрирование Q/K',
        'final_centered': 'Центрирование финальных эмбеддингов',
        'full_centered': 'Полное центрирование',
        'momentum_centered': 'Центрирование с momentum'
    }
    
    display_names = [pretty_names.get(name, name) for name in exp_names]
    
    # Создаем график
    plt.figure(figsize=(12, 8))
    
    # Цвета для разных типов экспериментов
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = plt.bar(range(len(exp_names)), final_losses, color=colors[:len(exp_names)])
    
    # Настройка графика
    plt.xlabel('Тип эксперимента', fontsize=12)
    plt.ylabel('Финальный validation loss', fontsize=12)
    plt.title('Сравнение результатов экспериментов с центрированием векторов', fontsize=14, pad=20)
    plt.xticks(range(len(exp_names)), display_names, rotation=45, ha='right')
    
    # Добавляем значения на столбцы
    for i, (bar, loss) in enumerate(zip(bars, final_losses)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Добавляем сетку
    plt.grid(True, alpha=0.3, axis='y')
    
    # Выделяем лучший результат
    best_idx = np.argmin(final_losses)
    bars[best_idx].set_color('#2ca02c')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('centering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return exp_names[best_idx], final_losses[best_idx]

def generate_samples_comparison():
    """Генерирует образцы текста для сравнения"""
    experiments = ['baseline', 'qk_centered', 'final_centered', 'full_centered', 'momentum_centered']
    
    print("\n🎭 Сравнение качества генерации текста:")
    print("=" * 60)
    
    for exp_name in experiments:
        out_dir = f"out-centering-{exp_name}"
        
        if os.path.exists(out_dir):
            print(f"\n--- {exp_name.upper()} ---")
            
            # Генерируем образец
            cmd = f"python sample.py --out_dir={out_dir} --num_samples=1 --max_new_tokens=100 --start='ROMEO:'"
            
            try:
                import subprocess
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Извлекаем сгенерированный текст
                    output_lines = result.stdout.split('\n')
                    in_sample = False
                    sample_text = []
                    
                    for line in output_lines:
                        if '---------------' in line:
                            in_sample = not in_sample
                            continue
                        if in_sample and line.strip():
                            sample_text.append(line.strip())
                    
                    if sample_text:
                        print(' '.join(sample_text))
                    else:
                        print("Не удалось извлечь образец")
                else:
                    print(f"Ошибка генерации: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("Таймаут при генерации")
            except Exception as e:
                print(f"Ошибка: {e}")

def main():
    print("📊 Анализ результатов экспериментов с центрированием векторов")
    print("=" * 70)
    
    # Парсим результаты
    results = parse_experiment_results()
    
    print("\n📈 Результаты экспериментов:")
    print("-" * 50)
    
    for exp_name, data in results.items():
        status = "✅" if data['checkpoint_exists'] else "❌"
        loss_str = f"{data['final_val_loss']:.4f}" if data['final_val_loss'] else "N/A"
        print(f"{status} {exp_name:20} | Final val loss: {loss_str}")
    
    # Создаем график сравнения
    try:
        best_exp, best_loss = create_comparison_plot(results)
        print(f"\n🏆 Лучший результат: {best_exp} (val loss: {best_loss:.4f})")
        print("📊 График сравнения сохранен: centering_comparison.png")
    except Exception as e:
        print(f"❌ Ошибка при создании графика: {e}")
    
    # Генерируем образцы для сравнения
    try:
        generate_samples_comparison()
    except Exception as e:
        print(f"❌ Ошибка при генерации образцов: {e}")
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main()
