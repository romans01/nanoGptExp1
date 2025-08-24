#!/usr/bin/env python3
"""
Создание сравнительных графиков для лучших моделей
Работает напрямую с checkpoint'ами и логами
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import datetime

def format_axis_labels(ax, axis='y'):
    """Форматирует подписи осей без научной нотации"""
    if axis == 'y':
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3f}' if x < 1 else f'{x:.1f}'))
    elif axis == 'x':
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))

def load_model_info(checkpoint_path):
    """Загружает информацию о модели из checkpoint'а"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model_args = checkpoint.get('model_args', {})
        config = checkpoint.get('config', {})
        best_val_loss = checkpoint.get('best_val_loss', None)
        iter_num = checkpoint.get('iter_num', None)
        
        # Определяем тип модели
        model_type = "Baseline"
        if model_args.get('use_centered_attention') and model_args.get('center_qk'):
            if model_args.get('center_final_output'):
                model_type = "Full Centered"
            else:
                model_type = "QK Centered"
        elif model_args.get('center_final_output'):
            model_type = "Final Centered"
        elif model_args.get('center_block_output'):
            model_type = "Block Centered"
        
        return {
            'model_type': model_type,
            'best_val_loss': best_val_loss,
            'iter_num': iter_num,
            'n_layer': model_args.get('n_layer', 'N/A'),
            'n_head': model_args.get('n_head', 'N/A'),
            'n_embd': model_args.get('n_embd', 'N/A'),
            'batch_size': config.get('batch_size', 'N/A'),
            'learning_rate': config.get('learning_rate', 'N/A'),
            'max_iters': config.get('max_iters', 'N/A'),
            'dataset': config.get('dataset', 'N/A')
        }
    except Exception as e:
        print(f"Ошибка загрузки {checkpoint_path}: {e}")
        return None

def create_results_comparison():
    """Создает сравнительный график результатов всех экспериментов"""
    
    # Ищем все checkpoint'ы
    model_dirs = [
        ('out-centering-bpe-baseline_bpe', 'Baseline'),
        ('out-centering-bpe-qk_centered_bpe', 'QK Centered'),
        ('out-centering-bpe-final_centered_bpe', 'Final Centered'),
        ('out-centering-bpe-full_centered_bpe', 'Full Centered'),
        ('out-centering-bpe-block_centered_bpe', 'Block Centered')
    ]
    
    results = []
    
    for model_dir, expected_name in model_dirs:
        checkpoint_path = f"{model_dir}/ckpt.pt"
        if os.path.exists(checkpoint_path):
            info = load_model_info(checkpoint_path)
            if info and info['best_val_loss'] is not None:
                results.append({
                    'name': expected_name,
                    'loss': info['best_val_loss'],
                    'info': info
                })
    
    if not results:
        print("❌ Не найдено checkpoint'ов для сравнения")
        return None
    
    # Сортируем по loss (лучшие первые)
    results.sort(key=lambda x: x['loss'])
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # График 1: Барный график validation loss
    names = [r['name'] for r in results]
    losses = [r['loss'] for r in results]
    
    colors = ['green' if i < 2 else 'orange' if i < 3 else 'red' for i in range(len(names))]
    bars = ax1.bar(range(len(names)), losses, color=colors, alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Model Type', fontsize=12)
    ax1.set_ylabel('Final Validation Loss', fontsize=12)
    ax1.set_title('Model Performance Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    format_axis_labels(ax1, 'y')
    
    # Добавляем значения на столбцы
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Выделяем лучший результат
    if results:
        best_bar = bars[0]
        best_bar.set_color('gold')
        best_bar.set_edgecolor('darkgoldenrod')
        best_bar.set_linewidth(3)
    
    # График 2: Улучшения относительно baseline
    baseline_loss = None
    for r in results:
        if r['name'] == 'Baseline':
            baseline_loss = r['loss']
            break
    
    if baseline_loss:
        improvements = []
        improvement_names = []
        improvement_colors = []
        
        for r in results:
            if r['name'] != 'Baseline':
                improvement = ((baseline_loss - r['loss']) / baseline_loss) * 100
                improvements.append(improvement)
                improvement_names.append(r['name'])
                improvement_colors.append('green' if improvement > 0 else 'red')
        
        bars2 = ax2.bar(range(len(improvement_names)), improvements, 
                       color=improvement_colors, alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Model Type', fontsize=12)
        ax2.set_ylabel('Improvement vs Baseline (%)', fontsize=12)
        ax2.set_title(f'Performance vs Baseline\n(Baseline loss: {baseline_loss:.3f})', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(improvement_names)))
        ax2.set_xticklabels(improvement_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        format_axis_labels(ax2, 'y')
        
        # Добавляем значения на столбцы
        for bar, imp in zip(bars2, improvements):
            height = bar.get_height()
            y_pos = height + (abs(height) * 0.05 if height >= 0 else -abs(height) * 0.05)
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{imp:+.1f}%', ha='center', 
                    va='bottom' if height >= 0 else 'top', 
                    fontweight='bold', fontsize=10)
    
    # Добавляем общую информацию
    info_text = f"""Experiment Details:
Architecture: {results[0]['info']['n_layer']} layers, {results[0]['info']['n_head']} heads, {results[0]['info']['n_embd']} embd
Dataset: {results[0]['info']['dataset']}
Max Iterations: {results[0]['info']['max_iters']}
Batch Size: {results[0]['info']['batch_size']}
Learning Rate: {results[0]['info']['learning_rate']}"""
    
    fig.text(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Место для информации
    
    # Сохраняем график
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"training_plots/models_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Сравнительный график сохранен: {save_path}")
    
    # Выводим результаты в консоль
    print(f"\n🏆 Результаты экспериментов:")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{status} {r['name']}: {r['loss']:.4f}")
        
        if baseline_loss and r['name'] != 'Baseline':
            improvement = ((baseline_loss - r['loss']) / baseline_loss) * 100
            print(f"   Улучшение vs Baseline: {improvement:+.2f}%")
    
    return save_path

def create_best_two_comparison():
    """Создает детальное сравнение двух лучших моделей"""
    
    models = [
        ('out-centering-bpe-baseline_bpe', 'Baseline'),
        ('out-centering-bpe-qk_centered_bpe', 'QK Centered')
    ]
    
    model_data = []
    
    for model_dir, name in models:
        checkpoint_path = f"{model_dir}/ckpt.pt"
        if os.path.exists(checkpoint_path):
            info = load_model_info(checkpoint_path)
            if info:
                model_data.append((name, info))
    
    if len(model_data) < 2:
        print("❌ Недостаточно данных для сравнения лучших моделей")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Заголовок
    fig.suptitle('Detailed Comparison: Best Models', fontsize=16, fontweight='bold')
    
    # График 1: Validation Loss сравнение
    names = [data[0] for data in model_data]
    losses = [data[1]['best_val_loss'] for data in model_data]
    colors = ['blue', 'red']
    
    bars1 = ax1.bar(names, losses, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Final Validation Loss')
    ax1.set_title('Final Validation Loss')
    ax1.grid(True, alpha=0.3, axis='y')
    format_axis_labels(ax1, 'y')
    
    for bar, loss in zip(bars1, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # График 2: Улучшение
    if len(losses) == 2:
        baseline_loss, qk_loss = losses
        improvement = ((baseline_loss - qk_loss) / baseline_loss) * 100
        
        ax2.bar(['QK Centered vs Baseline'], [improvement], 
               color='green' if improvement > 0 else 'red', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Performance Improvement')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        format_axis_labels(ax2, 'y')
        
        ax2.text(0, improvement + (abs(improvement) * 0.1 if improvement >= 0 else -abs(improvement) * 0.1),
                f'{improvement:+.2f}%', ha='center', 
                va='bottom' if improvement >= 0 else 'top', 
                fontweight='bold', fontsize=12)
    
    # График 3: Архитектурные параметры
    arch_params = ['n_layer', 'n_head', 'n_embd']
    baseline_arch = [model_data[0][1][param] for param in arch_params]
    qk_arch = [model_data[1][1][param] for param in arch_params]
    
    x = np.arange(len(arch_params))
    width = 0.35
    
    ax3.bar(x - width/2, baseline_arch, width, label='Baseline', color='blue', alpha=0.7)
    ax3.bar(x + width/2, qk_arch, width, label='QK Centered', color='red', alpha=0.7)
    
    ax3.set_xlabel('Architecture Parameters')
    ax3.set_ylabel('Value')
    ax3.set_title('Model Architecture Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(arch_params)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    format_axis_labels(ax3, 'y')
    
    # График 4: Сводная таблица
    ax4.axis('off')
    
    table_data = []
    headers = ['Parameter', 'Baseline', 'QK Centered']
    
    for param in ['best_val_loss', 'iter_num', 'n_layer', 'n_head', 'n_embd', 'batch_size', 'learning_rate']:
        row = [param.replace('_', ' ').title()]
        for _, info in model_data:
            value = info[param]
            if isinstance(value, float):
                if param == 'learning_rate':
                    row.append(f'{value:.0e}')
                else:
                    row.append(f'{value:.4f}')
            else:
                row.append(str(value))
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Выделяем заголовки
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Выделяем лучшие результаты
    if len(model_data) == 2 and model_data[1][1]['best_val_loss'] < model_data[0][1]['best_val_loss']:
        table[(1, 2)].set_facecolor('#FFD700')  # Золотой для лучшего результата
    
    ax4.set_title('Detailed Comparison Table', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"training_plots/best_models_detailed_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Детальное сравнение сохранено: {save_path}")
    
    return save_path

def main():
    """Основная функция"""
    import sys
    
    print("📊 Создание улучшенных графиков сравнения")
    print("=" * 50)
    
    # Создаем общее сравнение всех моделей
    comparison_path = create_results_comparison()
    
    # Создаем детальное сравнение лучших моделей
    detailed_path = create_best_two_comparison()
    
    print(f"\n✅ Графики созданы:")
    if comparison_path:
        print(f"   📊 Общее сравнение: {comparison_path}")
    if detailed_path:
        print(f"   🔍 Детальное сравнение: {detailed_path}")

if __name__ == "__main__":
    main()
