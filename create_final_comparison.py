#!/usr/bin/env python3
"""
Финальное сравнение всех результатов: 1000 vs 2000 итераций
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
            model_type = "QK Centered"
        
        return {
            'model_type': model_type,
            'best_val_loss': best_val_loss,
            'iter_num': iter_num,
            'max_iters': config.get('max_iters', 'N/A')
        }
    except Exception as e:
        print(f"Ошибка загрузки {checkpoint_path}: {e}")
        return None

def create_comprehensive_comparison():
    """Создает полное сравнение всех экспериментов"""
    
    # Все эксперименты
    experiments = [
        # 1000 итераций
        ('out-centering-bpe-baseline_bpe', 'Baseline 1K', 1000),
        ('out-centering-bpe-qk_centered_bpe', 'QK Centered 1K', 1000),
        # 2000 итераций  
        ('out-baseline_bpe_2k', 'Baseline 2K', 2000),
        ('out-qk_centered_bpe_2k', 'QK Centered 2K', 2000)
    ]
    
    results = []
    
    for model_dir, name, expected_iters in experiments:
        checkpoint_path = f"{model_dir}/ckpt.pt"
        if os.path.exists(checkpoint_path):
            info = load_model_info(checkpoint_path)
            if info and info['best_val_loss'] is not None:
                results.append({
                    'name': name,
                    'loss': info['best_val_loss'],
                    'iters': expected_iters,
                    'type': 'Baseline' if 'Baseline' in name else 'QK Centered'
                })
    
    if not results:
        print("❌ Не найдено checkpoint'ов для сравнения")
        return None
    
    # Создаем график
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Заголовок
    fig.suptitle('Comprehensive Results: 1000 vs 2000 Iterations', fontsize=16, fontweight='bold')
    
    # График 1: Все результаты
    names = [r['name'] for r in results]
    losses = [r['loss'] for r in results]
    colors = ['lightblue' if '1K' in name else 'darkblue' if 'Baseline' in name else 'lightcoral' if '1K' in name else 'darkred' for name in names]
    
    bars1 = ax1.bar(range(len(names)), losses, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Final Validation Loss')
    ax1.set_title('All Experiments Comparison')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    format_axis_labels(ax1, 'y')
    
    # Добавляем значения на столбцы
    for bar, loss in zip(bars1, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # График 2: Сравнение по итерациям
    baseline_1k = next((r['loss'] for r in results if r['name'] == 'Baseline 1K'), None)
    baseline_2k = next((r['loss'] for r in results if r['name'] == 'Baseline 2K'), None)
    qk_1k = next((r['loss'] for r in results if r['name'] == 'QK Centered 1K'), None)
    qk_2k = next((r['loss'] for r in results if r['name'] == 'QK Centered 2K'), None)
    
    if all(x is not None for x in [baseline_1k, baseline_2k, qk_1k, qk_2k]):
        x = np.arange(2)
        width = 0.35
        
        ax2.bar(x - width/2, [baseline_1k, baseline_2k], width, label='Baseline', color='blue', alpha=0.7)
        ax2.bar(x + width/2, [qk_1k, qk_2k], width, label='QK Centered', color='red', alpha=0.7)
        
        ax2.set_xlabel('Training Length')
        ax2.set_ylabel('Final Validation Loss')
        ax2.set_title('1000 vs 2000 Iterations')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['1000 iters', '2000 iters'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        format_axis_labels(ax2, 'y')
        
        # Добавляем значения
        for i, (b_val, q_val) in enumerate([(baseline_1k, qk_1k), (baseline_2k, qk_2k)]):
            ax2.text(i - width/2, b_val + b_val*0.01, f'{b_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax2.text(i + width/2, q_val + q_val*0.01, f'{q_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # График 3: Улучшения QK Centered vs Baseline
    improvements = []
    iter_labels = []
    
    if baseline_1k and qk_1k:
        imp_1k = ((baseline_1k - qk_1k) / baseline_1k) * 100
        improvements.append(imp_1k)
        iter_labels.append('1000 iters')
    
    if baseline_2k and qk_2k:
        imp_2k = ((baseline_2k - qk_2k) / baseline_2k) * 100
        improvements.append(imp_2k)
        iter_labels.append('2000 iters')
    
    if improvements:
        colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax3.bar(iter_labels, improvements, color=colors_imp, alpha=0.7, edgecolor='black')
        
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('QK Centered Improvement vs Baseline')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        format_axis_labels(ax3, 'y')
        
        # Добавляем значения
        for bar, imp in zip(bars3, improvements):
            height = bar.get_height()
            y_pos = height + (abs(height) * 0.1 if height >= 0 else -abs(height) * 0.1)
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{imp:+.2f}%', ha='center', 
                    va='bottom' if height >= 0 else 'top', 
                    fontweight='bold', fontsize=10)
    
    # График 4: Сводная таблица
    ax4.axis('off')
    
    table_data = []
    headers = ['Model', '1000 iters', '2000 iters', 'Improvement']
    
    # Baseline строка
    baseline_row = ['Baseline']
    baseline_row.append(f'{baseline_1k:.4f}' if baseline_1k else 'N/A')
    baseline_row.append(f'{baseline_2k:.4f}' if baseline_2k else 'N/A')
    if baseline_1k and baseline_2k:
        baseline_change = ((baseline_1k - baseline_2k) / baseline_1k) * 100
        baseline_row.append(f'{baseline_change:+.2f}%')
    else:
        baseline_row.append('N/A')
    table_data.append(baseline_row)
    
    # QK Centered строка
    qk_row = ['QK Centered']
    qk_row.append(f'{qk_1k:.4f}' if qk_1k else 'N/A')
    qk_row.append(f'{qk_2k:.4f}' if qk_2k else 'N/A')
    if qk_1k and qk_2k:
        qk_change = ((qk_1k - qk_2k) / qk_1k) * 100
        qk_row.append(f'{qk_change:+.2f}%')
    else:
        qk_row.append('N/A')
    table_data.append(qk_row)
    
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     bbox=[0, 0.3, 1, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Выделяем заголовки
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Выделяем лучшие результаты
    if qk_1k and baseline_1k and qk_1k < baseline_1k:
        table[(2, 1)].set_facecolor('#FFD700')  # QK Centered 1K лучше
    if qk_2k and baseline_2k and qk_2k < baseline_2k:
        table[(2, 2)].set_facecolor('#FFD700')  # QK Centered 2K лучше
    
    ax4.set_title('Summary Table', fontweight='bold', pad=20)
    
    # Добавляем выводы
    conclusions = []
    if baseline_1k and qk_1k:
        imp_1k = ((baseline_1k - qk_1k) / baseline_1k) * 100
        conclusions.append(f"• 1000 iters: QK Centered {imp_1k:+.2f}% vs Baseline")
    
    if baseline_2k and qk_2k:
        imp_2k = ((baseline_2k - qk_2k) / baseline_2k) * 100
        conclusions.append(f"• 2000 iters: QK Centered {imp_2k:+.2f}% vs Baseline")
    
    if len(improvements) == 2:
        if improvements[0] > improvements[1]:
            conclusions.append("• Improvement decreases with more training")
        else:
            conclusions.append("• Improvement increases with more training")
    
    conclusion_text = "Key Findings:\n" + "\n".join(conclusions)
    ax4.text(0.5, 0.1, conclusion_text, transform=ax4.transAxes, 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"training_plots/comprehensive_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Полное сравнение сохранено: {save_path}")
    
    # Выводим анализ
    print(f"\n🔍 АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("=" * 50)
    
    if baseline_1k and qk_1k and baseline_2k and qk_2k:
        print(f"1000 итераций:")
        print(f"  Baseline: {baseline_1k:.4f}")
        print(f"  QK Centered: {qk_1k:.4f}")
        print(f"  Улучшение: {((baseline_1k - qk_1k) / baseline_1k) * 100:+.2f}%")
        
        print(f"\n2000 итераций:")
        print(f"  Baseline: {baseline_2k:.4f}")
        print(f"  QK Centered: {qk_2k:.4f}")
        print(f"  Улучшение: {((baseline_2k - qk_2k) / baseline_2k) * 100:+.2f}%")
        
        print(f"\nВлияние дополнительного обучения:")
        baseline_change = ((baseline_1k - baseline_2k) / baseline_1k) * 100
        qk_change = ((qk_1k - qk_2k) / qk_1k) * 100
        print(f"  Baseline: {baseline_change:+.2f}% (1K→2K)")
        print(f"  QK Centered: {qk_change:+.2f}% (1K→2K)")
    
    return save_path

def main():
    create_comprehensive_comparison()

if __name__ == "__main__":
    main()
