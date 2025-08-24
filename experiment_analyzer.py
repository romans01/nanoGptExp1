#!/usr/bin/env python3
"""
Анализатор результатов универсальной системы экспериментов
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

class ExperimentAnalyzer:
    """Анализатор результатов экспериментов"""
    
    def __init__(self, results_file: str):
        """Инициализация с файлом результатов"""
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data['results']
        self.successful_results = [r for r in self.results if r['success']]
        
    def print_summary(self):
        """Выводит краткую сводку результатов"""
        
        print("📊 СВОДКА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
        print("=" * 60)
        
        total = len(self.results)
        successful = len(self.successful_results)
        failed = total - successful
        
        print(f"📈 Всего экспериментов: {total}")
        print(f"✅ Успешных: {successful}")
        print(f"❌ Неудачных: {failed}")
        
        if successful > 0:
            total_time = sum(r['time'] for r in self.successful_results)
            avg_time = total_time / successful
            print(f"⏱️  Среднее время: {avg_time:.1f}с")
            print(f"⏱️  Общее время: {total_time/60:.1f} мин")
            
        print(f"🧠 Базовая архитектура: {self.data['base_config']['n_layer']} слоев, {self.data['base_config']['n_head']} голов, {self.data['base_config']['n_embd']} эмбеддингов")
        
    def analyze_by_validation_loss(self, top_n: int = 10):
        """Анализирует результаты по validation loss"""
        
        print(f"\n🏆 ТОП-{top_n} ПО VALIDATION LOSS")
        print("=" * 60)
        
        # Фильтруем результаты с validation loss
        results_with_loss = [r for r in self.successful_results if 'final_val_loss' in r]
        
        if not results_with_loss:
            print("⚠️  Нет результатов с validation loss")
            return
            
        # Сортируем по validation loss
        results_with_loss.sort(key=lambda x: x['final_val_loss'])
        
        # Находим baseline для сравнения
        baseline_loss = None
        for r in results_with_loss:
            if 'baseline' in r['name'].lower():
                baseline_loss = r['final_val_loss']
                break
                
        print(f"📊 Baseline validation loss: {baseline_loss:.4f}" if baseline_loss else "⚠️  Baseline не найден")
        print()
        
        for i, result in enumerate(results_with_loss[:top_n], 1):
            val_loss = result['final_val_loss']
            name = result['name']
            description = result['description']
            
            # Вычисляем улучшение относительно baseline
            improvement = ""
            if baseline_loss and val_loss < baseline_loss:
                pct_improvement = ((baseline_loss - val_loss) / baseline_loss) * 100
                improvement = f" (+{pct_improvement:.2f}%)"
            elif baseline_loss and val_loss > baseline_loss:
                pct_degradation = ((val_loss - baseline_loss) / baseline_loss) * 100
                improvement = f" (-{pct_degradation:.2f}%)"
                
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{medal} {name}: {val_loss:.4f}{improvement}")
            print(f"   📝 {description}")
            print()
            
    def analyze_by_centering_type(self):
        """Анализирует результаты по типам центрирования"""
        
        print("\n🎯 АНАЛИЗ ПО ТИПАМ ЦЕНТРИРОВАНИЯ")
        print("=" * 60)
        
        # Группируем по типам центрирования
        centering_groups = {}
        
        for result in self.successful_results:
            if 'final_val_loss' not in result:
                continue
                
            config = result.get('config', {})
            
            # Определяем активные типы центрирования
            active_centering = []
            if config.get('center_qk', False):
                active_centering.append('QK')
            if config.get('center_v', False):
                active_centering.append('V')
            if config.get('center_mlp', False):
                active_centering.append('MLP')
            if config.get('center_embeddings', False):
                active_centering.append('Embeddings')
            if config.get('center_residual', False):
                active_centering.append('Residual')
                
            if not active_centering:
                centering_type = "None (Baseline)"
            else:
                centering_type = " + ".join(active_centering)
                
            if centering_type not in centering_groups:
                centering_groups[centering_type] = []
            centering_groups[centering_type].append(result)
            
        # Анализируем каждую группу
        for centering_type, group_results in centering_groups.items():
            if not group_results:
                continue
                
            losses = [r['final_val_loss'] for r in group_results]
            avg_loss = np.mean(losses)
            min_loss = np.min(losses)
            max_loss = np.max(losses)
            
            print(f"🎭 {centering_type}:")
            print(f"   📊 Экспериментов: {len(group_results)}")
            print(f"   📈 Средний loss: {avg_loss:.4f}")
            print(f"   🏆 Лучший loss: {min_loss:.4f}")
            if len(group_results) > 1:
                print(f"   📉 Худший loss: {max_loss:.4f}")
            print()
            
    def create_comparison_plot(self, output_file: Optional[str] = None):
        """Создает график сравнения результатов"""
        
        results_with_loss = [r for r in self.successful_results if 'final_val_loss' in r]
        
        if len(results_with_loss) < 2:
            print("⚠️  Недостаточно данных для графика")
            return
            
        # Сортируем по validation loss
        results_with_loss.sort(key=lambda x: x['final_val_loss'])
        
        # Подготавливаем данные
        names = [r['name'] for r in results_with_loss]
        losses = [r['final_val_loss'] for r in results_with_loss]
        
        # Создаем график
        plt.figure(figsize=(15, 8))
        
        # Цветовая схема
        colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'lightblue' 
                 for i in range(len(names))]
        
        bars = plt.bar(range(len(names)), losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Настройка графика
        plt.xlabel('Эксперименты', fontsize=12, fontweight='bold')
        plt.ylabel('Validation Loss', fontsize=12, fontweight='bold')
        plt.title('Сравнение результатов экспериментов с центрированием\n(Меньше = Лучше)', 
                 fontsize=14, fontweight='bold')
        
        # Поворачиваем названия экспериментов
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        
        # Добавляем значения на столбцы
        for i, (bar, loss) in enumerate(zip(bars, losses)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Добавляем медали для топ-3
            if i == 0:
                plt.text(bar.get_x() + bar.get_width()/2., height/2, '🥇', 
                        ha='center', va='center', fontsize=20)
            elif i == 1:
                plt.text(bar.get_x() + bar.get_width()/2., height/2, '🥈', 
                        ha='center', va='center', fontsize=20)
            elif i == 2:
                plt.text(bar.get_x() + bar.get_width()/2., height/2, '🥉', 
                        ha='center', va='center', fontsize=20)
        
        # Настройка сетки
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Плотная компоновка
        plt.tight_layout()
        
        # Сохранение
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'training_plots/universal_experiments_comparison_{timestamp}.png'
            
        os.makedirs('training_plots', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 График сохранен: {output_file}")
        
    def create_centering_heatmap(self, output_file: Optional[str] = None):
        """Создает тепловую карту эффективности разных типов центрирования"""
        
        results_with_loss = [r for r in self.successful_results if 'final_val_loss' in r]
        
        if len(results_with_loss) < 5:
            print("⚠️  Недостаточно данных для тепловой карты")
            return
            
        # Подготавливаем матрицу центрирования
        centering_types = ['center_qk', 'center_v', 'center_mlp', 'center_embeddings', 'center_residual']
        centering_names = ['QK', 'Value', 'MLP', 'Embeddings', 'Residual']
        
        # Создаем матрицу результатов
        matrix_data = []
        experiment_names = []
        
        for result in results_with_loss:
            config = result.get('config', {})
            row = [1 if config.get(ct, False) else 0 for ct in centering_types]
            matrix_data.append(row)
            experiment_names.append(result['name'])
            
        matrix_data = np.array(matrix_data)
        losses = np.array([r['final_val_loss'] for r in results_with_loss])
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Тепловая карта центрирования
        im1 = ax1.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto')
        ax1.set_xticks(range(len(centering_names)))
        ax1.set_xticklabels(centering_names)
        ax1.set_yticks(range(len(experiment_names)))
        ax1.set_yticklabels(experiment_names)
        ax1.set_title('Активные типы центрирования\n(Красный = Активен)', fontweight='bold')
        
        # Добавляем текст в ячейки
        for i in range(len(experiment_names)):
            for j in range(len(centering_names)):
                text = '✓' if matrix_data[i, j] else '✗'
                ax1.text(j, i, text, ha="center", va="center", 
                        color="white" if matrix_data[i, j] else "black", fontweight='bold')
        
        # График validation loss
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(losses)))
        bars = ax2.barh(range(len(experiment_names)), losses, color=colors)
        ax2.set_yticks(range(len(experiment_names)))
        ax2.set_yticklabels(experiment_names)
        ax2.set_xlabel('Validation Loss')
        ax2.set_title('Validation Loss по экспериментам\n(Меньше = Лучше)', fontweight='bold')
        
        # Добавляем значения на столбцы
        for i, (bar, loss) in enumerate(zip(bars, losses)):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{loss:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Сохранение
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'training_plots/centering_heatmap_{timestamp}.png'
            
        os.makedirs('training_plots', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🔥 Тепловая карта сохранена: {output_file}")
        
    def generate_report(self, output_file: Optional[str] = None):
        """Генерирует подробный отчет"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'training_plots/universal_experiments_report_{timestamp}.txt'
            
        os.makedirs('training_plots', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("📊 ПОДРОБНЫЙ ОТЧЕТ УНИВЕРСАЛЬНЫХ ЭКСПЕРИМЕНТОВ\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая информация
            f.write("📈 ОБЩАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"• Дата проведения: {self.data['timestamp']}\n")
            f.write(f"• Всего экспериментов: {len(self.results)}\n")
            f.write(f"• Успешных: {len(self.successful_results)}\n")
            f.write(f"• Базовая архитектура: {self.data['base_config']['n_layer']} слоев, {self.data['base_config']['n_head']} голов, {self.data['base_config']['n_embd']} эмбеддингов\n")
            f.write(f"• Общее время: {self.data['total_time']/60:.1f} минут\n\n")
            
            # Топ результаты
            results_with_loss = [r for r in self.successful_results if 'final_val_loss' in r]
            if results_with_loss:
                results_with_loss.sort(key=lambda x: x['final_val_loss'])
                
                f.write("🏆 ТОП-10 РЕЗУЛЬТАТОВ ПО VALIDATION LOSS:\n")
                f.write("-" * 50 + "\n")
                
                # Находим baseline
                baseline_loss = None
                for r in results_with_loss:
                    if 'baseline' in r['name'].lower():
                        baseline_loss = r['final_val_loss']
                        break
                        
                for i, result in enumerate(results_with_loss[:10], 1):
                    val_loss = result['final_val_loss']
                    name = result['name']
                    description = result['description']
                    
                    improvement = ""
                    if baseline_loss and val_loss != baseline_loss:
                        pct_change = ((baseline_loss - val_loss) / baseline_loss) * 100
                        improvement = f" ({pct_change:+.2f}%)"
                        
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                    f.write(f"{medal} {name}: {val_loss:.4f}{improvement}\n")
                    f.write(f"    📝 {description}\n\n")
                    
            # Детальные результаты
            f.write("\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
            f.write("=" * 50 + "\n")
            
            for result in self.successful_results:
                f.write(f"\n🧪 {result['name']}:\n")
                f.write(f"   📝 Описание: {result['description']}\n")
                f.write(f"   ⏱️  Время обучения: {result['time']:.1f}с\n")
                
                if 'final_val_loss' in result:
                    f.write(f"   📊 Validation Loss: {result['final_val_loss']:.4f}\n")
                if 'final_train_loss' in result:
                    f.write(f"   📈 Training Loss: {result['final_train_loss']:.4f}\n")
                    
                # Конфигурация центрирования
                config = result.get('config', {})
                centering_active = []
                if config.get('center_qk', False):
                    centering_active.append('QK')
                if config.get('center_v', False):
                    centering_active.append('Value')
                if config.get('center_mlp', False):
                    centering_active.append('MLP')
                if config.get('center_embeddings', False):
                    centering_active.append('Embeddings')
                if config.get('center_residual', False):
                    centering_active.append('Residual')
                    
                if centering_active:
                    f.write(f"   🎯 Центрирование: {', '.join(centering_active)}\n")
                    f.write(f"   🔧 Режим: {config.get('centering_mode', 'adaptive')}\n")
                else:
                    f.write(f"   🎯 Центрирование: Отсутствует (Baseline)\n")
                    
        print(f"📄 Подробный отчет сохранен: {output_file}")

def main():
    """Главная функция анализатора"""
    
    # Ищем последний файл результатов
    result_files = [f for f in os.listdir('.') if f.startswith('universal_experiments_results_')]
    
    if not result_files:
        print("❌ Файлы результатов не найдены")
        print("💡 Сначала запустите universal_experiment_system.py")
        return
        
    # Берем последний файл
    latest_file = sorted(result_files)[-1]
    print(f"📂 Анализируем файл: {latest_file}")
    
    # Создаем анализатор
    analyzer = ExperimentAnalyzer(latest_file)
    
    # Выполняем анализ
    analyzer.print_summary()
    analyzer.analyze_by_validation_loss()
    analyzer.analyze_by_centering_type()
    
    # Создаем визуализации
    analyzer.create_comparison_plot()
    analyzer.create_centering_heatmap()
    
    # Генерируем отчет
    analyzer.generate_report()
    
    print("\n🎉 Анализ завершен! Все файлы сохранены в training_plots/")

if __name__ == "__main__":
    main()
