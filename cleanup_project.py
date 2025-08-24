#!/usr/bin/env python3
"""
Умная очистка проекта от лишних файлов экспериментов
Сохраняет только необходимые файлы для будущих экспериментов
"""

import os
import shutil
import glob
from datetime import datetime
from typing import List, Dict, Set

class ProjectCleaner:
    """Умный очиститель проекта"""
    
    def __init__(self):
        self.files_to_keep = set()
        self.files_to_remove = set()
        self.dirs_to_remove = set()
        self.backup_created = False
        
    def analyze_project(self):
        """Анализирует проект и определяет, что оставить, а что удалить"""
        
        print("🔍 АНАЛИЗ ПРОЕКТА")
        print("=" * 50)
        
        # ФАЙЛЫ, КОТОРЫЕ ОБЯЗАТЕЛЬНО ОСТАВЛЯЕМ
        essential_files = {
            # Основные файлы проекта
            'model.py',                    # Оригинальная модель GPT
            'train.py',                    # Оригинальный скрипт обучения
            'sample.py',                   # Оригинальный скрипт генерации
            'configurator.py',             # Конфигуратор
            'bench.py',                    # Бенчмарки
            
            # Наши лучшие разработки
            'model_advanced_centering.py', # Лучшая модель с центрированием
            'train_advanced_centered.py',  # Скрипт для обучения с центрированием
            'sample_centered.py',          # Генерация для центрированных моделей
            
            # Универсальная система экспериментов
            'universal_experiment_system.py',  # Универсальная система
            'experiment_analyzer.py',          # Анализатор результатов
            
            # Финальные скрипты
            'run_top3_1000iters.py',          # Финальное тестирование
            'test_generation_quality_final.py', # Тестирование генерации
            'analyze_final_1000iters.py',      # Финальный анализ
            'final_generation_analysis.py',    # Анализ генерации
            
            # Документация
            'VECTOR_CENTERING_RESEARCH.md',
            'PRACTICAL_APPLICATIONS.md', 
            'PROJECT_SUMMARY.md',
            'MONITORING_README.md',
            
            # Системные файлы
            'README.md',
            'LICENSE',
            '.gitignore',
            'requirements.txt',
            
            # Jupyter notebooks
            'scaling_laws.ipynb',
            'transformer_sizing.ipynb'
        }
        
        # КОНФИГУРАЦИИ, КОТОРЫЕ ОСТАВЛЯЕМ
        essential_configs = {
            # Оригинальные конфигурации
            'config/train_shakespeare_char.py',
            'config/train_shakespeare_bpe.py', 
            'config/train_gpt2.py',
            'config/finetune_shakespeare.py',
            'config/eval_gpt2.py',
            'config/eval_gpt2_large.py',
            'config/eval_gpt2_medium.py',
            'config/eval_gpt2_xl.py',
            
            # Финальные лучшие конфигурации
            'config/final_aggressive_all_1000.py',  # Лучшая модель
            'config/final_value_centered_1000.py',  # Второй лучший
            'config/final_baseline_1000.py',        # Baseline для сравнения
        }
        
        # ПАПКИ С РЕЗУЛЬТАТАМИ, КОТОРЫЕ ОСТАВЛЯЕМ
        essential_result_dirs = {
            # Финальные лучшие модели
            'out-final-aggressive_all_1000',   # Лучшая модель
            'out-final-value_centered_1000',   # Вторая лучшая
            'out-final-baseline_1000',         # Baseline
            
            # Папки с данными
            'data',
            'assets',
            'training_plots'  # Все графики и анализы
        }
        
        self.files_to_keep.update(essential_files)
        self.files_to_keep.update(essential_configs)
        
        return essential_files, essential_configs, essential_result_dirs
    
    def scan_files_to_remove(self):
        """Сканирует файлы для удаления"""
        
        print("\n🗑️  ПОИСК ФАЙЛОВ ДЛЯ УДАЛЕНИЯ")
        print("=" * 50)
        
        # Временные и экспериментальные файлы
        patterns_to_remove = [
            'training_log_*.log',           # Старые логи
            'extracted_results_*.json',     # Временные результаты
            'universal_experiments_results_*.json',  # Старые результаты универсальных экспериментов
            'top4_500iters_results_*.json', # Результаты промежуточных тестов
            'final_top3_1000iters_results_*.json',  # Можем оставить только последний
            '*.pyc',                        # Скомпилированные Python файлы
            '__pycache__',                  # Кэш Python
        ]
        
        files_found = []
        
        for pattern in patterns_to_remove:
            matches = glob.glob(pattern, recursive=True)
            files_found.extend(matches)
            
        # Экспериментальные скрипты (оставляем только лучшие)
        experimental_scripts = [
            'test_centering.py',
            'test_advanced_centering.py', 
            'test_advanced_centering_simple.py',
            'train_centering_experiments.py',
            'train_centering_experiments_bpe.py',
            'advanced_centering_experiments.py',
            'run_best_advanced_1k.py',
            'train_best_models_2k.py',
            'create_comparison_plots.py',
            'create_final_comparison.py',
            'plot_training.py',
            'plot_training_advanced.py',
            'plot_training_improved.py',
            'train_with_logging.py',
            'analyze_training.sh',
            'compare_generation.py',
            'test_advanced_generation.py',
            'comprehensive_generation_test.py',
            'extract_results_from_checkpoints.py',
            'visualize_universal_results.py',
            'run_top4_500iters.py',
            'analyze_top4_500iters.py',
        ]
        
        for script in experimental_scripts:
            if os.path.exists(script):
                files_found.append(script)
        
        self.files_to_remove.update(files_found)
        
        print(f"📊 Найдено файлов для удаления: {len(files_found)}")
        return files_found
    
    def scan_configs_to_remove(self):
        """Сканирует конфигурации для удаления"""
        
        print("\n🗑️  ПОИСК КОНФИГУРАЦИЙ ДЛЯ УДАЛЕНИЯ")
        print("=" * 50)
        
        # Все конфигурации
        all_configs = glob.glob('config/*.py')
        
        # Определяем, какие оставить
        essential_configs = {
            'config/train_shakespeare_char.py',
            'config/train_shakespeare_bpe.py', 
            'config/train_gpt2.py',
            'config/finetune_shakespeare.py',
            'config/eval_gpt2.py',
            'config/eval_gpt2_large.py',
            'config/eval_gpt2_medium.py',
            'config/eval_gpt2_xl.py',
            'config/final_aggressive_all_1000.py',
            'config/final_value_centered_1000.py',
            'config/final_baseline_1000.py',
        }
        
        configs_to_remove = []
        for config in all_configs:
            if config not in essential_configs:
                configs_to_remove.append(config)
        
        self.files_to_remove.update(configs_to_remove)
        
        print(f"📊 Конфигураций всего: {len(all_configs)}")
        print(f"📊 Оставляем: {len(essential_configs)}")
        print(f"📊 Удаляем: {len(configs_to_remove)}")
        
        return configs_to_remove
    
    def scan_result_dirs_to_remove(self):
        """Сканирует папки с результатами для удаления"""
        
        print("\n🗑️  ПОИСК ПАПОК С РЕЗУЛЬТАТАМИ ДЛЯ УДАЛЕНИЯ")
        print("=" * 50)
        
        # Все папки out-*
        all_out_dirs = glob.glob('out-*')
        
        # Оставляем только финальные лучшие
        essential_dirs = {
            'out-final-aggressive_all_1000',
            'out-final-value_centered_1000', 
            'out-final-baseline_1000',
        }
        
        dirs_to_remove = []
        total_size = 0
        
        for out_dir in all_out_dirs:
            if out_dir not in essential_dirs:
                dirs_to_remove.append(out_dir)
                
                # Подсчитываем размер
                try:
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(out_dir)
                        for filename in filenames
                    )
                    total_size += dir_size
                except:
                    pass
        
        self.dirs_to_remove.update(dirs_to_remove)
        
        print(f"📊 Папок результатов всего: {len(all_out_dirs)}")
        print(f"📊 Оставляем: {len(essential_dirs)}")
        print(f"📊 Удаляем: {len(dirs_to_remove)}")
        print(f"💾 Освободим места: ~{total_size / (1024**3):.1f} ГБ")
        
        return dirs_to_remove, total_size
    
    def create_backup(self):
        """Создает резервную копию важных файлов"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f'backup_before_cleanup_{timestamp}'
        
        print(f"\n💾 СОЗДАНИЕ РЕЗЕРВНОЙ КОПИИ")
        print("=" * 50)
        print(f"📁 Папка: {backup_dir}")
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Копируем важные файлы
        important_files = [
            'universal_experiment_system.py',
            'final_generation_analysis.py',
            'analyze_final_1000iters.py',
        ]
        
        for file in important_files:
            if os.path.exists(file):
                shutil.copy2(file, backup_dir)
                print(f"   💾 {file}")
        
        # Копируем финальные результаты JSON
        result_files = glob.glob('final_top3_1000iters_results_*.json')
        for file in result_files:
            shutil.copy2(file, backup_dir)
            print(f"   💾 {file}")
        
        self.backup_created = True
        print(f"✅ Резервная копия создана")
        
        return backup_dir
    
    def preview_cleanup(self):
        """Показывает превью того, что будет удалено"""
        
        print(f"\n👀 ПРЕВЬЮ ОЧИСТКИ")
        print("=" * 50)
        
        essential_files, essential_configs, essential_dirs = self.analyze_project()
        files_to_remove = self.scan_files_to_remove()
        configs_to_remove = self.scan_configs_to_remove()
        dirs_to_remove, total_size = self.scan_result_dirs_to_remove()
        
        print(f"\n📊 СВОДКА:")
        print(f"   🗑️  Файлов к удалению: {len(self.files_to_remove)}")
        print(f"   🗑️  Папок к удалению: {len(self.dirs_to_remove)}")
        print(f"   💾 Места освободится: ~{total_size / (1024**3):.1f} ГБ")
        
        print(f"\n✅ ОСТАНЕТСЯ:")
        print(f"   📄 Основных файлов: {len(essential_files)}")
        print(f"   ⚙️  Конфигураций: {len(essential_configs)}")
        print(f"   📁 Папок результатов: {len(essential_dirs)}")
        
        return len(self.files_to_remove), len(self.dirs_to_remove), total_size
    
    def execute_cleanup(self, create_backup=True):
        """Выполняет очистку"""
        
        if create_backup:
            backup_dir = self.create_backup()
        
        print(f"\n🧹 ВЫПОЛНЕНИЕ ОЧИСТКИ")
        print("=" * 50)
        
        removed_files = 0
        removed_dirs = 0
        
        # Удаляем файлы
        for file in self.files_to_remove:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"   🗑️  {file}")
                    removed_files += 1
            except Exception as e:
                print(f"   ❌ Ошибка удаления {file}: {e}")
        
        # Удаляем папки
        for dir_path in self.dirs_to_remove:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"   🗑️  📁 {dir_path}")
                    removed_dirs += 1
            except Exception as e:
                print(f"   ❌ Ошибка удаления {dir_path}: {e}")
        
        print(f"\n✅ ОЧИСТКА ЗАВЕРШЕНА!")
        print(f"   🗑️  Удалено файлов: {removed_files}")
        print(f"   🗑️  Удалено папок: {removed_dirs}")
        
        if create_backup and self.backup_created:
            print(f"   💾 Резервная копия: {backup_dir}")
        
        return removed_files, removed_dirs

def main():
    """Главная функция"""
    
    print("🧹 УМНАЯ ОЧИСТКА ПРОЕКТА nanoGPT")
    print("=" * 60)
    print("🎯 Цель: Удалить лишние файлы экспериментов")
    print("💡 Сохранить: только необходимое для будущих экспериментов")
    print("=" * 60)
    
    cleaner = ProjectCleaner()
    
    # Показываем превью
    num_files, num_dirs, size_gb = cleaner.preview_cleanup()
    
    print(f"\n❓ ВЫПОЛНИТЬ ОЧИСТКУ?")
    print(f"   🗑️  Будет удалено: {num_files} файлов, {num_dirs} папок")
    print(f"   💾 Освободится: ~{size_gb / (1024**3):.1f} ГБ")
    print(f"   💾 Резервная копия: ДА")
    
    choice = input("\nПродолжить? (y/N): ").strip().lower()
    
    if choice in ['y', 'yes', 'да']:
        removed_files, removed_dirs = cleaner.execute_cleanup(create_backup=True)
        
        print(f"\n🎉 ПРОЕКТ ОЧИЩЕН!")
        print("✅ Оставлены только необходимые файлы для будущих экспериментов")
        print("✅ Лучшие модели и конфигурации сохранены")
        print("✅ Документация и анализы сохранены")
        
    else:
        print("\n❌ Очистка отменена")
        print("💡 Для выполнения очистки запустите скрипт снова и выберите 'y'")

if __name__ == "__main__":
    main()
