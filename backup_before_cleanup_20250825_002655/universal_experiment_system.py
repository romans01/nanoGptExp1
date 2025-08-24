#!/usr/bin/env python3
"""
Универсальная система экспериментов с центрированием
Обеспечивает одинаковые базовые характеристики для всех моделей
"""

import os
import json
import time
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

@dataclass
class BaseModelConfig:
    """Базовая конфигурация модели - одинаковая для всех экспериментов"""
    # Архитектура модели
    n_layer: int = 12
    n_head: int = 12  
    n_embd: int = 768
    dropout: float = 0.1
    
    # Данные
    dataset: str = 'shakespeare'
    gradient_accumulation_steps: int = 4
    batch_size: int = 8
    block_size: int = 256
    
    # Оптимизация
    learning_rate: float = 3e-4
    beta2: float = 0.99
    weight_decay: float = 1e-1
    
    # Система
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True

@dataclass 
class ExperimentConfig:
    """Конфигурация конкретного эксперимента"""
    name: str
    description: str
    max_iters: int
    
    # Центрирование
    center_qk: bool = False
    center_v: bool = False
    center_mlp: bool = False
    center_embeddings: bool = False
    center_residual: bool = False
    center_final_output: bool = False
    center_block_output: bool = False
    centering_mode: str = 'adaptive'
    
    # Дополнительные параметры
    eval_interval: Optional[int] = None
    log_interval: Optional[int] = None
    eval_iters: Optional[int] = None
    warmup_iters: Optional[int] = None
    lr_decay_iters: Optional[int] = None
    min_lr: Optional[float] = None

class UniversalExperimentSystem:
    """Универсальная система для проведения экспериментов"""
    
    def __init__(self, base_config: BaseModelConfig):
        self.base_config = base_config
        self.experiments: List[ExperimentConfig] = []
        self.results: Dict[str, Any] = {}
        
    def add_experiment(self, experiment: ExperimentConfig):
        """Добавляет эксперимент в систему"""
        self.experiments.append(experiment)
        
    def add_experiments_batch(self, experiments: List[ExperimentConfig]):
        """Добавляет несколько экспериментов"""
        self.experiments.extend(experiments)
        
    def generate_config_file(self, experiment: ExperimentConfig) -> str:
        """Генерирует файл конфигурации для эксперимента"""
        
        # Объединяем базовую конфигурацию с экспериментальной
        config_dict = asdict(self.base_config)
        exp_dict = asdict(experiment)
        
        # Удаляем служебные поля из эксперимента
        exp_dict.pop('name')
        exp_dict.pop('description')
        
        # Добавляем экспериментальные параметры
        config_dict.update({k: v for k, v in exp_dict.items() if v is not None})
        
        # Устанавливаем зависимые параметры
        config_dict['out_dir'] = f'out-universal-{experiment.name}'
        config_dict['always_save_checkpoint'] = True
        
        # Автоматические параметры на основе max_iters
        if experiment.eval_interval is None:
            config_dict['eval_interval'] = max(50, experiment.max_iters // 20)
        if experiment.log_interval is None:
            config_dict['log_interval'] = max(10, experiment.max_iters // 100)
        if experiment.eval_iters is None:
            config_dict['eval_iters'] = max(20, experiment.max_iters // 50)
        if experiment.warmup_iters is None:
            config_dict['warmup_iters'] = max(100, experiment.max_iters // 10)
        if experiment.lr_decay_iters is None:
            config_dict['lr_decay_iters'] = experiment.max_iters
        if experiment.min_lr is None:
            config_dict['min_lr'] = config_dict['learning_rate'] / 10
            
        # Создаем файл конфигурации
        config_path = f'config/universal_{experiment.name}.py'
        os.makedirs('config', exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(f"# Универсальная конфигурация: {experiment.description}\n")
            f.write(f"# Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in config_dict.items():
                if isinstance(value, str):
                    f.write(f"{key} = '{value}'\n")
                else:
                    f.write(f"{key} = {value}\n")
                    
        return config_path
        
    def run_experiment(self, experiment: ExperimentConfig) -> Dict[str, Any]:
        """Запускает один эксперимент"""
        
        print(f"\n🧪 ЭКСПЕРИМЕНТ: {experiment.name}")
        print("=" * 60)
        print(f"📝 Описание: {experiment.description}")
        print(f"📊 Итераций: {experiment.max_iters}")
        print(f"🧠 Архитектура: {self.base_config.n_layer} слоев, {self.base_config.n_head} голов, {self.base_config.n_embd} эмбеддингов")
        
        # Генерируем конфигурацию
        config_path = self.generate_config_file(experiment)
        print(f"📄 Конфиг: {config_path}")
        
        # Определяем скрипт обучения
        has_advanced_centering = any([
            experiment.center_v, experiment.center_mlp, 
            experiment.center_residual, experiment.center_embeddings
        ])
        
        if has_advanced_centering:
            train_script = 'train_advanced_centered.py'
        elif any([experiment.center_qk, experiment.center_final_output, experiment.center_block_output]):
            train_script = 'train_centered.py'
        else:
            train_script = 'train.py'
            
        print(f"🔧 Скрипт: {train_script}")
        
        # Запускаем обучение
        start_time = time.time()
        
        try:
            cmd = [
                'python', 'train_with_logging.py', 
                config_path, 
                f'--max_iters={experiment.max_iters}'
            ]
            
            print(f"🔄 Команда: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Эксперимент завершен за {training_time:.1f}с ({training_time/60:.1f} мин)")
                
                # Парсим результаты
                experiment_result = self.parse_experiment_results(experiment, training_time)
                return experiment_result
                
            else:
                print(f"❌ Ошибка обучения: {result.stderr[:200]}")
                return {
                    'name': experiment.name,
                    'success': False,
                    'error': result.stderr[:200],
                    'time': training_time
                }
                
        except subprocess.TimeoutExpired:
            print("⏱️  Таймаут эксперимента (60 мин)")
            return {
                'name': experiment.name,
                'success': False,
                'error': 'Timeout',
                'time': 3600
            }
        except Exception as e:
            print(f"❌ Исключение: {e}")
            return {
                'name': experiment.name,
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
            
    def parse_experiment_results(self, experiment: ExperimentConfig, training_time: float) -> Dict[str, Any]:
        """Парсит результаты эксперимента"""
        
        out_dir = f'out-universal-{experiment.name}'
        
        result = {
            'name': experiment.name,
            'description': experiment.description,
            'success': True,
            'time': training_time,
            'config': asdict(experiment),
            'base_config': asdict(self.base_config)
        }
        
        # Ищем лог файл
        log_files = [f for f in os.listdir('.') if f.startswith(f'training_log_universal_{experiment.name}')]
        if log_files:
            log_file = sorted(log_files)[-1]  # Берем последний
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Парсим последние метрики
                for line in reversed(lines):
                    if 'val_loss' in line and 'train_loss' in line:
                        # Простой парсинг метрик
                        parts = line.strip().split()
                        for i, part in enumerate(parts):
                            if part == 'val_loss' and i + 1 < len(parts):
                                try:
                                    result['final_val_loss'] = float(parts[i + 1])
                                except:
                                    pass
                            elif part == 'train_loss' and i + 1 < len(parts):
                                try:
                                    result['final_train_loss'] = float(parts[i + 1])
                                except:
                                    pass
                        break
                        
            except Exception as e:
                print(f"⚠️  Не удалось парсить лог: {e}")
                
        return result
        
    def run_all_experiments(self) -> Dict[str, Any]:
        """Запускает все эксперименты"""
        
        print("🧪 УНИВЕРСАЛЬНАЯ СИСТЕМА ЭКСПЕРИМЕНТОВ")
        print("=" * 80)
        print(f"🎯 Базовая архитектура: {self.base_config.n_layer} слоев, {self.base_config.n_head} голов, {self.base_config.n_embd} эмбеддингов")
        print(f"📊 Экспериментов: {len(self.experiments)}")
        
        total_iters = sum(exp.max_iters for exp in self.experiments)
        estimated_time = total_iters * 0.12  # ~0.12 сек на итерацию
        print(f"⏱️  Ожидаемое время: ~{estimated_time/60:.0f} минут")
        print("=" * 80)
        
        all_results = []
        start_time = time.time()
        
        for i, experiment in enumerate(self.experiments, 1):
            print(f"\nЭксперимент {i}/{len(self.experiments)}")
            result = self.run_experiment(experiment)
            all_results.append(result)
            
            # Небольшая пауза между экспериментами
            time.sleep(1)
            
        total_time = time.time() - start_time
        
        # Сохраняем результаты
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'base_config': asdict(self.base_config),
            'total_experiments': len(self.experiments),
            'total_time': total_time,
            'results': all_results
        }
        
        results_file = f'universal_experiments_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        print(f"\n🎉 ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
        print(f"⏱️  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
        print(f"💾 Результаты сохранены: {results_file}")
        
        # Анализируем результаты
        self.analyze_results(all_results)
        
        return results_summary
        
    def analyze_results(self, results: List[Dict[str, Any]]):
        """Анализирует результаты экспериментов"""
        
        print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 50)
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"✅ Успешных: {len(successful)}")
        print(f"❌ Неудачных: {len(failed)}")
        
        if successful:
            # Сортируем по validation loss
            successful_with_loss = [r for r in successful if 'final_val_loss' in r]
            if successful_with_loss:
                successful_with_loss.sort(key=lambda x: x['final_val_loss'])
                
                print(f"\n🏆 ТОП-5 ПО VALIDATION LOSS:")
                for i, result in enumerate(successful_with_loss[:5], 1):
                    val_loss = result['final_val_loss']
                    name = result['name']
                    description = result['description']
                    print(f"  {i}. {name}: {val_loss:.4f} - {description}")
                    
        if failed:
            print(f"\n❌ НЕУДАЧНЫЕ ЭКСПЕРИМЕНТЫ:")
            for result in failed:
                print(f"  • {result['name']}: {result['error']}")

def create_comprehensive_experiments() -> List[ExperimentConfig]:
    """Создает комплексный набор экспериментов"""
    
    experiments = []
    
    # 1. Базовые эксперименты
    experiments.extend([
        ExperimentConfig(
            name="baseline",
            description="Базовая модель без центрирования",
            max_iters=100
        ),
        ExperimentConfig(
            name="qk_centered",
            description="Только QK центрирование",
            max_iters=100,
            center_qk=True
        ),
        ExperimentConfig(
            name="value_centered", 
            description="Только Value центрирование",
            max_iters=100,
            center_v=True
        ),
        ExperimentConfig(
            name="embeddings_centered",
            description="Только Embeddings центрирование", 
            max_iters=100,
            center_embeddings=True
        )
    ])
    
    # 2. Комбинированные подходы
    experiments.extend([
        ExperimentConfig(
            name="qk_plus_value",
            description="QK + Value центрирование",
            max_iters=100,
            center_qk=True,
            center_v=True
        ),
        ExperimentConfig(
            name="qk_plus_embeddings",
            description="QK + Embeddings центрирование",
            max_iters=100,
            center_qk=True,
            center_embeddings=True
        ),
        ExperimentConfig(
            name="value_plus_embeddings",
            description="Value + Embeddings центрирование",
            max_iters=100,
            center_v=True,
            center_embeddings=True
        )
    ])
    
    # 3. Полные комбинации attention
    experiments.extend([
        ExperimentConfig(
            name="full_attention",
            description="Полное attention центрирование (Q+K+V)",
            max_iters=100,
            center_qk=True,
            center_v=True
        ),
        ExperimentConfig(
            name="attention_plus_embeddings",
            description="Полное attention + Embeddings",
            max_iters=100,
            center_qk=True,
            center_v=True,
            center_embeddings=True
        )
    ])
    
    # 4. MLP эксперименты
    experiments.extend([
        ExperimentConfig(
            name="mlp_centered",
            description="Только MLP центрирование",
            max_iters=100,
            center_mlp=True
        ),
        ExperimentConfig(
            name="qk_plus_mlp",
            description="QK + MLP центрирование",
            max_iters=100,
            center_qk=True,
            center_mlp=True
        ),
        ExperimentConfig(
            name="full_attention_plus_mlp",
            description="Полное attention + MLP",
            max_iters=100,
            center_qk=True,
            center_v=True,
            center_mlp=True
        )
    ])
    
    # 5. Residual эксперименты
    experiments.extend([
        ExperimentConfig(
            name="residual_centered",
            description="Только Residual центрирование",
            max_iters=100,
            center_residual=True
        ),
        ExperimentConfig(
            name="embeddings_plus_residual",
            description="Embeddings + Residual центрирование",
            max_iters=100,
            center_embeddings=True,
            center_residual=True
        )
    ])
    
    # 6. Агрессивные комбинации
    experiments.extend([
        ExperimentConfig(
            name="conservative_all",
            description="Консервативное: QK + Embeddings + MLP",
            max_iters=100,
            center_qk=True,
            center_embeddings=True,
            center_mlp=True
        ),
        ExperimentConfig(
            name="aggressive_all",
            description="Агрессивное: все кроме Residual",
            max_iters=100,
            center_qk=True,
            center_v=True,
            center_embeddings=True,
            center_mlp=True
        ),
        ExperimentConfig(
            name="maximum_centering",
            description="МАКСИМУМ: все виды центрирования",
            max_iters=100,
            center_qk=True,
            center_v=True,
            center_embeddings=True,
            center_mlp=True,
            center_residual=True
        )
    ])
    
    return experiments

def create_mode_experiments() -> List[ExperimentConfig]:
    """Создает эксперименты с разными режимами центрирования"""
    
    experiments = []
    modes = ['simple', 'adaptive', 'learnable_center', 'momentum']
    
    for mode in modes:
        experiments.extend([
            ExperimentConfig(
                name=f"qk_{mode}",
                description=f"QK центрирование в режиме {mode}",
                max_iters=500,
                center_qk=True,
                centering_mode=mode
            ),
            ExperimentConfig(
                name=f"embeddings_{mode}",
                description=f"Embeddings центрирование в режиме {mode}",
                max_iters=500,
                center_embeddings=True,
                centering_mode=mode
            )
        ])
    
    return experiments

def main():
    """Главная функция для запуска экспериментов"""
    
    print("🧪 УНИВЕРСАЛЬНАЯ СИСТЕМА ЭКСПЕРИМЕНТОВ С ЦЕНТРИРОВАНИЕМ")
    print("=" * 80)
    
    # Базовая конфигурация - одинаковая для всех
    base_config = BaseModelConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1
    )
    
    # Создаем систему
    system = UniversalExperimentSystem(base_config)
    
    # Выбираем набор экспериментов
    print("Выберите набор экспериментов:")
    print("1. Комплексные эксперименты (17 экспериментов, ~3 мин)")
    print("2. Эксперименты с режимами (8 экспериментов, ~1 мин)")
    print("3. Быстрые тесты (4 эксперимента, ~1 мин)")
    
    choice = input("Ваш выбор (1-3): ").strip()
    
    if choice == "1":
        experiments = create_comprehensive_experiments()
    elif choice == "2":
        experiments = create_mode_experiments()
    elif choice == "3":
        experiments = [
            ExperimentConfig("baseline_quick", "Быстрый baseline", 200),
            ExperimentConfig("qk_quick", "Быстрый QK", 200, center_qk=True),
            ExperimentConfig("embeddings_quick", "Быстрый Embeddings", 200, center_embeddings=True),
            ExperimentConfig("combined_quick", "Быстрый комбинированный", 200, center_qk=True, center_embeddings=True)
        ]
    else:
        print("Неверный выбор, используем быстрые тесты")
        experiments = create_comprehensive_experiments()[:4]
    
    # Добавляем эксперименты
    system.add_experiments_batch(experiments)
    
    # Запускаем все эксперименты
    results = system.run_all_experiments()
    
    return results

if __name__ == "__main__":
    main()
