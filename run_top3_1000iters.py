#!/usr/bin/env python3
"""
Финальное тестирование топ-3 методов на 1000 итераций
"""

import os
import json
import time
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента"""
    name: str
    description: str
    max_iters: int = 1000
    
    # Центрирование
    center_qk: bool = False
    center_v: bool = False
    center_mlp: bool = False
    center_embeddings: bool = False
    center_residual: bool = False
    center_final_output: bool = False
    center_block_output: bool = False
    centering_mode: str = 'adaptive'

@dataclass
class BaseModelConfig:
    """Базовая конфигурация модели"""
    n_layer: int = 12
    n_head: int = 12  
    n_embd: int = 768
    dropout: float = 0.1
    
    dataset: str = 'shakespeare'
    gradient_accumulation_steps: int = 4
    batch_size: int = 8
    block_size: int = 256
    
    learning_rate: float = 3e-4
    beta2: float = 0.99
    weight_decay: float = 1e-1
    
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True

def create_top3_experiments() -> List[ExperimentConfig]:
    """Создает топ-3 эксперимента по результатам 500 итераций"""
    
    return [
        ExperimentConfig(
            name="baseline_1000",
            description="🥇 ЛУЧШИЙ: Baseline без центрирования (1000 итераций)"
        ),
        ExperimentConfig(
            name="value_centered_1000",
            description="🥈 2-е место: Value центрирование (1000 итераций)",
            center_v=True
        ),
        ExperimentConfig(
            name="aggressive_all_1000", 
            description="🥉 3-е место: Агрессивное все кроме Residual (1000 итераций)",
            center_qk=True,
            center_v=True,
            center_embeddings=True,
            center_mlp=True
        )
    ]

def generate_config_file(experiment: ExperimentConfig, base_config: BaseModelConfig) -> str:
    """Генерирует файл конфигурации для 1000 итераций"""
    
    # Объединяем конфигурации
    config_dict = asdict(base_config)
    exp_dict = asdict(experiment)
    
    # Удаляем служебные поля
    exp_dict.pop('name')
    exp_dict.pop('description')
    
    # Добавляем экспериментальные параметры
    config_dict.update({k: v for k, v in exp_dict.items() if v is not None})
    
    # Настройки для 1000 итераций
    config_dict.update({
        'out_dir': f'out-final-{experiment.name}',
        'always_save_checkpoint': True,
        'eval_interval': 50,   # Каждые 50 итераций
        'log_interval': 10,    # Каждые 10 итераций
        'eval_iters': 50,      # 50 итераций для оценки
        'warmup_iters': 100,   # 10% от общего количества
        'lr_decay_iters': experiment.max_iters,
        'min_lr': config_dict['learning_rate'] / 10,
        
        # Отключаем wandb
        'wandb_log': False,
        'wandb_project': 'nanogpt-final',
        'wandb_run_name': experiment.name,
        
        # Дополнительные настройки для стабильности
        'beta1': 0.9,
        'grad_clip': 1.0,
    })
    
    # Создаем файл конфигурации
    config_path = f'config/final_{experiment.name}.py'
    os.makedirs('config', exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(f"# Финальная конфигурация: {experiment.description}\n")
        f.write(f"# Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ на 1000 итераций\n\n")
        
        for key, value in config_dict.items():
            if isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            else:
                f.write(f"{key} = {value}\n")
                
    return config_path

def run_experiment_with_detailed_logging(experiment: ExperimentConfig, base_config: BaseModelConfig) -> Dict[str, Any]:
    """Запускает эксперимент с детальным логированием"""
    
    print(f"\n🏆 ФИНАЛЬНЫЙ ЭКСПЕРИМЕНТ: {experiment.name}")
    print("=" * 80)
    print(f"📝 Описание: {experiment.description}")
    print(f"📊 Итераций: {experiment.max_iters}")
    print(f"🧠 Архитектура: {base_config.n_layer} слоев, {base_config.n_head} голов, {base_config.n_embd} эмбеддингов")
    
    # Генерируем конфигурацию
    config_path = generate_config_file(experiment, base_config)
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
    
    # Создаем уникальный лог файл
    log_file = f'training_log_final_{experiment.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Запускаем обучение
    start_time = time.time()
    
    try:
        cmd = [
            'python', train_script,
            config_path,
            f'--max_iters={experiment.max_iters}',
            '--wandb_log=False'
        ]
        
        print(f"🔄 Команда: {' '.join(cmd)}")
        print(f"📝 Лог файл: {log_file}")
        print(f"⏱️  Ожидаемое время: ~{experiment.max_iters * 0.12 / 60:.1f} минут")
        
        # Запускаем с сохранением вывода и показом прогресса
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                universal_newlines=True
            )
            
            # Показываем прогресс в реальном времени
            last_progress_time = time.time()
            for line in process.stdout:
                f.write(line)
                f.flush()
                
                # Показываем прогресс каждые 30 секунд
                current_time = time.time()
                if current_time - last_progress_time > 30:
                    elapsed = current_time - start_time
                    print(f"   ⏱️  Прошло: {elapsed/60:.1f} мин, продолжается...")
                    last_progress_time = current_time
                
                # Показываем важные строки
                if 'step' in line and ('val_loss' in line or 'train_loss' in line):
                    print(f"   📊 {line.strip()}")
            
            process.wait()
            
        training_time = time.time() - start_time
        
        if process.returncode == 0:
            print(f"✅ Эксперимент завершен за {training_time:.1f}с ({training_time/60:.1f} мин)")
            
            # Парсим результаты
            experiment_result = parse_final_results(experiment, training_time, log_file)
            return experiment_result
            
        else:
            print(f"❌ Ошибка обучения")
            return {
                'name': experiment.name,
                'success': False,
                'error': 'Training failed',
                'time': training_time,
                'log_file': log_file
            }
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return {
            'name': experiment.name,
            'success': False,
            'error': str(e),
            'time': time.time() - start_time,
            'log_file': log_file
        }

def parse_final_results(experiment: ExperimentConfig, training_time: float, log_file: str) -> Dict[str, Any]:
    """Парсит финальные результаты"""
    
    result = {
        'name': experiment.name,
        'description': experiment.description,
        'success': True,
        'time': training_time,
        'config': asdict(experiment),
        'log_file': log_file
    }
    
    # Парсим лог файл для получения истории обучения
    training_history = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'step' in line and 'val_loss' in line and 'train_loss' in line:
                    try:
                        # Парсим строку типа: "step 100: train loss 4.1234, val loss 4.5678"
                        parts = line.split()
                        step = None
                        train_loss = None
                        val_loss = None
                        
                        for i, part in enumerate(parts):
                            if part == 'step' and i + 1 < len(parts):
                                step_str = parts[i + 1].rstrip(':')
                                step = int(step_str)
                            elif part == 'loss' and i > 0 and parts[i-1] == 'train':
                                if i + 1 < len(parts):
                                    train_loss_str = parts[i + 1].rstrip(',')
                                    train_loss = float(train_loss_str)
                            elif part == 'loss' and i > 0 and parts[i-1] == 'val':
                                if i + 1 < len(parts):
                                    val_loss_str = parts[i + 1].rstrip(',')
                                    val_loss = float(val_loss_str)
                        
                        if step is not None and train_loss is not None and val_loss is not None:
                            training_history.append({
                                'step': step,
                                'train_loss': train_loss,
                                'val_loss': val_loss
                            })
                            
                    except (ValueError, IndexError):
                        continue
                        
        result['training_history'] = training_history
        
        # Финальные метрики
        if training_history:
            final_metrics = training_history[-1]
            result['final_train_loss'] = final_metrics['train_loss']
            result['final_val_loss'] = final_metrics['val_loss']
            result['final_step'] = final_metrics['step']
            
    except Exception as e:
        print(f"⚠️  Ошибка парсинга лога: {e}")
    
    # Парсим чекпоинт
    out_dir = f'out-final-{experiment.name}'
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    
    if os.path.exists(ckpt_path):
        try:
            import torch
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            if 'best_val_loss' in checkpoint:
                result['checkpoint_val_loss'] = float(checkpoint['best_val_loss'])
            if 'iter_num' in checkpoint:
                result['checkpoint_final_iter'] = int(checkpoint['iter_num'])
                
            print(f"📊 Чекпоинт: val_loss={checkpoint.get('best_val_loss', 'N/A'):.4f}, iter={checkpoint.get('iter_num', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️  Ошибка загрузки чекпоинта: {e}")
    
    return result

def main():
    """Главная функция"""
    
    print("🏆 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ ТОП-3 НА 1000 ИТЕРАЦИЙ")
    print("=" * 80)
    print("🎯 Цель: Окончательная проверка лучших методов на максимальной дистанции")
    print("📊 Итераций: 1000 каждый")
    print("🧠 Модель: 12 слоев, 12 голов, 768 эмбеддингов")
    print("🔬 Проверяем гипотезу о долгосрочной эффективности baseline")
    print("=" * 80)
    
    # Базовая конфигурация
    base_config = BaseModelConfig()
    
    # Создаем эксперименты
    experiments = create_top3_experiments()
    
    print(f"📋 Финальные эксперименты:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")
        print(f"     📝 {exp.description}")
    
    estimated_time = len(experiments) * 1000 * 0.12  # ~0.12 сек на итерацию
    print(f"⏱️  Ожидаемое время: ~{estimated_time/60:.0f} минут")
    print("=" * 80)
    
    # Запускаем эксперименты
    all_results = []
    start_time = time.time()
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n🚀 Эксперимент {i}/{len(experiments)}")
        result = run_experiment_with_detailed_logging(experiment, base_config)
        all_results.append(result)
        
        # Пауза между экспериментами
        if i < len(experiments):
            print(f"⏸️  Пауза 5 секунд перед следующим экспериментом...")
            time.sleep(5)
        
    total_time = time.time() - start_time
    
    # Сохраняем результаты
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'final_top3_1000iters',
        'base_config': asdict(base_config),
        'total_experiments': len(experiments),
        'total_time': total_time,
        'results': all_results
    }
    
    results_file = f'final_top3_1000iters_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
        
    print(f"\n🎉 ВСЕ ФИНАЛЬНЫЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print(f"⏱️  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
    print(f"💾 Результаты сохранены: {results_file}")
    
    # Финальный анализ
    analyze_final_results(all_results)
    
    return results_summary

def analyze_final_results(results: List[Dict[str, Any]]):
    """Финальный анализ результатов"""
    
    print(f"\n🏆 ФИНАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ (1000 ИТЕРАЦИЙ)")
    print("=" * 70)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Успешных: {len(successful)}")
    print(f"❌ Неудачных: {len(failed)}")
    
    if successful:
        # Сортируем по validation loss
        successful_with_loss = [r for r in successful if 'final_val_loss' in r]
        if successful_with_loss:
            successful_with_loss.sort(key=lambda x: x['final_val_loss'])
            
            print(f"\n🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (1000 итераций):")
            print("-" * 50)
            
            baseline_loss = None
            for r in successful_with_loss:
                if 'baseline' in r['name']:
                    baseline_loss = r['final_val_loss']
                    break
            
            for i, result in enumerate(successful_with_loss, 1):
                val_loss = result['final_val_loss']
                name = result['name']
                description = result['description']
                time_min = result['time'] / 60
                
                # Вычисляем улучшение относительно baseline
                improvement = ""
                if baseline_loss and val_loss != baseline_loss:
                    pct_change = ((baseline_loss - val_loss) / baseline_loss) * 100
                    improvement = f" ({pct_change:+.2f}%)"
                
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                print(f"{medal} {name}: {val_loss:.4f}{improvement}")
                print(f"    📝 {description}")
                print(f"    ⏱️  Время: {time_min:.1f} минут")
                
                # Показываем историю конвергенции
                if 'training_history' in result and result['training_history']:
                    history = result['training_history']
                    if len(history) >= 2:
                        start_loss = history[0]['val_loss']
                        final_loss = history[-1]['val_loss']
                        convergence = start_loss - final_loss
                        print(f"    📈 Конвергенция: {start_loss:.4f} → {final_loss:.4f} (улучшение {convergence:.4f})")
                print()
                
    if failed:
        print(f"\n❌ НЕУДАЧНЫЕ ЭКСПЕРИМЕНТЫ:")
        for result in failed:
            print(f"  • {result['name']}: {result['error']}")
            
    print(f"\n🔬 ВЫВОДЫ:")
    print("• Проверили гипотезу о долгосрочной эффективности на максимальной дистанции")
    print("• Получили окончательные результаты для научных выводов")
    print("• Данные готовы для финального отчета и публикации")

if __name__ == "__main__":
    main()
