#!/usr/bin/env python3
"""
Скрипт для экспериментов с центрированием векторов в nanoGPT
Тестирует разные варианты центрирования и сравнивает результаты
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Эксперименты для тестирования
EXPERIMENTS = [
    {
        'name': 'baseline',
        'description': 'Базовая модель без центрирования',
        'config': {
            'use_centered_attention': False,
            'center_qk': False,
            'center_block_output': False,
            'center_final_output': False,
        }
    },
    {
        'name': 'qk_centered',
        'description': 'Центрирование только query/key в attention',
        'config': {
            'use_centered_attention': True,
            'center_qk': True,
            'center_block_output': False,
            'center_final_output': False,
            'centering_mode': 'adaptive'
        }
    },
    {
        'name': 'final_centered',
        'description': 'Центрирование только финальных эмбеддингов',
        'config': {
            'use_centered_attention': False,
            'center_qk': False,
            'center_block_output': False,
            'center_final_output': True,
            'centering_mode': 'adaptive'
        }
    },
    {
        'name': 'full_centered',
        'description': 'Полное центрирование (QK + финальные эмбеддинги)',
        'config': {
            'use_centered_attention': True,
            'center_qk': True,
            'center_block_output': False,
            'center_final_output': True,
            'centering_mode': 'adaptive'
        }
    },
    {
        'name': 'momentum_centered',
        'description': 'Центрирование с momentum',
        'config': {
            'use_centered_attention': True,
            'center_qk': True,
            'center_block_output': False,
            'center_final_output': True,
            'centering_mode': 'momentum'
        }
    }
]

def create_config_file(experiment, base_config_path='config/train_shakespeare_char.py'):
    """Создает конфигурационный файл для эксперимента"""
    
    # Читаем базовый конфиг
    with open(base_config_path, 'r') as f:
        base_config = f.read()
    
    # Создаем новый конфиг
    config_lines = []
    config_lines.append(f"# Эксперимент: {experiment['name']}")
    config_lines.append(f"# {experiment['description']}")
    config_lines.append("")
    
    # Добавляем базовые настройки
    config_lines.append("# Базовые настройки")
    config_lines.append(f"out_dir = 'out-centering-{experiment['name']}'")
    config_lines.append("eval_interval = 250")
    config_lines.append("eval_iters = 200") 
    config_lines.append("log_interval = 10")
    config_lines.append("always_save_checkpoint = False")
    config_lines.append("")
    config_lines.append("wandb_log = False")
    config_lines.append(f"wandb_project = 'centering-experiments'")
    config_lines.append(f"wandb_run_name = '{experiment['name']}'")
    config_lines.append("")
    config_lines.append("dataset = 'shakespeare_char'")
    config_lines.append("gradient_accumulation_steps = 1")
    config_lines.append("batch_size = 64")
    config_lines.append("block_size = 256")
    config_lines.append("")
    config_lines.append("# Архитектура модели")
    config_lines.append("n_layer = 6")
    config_lines.append("n_head = 6")
    config_lines.append("n_embd = 384")
    config_lines.append("dropout = 0.2")
    config_lines.append("")
    config_lines.append("# Параметры обучения")
    config_lines.append("learning_rate = 1e-3")
    config_lines.append("max_iters = 1000  # Короткие эксперименты")
    config_lines.append("lr_decay_iters = 1000")
    config_lines.append("min_lr = 1e-4")
    config_lines.append("beta2 = 0.99")
    config_lines.append("warmup_iters = 100")
    config_lines.append("")
    
    # Добавляем параметры центрирования
    config_lines.append("# Параметры центрирования")
    for key, value in experiment['config'].items():
        if isinstance(value, str):
            config_lines.append(f"{key} = '{value}'")
        else:
            config_lines.append(f"{key} = {value}")
    
    config_content = '\n'.join(config_lines)
    
    # Сохраняем конфиг
    config_path = f"config/centering_{experiment['name']}.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_experiment(experiment, max_iters=1000):
    """Запускает один эксперимент"""
    print(f"\n🚀 Запускаю эксперимент: {experiment['name']}")
    print(f"📝 Описание: {experiment['description']}")
    
    # Создаем конфиг
    config_path = create_config_file(experiment)
    print(f"📄 Конфиг: {config_path}")
    
    # Запускаем обучение
    cmd = [
        'python', 'train_with_logging.py',
        config_path,
        f'--max_iters={max_iters}'
    ]
    
    print(f"🔄 Команда: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 час таймаут
        
        if result.returncode == 0:
            duration = time.time() - start_time
            print(f"✅ Эксперимент {experiment['name']} завершен за {duration:.1f}с")
            return True, duration
        else:
            print(f"❌ Эксперимент {experiment['name']} завершился с ошибкой:")
            print(result.stderr)
            return False, 0
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Эксперимент {experiment['name']} превысил таймаут")
        return False, 0
    except KeyboardInterrupt:
        print(f"🛑 Эксперимент {experiment['name']} прерван пользователем")
        return False, 0

def analyze_results():
    """Анализирует результаты всех экспериментов"""
    print("\n📊 Анализ результатов экспериментов:")
    
    results = []
    
    for experiment in EXPERIMENTS:
        out_dir = f"out-centering-{experiment['name']}"
        
        if os.path.exists(out_dir):
            # Ищем последний checkpoint
            checkpoints = list(Path(out_dir).glob("*.pt"))
            if checkpoints:
                print(f"✅ {experiment['name']}: Найден checkpoint")
                
                # Пытаемся найти логи в training_plots
                log_files = list(Path("training_plots").glob(f"*{experiment['name']}*.txt"))
                if log_files:
                    # Читаем последний отчет
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    try:
                        with open(latest_log, 'r') as f:
                            content = f.read()
                            # Извлекаем финальный loss
                            for line in content.split('\n'):
                                if 'Финальный training loss:' in line:
                                    final_loss = float(line.split(':')[1].strip())
                                    results.append({
                                        'name': experiment['name'],
                                        'description': experiment['description'],
                                        'final_loss': final_loss
                                    })
                                    break
                    except:
                        pass
            else:
                print(f"❌ {experiment['name']}: Checkpoint не найден")
        else:
            print(f"❌ {experiment['name']}: Директория не найдена")
    
    # Сортируем по loss
    if results:
        results.sort(key=lambda x: x['final_loss'])
        
        print(f"\n🏆 Рейтинг экспериментов по финальному loss:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']}: {result['final_loss']:.4f} - {result['description']}")
    
    return results

def generate_samples():
    """Генерирует образцы текста для сравнения качества"""
    print("\n🎭 Генерация образцов текста:")
    
    for experiment in EXPERIMENTS:
        out_dir = f"out-centering-{experiment['name']}"
        
        if os.path.exists(out_dir):
            print(f"\n--- {experiment['name']} ---")
            cmd = [
                'python', 'sample.py',
                f'--out_dir={out_dir}',
                '--num_samples=1',
                '--max_new_tokens=150'
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    # Извлекаем сгенерированный текст
                    output_lines = result.stdout.split('\n')
                    in_sample = False
                    for line in output_lines:
                        if '---------------' in line:
                            in_sample = not in_sample
                            continue
                        if in_sample and line.strip():
                            print(line)
                else:
                    print(f"Ошибка генерации для {experiment['name']}")
            except:
                print(f"Таймаут генерации для {experiment['name']}")

def main():
    print("🧪 Эксперименты с центрированием векторов в nanoGPT")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            analyze_results()
            return
        elif sys.argv[1] == 'sample':
            generate_samples()
            return
        elif sys.argv[1] == 'quick':
            max_iters = 500
        else:
            max_iters = int(sys.argv[1])
    else:
        max_iters = 1000
    
    print(f"🎯 Каждый эксперимент: {max_iters} итераций")
    print(f"📊 Всего экспериментов: {len(EXPERIMENTS)}")
    
    # Подготавливаем данные
    if not os.path.exists('data/shakespeare_char/train.bin'):
        print("📥 Подготавливаю данные Shakespeare...")
        subprocess.run(['python', 'data/shakespeare_char/prepare.py'])
    
    successful_experiments = 0
    total_time = 0
    
    # Запускаем эксперименты
    for i, experiment in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*60}")
        print(f"Эксперимент {i}/{len(EXPERIMENTS)}")
        
        success, duration = run_experiment(experiment, max_iters)
        if success:
            successful_experiments += 1
            total_time += duration
    
    print(f"\n{'='*60}")
    print(f"🎉 Завершено экспериментов: {successful_experiments}/{len(EXPERIMENTS)}")
    print(f"⏱️  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
    
    if successful_experiments > 0:
        print(f"\n📊 Для анализа результатов запустите:")
        print(f"python {sys.argv[0]} analyze")
        print(f"\n🎭 Для генерации образцов запустите:")
        print(f"python {sys.argv[0]} sample")

if __name__ == "__main__":
    main()
