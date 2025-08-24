#!/usr/bin/env python3
"""
Эксперименты с центрированием векторов на базе train_shakespeare_bpe.py
Более серьезные эксперименты с BPE токенизацией и 1000 итераций
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Эксперименты для тестирования (базируемся на train_shakespeare_bpe.py)
EXPERIMENTS = [
    {
        'name': 'baseline_bpe',
        'description': 'Базовая BPE модель без центрирования (24 слоя)',
        'config': {
            'use_centered_attention': False,
            'center_qk': False,
            'center_block_output': False,
            'center_final_output': False,
        }
    },
    {
        'name': 'qk_centered_bpe',
        'description': 'BPE модель с центрированием query/key в attention',
        'config': {
            'use_centered_attention': True,
            'center_qk': True,
            'center_block_output': False,
            'center_final_output': False,
            'centering_mode': 'adaptive'
        }
    },
    {
        'name': 'final_centered_bpe',
        'description': 'BPE модель с центрированием финальных эмбеддингов',
        'config': {
            'use_centered_attention': False,
            'center_qk': False,
            'center_block_output': False,
            'center_final_output': True,
            'centering_mode': 'adaptive'
        }
    },
    {
        'name': 'full_centered_bpe',
        'description': 'BPE модель с полным центрированием (QK + финальные)',
        'config': {
            'use_centered_attention': True,
            'center_qk': True,
            'center_block_output': False,
            'center_final_output': True,
            'centering_mode': 'adaptive'
        }
    },
    {
        'name': 'momentum_centered_bpe',
        'description': 'BPE модель с центрированием momentum (лучший из предыдущих)',
        'config': {
            'use_centered_attention': True,
            'center_qk': True,
            'center_block_output': False,
            'center_final_output': True,
            'centering_mode': 'momentum'
        }
    },
    {
        'name': 'block_centered_bpe',
        'description': 'BPE модель с центрированием выходов блоков',
        'config': {
            'use_centered_attention': False,
            'center_qk': False,
            'center_block_output': True,
            'center_final_output': False,
            'centering_mode': 'adaptive'
        }
    }
]

def create_config_file(experiment, base_config_path='config/train_shakespeare_bpe.py'):
    """Создает конфигурационный файл для эксперимента на базе BPE конфига"""
    
    # Читаем базовый конфиг
    with open(base_config_path, 'r') as f:
        base_config = f.read()
    
    # Создаем новый конфиг
    config_lines = []
    config_lines.append(f"# Эксперимент с центрированием: {experiment['name']}")
    config_lines.append(f"# {experiment['description']}")
    config_lines.append("# Базируется на train_shakespeare_bpe.py")
    config_lines.append("")
    
    # Добавляем базовые настройки из оригинального конфига
    config_lines.append("# Базовые настройки")
    config_lines.append(f"out_dir = 'out-centering-bpe-{experiment['name']}'")
    config_lines.append("eval_interval = 50  # чаще проверяем для 1000 итераций")
    config_lines.append("eval_iters = 20") 
    config_lines.append("log_interval = 10")
    config_lines.append("always_save_checkpoint = True  # Сохраняем checkpoint для анализа")
    config_lines.append("")
    config_lines.append("wandb_log = False")
    config_lines.append(f"wandb_project = 'centering-bpe-experiments'")
    config_lines.append(f"wandb_run_name = '{experiment['name']}'")
    config_lines.append("")
    config_lines.append("# Используем BPE токенизацию (как GPT-2), но обучаем с нуля")
    config_lines.append("dataset = 'shakespeare'")
    config_lines.append("init_from = 'scratch'  # с нуля, а не от GPT-2")
    config_lines.append("")
    config_lines.append("# Параметры модели")
    config_lines.append("gradient_accumulation_steps = 4")
    config_lines.append("batch_size = 8")
    config_lines.append("block_size = 512  # контекст 512 токенов")
    config_lines.append("")
    config_lines.append("# Архитектура - как в оригинальном BPE конфиге")
    config_lines.append("n_layer = 24")
    config_lines.append("n_head = 12")
    config_lines.append("n_embd = 768")
    config_lines.append("dropout = 0.1")
    config_lines.append("")
    config_lines.append("# Параметры обучения")
    config_lines.append("learning_rate = 3e-4")
    config_lines.append("max_iters = 1000  # Эксперименты с 1000 итераций")
    config_lines.append("lr_decay_iters = 1000")
    config_lines.append("min_lr = 3e-5")
    config_lines.append("beta2 = 0.95")
    config_lines.append("warmup_iters = 100  # меньше для 1000 итераций")
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
    config_path = f"config/centering_bpe_{experiment['name']}.py"
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 часа таймаут
        
        if result.returncode == 0:
            duration = time.time() - start_time
            print(f"✅ Эксперимент {experiment['name']} завершен за {duration:.1f}с ({duration/60:.1f} мин)")
            return True, duration
        else:
            print(f"❌ Эксперимент {experiment['name']} завершился с ошибкой:")
            print(result.stderr)
            return False, 0
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Эксперимент {experiment['name']} превысил таймаут (2 часа)")
        return False, 0
    except KeyboardInterrupt:
        print(f"🛑 Эксперимент {experiment['name']} прерван пользователем")
        return False, 0

def analyze_results():
    """Анализирует результаты всех BPE экспериментов"""
    print("\n📊 Анализ результатов BPE экспериментов:")
    
    results = []
    
    for experiment in EXPERIMENTS:
        out_dir = f"out-centering-bpe-{experiment['name']}"
        
        if os.path.exists(out_dir):
            # Ищем последний checkpoint
            checkpoints = list(Path(out_dir).glob("*.pt"))
            if checkpoints:
                print(f"✅ {experiment['name']}: Найден checkpoint")
                
                # Пытаемся извлечь loss из checkpoint
                try:
                    import torch
                    checkpoint_path = checkpoints[0]
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    final_loss = checkpoint.get('best_val_loss', None)
                    
                    if final_loss:
                        results.append({
                            'name': experiment['name'],
                            'description': experiment['description'],
                            'final_loss': final_loss
                        })
                except Exception as e:
                    print(f"⚠️  Ошибка чтения checkpoint для {experiment['name']}: {e}")
            else:
                print(f"❌ {experiment['name']}: Checkpoint не найден")
        else:
            print(f"❌ {experiment['name']}: Директория не найдена")
    
    # Сортируем по loss
    if results:
        results.sort(key=lambda x: x['final_loss'])
        
        print(f"\n🏆 Рейтинг BPE экспериментов по финальному validation loss:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']}: {result['final_loss']:.4f} - {result['description']}")
            
        # Вычисляем улучшения относительно baseline
        baseline_loss = None
        for result in results:
            if 'baseline' in result['name']:
                baseline_loss = result['final_loss']
                break
        
        if baseline_loss:
            print(f"\n📈 Улучшения относительно baseline ({baseline_loss:.4f}):")
            for result in results:
                if 'baseline' not in result['name']:
                    improvement = ((baseline_loss - result['final_loss']) / baseline_loss) * 100
                    sign = "+" if improvement > 0 else ""
                    print(f"  {result['name']}: {sign}{improvement:.2f}%")
    
    return results

def generate_samples():
    """Генерирует образцы текста для сравнения качества BPE моделей"""
    print("\n🎭 Генерация образцов текста (BPE модели):")
    
    for experiment in EXPERIMENTS:
        out_dir = f"out-centering-bpe-{experiment['name']}"
        
        if os.path.exists(out_dir):
            print(f"\n--- {experiment['name'].upper()} ---")
            cmd = [
                'python', 'sample.py',
                f'--out_dir={out_dir}',
                '--num_samples=1',
                '--max_new_tokens=100',
                '--start=ROMEO:'
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
    print("🧪 Эксперименты с центрированием векторов - BPE модели")
    print("=" * 70)
    print("📚 Базируемся на config/train_shakespeare_bpe.py")
    print("🎯 Архитектура: 24 слоя, 12 голов, 768 эмбеддингов")
    print("🔤 Токенизация: BPE (как GPT-2)")
    print("🚀 Обучение: с нуля (scratch)")
    
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
    
    # Подготавливаем данные Shakespeare BPE
    if not os.path.exists('data/shakespeare/train.bin'):
        print("📥 Подготавливаю данные Shakespeare BPE...")
        subprocess.run(['python', 'data/shakespeare/prepare.py'])
    
    successful_experiments = 0
    total_time = 0
    
    # Запускаем эксперименты
    for i, experiment in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"Эксперимент {i}/{len(EXPERIMENTS)}")
        
        success, duration = run_experiment(experiment, max_iters)
        if success:
            successful_experiments += 1
            total_time += duration
    
    print(f"\n{'='*70}")
    print(f"🎉 Завершено экспериментов: {successful_experiments}/{len(EXPERIMENTS)}")
    print(f"⏱️  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
    
    if successful_experiments > 0:
        print(f"\n📊 Для анализа результатов запустите:")
        print(f"python {sys.argv[0]} analyze")
        print(f"\n🎭 Для генерации образцов запустите:")
        print(f"python {sys.argv[0]} sample")

if __name__ == "__main__":
    main()
