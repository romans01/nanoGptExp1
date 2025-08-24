#!/usr/bin/env python3
"""
Эксперимент с 2 лучшими моделями на 2000 итераций
Сравниваем baseline vs qk_centered для более точной оценки
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Только 2 лучших эксперимента
EXPERIMENTS = [
    {
        'name': 'baseline_bpe_2k',
        'description': 'Базовая BPE модель без центрирования (2000 итераций)',
        'config': {
            'use_centered_attention': False,
            'center_qk': False,
            'center_block_output': False,
            'center_final_output': False,
        }
    },
    {
        'name': 'qk_centered_bpe_2k',
        'description': 'BPE модель с центрированием query/key (2000 итераций)',
        'config': {
            'use_centered_attention': True,
            'center_qk': True,
            'center_block_output': False,
            'center_final_output': False,
            'centering_mode': 'adaptive'
        }
    }
]

def create_config_file(experiment, base_config_path='config/train_shakespeare_bpe.py'):
    """Создает конфигурационный файл для эксперимента на 2000 итераций"""
    
    config_lines = []
    config_lines.append(f"# Эксперимент 2K итераций: {experiment['name']}")
    config_lines.append(f"# {experiment['description']}")
    config_lines.append("# Базируется на train_shakespeare_bpe.py")
    config_lines.append("")
    
    # Добавляем базовые настройки
    config_lines.append("# Базовые настройки")
    config_lines.append(f"out_dir = 'out-{experiment['name']}'")
    config_lines.append("eval_interval = 100  # чаще проверяем для 2000 итераций")
    config_lines.append("eval_iters = 50") 
    config_lines.append("log_interval = 10")
    config_lines.append("always_save_checkpoint = True  # Сохраняем checkpoint для анализа")
    config_lines.append("")
    config_lines.append("wandb_log = False")
    config_lines.append(f"wandb_project = 'centering-2k-experiments'")
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
    config_lines.append("# Параметры обучения - 2000 итераций")
    config_lines.append("learning_rate = 3e-4")
    config_lines.append("max_iters = 2000  # Длинные эксперименты")
    config_lines.append("lr_decay_iters = 2000")
    config_lines.append("min_lr = 3e-5")
    config_lines.append("beta2 = 0.95")
    config_lines.append("warmup_iters = 200  # больше для 2000 итераций")
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
    config_path = f"config/{experiment['name']}.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_experiment(experiment, max_iters=2000):
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)  # 3 часа таймаут
        
        if result.returncode == 0:
            duration = time.time() - start_time
            print(f"✅ Эксперимент {experiment['name']} завершен за {duration:.1f}с ({duration/60:.1f} мин)")
            return True, duration
        else:
            print(f"❌ Эксперимент {experiment['name']} завершился с ошибкой:")
            print(result.stderr)
            return False, 0
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Эксперимент {experiment['name']} превысил таймаут (3 часа)")
        return False, 0
    except KeyboardInterrupt:
        print(f"🛑 Эксперимент {experiment['name']} прерван пользователем")
        return False, 0

def analyze_results():
    """Анализирует результаты 2K экспериментов"""
    print("\n📊 Анализ результатов 2K экспериментов:")
    
    results = []
    
    for experiment in EXPERIMENTS:
        out_dir = f"out-{experiment['name']}"
        
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
        
        print(f"\n🏆 Рейтинг 2K экспериментов по финальному validation loss:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']}: {result['final_loss']:.4f} - {result['description']}")
            
        # Вычисляем улучшения
        if len(results) == 2:
            baseline_loss = None
            qk_loss = None
            
            for result in results:
                if 'baseline' in result['name']:
                    baseline_loss = result['final_loss']
                elif 'qk_centered' in result['name']:
                    qk_loss = result['final_loss']
            
            if baseline_loss and qk_loss:
                improvement = ((baseline_loss - qk_loss) / baseline_loss) * 100
                print(f"\n📈 QK Centered улучшение относительно Baseline:")
                print(f"   Baseline: {baseline_loss:.4f}")
                print(f"   QK Centered: {qk_loss:.4f}")
                print(f"   Улучшение: {improvement:+.2f}%")
    
    return results

def compare_generation():
    """Сравнивает качество генерации 2K моделей"""
    print("\n🎭 Сравнение генерации 2K моделей:")
    
    prompts = ["ROMEO:", "JULIET:", "To be or not to be"]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"🎯 Промпт: '{prompt}'")
        print(f"{'='*60}")
        
        for experiment in EXPERIMENTS:
            out_dir = f"out-{experiment['name']}"
            
            if os.path.exists(f"{out_dir}/ckpt.pt"):
                print(f"\n🎭 {experiment['name'].upper()}:")
                print("-" * 40)
                
                cmd = [
                    'python', 'sample_centered.py',
                    f'--out_dir={out_dir}',
                    '--num_samples=1',
                    '--max_new_tokens=100',
                    f'--start={prompt}',
                    '--temperature=0.8'
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        # Извлекаем сгенерированный текст
                        output_lines = result.stdout.split('\n')
                        
                        start_found = False
                        for line in output_lines:
                            if prompt in line and not start_found:
                                start_found = True
                                print(line)
                            elif start_found and '---------------' in line:
                                break
                            elif start_found:
                                print(line)
                    else:
                        print("Ошибка генерации")
                        
                except:
                    print("Таймаут генерации")

def main():
    print("🧪 Эксперименты с 2 лучшими моделями - 2000 итераций")
    print("=" * 70)
    print("🎯 Сравниваем: Baseline vs QK Centered")
    print("📊 Итераций: 2000 каждый")
    print("⏱️  Ожидаемое время: ~1 час")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            analyze_results()
            return
        elif sys.argv[1] == 'compare':
            compare_generation()
            return
    
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
        
        success, duration = run_experiment(experiment, 2000)
        if success:
            successful_experiments += 1
            total_time += duration
    
    print(f"\n{'='*70}")
    print(f"🎉 Завершено экспериментов: {successful_experiments}/{len(EXPERIMENTS)}")
    print(f"⏱️  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
    
    if successful_experiments > 0:
        print(f"\n📊 Для анализа результатов запустите:")
        print(f"python {sys.argv[0]} analyze")
        print(f"\n🎭 Для сравнения генерации запустите:")
        print(f"python {sys.argv[0]} compare")

if __name__ == "__main__":
    main()
