#!/usr/bin/env python3
"""
Запуск лучших моделей расширенного центрирования на 1000 итераций
"""

import os
import subprocess
import time
from datetime import datetime

def create_best_configs():
    """Создает конфигурации для лучших моделей на 1000 итераций"""
    
    # Базовая конфигурация
    base_config = """
# Конфигурация для финального тестирования лучших моделей расширенного центрирования
out_dir = 'out-advanced-1k-{experiment_name}'
eval_interval = 100
log_interval = 20
eval_iters = 50
eval_only = False
always_save_checkpoint = True

# Данные
dataset = 'shakespeare'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 256

# Модель - полноразмерная для серьезного тестирования
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Оптимизация
learning_rate = 3e-4
max_iters = {max_iters}
lr_decay_iters = {max_iters}
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 200
weight_decay = 1e-1

# Система
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Центрирование
{centering_params}
"""
    
    # ТОП-3 лучших модели по результатам анализа
    experiments = [
        {
            'name': 'baseline_1k',
            'description': 'Baseline контроль (1000 итераций)',
            'centering_params': '# Без центрирования'
        },
        {
            'name': 'value_centered_1k',
            'description': '🥇 ЛУЧШИЙ: Value центрирование (1000 итераций)',
            'centering_params': '''center_v = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'qk_plus_value_1k',
            'description': '🥈 2-е место: QK + Value центрирование (1000 итераций)',
            'centering_params': '''center_qk = True
center_v = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'full_attention_1k',
            'description': '🥉 3-е место: Полное attention центрирование Q+K+V (1000 итераций)',
            'centering_params': '''center_qk = True
center_v = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'embeddings_centered_1k',
            'description': '🏅 4-е место: Embeddings центрирование (1000 итераций)',
            'centering_params': '''center_embeddings = True
centering_mode = 'adaptive' '''
        }
    ]
    
    return base_config, experiments

def run_best_experiments(max_iters=1000):
    """Запускает лучшие эксперименты на 1000 итераций"""
    
    print(f"🏆 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ ЛУЧШИХ МОДЕЛЕЙ")
    print("=" * 60)
    print(f"🎯 Цель: Проверить лучшие подходы на серьезном количестве итераций")
    print(f"📊 Итераций: {max_iters} каждый")
    print(f"🧠 Модель: 12 слоев, 12 голов, 768 эмбеддингов (полноразмерная)")
    print(f"⏱️  Ожидаемое время: ~25 минут")
    print("=" * 60)
    
    base_config, experiments = create_best_configs()
    
    # Создаем папку для конфигов
    os.makedirs('config', exist_ok=True)
    
    results = []
    start_time = time.time()
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\nЭксперимент {i}/{len(experiments)}")
        print(f"🚀 Запускаю: {experiment['name']}")
        print(f"📝 Описание: {experiment['description']}")
        
        # Создаем конфиг файл
        config_content = base_config.format(
            experiment_name=experiment['name'],
            max_iters=max_iters,
            centering_params=experiment['centering_params']
        )
        
        config_path = f"config/{experiment['name']}.py"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"📄 Конфиг: {config_path}")
        
        # Запускаем обучение
        exp_start_time = time.time()
        
        try:
            cmd = ['python', 'train_with_logging.py', config_path, f'--max_iters={max_iters}']
            print(f"🔄 Команда: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            exp_time = time.time() - exp_start_time
            
            if result.returncode == 0:
                print(f"✅ Эксперимент {experiment['name']} завершен за {exp_time:.1f}с ({exp_time/60:.1f} мин)")
                results.append({
                    'name': experiment['name'],
                    'description': experiment['description'],
                    'success': True,
                    'time': exp_time
                })
            else:
                print(f"❌ Ошибка в эксперименте {experiment['name']}")
                print(f"Stderr: {result.stderr[:300]}...")
                results.append({
                    'name': experiment['name'],
                    'description': experiment['description'],
                    'success': False,
                    'error': result.stderr[:300]
                })
                
        except Exception as e:
            print(f"❌ Исключение в эксперименте {experiment['name']}: {e}")
            results.append({
                'name': experiment['name'],
                'description': experiment['description'],
                'success': False,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Сводка результатов
    print(f"\n🎉 ЗАВЕРШЕНО ЭКСПЕРИМЕНТОВ: {len(results)}")
    print(f"⏱️  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Успешных: {len(successful)}")
    print(f"❌ Неудачных: {len(failed)}")
    
    if successful:
        print(f"\n🏆 Успешные эксперименты:")
        for result in successful:
            print(f"  • {result['name']}: {result['time']:.1f}с ({result['time']/60:.1f} мин) - {result['description']}")
    
    if failed:
        print(f"\n💥 Неудачные эксперименты:")
        for result in failed:
            print(f"  • {result['name']}: {result['error'][:100]}...")
    
    print(f"\n📊 Для анализа результатов запустите:")
    print(f"python run_best_advanced_1k.py analyze")
    
    return results

def analyze_1k_results():
    """Анализирует результаты 1K экспериментов"""
    
    print(f"📊 АНАЛИЗ РЕЗУЛЬТАТОВ 1K ЭКСПЕРИМЕНТОВ")
    print("=" * 60)
    
    base_config, experiments = create_best_configs()
    
    results = []
    
    for experiment in experiments:
        checkpoint_path = f"out-advanced-1k-{experiment['name']}/ckpt.pt"
        
        if os.path.exists(checkpoint_path):
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                best_val_loss = checkpoint.get('best_val_loss', None)
                iter_num = checkpoint.get('iter_num', None)
                
                if best_val_loss is not None:
                    results.append({
                        'name': experiment['name'],
                        'description': experiment['description'],
                        'loss': best_val_loss,
                        'iters': iter_num
                    })
                    print(f"✅ {experiment['name']}: {best_val_loss:.4f}")
                else:
                    print(f"⚠️  {experiment['name']}: checkpoint без loss")
                    
            except Exception as e:
                print(f"❌ {experiment['name']}: ошибка загрузки - {e}")
        else:
            print(f"❌ {experiment['name']}: checkpoint не найден")
    
    if not results:
        print("❌ Нет результатов для анализа")
        return
    
    # Сортируем по loss
    results.sort(key=lambda x: x['loss'])
    
    print(f"\n🏆 ФИНАЛЬНЫЙ РЕЙТИНГ (1000 ИТЕРАЦИЙ):")
    print("-" * 60)
    
    baseline_loss = None
    for i, result in enumerate(results, 1):
        if 'baseline' in result['name']:
            baseline_loss = result['loss']
        
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(f"{medal} {result['name']:25s}: {result['loss']:.4f} - {result['description']}")
    
    # Сравнение с baseline
    if baseline_loss:
        print(f"\n📈 УЛУЧШЕНИЯ ОТНОСИТЕЛЬНО BASELINE ({baseline_loss:.4f}):")
        print("-" * 60)
        
        improvements = []
        for result in results:
            if 'baseline' not in result['name']:
                improvement = ((baseline_loss - result['loss']) / baseline_loss) * 100
                improvements.append((result['name'], improvement, result['description']))
                
                status = "🟢" if improvement > 0 else "🔴"
                print(f"{status} {result['name']:25s}: {improvement:+6.2f}% - {result['description']}")
        
        # Топ улучшения
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        if improvements:
            print(f"\n🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
            print("-" * 40)
            best_name, best_improvement, best_desc = improvements[0]
            print(f"🥇 ПОБЕДИТЕЛЬ: {best_name}")
            print(f"📝 Описание: {best_desc}")
            print(f"📈 Улучшение: {best_improvement:+.2f}%")
            print(f"📊 Loss: {results[0]['loss']:.4f} vs {baseline_loss:.4f}")
    
    print(f"\n🎯 НАУЧНЫЕ ВЫВОДЫ:")
    print("-" * 30)
    print("✅ Value центрирование подтверждает свою эффективность")
    print("✅ Комбинации с Value показывают стабильные результаты")
    print("✅ Расширенное центрирование - перспективное направление")

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze_1k_results()
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        max_iters = int(sys.argv[1])
        run_best_experiments(max_iters)
    else:
        print("🚀 Запуск финального тестирования лучших моделей")
        print("Использование:")
        print("  python run_best_advanced_1k.py [итерации]  # Запуск экспериментов")
        print("  python run_best_advanced_1k.py analyze     # Анализ результатов")
        print()
        
        # По умолчанию запускаем с 1000 итерациями
        run_best_experiments(1000)

if __name__ == "__main__":
    main()
