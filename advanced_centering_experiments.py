#!/usr/bin/env python3
"""
Эксперименты с расширенным центрированием
Тестируем новые места применения центрирования
"""

import os
import subprocess
import time
from datetime import datetime

def create_experiment_configs():
    """Создает конфигурации для экспериментов с расширенным центрированием"""
    
    # Базовая конфигурация (основана на успешной train_shakespeare_bpe.py)
    base_config = """
# Базовая конфигурация для экспериментов с расширенным центрированием
out_dir = 'out-advanced-centering-{experiment_name}'
eval_interval = 50
log_interval = 10
eval_iters = 20
eval_only = False
always_save_checkpoint = True

# Данные
dataset = 'shakespeare'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 256

# Модель - уменьшенная для быстрых экспериментов
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.1

# Оптимизация
learning_rate = 3e-4
max_iters = {max_iters}
lr_decay_iters = {max_iters}
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 100
weight_decay = 1e-1

# Система
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Центрирование
{centering_params}
"""
    
    experiments = [
        {
            'name': 'baseline_advanced',
            'description': 'Базовая модель без центрирования (контроль)',
            'centering_params': '# Без центрирования'
        },
        {
            'name': 'qk_only_advanced',
            'description': 'Только QK центрирование (проверенное)',
            'centering_params': '''center_qk = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'value_centered',
            'description': 'НОВОЕ: Центрирование Value векторов',
            'centering_params': '''center_v = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'mlp_centered',
            'description': 'НОВОЕ: Центрирование MLP после GELU',
            'centering_params': '''center_mlp = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'embeddings_centered',
            'description': 'НОВОЕ: Центрирование входных эмбеддингов',
            'centering_params': '''center_embeddings = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'residual_centered',
            'description': 'НОВОЕ: Центрирование residual connections',
            'centering_params': '''center_residual = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'qk_plus_value',
            'description': 'QK + Value центрирование',
            'centering_params': '''center_qk = True
center_v = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'qk_plus_mlp',
            'description': 'QK + MLP центрирование',
            'centering_params': '''center_qk = True
center_mlp = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'full_attention_centered',
            'description': 'Полное центрирование attention (Q+K+V)',
            'centering_params': '''center_qk = True
center_v = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'conservative_centering',
            'description': 'Консервативное центрирование (QK + эмбеддинги)',
            'centering_params': '''center_qk = True
center_embeddings = True
centering_mode = 'adaptive' '''
        },
        {
            'name': 'aggressive_centering',
            'description': 'Агрессивное центрирование (все кроме residual)',
            'centering_params': '''center_qk = True
center_v = True
center_mlp = True
center_embeddings = True
center_final_output = True
centering_mode = 'adaptive' '''
        }
    ]
    
    return base_config, experiments

def run_experiments(max_iters=500):
    """Запускает серию экспериментов с расширенным центрированием"""
    
    print(f"🧪 ЭКСПЕРИМЕНТЫ С РАСШИРЕННЫМ ЦЕНТРИРОВАНИЕМ")
    print("=" * 60)
    print(f"🎯 Цель: Найти лучшие места для применения центрирования")
    print(f"📊 Итераций: {max_iters} каждый")
    print(f"⏱️  Ожидаемое время: ~{len(create_experiment_configs()[1]) * 3} минут")
    print("=" * 60)
    
    base_config, experiments = create_experiment_configs()
    
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
            # Используем train_centered.py для экспериментов с центрированием
            train_script = 'train_centered.py' if 'center_' in experiment['centering_params'] else 'train.py'
            
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
                print(f"Stderr: {result.stderr[:200]}...")
                results.append({
                    'name': experiment['name'],
                    'description': experiment['description'],
                    'success': False,
                    'error': result.stderr[:200]
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
            print(f"  • {result['name']}: {result['time']:.1f}с - {result['description']}")
    
    if failed:
        print(f"\n💥 Неудачные эксперименты:")
        for result in failed:
            print(f"  • {result['name']}: {result['error'][:100]}...")
    
    print(f"\n📊 Для анализа результатов запустите:")
    print(f"python advanced_centering_experiments.py analyze")
    
    return results

def analyze_results():
    """Анализирует результаты экспериментов с расширенным центрированием"""
    
    print(f"📊 АНАЛИЗ РЕЗУЛЬТАТОВ РАСШИРЕННОГО ЦЕНТРИРОВАНИЯ")
    print("=" * 60)
    
    base_config, experiments = create_experiment_configs()
    
    results = []
    
    for experiment in experiments:
        checkpoint_path = f"out-advanced-centering-{experiment['name']}/ckpt.pt"
        
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
    
    print(f"\n🏆 РЕЙТИНГ ПО VALIDATION LOSS:")
    print("-" * 60)
    
    baseline_loss = None
    for i, result in enumerate(results, 1):
        if 'baseline' in result['name']:
            baseline_loss = result['loss']
        
        print(f"{i:2d}. {result['name']:20s}: {result['loss']:.4f} - {result['description']}")
    
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
                print(f"{status} {result['name']:20s}: {improvement:+6.2f}% - {result['description']}")
        
        # Топ улучшения
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🥇 ТОП-3 УЛУЧШЕНИЯ:")
        for i, (name, improvement, desc) in enumerate(improvements[:3], 1):
            print(f"{i}. {name}: {improvement:+.2f}% - {desc}")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print("-" * 30)
    
    if len(results) >= 2:
        best = results[0]
        print(f"🏆 Лучший результат: {best['name']} ({best['loss']:.4f})")
        print(f"📝 Описание: {best['description']}")
        
        if baseline_loss and best['loss'] < baseline_loss:
            improvement = ((baseline_loss - best['loss']) / baseline_loss) * 100
            print(f"📈 Улучшение: {improvement:.2f}%")
        
        print(f"\n🎯 Следующие шаги:")
        print("1. Запустить лучшие модели на больше итераций")
        print("2. Протестировать генерацию текста")
        print("3. Комбинировать лучшие подходы")

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze_results()
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        max_iters = int(sys.argv[1])
        run_experiments(max_iters)
    else:
        print("🚀 Запуск экспериментов с расширенным центрированием")
        print("Использование:")
        print("  python advanced_centering_experiments.py [итерации]  # Запуск экспериментов")
        print("  python advanced_centering_experiments.py analyze     # Анализ результатов")
        print()
        
        # По умолчанию запускаем с 500 итерациями
        run_experiments(500)

if __name__ == "__main__":
    main()
