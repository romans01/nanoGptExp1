#!/usr/bin/env python3
"""
Тестирование расширенного центрирования
"""

import torch
import torch.nn as nn
from model_advanced_centering import AdvancedGPT, AdvancedGPTConfig

def test_centering_modes():
    """Тестируем различные режимы центрирования"""
    print("🧪 Тестирование расширенного центрирования")
    print("=" * 50)
    
    # Базовая конфигурация для тестов
    base_config = AdvancedGPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False
    )
    
    # Тестовые данные
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
    
    # Тестируем разные комбинации центрирования
    test_configs = [
        {
            'name': 'Baseline (без центрирования)',
            'params': {}
        },
        {
            'name': 'QK Centered (проверенный)',
            'params': {'center_qk': True}
        },
        {
            'name': 'Value Centered (НОВОЕ)',
            'params': {'center_v': True}
        },
        {
            'name': 'MLP Centered (НОВОЕ)',
            'params': {'center_mlp': True}
        },
        {
            'name': 'Embeddings Centered (НОВОЕ)',
            'params': {'center_embeddings': True}
        },
        {
            'name': 'Residual Centered (НОВОЕ)',
            'params': {'center_residual': True}
        },
        {
            'name': 'Full Advanced Centering',
            'params': {
                'center_qk': True,
                'center_v': True,
                'center_mlp': True,
                'center_embeddings': True,
                'center_residual': True,
                'center_final_output': True
            }
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n🔬 Тестируем: {test_config['name']}")
        
        # Создаем конфиг
        config = AdvancedGPTConfig(**{**base_config.__dict__, **test_config['params']})
        
        try:
            # Создаем модель
            model = AdvancedGPT(config)
            model.eval()
            
            # Прямой проход
            with torch.no_grad():
                logits, loss = model(x)
                
                # Проход с статистикой центрирования
                logits_stats, loss_stats, centering_stats = model(x, return_centering_stats=True)
                
                # Проверяем, что результаты одинаковые
                assert torch.allclose(logits, logits_stats, atol=1e-6), "Результаты с/без статистики должны совпадать"
                
                print(f"  ✅ Форма выхода: {logits.shape}")
                print(f"  📊 Статистик центрирования: {len(centering_stats)}")
                
                # Выводим статистики центрирования
                if centering_stats:
                    print("  📈 Статистики центрирования:")
                    for stat_name, stats in centering_stats[:3]:  # Показываем первые 3
                        print(f"    {stat_name}: norm_mean={stats['centered_norm_mean']:.4f}, center_norm={stats['center_norm']:.4f}")
                    if len(centering_stats) > 3:
                        print(f"    ... и еще {len(centering_stats) - 3} статистик")
                
                # Проверяем градиенты на отдельной модели
                model_grad = AdvancedGPT(config)
                model_grad.train()
                logits_grad, _ = model_grad(x)
                loss_dummy = logits_grad.sum()
                loss_dummy.backward()
                
                grad_norms = []
                for name, param in model_grad.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                
                avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
                print(f"  🎯 Средняя норма градиентов: {avg_grad_norm:.6f}")
                
                results.append({
                    'name': test_config['name'],
                    'output_shape': logits.shape,
                    'centering_stats_count': len(centering_stats),
                    'avg_grad_norm': avg_grad_norm,
                    'success': True
                })
                
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            results.append({
                'name': test_config['name'],
                'success': False,
                'error': str(e)
            })
    
    # Сводка результатов
    print(f"\n📊 СВОДКА РЕЗУЛЬТАТОВ:")
    print("=" * 50)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Успешных тестов: {len(successful)}/{len(results)}")
    
    if successful:
        print("\n🏆 Успешные конфигурации:")
        for result in successful:
            print(f"  • {result['name']}: {result['centering_stats_count']} статистик, grad_norm={result['avg_grad_norm']:.6f}")
    
    if failed:
        print(f"\n❌ Неудачные тесты: {len(failed)}")
        for result in failed:
            print(f"  • {result['name']}: {result['error']}")
    
    return results

def test_centering_modes_comparison():
    """Сравниваем разные режимы центрирования"""
    print(f"\n🔬 Сравнение режимов центрирования")
    print("=" * 50)
    
    config = AdvancedGPTConfig(
        block_size=32,
        vocab_size=500,
        n_layer=1,
        n_head=2,
        n_embd=64,
        center_qk=True,
        centering_mode='adaptive'  # Будем менять этот параметр
    )
    
    modes = ['simple', 'adaptive', 'learnable_center', 'momentum']
    x = torch.randint(0, config.vocab_size, (1, 16))
    
    for mode in modes:
        print(f"\n🎯 Режим: {mode}")
        config.centering_mode = mode
        
        try:
            model = AdvancedGPT(config)
            model.eval()
            
            with torch.no_grad():
                logits, loss, stats = model(x, return_centering_stats=True)
                
                print(f"  ✅ Работает, статистик: {len(stats)}")
                if stats:
                    first_stat = stats[0][1]
                    print(f"  📊 Первая статистика: center_norm={first_stat['center_norm']:.4f}")
                    
        except Exception as e:
            print(f"  ❌ Ошибка в режиме {mode}: {e}")

def main():
    print("🚀 Запуск тестирования расширенного центрирования")
    
    # Основные тесты
    results = test_centering_modes()
    
    # Тесты режимов
    test_centering_modes_comparison()
    
    print(f"\n🎉 Тестирование завершено!")
    
    # Рекомендации
    successful_count = len([r for r in results if r['success']])
    if successful_count == len(results):
        print("✅ Все тесты прошли успешно! Можно переходить к экспериментам.")
    else:
        print("⚠️  Есть проблемы, нужно исправить перед экспериментами.")

if __name__ == "__main__":
    main()
