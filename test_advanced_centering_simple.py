#!/usr/bin/env python3
"""
Упрощенное тестирование расширенного центрирования
"""

import torch
from model_advanced_centering import AdvancedGPT, AdvancedGPTConfig

def test_all_centering_modes():
    """Быстрый тест всех режимов центрирования"""
    print("🧪 Тестирование всех режимов центрирования")
    print("=" * 50)
    
    config = AdvancedGPTConfig(
        block_size=32,
        vocab_size=500,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False
    )
    
    x = torch.randint(0, config.vocab_size, (1, 16))
    
    # Тестируем все новые возможности
    test_configs = [
        ('Baseline', {}),
        ('QK Centered', {'center_qk': True}),
        ('Value Centered', {'center_v': True}),
        ('MLP Centered', {'center_mlp': True}),
        ('Embeddings Centered', {'center_embeddings': True}),
        ('Residual Centered', {'center_residual': True}),
        ('Full Centering', {
            'center_qk': True,
            'center_v': True,
            'center_mlp': True,
            'center_embeddings': True,
            'center_residual': True,
            'center_final_output': True
        })
    ]
    
    results = []
    
    for name, params in test_configs:
        print(f"\n🔬 {name}:")
        
        test_config = AdvancedGPTConfig(**{**config.__dict__, **params})
        
        try:
            model = AdvancedGPT(test_config)
            model.eval()
            
            # Прямой проход
            with torch.no_grad():
                logits, loss, stats = model(x, return_centering_stats=True)
                
                print(f"  ✅ Выход: {logits.shape}")
                print(f"  📊 Статистик: {len(stats)}")
                
                if stats:
                    # Показываем первые несколько статистик
                    for i, (stat_name, stat_data) in enumerate(stats[:2]):
                        print(f"    {stat_name}: norm={stat_data['centered_norm_mean']:.3f}")
                    if len(stats) > 2:
                        print(f"    ... и еще {len(stats) - 2}")
                
                results.append((name, len(stats), True))
                
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            results.append((name, 0, False))
    
    # Сводка
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print("=" * 30)
    
    successful = [r for r in results if r[2]]
    print(f"✅ Работает: {len(successful)}/{len(results)}")
    
    for name, stats_count, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}: {stats_count} статистик")
    
    return len(successful) == len(results)

def main():
    print("🚀 Быстрое тестирование расширенного центрирования")
    
    success = test_all_centering_modes()
    
    if success:
        print(f"\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ! Готово к экспериментам!")
        print(f"\n🎯 НОВЫЕ ВОЗМОЖНОСТИ:")
        print("• center_v: Центрирование Value векторов в attention")
        print("• center_mlp: Центрирование после GELU в MLP")
        print("• center_embeddings: Центрирование входных эмбеддингов")
        print("• center_residual: Центрирование residual connections")
        print("• center_final_output: Центрирование финального выхода")
        
        print(f"\n📝 Пример использования:")
        print("config = AdvancedGPTConfig(")
        print("    center_qk=True,      # Проверенное QK центрирование")
        print("    center_v=True,       # НОВОЕ: Value центрирование")
        print("    center_mlp=True,     # НОВОЕ: MLP центрирование")
        print("    centering_mode='adaptive'")
        print(")")
    else:
        print(f"\n⚠️  Есть проблемы, нужно исправить")

if __name__ == "__main__":
    main()
