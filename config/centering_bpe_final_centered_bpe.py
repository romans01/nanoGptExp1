# Эксперимент с центрированием: final_centered_bpe
# BPE модель с центрированием финальных эмбеддингов
# Базируется на train_shakespeare_bpe.py

# Базовые настройки
out_dir = 'out-centering-bpe-final_centered_bpe'
eval_interval = 50  # чаще проверяем для 1000 итераций
eval_iters = 20
log_interval = 10
always_save_checkpoint = True  # Сохраняем checkpoint для анализа

wandb_log = False
wandb_project = 'centering-bpe-experiments'
wandb_run_name = 'final_centered_bpe'

# Используем BPE токенизацию (как GPT-2), но обучаем с нуля
dataset = 'shakespeare'
init_from = 'scratch'  # с нуля, а не от GPT-2

# Параметры модели
gradient_accumulation_steps = 4
batch_size = 8
block_size = 512  # контекст 512 токенов

# Архитектура - как в оригинальном BPE конфиге
n_layer = 24
n_head = 12
n_embd = 768
dropout = 0.1

# Параметры обучения
learning_rate = 3e-4
max_iters = 1000  # Эксперименты с 1000 итераций
lr_decay_iters = 1000
min_lr = 3e-5
beta2 = 0.95
warmup_iters = 100  # меньше для 1000 итераций

# Параметры центрирования
use_centered_attention = False
center_qk = False
center_block_output = False
center_final_output = True
centering_mode = 'adaptive'