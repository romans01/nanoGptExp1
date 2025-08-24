# Эксперимент 2K итераций: qk_centered_bpe_2k
# BPE модель с центрированием query/key (2000 итераций)
# Базируется на train_shakespeare_bpe.py

# Базовые настройки
out_dir = 'out-qk_centered_bpe_2k'
eval_interval = 100  # чаще проверяем для 2000 итераций
eval_iters = 50
log_interval = 10
always_save_checkpoint = True  # Сохраняем checkpoint для анализа

wandb_log = False
wandb_project = 'centering-2k-experiments'
wandb_run_name = 'qk_centered_bpe_2k'

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

# Параметры обучения - 2000 итераций
learning_rate = 3e-4
max_iters = 2000  # Длинные эксперименты
lr_decay_iters = 2000
min_lr = 3e-5
beta2 = 0.95
warmup_iters = 200  # больше для 2000 итераций

# Параметры центрирования
use_centered_attention = True
center_qk = True
center_block_output = False
center_final_output = False
centering_mode = 'adaptive'