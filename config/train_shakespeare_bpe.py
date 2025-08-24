# Обучение BPE модели на Шекспире С НУЛЯ
# В отличие от finetune_shakespeare.py, здесь мы начинаем с нуля

out_dir = 'out-shakespeare-bpe'
eval_interval = 25
eval_iters = 20
log_interval = 10

# Обучение с нуля, не fine-tuning
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-bpe'
wandb_run_name = 'from-scratch'

# Используем BPE токенизацию (как GPT-2), но обучаем с нуля
dataset = 'shakespeare'
init_from = 'scratch'  # ← КЛЮЧЕВОЕ ОТЛИЧИЕ: с нуля, а не от GPT-2

# Параметры модели - делаем поменьше для быстрого обучения
gradient_accumulation_steps = 4
batch_size = 8
block_size = 512  # контекст 512 токенов (не символов!)

# Маленькая GPT модель для быстрого обучения
n_layer = 24
n_head = 12
n_embd = 768
dropout = 0.1

# Параметры обучения
learning_rate = 3e-4
max_iters = 2000
lr_decay_iters = 2000
min_lr = 3e-5
beta2 = 0.95

warmup_iters = 200
