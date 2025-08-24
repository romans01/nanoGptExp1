# Финальная конфигурация: 🥉 3-е место: Агрессивное все кроме Residual (1000 итераций)
# Создано: 2025-08-25 00:02:04
# ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ на 1000 итераций

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
dataset = 'shakespeare'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 256
learning_rate = 0.0003
beta2 = 0.99
weight_decay = 0.1
device = 'cuda'
dtype = 'bfloat16'
compile = True
max_iters = 1000
center_qk = True
center_v = True
center_mlp = True
center_embeddings = True
center_residual = False
center_final_output = False
center_block_output = False
centering_mode = 'adaptive'
out_dir = 'out-final-aggressive_all_1000'
always_save_checkpoint = True
eval_interval = 50
log_interval = 10
eval_iters = 50
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 2.9999999999999997e-05
wandb_log = False
wandb_project = 'nanogpt-final'
wandb_run_name = 'aggressive_all_1000'
beta1 = 0.9
grad_clip = 1.0
