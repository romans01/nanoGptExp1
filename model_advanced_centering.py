"""
nanoGPT model with advanced vector centering
Расширенная версия с центрированием в разных частях архитектуры
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class VectorCentering(nn.Module):
    """
    Универсальный модуль для центрирования векторов
    Поддерживает различные режимы центрирования
    """
    
    def __init__(self, config, mode='adaptive', momentum=0.9, feature_dim=None):
        super().__init__()
        self.mode = mode
        self.eps = 1e-8
        self.momentum = momentum
        
        # Определяем размерность признаков
        if feature_dim is not None:
            self.feature_dim = feature_dim
        else:
            self.feature_dim = config.n_embd
        
        if mode == 'simple':
            # Простое центрирование по батчу
            pass
        elif mode == 'adaptive':
            # Адаптивное центрирование с running statistics
            self.register_buffer('running_center', torch.zeros(self.feature_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        elif mode == 'learnable_center':
            # Обучаемый центр
            self.learned_center = nn.Parameter(torch.zeros(self.feature_dim))
        elif mode == 'momentum':
            # Momentum-based центрирование
            self.register_buffer('momentum_center', torch.zeros(self.feature_dim))
        else:
            raise ValueError(f"Unknown centering mode: {mode}")
    
    def forward(self, x, return_stats=False):
        original_shape = x.shape
        
        # Приводим к 2D: (batch_size * seq_len [* num_heads], features)
        if len(x.shape) == 4:  # attention heads: (B, nh, T, hs)
            B, nh, T, hs = x.shape
            x_flat = x.reshape(B * nh * T, hs)
        else:  # regular embeddings: (B, T, C)
            B, T, C = x.shape
            x_flat = x.reshape(B * T, C)
        
        # Вычисляем центр в зависимости от режима
        if self.mode == 'simple':
            center = x_flat.mean(dim=0)
        elif self.mode == 'adaptive':
            if self.training:
                center = x_flat.mean(dim=0)
                # Обновляем running statistics
                if self.num_batches_tracked == 0:
                    self.running_center.copy_(center)
                else:
                    self.running_center.mul_(self.momentum).add_(center, alpha=1-self.momentum)
                self.num_batches_tracked += 1
            else:
                center = self.running_center
        elif self.mode == 'learnable_center':
            center = self.learned_center
        elif self.mode == 'momentum':
            center = x_flat.mean(dim=0)
            if self.training:
                self.momentum_center.mul_(self.momentum).add_(center, alpha=1-self.momentum)
            center = self.momentum_center
        
        # Центрируем
        x_centered = x_flat - center.unsqueeze(0)
        
        # Перенормировка
        norms = torch.norm(x_centered, dim=-1, keepdim=True)
        x_normalized = x_centered / (norms + self.eps)
        
        # Возвращаем в исходную форму
        x_out = x_normalized.reshape(original_shape)
        
        if return_stats:
            center_norm = torch.norm(center).item() if center is not None else 0.0
            stats = {
                'original_norm_mean': torch.norm(x_flat, dim=-1).mean().item(),
                'centered_norm_mean': norms.mean().item(),
                'center_norm': center_norm
            }
            return x_out, stats
        
        return x_out

class AdvancedCenteredCausalSelfAttention(nn.Module):
    """
    Attention с расширенным центрированием Q, K, V векторов
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Центрирование для query, key и value
        self.center_qk = config.center_qk
        self.center_v = config.center_v  # НОВОЕ!
        
        # Размерность головы attention
        head_size = config.n_embd // config.n_head
        
        if self.center_qk:
            self.q_centering = VectorCentering(config, mode=config.centering_mode, feature_dim=head_size)
            self.k_centering = VectorCentering(config, mode=config.centering_mode, feature_dim=head_size)
        
        if self.center_v:
            self.v_centering = VectorCentering(config, mode=config.centering_mode, feature_dim=head_size)
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, return_centering_stats=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        centering_stats = []
        
        # Центрирование query и key
        if self.center_qk:
            if return_centering_stats:
                q, q_stats = self.q_centering(q, return_stats=True)
                k, k_stats = self.k_centering(k, return_stats=True)
                centering_stats.extend([('q', q_stats), ('k', k_stats)])
            else:
                q = self.q_centering(q)
                k = self.k_centering(k)
        
        # НОВОЕ: Центрирование value векторов
        if self.center_v:
            if return_centering_stats:
                v, v_stats = self.v_centering(v, return_stats=True)
                centering_stats.append(('v', v_stats))
            else:
                v = self.v_centering(v)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        if return_centering_stats:
            return y, centering_stats
        return y

class CenteredMLP(nn.Module):
    """
    MLP с центрированием после GELU активации
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # НОВОЕ: Центрирование после GELU
        self.center_mlp = config.center_mlp
        if self.center_mlp:
            # Используем feature_dim для MLP центрирования (4x больше размерность)
            self.mlp_centering = VectorCentering(config, mode=config.centering_mode, feature_dim=4 * config.n_embd)

    def forward(self, x, return_centering_stats=False):
        x = self.c_fc(x)
        x = self.gelu(x)
        
        centering_stats = []
        
        # НОВОЕ: Центрирование после GELU активации
        if self.center_mlp:
            if return_centering_stats:
                x, mlp_stats = self.mlp_centering(x, return_stats=True)
                centering_stats.append(('mlp_gelu', mlp_stats))
            else:
                x = self.mlp_centering(x)
        
        x = self.c_proj(x)
        x = self.dropout(x)
        
        if return_centering_stats:
            return x, centering_stats
        return x

class AdvancedBlock(nn.Module):
    """
    Transformer блок с расширенным центрированием
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = AdvancedCenteredCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = CenteredMLP(config)
        
        # НОВОЕ: Центрирование residual connections
        self.center_residual = config.center_residual
        if self.center_residual:
            self.residual_centering_1 = VectorCentering(config, mode=config.centering_mode)
            self.residual_centering_2 = VectorCentering(config, mode=config.centering_mode)

    def forward(self, x, return_centering_stats=False):
        centering_stats = []
        
        # Attention block
        if return_centering_stats:
            attn_out, attn_stats = self.attn(self.ln_1(x), return_centering_stats=True)
            centering_stats.extend(attn_stats)
        else:
            attn_out = self.attn(self.ln_1(x))
        
        x = x + attn_out
        
        # НОВОЕ: Центрирование после первого residual connection
        if self.center_residual:
            if return_centering_stats:
                x, res1_stats = self.residual_centering_1(x, return_stats=True)
                centering_stats.append(('residual_1', res1_stats))
            else:
                x = self.residual_centering_1(x)
        
        # MLP block
        if return_centering_stats:
            mlp_out, mlp_stats = self.mlp(self.ln_2(x), return_centering_stats=True)
            centering_stats.extend(mlp_stats)
        else:
            mlp_out = self.mlp(self.ln_2(x))
        
        x = x + mlp_out
        
        # НОВОЕ: Центрирование после второго residual connection
        if self.center_residual:
            if return_centering_stats:
                x, res2_stats = self.residual_centering_2(x, return_stats=True)
                centering_stats.append(('residual_2', res2_stats))
            else:
                x = self.residual_centering_2(x)
        
        if return_centering_stats:
            return x, centering_stats
        return x

@dataclass
class AdvancedGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # Расширенные параметры центрирования
    center_qk: bool = False              # Центрировать query/key (уже работает)
    center_v: bool = False               # НОВОЕ: Центрировать value векторы
    center_mlp: bool = False             # НОВОЕ: Центрировать MLP после GELU
    center_residual: bool = False        # НОВОЕ: Центрировать residual connections
    center_embeddings: bool = False      # НОВОЕ: Центрировать входные эмбеддинги
    center_final_output: bool = False    # Центрировать финальные эмбеддинги
    centering_mode: str = 'adaptive'     # 'simple', 'adaptive', 'learnable_center', 'momentum'

class AdvancedGPT(nn.Module):
    """
    GPT с расширенным центрированием в разных частях архитектуры
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([AdvancedBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for argument 'parameters'"
        # ignore these, it's a harmless warning
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # НОВОЕ: Центрирование входных эмбеддингов
        self.center_embeddings = config.center_embeddings
        if self.center_embeddings:
            self.embedding_centering = VectorCentering(config, mode=config.centering_mode)

        # Центрирование финальных эмбеддингов
        self.center_final = config.center_final_output
        if self.center_final:
            self.final_centering = VectorCentering(config, mode=config.centering_mode)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_centering_stats=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        centering_stats = []
        
        # НОВОЕ: Центрирование входных эмбеддингов
        if self.center_embeddings:
            if return_centering_stats:
                x, emb_stats = self.embedding_centering(x, return_stats=True)
                centering_stats.append(('embeddings', emb_stats))
            else:
                x = self.embedding_centering(x)
        
        # Проход через блоки
        for i, block in enumerate(self.transformer.h):
            if return_centering_stats:
                x, block_stats = block(x, return_centering_stats=True)
                for stat_name, stat_data in block_stats:
                    centering_stats.append((f'block_{i}_{stat_name}', stat_data))
            else:
                x = block(x)
            
        x = self.transformer.ln_f(x)
        
        # Центрирование финальных эмбеддингов
        if self.center_final:
            if return_centering_stats:
                x, final_stats = self.final_centering(x, return_stats=True)
                centering_stats.append(('final', final_stats))
            else:
                x = self.final_centering(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_centering_stats:
            return logits, loss, centering_stats
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will leave
        # 'transformer.wte.weight' in no_decay set, which is what we want.
        decay.discard('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Отдельно обрабатываем параметры центрирования
        centering_params = []
        centering_param_names = set()
        for pn, p in param_dict.items():
            if 'centering' in pn:
                centering_params.append(p)
                centering_param_names.add(pn)
        
        # Исключаем параметры центрирования из основных групп
        decay_params = [param_dict[pn] for pn in sorted(list(decay)) if pn not in centering_param_names]
        nodecay_params = [param_dict[pn] for pn in sorted(list(no_decay)) if pn not in centering_param_names]
        
        # Проверяем, что все параметры учтены
        all_accounted = len(decay_params) + len(nodecay_params) + len(centering_params)
        assert all_accounted == len(param_dict), f"Параметры не совпадают: {all_accounted} vs {len(param_dict)}"
        
        # create the pytorch optimizer object
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Добавляем группу для параметров центрирования с пониженным learning rate
        if centering_params:
            optim_groups.append({
                'params': centering_params, 
                'weight_decay': 0.0, 
                'lr': learning_rate * 0.1  # Пониженный learning rate для центрирования
            })
        
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# Для обратной совместимости
GPTConfig = AdvancedGPTConfig
GPT = AdvancedGPT
