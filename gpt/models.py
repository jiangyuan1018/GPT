import torch
import torch.nn as nn
from torch.nn import functional as F

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4*n_embd,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        mask = torch.triu(torch.ones(block_size, block_size) * float('-inf'), diagonal=1)
        self.register_buffer("mask", mask)
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # 配置
        self.block_size = block_size
        self.device = device
        self.config = {
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'dropout': dropout
        }
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 获取 token 和位置嵌入
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        
        # 应用 transformer 编码器，使用因果掩码确保只关注前面的 token
        x = self.transformer(x, mask=self.mask[:T, :T])  # (B,T,C)
        
        # 最终层标准化和输出
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        # 计算损失
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=0.9):
        """生成文本，支持温度调节、Top-K 和 Top-P 采样"""
        for _ in range(max_new_tokens):
            # 截取最后 block_size 个 token
            idx_cond = idx[:, -self.block_size:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 应用温度
            
            # 如果使用 Top-K 采样
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 如果使用 Top-P 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率质量超过阈值的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 将第一个 Token 的移除标记设为 False，确保至少有一个 Token 可用
                sorted_indices_to_remove[:, 0] = False
                
                # 散点索引回原始 logits
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

def get_batch(data, split, batch_size, block_size, device):
    """生成一批训练数据"""
    if split == 'train':
        data_split = data['train']
    else:
        data_split = data['val']
        
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, data, eval_iters, batch_size, block_size, device):
    """评估模型损失"""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, split, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out

def save_model(model, optimizer, epoch, val_loss, filename, logger=None):
    """保存模型"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': model.config,
    }
    torch.save(state, filename)
    if logger:
        logger.info(f"Model saved to {filename}")
    else:
        print(f"Model saved to {filename}")

def load_model(model_path, device):
    """加载模型"""
    # 加载状态字典
    state = torch.load(model_path, map_location=device)
    config = state['config']
    
    # 创建模型实例
    model = TransformerLanguageModel(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout'],
        device=device
    )
    
    # 加载参数
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    print(f"Model loaded from {model_path}")
    print(f"Validation loss: {state['val_loss']:.4f}, Training iteration: {state['epoch']}")
    
    return model
