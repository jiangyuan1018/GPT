# language_model/train.py
import torch
import matplotlib.pyplot as plt
import os
import json
import argparse
import logging
from tqdm import tqdm
import sys

from .models import TransformerLanguageModel, get_batch, estimate_loss, save_model
from .tokenizers import init_tokenizer, save_tokenizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Language Model Training Script')
    parser.add_argument('--input_file', type=str, required=True, help='Input text file path')
    parser.add_argument('--tokenizer', type=str, default='bpe', choices=['char', 'bpe'], help='Tokenizer type: char or bpe')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Target vocabulary size for BPE tokenizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--block_size', type=int, default=64, help='Context length')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout rate')
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of evaluation iterations')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints2', help='Model save directory')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_file', type=str, default='training.log', help='Log file path')
    return parser.parse_args()

def setup_logger(log_file):
    """设置日志记录器"""
    # 创建logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器（仅输出重要信息）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 只有警告及以上级别才输出到控制台
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    # 解析参数
    if len(sys.argv) == 1:
        # 没有参数时显示帮助
        parse_args().print_help()
        return
        
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 设置日志记录器
    log_file = os.path.join(args.checkpoint_dir, args.log_file)
    logger = setup_logger(log_file)
    
    logger.info("=" * 50)
    logger.info(f"Starting new model training using {args.tokenizer} tokenizer")
    logger.info("Configuration parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # 保存配置
    with open(os.path.join(args.checkpoint_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    # 读取数据
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info(f"Total characters in dataset: {len(text)}")
    logger.info(f"First 100 characters: {text[:100]}...")
    
    # 初始化分词器
    encode, decode, vocab_size, tokenizer_data = init_tokenizer(
        text, args.tokenizer, args.vocab_size
    )
    
    # 保存分词器
    save_tokenizer(args.tokenizer, tokenizer_data, args.checkpoint_dir)
    logger.info(f"{args.tokenizer} tokenizer saved")
    
    # 测试分词效果
    test_str = "张无忌"
    encoded = encode(test_str)
    decoded = decode(encoded)
    logger.info(f"Tokenization test:")
    logger.info(f"  Encoded '{test_str}': {encoded}")
    logger.info(f"  Decoded back: '{decoded}'")
    
    # 数据集划分
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    data_dict = {'train': train_data, 'val': val_data}
    logger.info(f"Dataset split: Training set {len(train_data)} samples, Validation set {len(val_data)} samples")
    
    # 初始化模型
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
        device=args.device
    )
    model = model.to(args.device)
    
    # 打印模型信息
    param_count = sum(p.numel() for p in model.parameters())/1e6
    logger.info(f"Model parameter count: {param_count:.2f}M")
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 训练准备
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 使用进度条显示训练进度
    logger.info("Starting training...")
    print(f"Starting training - See log file for details: {log_file}")
    
    # 训练循环
    pbar = tqdm(range(args.max_iters), desc="Training progress")
    for iter in pbar:
        # 评估
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(
                model, data_dict, args.eval_iters, 
                args.batch_size, args.block_size, args.device
            )
            logger.info(f"Step {iter}: Training loss {losses['train']:.4f}, Validation loss {losses['val']:.4f}")
            
            # 更新进度条描述
            pbar.set_description(f"Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")
            
            # 记录损失
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            # 检查是否是最佳模型
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                
                # 保存最佳模型
                save_model(
                    model, optimizer, iter, best_val_loss,
                    os.path.join(args.checkpoint_dir, f'best_model.pt'),
                    logger
                )
                logger.info(f"Found new best model, validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping: Validation loss not improved for {args.patience} evaluations")
                    break
        
        # 训练步骤
        xb, yb = get_batch(data_dict, 'train', args.batch_size, args.block_size, args.device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # 保存最终模型
    save_model(
        model, optimizer, args.max_iters-1, best_val_loss,
        os.path.join(args.checkpoint_dir, f'final_model.pt'),
        logger
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Count')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({args.tokenizer} tokenizer)')
    plt.legend()
    loss_curve_path = os.path.join(args.checkpoint_dir, f'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    logger.info(f"Loss curve saved to {loss_curve_path}")
    
    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model and tokenizer saved in {args.checkpoint_dir}")
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
