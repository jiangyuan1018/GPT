import torch
import os
import argparse
import sys
from .models import load_model
from .tokenizers import load_tokenizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Language Model Text Generation Script')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints2', help='Model save directory')
    parser.add_argument('--model_file', type=str, default='best_model.pt', help='Model file to load')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Generation device')
    parser.add_argument('--prompt', type=str, default=None, help='Initial prompt text, use interactive mode if not provided')
    parser.add_argument('--max_tokens', type=int, default=500, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top-K sampling, 0 to disable')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-P (nucleus sampling), 1.0 to disable')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def generate_text(model, encode, decode, prompt, max_tokens, temperature, top_k, top_p, device):
    """生成文本"""
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成文本
    print(f"\nGenerating...")
    with torch.no_grad():
        generated_indices = model.generate(
            context, 
            max_tokens, 
            temperature=temperature,
            top_k=top_k, 
            top_p=top_p
        )
    
    # 解码
    generated_text = decode(generated_indices[0].tolist())
    return generated_text

def interactive_generation(model, encode, decode, device, max_tokens=500, temperature=1.0, top_k=0, top_p=0.9):
    """交互式生成文本"""
    print("\n==== Interactive Text Generation ====")
    print("Enter a prompt text to generate, type 'exit' to quit, or 'settings' to change generation parameters")
    
    # 当前参数
    params = {
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p
    }
    
    while True:
        # 显示当前参数
        print(f"\nCurrent settings: max_tokens={params['max_tokens']}, temperature={params['temperature']}, top_k={params['top_k']}, top_p={params['top_p']}")
        prompt = input("\nEnter prompt: ")
        
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        
        elif prompt.lower() == 'settings':
            # 修改设置
            try:
                params['max_tokens'] = int(input(f"Max tokens (current: {params['max_tokens']}): ") or params['max_tokens'])
                params['temperature'] = float(input(f"Temperature (current: {params['temperature']}): ") or params['temperature'])
                params['top_k'] = int(input(f"Top-K (current: {params['top_k']}, 0=disabled): ") or params['top_k'])
                params['top_p'] = float(input(f"Top-P (current: {params['top_p']}, 1.0=disabled): ") or params['top_p'])
                print("Settings updated!")
            except ValueError:
                print("Invalid input, using original parameters")
            continue
        
        # 生成文本
        generated = generate_text(
            model, encode, decode, 
            prompt, 
            params['max_tokens'], 
            params['temperature'], 
            params['top_k'], 
            params['top_p'],
            device
        )
        
        # 输出结果
        print("\nGenerated text:")
        print(generated)
        print("-" * 50)

def main():
    '''
    if len(sys.argv) == 1:
        parse_args().print_help()
        return
    '''
        
    args = parse_args()

    torch.manual_seed(args.seed)
    
    encode, decode, tokenizer_type = load_tokenizer(args.checkpoint_dir)
    print(f"Loaded {tokenizer_type} tokenizer")
    model_path = os.path.join(args.checkpoint_dir, args.model_file)
    model = load_model(model_path, args.device)
    
    if args.prompt:
        generated_text = generate_text(
            model, encode, decode, 
            args.prompt, 
            args.max_tokens, 
            args.temperature, 
            args.top_k, 
            args.top_p,
            args.device
        )
        print("\nGenerated text:")
        print(generated_text)
    else:
        interactive_generation(
            model, encode, decode, 
            args.device,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p
        )

if __name__ == "__main__":
    main()
