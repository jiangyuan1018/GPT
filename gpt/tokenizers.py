import torch
import os
import re
import collections
from collections import Counter, defaultdict

## 字符级分词器实现
def init_char_tokenizer(text):
    """初始化字符级分词器"""
    print("初始化字符级分词器...")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"文本中不同字符数量: {vocab_size}")
    
    # 创建映射字典
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    # 定义编码和解码函数
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return encode, decode, vocab_size, (stoi, itos)

## BPE分词（效果较差）
def train_bpe(text, target_vocab_size):
    """训练BPE分词器"""
    print("开始训练BPE分词器...")
    
    # 文本预处理，添加结束符号</w>
    processed_text = ""
    for line in text.split('\n'):
        words = line.strip().split()
        # 给每个单词添加结束符号
        processed_text += ' '.join([word + '</w>' for word in words]) + ' '
    
    # 初始化词汇表
    def get_vocab(processed_text):
        """构建初始词汇表，按单词频率统计"""
        vocab = defaultdict(int)
        for word in processed_text.strip().split():
            # 将单词拆分为字符序列，并添加空格
            vocab[' '.join(list(word[:-4])) + ' </w>'] += 1
        return vocab
    
    # 获取字符对统计
    def get_stats(vocab):
        """计算所有相邻字符对的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    # 合并词汇表
    def merge_vocab(pair, v_in):
        """根据最佳字符对合并词汇表"""
        v_out = {}
        bigram = re.escape(' '.join(pair))
        # 正则表达式确保只合并完整的字符对
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word, freq in v_in.items():
            # 合并单词中的字符对
            new_word = p.sub(''.join(pair), word)
            v_out[new_word] = freq
        return v_out
    
    # 获取词汇表中的所有token
    def get_tokens_from_vocab(vocab):
        """从词汇表中提取token及其频率"""
        tokens_frequencies = defaultdict(int)
        vocab_tokenization = {}
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens_frequencies[token] += freq
            # 记录词汇表中每个单词对应的token序列
            vocab_tokenization[''.join(word_tokens)] = word_tokens
        return tokens_frequencies, vocab_tokenization
    
    # 测量token长度
    def measure_token_length(token):
        """计算token的实际长度"""
        if token[-4:] == '</w>':
            return len(token[:-4]) + 1
        else:
            return len(token)
    
    # 对token按长度和频率排序
    def sort_tokens(tokens_frequencies):
        """按token长度和频率排序"""
        sorted_tokens_tuple = sorted(
            tokens_frequencies.items(), 
            key=lambda item: (measure_token_length(item[0]), item[1]), 
            reverse=True
        )
        sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
        return sorted_tokens
    
    # 对未知单词进行分词
    def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
        """使用排序后的token列表对单词进行分词"""
        if string == '':
            return []
        if sorted_tokens == []:
            return [unknown_token] * len(string)
            
        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace('.', '[.]'))
            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            
            if len(matched_positions) == 0:
                continue
                
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]
            substring_start_position = 0
            
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += tokenize_word(substring, sorted_tokens[i+1:], unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
                
            remaining_substring = string[substring_start_position:]
            string_tokens += tokenize_word(remaining_substring, sorted_tokens[i+1:], unknown_token)
            break
        else:
            string_tokens = [unknown_token] * len(string)
            
        return string_tokens
    
    # 开始BPE训练过程
    vocab = get_vocab(processed_text)
    print(f"初始词汇表大小: {len(vocab)}")
    
    # 初始tokens
    tokens = defaultdict(int)
    for word, freq in vocab.items():
        for token in word.split():
            tokens[token] += freq
    
    print(f"初始token数量: {len(tokens)}")
    
    # 执行BPE合并操作
    num_merges = target_vocab_size - len(tokens)
    print(f"将执行{num_merges}次合并操作")
    
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
            
        # 选择频率最高的字符对
        best = max(pairs, key=pairs.get)
        new_token = ''.join(best)
        
        # 应用合并规则
        vocab = merge_vocab(best, vocab)
        
        # 更新token计数
        tokens[new_token] = pairs[best]
        tokens[best[0]] -= pairs[best]
        tokens[best[1]] -= pairs[best]
        
        # 打印进度
        if (i+1) % 500 == 0:
            print(f"合并进度: {i+1}/{num_merges}, 最新合并: {best} -> {new_token}")
    
    # 获取最终的tokens和词汇表标记化
    tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
    sorted_tokens = sort_tokens(tokens_frequencies)
    
    print(f"BPE训练完成，词汇表大小: {len(tokens_frequencies)}")
    
    # 创建编码和解码函数
    def encode_with_bpe(text):
        """使用BPE将文本编码为token ID序列"""
        encoded = []
        for word in text.split():
            word = word + "</w>"
            
            if word in vocab_tokenization:
                word_tokens = vocab_tokenization[word]
            else:
                word_tokens = tokenize_word(word, sorted_tokens, unknown_token='</u>')
                
            for token in word_tokens:
                encoded.append(stoi[token])
        return encoded

    def decode_with_bpe(ids):
        """将token ID序列解码为文本"""
        tokens = [itos[id] for id in ids if id in itos]
        text = ''
        current_word = ''
        
        for token in tokens:
            if token == '</u>':
                # 处理未知token
                text += '?'
            elif token.endswith('</w>'):
                # 处理单词结束
                current_word += token[:-4]
                text += current_word + ' '
                current_word = ''
            else:
                # 处理普通token
                current_word += token
                
        return text.strip()
    
    # 创建token到ID的映射
    all_tokens = [token for token in tokens_frequencies.keys()]
    stoi = {token: i for i, token in enumerate(all_tokens)}
    itos = {i: token for i, token in enumerate(all_tokens)}
    
    # 添加未知token
    if '</u>' not in stoi:
        stoi['</u>'] = len(stoi)
        itos[len(itos)] = '</u>'
    
    # 保存额外信息，用于后续加载
    tokenizer_data = (stoi, itos, sorted_tokens, vocab_tokenization)
    
    return encode_with_bpe, decode_with_bpe, len(stoi), tokenizer_data

## 保存和加载函数
def save_tokenizer(tokenizer_type, tokenizer_data, checkpoint_dir, prefix=''):
    """保存分词器"""
    tokenizer_dir = os.path.join(checkpoint_dir, f"{prefix}tokenizer_{tokenizer_type}")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    if tokenizer_type == 'char':
        stoi, itos = tokenizer_data
        torch.save({'stoi': stoi, 'itos': itos}, os.path.join(tokenizer_dir, 'tokenizer.pt'))
    elif tokenizer_type == 'bpe':
        stoi, itos, sorted_tokens, vocab_tokenization = tokenizer_data
        torch.save({
            'stoi': stoi, 
            'itos': itos, 
            'sorted_tokens': sorted_tokens,
            'vocab_tokenization': vocab_tokenization
        }, os.path.join(tokenizer_dir, 'tokenizer.pt'))
    
    print(f"分词器已保存到 {tokenizer_dir}")

def load_tokenizer(checkpoint_dir, override_tokenizer=None):
    """加载分词器"""
    import json
    try:
        with open(os.path.join(checkpoint_dir, 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 使用指定的分词器类型
        tokenizer_type = override_tokenizer or config.get('tokenizer', 'char')
        
        # 检查分词器目录是否存在
        tokenizer_dir = os.path.join(checkpoint_dir, f"tokenizer_{tokenizer_type}")
        if not os.path.exists(tokenizer_dir):
            alt_type = 'bpe' if tokenizer_type == 'char' else 'char'
            alt_dir = os.path.join(checkpoint_dir, f"tokenizer_{alt_type}")
            
            if os.path.exists(alt_dir):
                print(f"警告: 找不到{tokenizer_type}分词器，回退到{alt_type}分词器")
                tokenizer_type = alt_type
                tokenizer_dir = alt_dir
            else:
                raise FileNotFoundError(f"在{checkpoint_dir}中找不到char或bpe分词器目录")
        
        # 加载分词器文件
        tokenizer_file = os.path.join(tokenizer_dir, 'tokenizer.pt')
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"分词器文件不存在: {tokenizer_file}")
        
        data = torch.load(tokenizer_file)
        
        if tokenizer_type == 'char':
            stoi, itos = data['stoi'], data['itos']
            encode = lambda s: [stoi[c] if c in stoi else stoi.get('<UNK>', 0) for c in s]
            decode = lambda l: ''.join([itos.get(i, '<UNK>') for i in l])
            return encode, decode, tokenizer_type
        
        elif tokenizer_type == 'bpe':
            stoi = data['stoi']
            itos = data['itos']
            sorted_tokens = data['sorted_tokens']
            vocab_tokenization = data['vocab_tokenization']
            
            def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
                """使用排序后的token列表对单词进行分词"""
                if string == '':
                    return []
                if sorted_tokens == []:
                    return [unknown_token] * len(string)
                    
                string_tokens = []
                for i in range(len(sorted_tokens)):
                    token = sorted_tokens[i]
                    token_reg = re.escape(token.replace('.', '[.]'))
                    matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
                    
                    if len(matched_positions) == 0:
                        continue
                        
                    substring_end_positions = [matched_position[0] for matched_position in matched_positions]
                    substring_start_position = 0
                    
                    for substring_end_position in substring_end_positions:
                        substring = string[substring_start_position:substring_end_position]
                        string_tokens += tokenize_word(substring, sorted_tokens[i+1:], unknown_token)
                        string_tokens += [token]
                        substring_start_position = substring_end_position + len(token)
                        
                    remaining_substring = string[substring_start_position:]
                    string_tokens += tokenize_word(remaining_substring, sorted_tokens[i+1:], unknown_token)
                    break
                else:
                    string_tokens = [unknown_token] * len(string)
                    
                return string_tokens
            
            def encode_with_bpe(text):
                """使用BPE将文本编码为token ID序列"""
                encoded = []
                for word in text.split():
                    word = word + "</w>"
                    
                    if word in vocab_tokenization:
                        word_tokens = vocab_tokenization[word]
                    else:
                        word_tokens = tokenize_word(word, sorted_tokens, unknown_token='</u>')
                        
                    for token in word_tokens:
                        encoded.append(stoi[token] if token in stoi else stoi['</u>'])
                return encoded

            def decode_with_bpe(ids):
                """将token ID序列解码为文本"""
                tokens = [itos[id] if id in itos else '</u>' for id in ids]
                text = ''
                current_word = ''
                
                for token in tokens:
                    if token == '</u>':
                        # 处理未知token
                        text += '?'
                    elif token.endswith('</w>'):
                        # 处理单词结束
                        current_word += token[:-4]
                        text += current_word + ' '
                        current_word = ''
                    else:
                        # 处理普通token
                        current_word += token
                        
                return text.strip()
            
            return encode_with_bpe, decode_with_bpe, tokenizer_type
        
        else:
            raise ValueError(f"不支持的分词器类型: {tokenizer_type}")
            
    except Exception as e:
        print(f"加载分词器时出错: {e}")
        print("尝试返回默认的字符级分词器...")
        
        # 提供一个简单的默认字符级分词器作为回退
        vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-:;'\"()[]{}，。！？、：；''""（）【】《》")
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for i, ch in enumerate(vocab)}
        
        encode = lambda s: [stoi[c] if c in stoi else stoi[' '] for c in s]
        decode = lambda l: ''.join([itos[i] if i in itos else ' ' for i in l])
        
        return encode, decode, 'char'

def init_tokenizer(text, tokenizer_type='char', target_vocab_size=5000):
    """初始化分词器，支持字符级和BPE"""
    if tokenizer_type == 'char':
        return init_char_tokenizer(text)
    elif tokenizer_type == 'bpe':
        return train_bpe(text, target_vocab_size)
    else:
        raise ValueError(f"不支持的分词器类型: {tokenizer_type}")
