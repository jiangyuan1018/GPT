# 简易 GPT 实现

这是一个基于 Transformer 架构的 gpt 生成模型项目，支持字符级和 BPE 两种分词方法，使用 PyTorch 实现。

## 项目结构

```
gpt/
│
├── __init__.py        # 包初始化文件
├── models.py          # 模型定义
├── tokenizers.py      # 分词器实现（字符级和BPE）
├── train.py           # 训练脚本
└── generate.py        # 生成脚本

# 入口脚本
train_model.py         # 训练入口
generate_text.py       # 生成入口
```

## 安装

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/bigict/GPT.git
cd GPT
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

或直接安装必要的库：

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install tqdm
```

## 使用方法

### 1. 准备数据

将你的训练文本文件准备好，例如中文小说、文章集合等。以倚天屠龙记为例

### 2. 训练模型

#### 使用字符级分词训练：

```bash
python train_model.py --input_file 金庸-倚天屠龙记txt精校版_utf-8.txt --tokenizer char
```

#### 使用 BPE 分词训练：

```bash
python train_model.py --input_file 金庸-倚天屠龙记txt精校版_utf-8.txt --tokenizer bpe --vocab_size 5000
```

### 3. 生成文本

#### 交互式生成：

```bash
python generate_text.py --checkpoint_dir ./checkpoints
```

#### 使用提示词生成：

```bash
python generate_text.py --checkpoint_dir ./checkpoints --prompt "张无忌说：" --max_tokens 1000
```

### 4. 交互模式特殊命令

- 输入 `exit` 退出程序
- 输入 `settings` 修改生成参数（最大 token 数、温度、Top-K、Top-P）
