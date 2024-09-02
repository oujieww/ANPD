import os
os.environ['TRANSFORMERS_CACHE'] = '/home/UserData/huggingface_cache'
os.environ['http_proxy'] = 'http://192.168.3.206:20171/'
os.environ['https_proxy'] = 'http://192.168.3.206:20171/'
os.environ['HTTP_PROXY'] = 'http://192.168.3.206:20171/'
os.environ['HTTPS_PROXY'] = 'http://192.168.3.206:20171/'

import pickle
from collections import defaultdict, deque
import random
import argparse
import torch
from modeling_llama import LlamaForCausalLM
from modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from modeling_chatglm import ChatGLMForConditionalGeneration
from decoding_ml_ngram import clip_input, infer_input_ids
from rouge_score import rouge_scorer
import numpy as np
import transformers
# from collections import defaultdict
from accelerate import infer_auto_device_map, init_empty_weights
import deepspeed
from datasets import load_dataset

seed=42
torch.manual_seed(seed)
np.random.seed(seed)

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
instruction="Summarize the following articles."

class NGramModel:
    def __init__(self, text, n=2):
        self.model = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)
        self.prev_words = deque(maxlen=n-1)
        self.n = n
        self._process_text(text)
        self.default_token_id = "1"

    def _process_text(self, text):
        tokens = text.split()
        # Initialize deque with the beginning of the text
        self.prev_words.extend(tokens[:self.n-1])
        for i in range(self.n-1, len(tokens)):
            self.update(tokens[i])

    def update(self, new_word):
        prev_words_tuple = tuple(self.prev_words)
        if prev_words_tuple:
            self.model[prev_words_tuple][new_word] += 1
            self.total_counts[prev_words_tuple] += 1
        self.prev_words.append(new_word)

    def forward(self, current_words, method='greedy', topk=1, seed=None):
        # if not isinstance(current_words,tuple):
        #     current_words=(current_words,)
        # # print(current_words)
        if seed is not None:
            random.seed(seed)
        
        current_words_tuple = tuple(current_words[-(self.n - 1):])  # Only consider the last (n-1) words
        
        # current_words_tuple = tuple(current_words)
        # Handle the case when current_words is not in the model
        if current_words_tuple not in self.model:
            return [self.default_token_id] * topk

        next_word_candidates = list(self.model[current_words_tuple].keys())
        next_word_probabilities = list(self.model[current_words_tuple].values())

        # Handle the case when there are no candidates
        if not next_word_candidates:
            return [self.default_token_id] * topk

        if method == 'greedy':
            # Get indices of top-k probabilities
            # topk_indices = sorted(range(len(next_word_probabilities)), key=lambda i: next_word_probabilities[i])[-topk:]
            # next_words = [next_word_candidates[i] for i in reversed(topk_indices)]
            # Get indices of top-k probabilities
            topk_indices = np.argsort(next_word_probabilities)[-topk:]
            # print(np.argsort(next_word_probabilities))
            next_words = [next_word_candidates[i] for i in topk_indices]
            """
            np.argsort 会根据值在原列表中的位置排序，所以具有相同概率的词，位置在前的会先被选中。
            list.index(max(next_word_probabilities)) 只会返回第一个找到的最大值的索引，所以它会返回具有最大概率且在列表中首次出现的词。
            """
        elif method == 'random':
            next_words = random.choices(
                next_word_candidates,
                weights=next_word_probabilities,
                k=topk)
            """
            # 将概率列表转换成PyTorch Tensor
            probabilities_tensor = torch.tensor(next_word_probabilities, dtype=torch.float)

            # 使用torch.multinomial进行采样
            # 第一个参数是概率权重，第二个参数是采样的数量
            # replacement=True表示可以取相同的元素多次
            indices = torch.multinomial(probabilities_tensor, topk, replacement=True)

            # 根据indices获取对应的词
            next_words = [next_word_candidates[i] for i in indices]
            
            
            # 假设 next_word_probabilities 是一个与候选词对应的概率列表
            # 并且它们的和为1
            # topk 是你想要采样的次数

            # 将概率列表转换成NumPy数组，并确保其和为1
            probabilities = np.array(next_word_probabilities)
            probabilities /= probabilities.sum()

            # 使用numpy.random.multinomial进行采样
            # 第一个参数是每次试验中每个事件发生的次数，这里我们每次只抽取一次样本
            # 第二个参数是概率列表
            # 第三个参数是试验的次数
            samples = np.random.multinomial(1, probabilities, size=topk)

            # 根据samples获取对应的词
            # samples 每一行代表一次试验的结果，非零索引即为抽取的词的索引
            next_words_indices = samples.nonzero()[1]
            next_words = [next_word_candidates[i] for i in next_words_indices]

            """
        else:
            raise ValueError("Invalid method. Choose 'greedy' or 'random'.")
        if len(next_words) < topk:
            next_words.extend([self.default_token_id] * (topk - len(next_words)))
        return next_words
    
class MultiLevelNgram:
    def __init__(self, text, max_n=3):
        self.models = [NGramModel(text, n=i) for i in range(2, max_n + 1)]
        self.max_n = max_n
        self._process_text(text)

    def _process_text(self, text):
        tokens = text.split()
        for word in tokens:
            self.update(word)

    def update(self, new_word):
        for model in self.models:
            model.update(new_word)

    def forward(self, current_words, method='greedy', topk=1, seed=None):
        for model in reversed(self.models):  # Start from the highest n-gram
            next_words = model.forward(current_words, method=method, topk=topk, seed=seed)
            if next_words[0] != model.default_token_id:
                return next_words
        return [model.default_token_id] * topk  # If no model returns a valid word, return the default token
