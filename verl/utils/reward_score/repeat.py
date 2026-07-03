import re
from tqdm import tqdm
import copy
import math

import numpy as np
from multiprocessing import Pool
from pydivsufsort import divsufsort, kasai
from collections import Counter

MIN_OCCURRENCE_NUM = 6  # 从20降低到6，更敏感地检测重复
MIN_OCCURRENCE_NUM_small = 2
MIN_NGRAM_NUM = 2
INTERVEL_MIN = 3

MIN_LINE_OCCURRENCE_NUM=6  # 从10降低到6
MIN_LINE_LENGTH=20

from transformers import AutoTokenizer

class SparseTable:
    def __init__(self, arr, rank):
        self.n = len(arr)
        self.arr = arr
        self.rank = rank
        self.build_sparse_table()

    def build_sparse_table(self):
        logn = math.ceil(math.log2(self.n)) + 1
        self.sparse_table = [[0] * logn for _ in range(self.n)]
        for i in range(self.n):
            self.sparse_table[i][0] = self.arr[i]
        for j in range(1, logn):
            for i in range(self.n - (1 << j) + 1):
                self.sparse_table[i][j] = min(self.sparse_table[i][j - 1], self.sparse_table[i + (1 << (j - 1))][j - 1])

    def query(self, l, r):
        l, r = self.rank[l], self.rank[r]
        if l > r:
            l, r = r, l
        l += 1
        j = math.floor(math.log2(r - l + 1))
        return min(self.sparse_table[l][j], self.sparse_table[r - (1 << j) + 1][j])


def detect_successive_repetition(tokens, tokenizer):
    assert MIN_OCCURRENCE_NUM > 1
    if len(tokens) < MIN_NGRAM_NUM * 2:
        return False, ""
    tokens = np.array(tokens).astype(np.int32)
    sa = divsufsort(tokens)
    rank = [0] * len(tokens)
    for i in range(len(tokens)):
        rank[sa[i]] = i
    lcp = kasai(tokens, sa)
    lcp = np.roll(lcp, 1)
    lcp[0] = len(tokens) + 1
    st_table = SparseTable(lcp, rank)
    
    for i in range(MIN_NGRAM_NUM, len(tokens) // 2 + 1):
        for j in range(i, len(tokens), i):
            sublen = st_table.query(j - i, j)
            if sublen < i:
                continue
            num = (i + sublen) // i
            if sublen % i != 0:
                offset = i - sublen % i
                if j - i - offset >= 0 and st_table.query(j - i - offset, j - offset) >= offset:
                    num += 1
                    j -= offset
            if num >= MIN_OCCURRENCE_NUM:
                subtoken = tokenizer.decode(tokens[j - i: j])
                return (True, subtoken)
    return (False, "")


def detect_successive_repetition_thinking_withngram(text, tokens, tokenizer, n=20):

    def generate_ngrams(text, n):
        if len(text) <= n:
            return [tuple(text)]
        ngrams = (tuple(text[i: i + n]) for i in range(len(text) - n + 1))
        return ngrams

    def count_ngrams(text, n):
        """统计n-gram出现次数"""
        ngrams = generate_ngrams(text, n)
        return Counter(ngrams)
    
    def filter_high_frequency_ngrams(counter, threshold):
        """过滤出现次数超过阈值的n-gram"""
        return {ngram: count for ngram, count in counter.items() if count > threshold}
    
    # original_text = copy.deepcopy(text)
    # tokens = tokenizer(original_text, padding=False)['input_ids']
    
    # 边界检查：如果tokens太短，无法形成n-gram
    if len(tokens) < n:
        return False, ""
    
    counted_ngrams = count_ngrams(tokens, n)
    
    # 检查Counter是否为空
    if len(counted_ngrams) == 0:
        return False, ""
    
    most_common_ngram = counted_ngrams.most_common(1)[0]
    if most_common_ngram[1] >= 6:  # 从20降低到6，更敏感地检测n-gram重复
        # seq = most_common_ngram[0].replace(" ","")
        # 复用已tokenize的结果，避免重复tokenize
        has_repetition, seq = detect_successive_repetition(tokens, tokenizer)
        if has_repetition == True:
            return True, seq
        else:
            return False, ""
    else:
        return False, ""

def detect_line_exactmatch_repetition(text):
    text_splits = text.split("\n")
    text_splits_counter = Counter(text_splits)
    for seq, cnt in text_splits_counter.items():
        # 调整为6次：同一行出现≥6次（长度≥20）或≥6次（长度≥50）
        if (cnt >= 6 and len(seq)>=20) or (cnt >= 6 and len(seq)>=50):
            # print(x)
            return True, seq
    return False, ""


def detect_repetition(text, text_ids, tokenizer):
    has_repetition_linematch, repeat_seq_linematch = detect_line_exactmatch_repetition(text)
    if has_repetition_linematch == True:
        return has_repetition_linematch, repeat_seq_linematch, "line_match"
    has_repetition_ngram, repeat_seq_ngram = detect_successive_repetition_thinking_withngram(text, text_ids, tokenizer)
    if has_repetition_ngram == True:
        return has_repetition_ngram, repeat_seq_ngram, "ngram"
    return False, "", "None"


def work(examples):
    print(len(examples))
    for e in tqdm(examples):
        has_repetition, seq, repetition_reason = detect_repetition(e['messages'][-1]['content'].split("</think>")[0])
        e['has_repetition'] = has_repetition
        e['repetition_related_seq'] = seq
        e['repetition_type'] = repetition_reason
    return examples


def detect_repetition_parallel(examples):
    if len(examples) == 0:
        return []
    tobe_solved_examples = []
    group_num = 20
    if len(examples) < group_num:
        group_num = len(examples)
    batch_size = len(examples) // group_num
    for i in range(group_num):
        if i == group_num-1:
            tobe_solved_examples.append(examples[i*batch_size:])
        else:
            tobe_solved_examples.append(examples[i*batch_size:(i+1)*batch_size])

    with Pool(group_num) as p:
        new_examples = p.map(work, tobe_solved_examples)
    ret_examples = []
    for tmp in new_examples:
        ret_examples += tmp
    return ret_examples


def check_response_valid(response):
    keywords = ["limited time", "time is limited", "time constraint", "时间有限", "时间限制", "i guess the answer", "我猜答案"]
    for keyword in keywords:
        if keyword in response:
            return False 
    return True 


def remove_latex_environments(text):
    """
    移除文本中的LaTeX环境内容（行内公式、行间公式、$$公式、$公式）
    支持处理跨多行的情况
    """
    pattern = r'''
        \\\(              # 匹配 \( 开始符
            .*?           # 非贪婪匹配任意字符（包括换行，因re.DOTALL）
        \\\)              # 匹配 \) 结束符
    |                     # 或
        \\\[              # 匹配 \[ 开始符
            .*?           # 非贪婪匹配任意字符
        \\\]              # 匹配 \] 结束符
    |                     # 或
        \$\$              # 匹配 $$ 开始符
            .*?           # 非贪婪匹配任意字符
        \$\$              # 匹配 $$ 结束符
    |                     # 或
        \$                # 匹配 $ 开始符（行内公式）
            .*?           # 非贪婪匹配任意字符
        \$                # 匹配 $ 结束符
    '''
    return re.sub(pattern, '', text, flags=re.DOTALL | re.VERBOSE)


def check_codeswitch_valid(response):
    if response.count("</think>") != 1:
        return False
    
    keywords = [' can ', ' could ', ' should ', ' will ', ' would ', ' may ', ' might ' , " let ", " actually ", " wait ", " alternatively ", " we ", " sure ", " must ", " need ", " used ", " have ", " had "]
    for keyword in keywords:
        if keyword in response:
            return False
    
    return True

def check_repetition_valid(response, response_ids, tokenizer):
    """
    检查回答是否有效（无重复）
    
    Returns:
        bool: True表示有效（无重复），False表示无效（有重复）
    """
    if len(response.strip()) == 0:
        return True  # 空文本视为有效（无重复）
    has_repetition, repeat_str, _ = detect_repetition(response, response_ids, tokenizer)
    return not has_repetition  # 有重复返回False（无效），无重复返回True（有效）
