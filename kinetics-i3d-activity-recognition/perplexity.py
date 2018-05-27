# calculate blue score
from __future__ import print_function
import time
import random
import math
import sys
import argparse
from nltk.translate.bleu_score import sentence_bleu

start = time.time()

from collections import Counter, defaultdict
import numpy as np

goldenTest, predicted = None, None

def calculate_bleu_score(natural, caption):
    score = sentence_bleu(natural, caption)
    return score 

def main():
    natural = "a group of giraffes standing next to each other"
    caption = "a group of monkeys walking next to each other"
    print (calculate_bleu_score(natural, caption))

main()
