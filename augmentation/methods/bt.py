import os
import pdb
import sys
import json
import torch
import random

from transformers import MarianMTModel, MarianTokenizer


model_name1 = 'Helsinki-NLP/opus-mt-en-de'
model_name2 = 'Helsinki-NLP/opus-mt-de-en'

tokenizer1 = MarianTokenizer.from_pretrained(model_name1)
model1 = MarianMTModel.from_pretrained(model_name1).to('cuda')

tokenizer2 = MarianTokenizer.from_pretrained(model_name2)
model2 = MarianMTModel.from_pretrained(model_name2).to('cuda')


def generate_new_text(text, model, tokenizer):
    model = model.to("cuda")
    tokenized = tokenizer(text, return_tensors="pt", padding=True).to("cuda")
    translated = model.generate(**tokenized)
    translated_texts = tokenizer.decode(translated[0], skip_special_tokens = True)
    return translated_texts

    

def make_back_trans(text):
    de_data= generate_new_text(text, model1, tokenizer1)
    en_data= generate_new_text(de_data, model2, tokenizer2)
    return en_data
            

def run_bt(text):
    return make_back_trans(text)