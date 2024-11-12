import torch
from transformers import AutoTokenizer, OPTForCausalLM

class ModelHandler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
        self.model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")
        
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
