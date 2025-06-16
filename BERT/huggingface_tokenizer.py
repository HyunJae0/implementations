from transformers import BertTokenizerFast

class HuggingFaceTokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def get_token_id(self, token):
        return self.tokenizer.get_vocab()[token]

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_ids_to_sentence(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)