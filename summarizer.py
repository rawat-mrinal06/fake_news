import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Summarizer:

    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-large')

        # if USE_GPU:
        self.model.to('cuda')

    def generate_summary(self, sentence):
        input_ids = self.tokenizer.encode('summarize: {}'.format(sentence), truncation=True)
        tokens_tensor = torch.tensor([input_ids]).to('cuda')

        # generate text until the output length (which includes the context length) reaches 50
        generated_ids = self.model.generate(input_ids=tokens_tensor,
                                            max_length=250,
                                            num_beams=5,
                                            no_repeat_ngram_size=10,
                                            repetition_penalty=2.0)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def generate_batch_summaries(self, sentences, max_len=250):
        input_ids = self.tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors='pt')
        tokens_tensor = input_ids['input_ids'].cuda()

        # generate text until the output length (which includes the context length) reaches 50
        generated_ids = self.model.generate(input_ids=tokens_tensor,
                                            max_length=max_len,
                                            num_beams=5,
                                            no_repeat_ngram_size=10,
                                            repetition_penalty=2.0)
        return [self.tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen in
                generated_ids]


summarizer = Summarizer()

