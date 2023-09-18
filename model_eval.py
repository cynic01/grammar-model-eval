import spacy
import subprocess
from pathlib import Path
wd = Path(__file__).parent.resolve()

class FileEval:
    def __init__(self,
                 nlp=None,
                 tokenizer=None,
                 scorer=None):
        self.nlp = nlp or spacy.load('en')
        self.tokenizer = tokenizer or self.nlp.tokenizer
        self.scorer = scorer #or SerrantScorer(nlp=self.nlp)
    
    def evaluate(self,
                 orig=Path(wd / "orig.txt"),
                 orig_tokenized=False,
                 model_results=Path(wd / "model_results.txt"),
                 model_results_tokenized=False,
                 label=Path(wd / "label.txt"),
                 label_tokenized=False,
                 model_m2_out=Path(wd / "model_m2_out.txt"),
                 label_m2_out=Path(wd / "label_m2_out.txt")):
        # convert to word tokenized format
        if not orig_tokenized:
            with open(orig, 'r') as forig:
                orig_lines = forig.readlines()
            orig_tokens_text = [" ".join([token.text for token in self.tokenizer(sent.strip())]) + "\n" for sent in orig_lines]
            orig = Path(wd / "orig_tmp.txt")
            with open(orig, 'w') as ftmp:
                ftmp.writelines(orig_tokens_text)
        if not model_results_tokenized:
            with open(model_results, 'r') as fmodel_results:
                model_results_lines = fmodel_results.readlines()
            model_results_tokens_text = [" ".join([token.text for token in self.tokenizer(sent.strip())]) + "\n" for sent in model_results_lines]
            model_results = Path(wd / "model_results_tmp.txt")
            with open(model_results, 'w') as ftmp:
                ftmp.writelines(model_results_tokens_text)
        if not label_tokenized:
            with open(label, 'r') as flabel:
                label_lines = flabel.readlines()
            label_tokens_text = [" ".join([token.text for token in self.tokenizer(sent.strip())]) + "\n" for sent in label_lines]
            label = Path(wd / "label_tmp.txt")
            with open(label, 'w') as ftmp:
                ftmp.writelines(label_tokens_text)
        # generate annotated m2 files
        proc_model = subprocess.run(["serrant_parallel", "-orig", str(orig), "-cor", str(model_results), "-out", str(model_m2_out)])
        proc_label = subprocess.run(["serrant_parallel", "-orig", str(orig), "-cor", str(label), "-out", str(label_m2_out)])
        # evaluate model results against label
        proc_eval = subprocess.run(["serrant_compare", "-hyp", str(model_m2_out), "-ref", str(label_m2_out)])

class ModelEval:
    def __init__(self,
                 nlp=None,
                 model=None,
                 tokenizer=None,
                 scorer=None
                 ):
        self.nlp = nlp or spacy.load('en')
        self.model = model #or GPT2GEC()
        self.tokenizer = tokenizer or self.nlp.tokenizer
        self.scorer = scorer #or SerrantScorer(nlp=self.nlp)
    
    def evaluate(self,
                 orig=Path(wd / "orig.txt"),
                 label=Path(wd / "label.txt")):
        with open(orig, 'r') as forig:
            input_lines = forig.readlines()
        # tokenization
        if not self.model.has_tokenizer_detokenizer:
            original_tokens = []
            for sent in input_lines:
                tokens = self.tokenizer.tokenize(sent)
                original_tokens.append(tokens)
            model_in = original_tokens
        else:
            model_in = input_lines
        # predict
        model_results = self.model.predict(model_in)

if __name__ == '__main__':
    fe = FileEval()
    fe.evaluate()