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

    def tokenize(self, in_filename: Path, out_filename: Path=None):
        """Converts file defined by in_filename to word tokenized format"""
        out_filename = out_filename or in_filename.with_suffix(".tok")
        with open(in_filename, 'r') as fin:
            orig_lines = fin.readlines()
        orig_tokens_text = [" ".join([token.text for token in self.tokenizer(sent.strip())]) + "\n" for sent in orig_lines]
        with open(out_filename, 'w') as fout:
            fout.writelines(orig_tokens_text)
        return out_filename
    
    def generate_m2(self, original_tok: Path, corrected_tok: Path, out_filename: Path=None):
        """Generate annotated m2 file from correction"""
        out_filename = out_filename or corrected_tok.with_suffix(".m2")
        proc = subprocess.run(["serrant_parallel", "-orig", str(original_tok), "-cor", str(corrected_tok), "-out", str(out_filename)])
        return out_filename
    
    def serrant_evaluate(self, model_m2: Path, label_m2: Path):
        """Evaluate model results against label using serrant"""
        proc = subprocess.run(["serrant_compare", "-hyp", str(model_m2), "-ref", str(label_m2)])
    
    def m2scorer_evaluate(self, model_tok, label_m2):
        """Evaluate model results against label using m2scorer"""
        proc = subprocess.run([str(Path(wd / "m2scorer" / "m2scorer")), str(model_tok), str(label_m2)])
        

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
    # Evaluate against our essay data
    orig = Path(wd / "ielts_essays" / "orig.txt")
    model_results = Path(wd / "ielts_essays" / "model_results.txt")
    label = Path(wd / "ielts_essays" / "label.txt")

    fe = FileEval()

    orig_tok = fe.tokenize(orig)
    model_tok = fe.tokenize(model_results)
    label_tok = fe.tokenize(label)

    model_m2 = fe.generate_m2(orig_tok, model_tok)
    label_m2 = fe.generate_m2(orig_tok, label_tok)

    fe.serrant_evaluate(model_m2, label_m2)
    fe.m2scorer_evaluate(model_tok, label_m2)