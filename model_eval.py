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
    
    def evaluate(self,
                 orig: Path,
                 model: Path,
                 label: Path,
                 model_m2_out: Path=None,
                 label_m2_out: Path=None):
        # generate annotated m2 files
        model_m2_out = model_m2_out or model.with_suffix(".m2")
        label_m2_out = label_m2_out or label.with_suffix(".m2")
        proc_model = subprocess.run(["serrant_parallel", "-orig", str(orig), "-cor", str(model), "-out", str(model_m2_out)])
        proc_label = subprocess.run(["serrant_parallel", "-orig", str(orig), "-cor", str(label), "-out", str(label_m2_out)])
        # evaluate model results against label
        proc_eval_serrant = subprocess.run(["serrant_compare", "-hyp", str(model_m2_out), "-ref", str(label_m2_out)])
        proc_eval_m2scorer = subprocess.run([str(Path(wd / "m2scorer" / "m2scorer")), str(model), str(label_m2_out)])

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
    orig = Path(wd / "ielts_essays" / "orig.txt")
    model_results = Path(wd / "ielts_essays" / "model_results.txt")
    label = Path(wd / "ielts_essays" / "label.txt")

    fe = FileEval()

    orig_tok = fe.tokenize(orig)
    model_tok = fe.tokenize(model_results)
    label_tok = fe.tokenize(label)

    fe.evaluate(orig=orig_tok,
                model=model_tok,
                label=label_tok)