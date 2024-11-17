from typing import Any, Dict
from transformers import EvalPrediction
import evaluate
import re

from aat.training.evaluate import compute_validation_metrics

class ComputeMetrics():
    def __init__(self, tokenizer):
        
        self.tokenizer = tokenizer
        self.captioning_metrics = evaluate.combine(
            [
                evaluate.load("bleu", keep_in_memory=True),
                evaluate.load("rouge", keep_in_memory=True),
                evaluate.load("meteor", keep_in_memory=True),
            ]
        )
        self.wer_compute = evaluate.load("wer")

        return

    def __call__(self, predictions=None, label_ids=None, losses=None, inputs=None, prefix_ids=None, generated_ids=None, **kwargs) -> Dict:
        inputs_ids = inputs
        generations_normalized = []
        
        prefixes_text = self.tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)
        
        generated_sentences = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for sentence in generated_sentences:
            sentence: str
            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("\n", " ")
            sentence = sentence.strip()
            sentence = sentence.rstrip()
            sentence = sentence.lower()
            
            sentence = re.sub(r"[^a-z\s\-\d]", "", sentence)
            
            generations_normalized.append(sentence)

        all_references = self.tokenizer.batch_decode(inputs_ids, skip_special_tokens=True)

        target_generations_normalized = []
        for i, (prefix, reference) in enumerate(zip(prefixes_text, all_references)):
            reference: str
            reference = reference[len(prefix):]
            reference = reference.replace("\n", " ")
            reference = reference.rstrip()
            reference = reference.strip()
            reference = reference.lower()

            reference = re.sub(r"[^a-z\s\-\d]", "", reference)

            target_generations_normalized.append([ reference ])

        assert len(generations_normalized) > 0, f"len(generations)={len(generations_normalized)}"
        assert len(target_generations_normalized) == len(generations_normalized), f"len(target_generations) == len(generations): {len(target_generations_normalized)} == {len(generations_normalized)}"

        validation_metrics = compute_validation_metrics(generations_normalized, target_generations_normalized, wer_compute=self.wer_compute, captioning_metrics=self.captioning_metrics)
        print("compute metrics:", validation_metrics)

        return validation_metrics


