import torch
from typing import Any, Dict
from transformers import EvalPrediction
import evaluate
from evaluate import Metric
import re
from typing import List, Optional

from transformers import logging

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
        
        regexp_substitude = False
        
        prefixes_text = self.tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)
        
        generated_sentences = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for sentence in generated_sentences:
            sentence: str
            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("\n", " ")
            sentence = sentence.strip()
            sentence = sentence.rstrip()
            sentence = sentence.lower()
            
            if regexp_substitude:
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

            if regexp_substitude:
                reference = re.sub(r"[^a-z\s\-\d]", "", reference)

            target_generations_normalized.append([ reference ])

        assert len(generations_normalized) > 0, f"len(generations)={len(generations_normalized)}"
        assert len(target_generations_normalized) == len(generations_normalized), f"len(target_generations) == len(generations): {len(target_generations_normalized)} == {len(generations_normalized)}"

        validation_metrics = self.compute_validation_metrics(generations_normalized, target_generations_normalized, wer_compute=self.wer_compute, captioning_metrics=self.captioning_metrics)
        print("compute metrics:", validation_metrics)

        return validation_metrics


    @torch.no_grad()
    def compute_validation_metrics(self, generations: List[str], references: List[List[str]], wer_compute: Optional[Metric]=None, captioning_metrics: Optional[Metric]=None):

        wer_references = [ x[0] for x in references ]
        
        logger = logging.get_logger("compute_metrics")

        logger.warning(f"generations {generations[:10]}")
        logger.warning(f"wer_references {wer_references[:10]}")

        wer_score = 0.0
        if wer_compute is not None:
            wer_score = wer_compute.compute(predictions=generations, references=wer_references)

        validation_metrics = {
            "wer": wer_score
        }

        try:
            if captioning_metrics is not None:
                evaluate_bleu_results = captioning_metrics.compute(predictions=generations, references=references)
                logger.info(f"evaluate_bleu_results {evaluate_bleu_results}")

                validation_metrics["evaluate_bleu"] = evaluate_bleu_results['bleu'] * 100
                validation_metrics["evaluate_rouge1"] = evaluate_bleu_results['rouge1']
                validation_metrics["evaluate_rouge2"] = evaluate_bleu_results['rouge2']
                validation_metrics["evaluate_rougeL"] = evaluate_bleu_results['rougeL']
                validation_metrics["evaluate_rougeLsum"] = evaluate_bleu_results['rougeLsum']
                validation_metrics["evaluate_meteor"] = evaluate_bleu_results['meteor']
        except Exception as e:
            print("Catch eval exception", e)

        return validation_metrics

