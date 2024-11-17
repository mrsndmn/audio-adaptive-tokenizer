from typing import List, Optional

import logging

import torch

from evaluate import Metric


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


@torch.no_grad()
def compute_validation_metrics(generations: List[str], references: List[List[str]], wer_compute: Optional[Metric]=None, captioning_metrics: Optional[Metric]=None):

    wer_references = [ x[0] for x in references ]

    logger.info(f"generations {generations[:10]}")
    logger.info(f"wer_references {wer_references[:10]}")

    wer_score = 0.0
    if wer_compute is not None:
        wer_score = wer_compute.compute(predictions=generations, references=wer_references)

    validation_metrics = {
        "validation/wer": wer_score
    }

    try:
        if captioning_metrics is not None:
            evaluate_bleu_results = captioning_metrics.compute(predictions=generations, references=references)
            logger.info(f"evaluate_bleu_results {evaluate_bleu_results}")

            validation_metrics["validation/evaluate_bleu"] = evaluate_bleu_results['bleu'] * 100
            validation_metrics["validation/evaluate_rouge1"] = evaluate_bleu_results['rouge1']
            validation_metrics["validation/evaluate_rouge2"] = evaluate_bleu_results['rouge2']
            validation_metrics["validation/evaluate_rougeL"] = evaluate_bleu_results['rougeL']
            validation_metrics["validation/evaluate_rougeLsum"] = evaluate_bleu_results['rougeLsum']
            validation_metrics["validation/evaluate_meteor"] = evaluate_bleu_results['meteor']
    except Exception as e:
        print("Catch eval exception", e)

    return validation_metrics

