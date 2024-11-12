import openai
import logging
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness, context_recall
import os
from datasets import Dataset

class RAGASEvaluator:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수 'OPENAI_API_KEY'를 설정하십시오.")

    def evaluate_with_ragas(self, results):
        # 평가할 데이터 샘플 생성
        data_samples = {
            'question': [res["question"] for res in results],
            'answer': [res["answer"] for res in results],
            'contexts': [res["contexts"] for res in results],
            'ground_truth': [res["ground_truth"] for res in results]
        }
        
        dataset = Dataset.from_dict(data_samples)

        # 평가 수행
        score = evaluate(dataset, metrics=[answer_relevancy, context_precision, faithfulness, context_recall])
        logging.info("평가 점수: %s", score)
        return score
