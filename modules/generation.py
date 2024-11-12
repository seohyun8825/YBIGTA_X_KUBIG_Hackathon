from transformers import pipeline
import faiss
from embedding import EmbeddingHandler

class AnswerGenerator:
    def __init__(self, model_handler, chunk_size=512, answer_aggregation="concat"):
        self.model_handler = model_handler
        self.chunk_size = chunk_size
        self.answer_aggregation = answer_aggregation
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            device=0 if model_handler.device == 'cuda' else -1,
            pad_token_id=model_handler.tokenizer.eos_token_id
        )

    def generate_answer_and_collect_results(self, question, data, top_k=10):
        # 질문을 임베딩하여 벡터화한 뒤, L2 정규화를 수행한다
        query_vector = self.model_handler.get_embedding(question).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # 저장된 FAISS 인덱스를 불러오고, top-k개의 유사한 문서를 검색한다
        index = faiss.read_index("faiss_index.bin")
        distances, indices = index.search(query_vector, k=top_k)
        
        # 검색된 문서의 내용을 모아서 context로 사용한다
        contexts = [
            data[int(i)]["documents"] if isinstance(data[int(i)]["documents"], str)
            else " ".join(data[int(i)]["documents"])
            for i in indices[0]
        ]

        # 모든 context를 하나의 텍스트로 합치고, chunk 크기에 따라 토큰을 나눈다
        context_text = " ".join(contexts)
        tokenized_text = self.model_handler.tokenizer(context_text, truncation=False, return_tensors="pt")
        token_chunks = tokenized_text["input_ids"][0].split(self.chunk_size)
        
        # 각 chunk에 대해 부분 답변을 생성한다
        partial_answers = []
        for chunk in token_chunks:
            prompt = f"Q: {question}\nContext: {self.model_handler.tokenizer.decode(chunk)}\nA:"
            inputs = self.model_handler.tokenizer(prompt, return_tensors="pt").to(self.model_handler.device)
            generated_tokens = self.model_handler.model.generate(inputs["input_ids"], max_new_tokens=50)
            partial_answer = self.model_handler.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).split("A:")[-1].strip()
            partial_answers.append(partial_answer)

        # 지정된 결합 방식을 사용하여 부분 답변을 하나의 답변으로 결합한다
        if self.answer_aggregation == "concat":
            answer = " ".join(partial_answers)
        elif self.answer_aggregation == "majority_vote":
            answer = max(set(partial_answers), key=partial_answers.count)  # 가장 자주 등장한 답변을 선택
        elif self.answer_aggregation == "best_match":
            answer = sorted(partial_answers, key=lambda x: self.model_handler.get_embedding(x) @ query_vector.T)[0]  # 임베딩 유사도에 따라 가장 적합한 답변 선택
        else:
            answer = " ".join(partial_answers)  # 부분 답변을 단순히 연결

        # return
        return {
            "question": question,     # 원본 질문
            "answer": answer,         # 생성된 최종 답변
            "contexts": contexts,     # 검색된 문서의 context들 리스트

            # "ground_truth": data[0].get("response", None)
        }
