import os
import torch
import json
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

# 작업 디렉토리 설정
base_path = "fiqa"  # 데이터가 저장된 디렉토리 경로 (절대 경로 또는 상대 경로로 수정)
save_path = "model_data_eng"  # 모델 저장 경로

# 디렉토리 목록 확인
print(os.listdir(base_path))

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# DPR Class
class DPR:
    def __init__(self, model_path: tuple):
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0]).to(device).eval()

        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1]).to(device).eval()

    def encode_queries(self, queries, batch_size=16):
        query_embeddings = []
        with torch.no_grad():
            for start in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
                batch = queries[start:start + batch_size]
                encoded = self.q_tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
                model_out = self.q_model(**encoded)
                query_embeddings.extend(model_out.pooler_output.cpu())
        return torch.stack(query_embeddings)

    def encode_corpus(self, corpus, batch_size=8):
        corpus_embeddings = []
        with torch.no_grad():
            for start in tqdm(range(0, len(corpus), batch_size), desc="Encoding corpus"):
                batch = corpus[start:start + batch_size]
                titles = [doc.get("title", "") for doc in batch]
                texts = [doc["text"] for doc in batch]
                encoded = self.ctx_tokenizer(titles, texts, truncation="longest_first", padding=True, max_length=512, return_tensors="pt").to(device)
                model_out = self.ctx_model(**encoded)
                corpus_embeddings.extend(model_out.pooler_output.cpu())
        return torch.stack(corpus_embeddings)


# 데이터 로드 함수
def load_beir_data(base_path):
    # Load corpus
    corpus = {}
    with open(os.path.join(base_path, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc["text"]}

    # Load queries
    queries = {}
    with open(os.path.join(base_path, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line)
            queries[query["_id"]] = query["text"]

    # Load qrels (Ground truth relevance)
    qrels = {}
    with open(os.path.join(base_path, "qrels/dev.tsv"), "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            query_id, corpus_id, score = line.strip().split("\t")  # query-id, corpus-id, score로 수정
            if query_id not in qrels:
                qrels[query_id] = []
            qrels[query_id].append((corpus_id, int(score)))  # doc_id -> corpus_id로 수정

    return corpus, queries, qrels

# 문서 임베딩 저장
def save_corpus_embeddings(dpr, corpus, save_path):
    os.makedirs(save_path, exist_ok=True)
    corpus_list = list(corpus.values())
    corpus_embeddings = dpr.encode_corpus(corpus_list)
    torch.save(corpus_embeddings, os.path.join(save_path, "corpus_embeddings.pt"))
    torch.save(corpus_list, os.path.join(save_path, "corpus_metadata.pt"))

# 검색 및 평가
def evaluate(dpr, queries, corpus, qrels, save_path):
    corpus_embeddings = torch.load(os.path.join(save_path, "corpus_embeddings.pt"))
    corpus_metadata = torch.load(os.path.join(save_path, "corpus_metadata.pt"))
    corpus_ids = list(corpus.keys())

    total_correct = 0
    total_queries = 0

    for query_id, query_text in tqdm(queries.items(), desc="Evaluating queries"):
        if query_id not in qrels:
            continue  # Skip if no ground truth

        query_embedding = dpr.encode_queries([query_text])
        scores = torch.matmul(query_embedding, corpus_embeddings.T)
        top_k = torch.topk(scores, k=5)

        retrieved_docs = [corpus_ids[idx] for idx in top_k.indices[0]]
        ground_truth_docs = {corpus_id for corpus_id, _ in qrels[query_id]}  # corpus_id로 수정

        correct = sum(1 for doc in retrieved_docs if doc in ground_truth_docs)
        total_correct += correct
        total_queries += 1

    accuracy = total_correct / total_queries if total_queries > 0 else 0
    print(f"\n🔹 평가 결과: Top-5 Accuracy = {accuracy:.4f}")


# 실행 코드
base_path = "fiqa"  # 데이터 경로 (절대 경로 또는 상대 경로로 수정)
save_path = "model_data_eng"  # 모델 저장 경로

print("데이터 로드 중...")
corpus, queries, qrels = load_beir_data(base_path)

print("모델 초기화 중...")
dpr = DPR(model_path=("facebook/dpr-question_encoder-single-nq-base",
                      "facebook/dpr-ctx_encoder-single-nq-base"))

print("문서 임베딩 생성 및 저장 중...")
save_corpus_embeddings(dpr, corpus, save_path)

print("모델 평가 중...")
evaluate(dpr, queries, corpus, qrels, save_path)