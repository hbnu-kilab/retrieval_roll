import os
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# DPR Class
class DPR:
    def __init__(self, model_path: tuple):
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0], ignore_mismatched_sizes=True).to(device).eval()

        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1], ignore_mismatched_sizes=True).to(device).eval()

    def encode_queries(self, queries):
        with torch.no_grad():
            encoded = self.q_tokenizer(queries, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
            return self.q_model(**encoded).pooler_output.cpu()

    def encode_corpus(self, corpus, batch_size=32):
        titles = [doc.get("title", "") for doc in corpus]
        texts = [doc["text"] for doc in corpus]

        dataset = list(zip(titles, texts))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding corpus"):
                batch_titles, batch_texts = batch
                encoded = self.ctx_tokenizer(batch_titles, batch_texts, truncation="longest_first",
                                             padding=True, max_length=512, return_tensors="pt").to(device)
                batch_embeddings = self.ctx_model(**encoded).pooler_output.cpu()
                embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0)

# 데이터 로드 함수
def load_beir_data(base_path):
    with open(os.path.join(base_path, "corpus.jsonl"), "r", encoding="utf-8") as f:
        corpus = {doc["_id"]: {"title": doc.get("title", ""), "text": doc["text"]} for doc in map(json.loads, f)}

    with open(os.path.join(base_path, "queries.jsonl"), "r", encoding="utf-8") as f:
        queries = {query["_id"]: query["text"] for query in map(json.loads, f)}

    qrels = {}
    with open(os.path.join(base_path, "qrels/dev.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, corpus_id, _ = line.strip().split("\t")
            qrels.setdefault(query_id, []).append(corpus_id)

    return corpus, queries, qrels

# 문서 임베딩 저장
def save_corpus_embeddings(dpr, corpus, save_path, batch_size=32):
    os.makedirs(save_path, exist_ok=True)
    corpus_list = list(corpus.values())
    torch.save(dpr.encode_corpus(corpus_list, batch_size=batch_size), os.path.join(save_path, "corpus_embeddings.pt"))
    torch.save(list(corpus.keys()), os.path.join(save_path, "corpus_ids.pt"))

# 검색 및 평가
def evaluate(dpr, queries, qrels, save_path):
    corpus_embeddings = torch.load(os.path.join(save_path, "corpus_embeddings.pt"))
    corpus_ids = torch.load(os.path.join(save_path, "corpus_ids.pt"))

    total_correct = sum(
        sum(1 for doc in torch.topk(torch.matmul(dpr.encode_queries([query_text]), corpus_embeddings.T), k=5).indices[0] if corpus_ids[doc] in qrels.get(query_id, []))
        for query_id, query_text in tqdm(queries.items(), desc="Evaluating queries")
        if query_id in qrels
    )

    accuracy = total_correct / len(qrels) if qrels else 0
    print(f"\n🔹 평가 결과: Top-5 Accuracy = {accuracy:.4f}")

# 실행 코드
base_path, save_path = "fiqa", "model_data_eng"
corpus, queries, qrels = load_beir_data(base_path)
dpr = DPR(("facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-ctx_encoder-single-nq-base"))
save_corpus_embeddings(dpr, corpus, save_path, batch_size=32)  # 배치 크기 조절 가능
evaluate(dpr, queries, qrels, save_path)
