import json
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

# 데이터 로드 코드
# JSONL 파일 로드 함수
def load_jsonl_c(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {json.loads(line)["id"]: json.loads(line) for line in f}

def load_jsonl_q(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {json.loads(line)["query_id"]: json.loads(line) for line in f}

def load_qrels(file_path):
    qrels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = data["query_id"]
            doc_ids = [passage["docid"] for passage in data.get("positive_passages", [])]
            qrels[qid] = doc_ids  # 여러 개의 문서를 리스트로 저장
    return qrels

# 데이터 로드
corpus = load_jsonl_c("/home/kilab_ndw/MrTydi/doc_512.jsonl")
queries = load_jsonl_q("/home/kilab_ndw/MrTydi/test.jsonl")
qrels = load_qrels("/home/kilab_ndw/MrTydi/test.jsonl")  # test 셋 사용

# 평가 데이터 준비
eval_data = []
for qid, docids in qrels.items():
    if qid not in queries:
        print(f"Warning: Query ID {qid} not found in queries.")
        continue
    query_text = queries[qid]["query"]

    for docid in docids:
        if docid not in corpus:
            print(f"Warning: Document ID {docid} not found in corpus.")
            continue
        doc_text = corpus[docid]["contents"]
        eval_data.append((query_text, doc_text, 1))  # 1은 긍정적인 문서 (relevant)

# DPR 모델 및 데이터셋 생성
# 저장된 모델과 토크나이저 로드
question_encoder = DPRQuestionEncoder.from_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_question_encoder")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_question_tokenizer")

context_encoder = DPRContextEncoder.from_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_context_encoder")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_context_tokenizer")

class DPRDataset(Dataset):
    def __init__(self, data, question_tokenizer, context_tokenizer):
        self.data = data
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, context, _ = self.data[idx]

        query_enc = self.question_tokenizer(query, padding="max_length", truncation=True, return_tensors="pt", max_length=512)
        context_enc = self.context_tokenizer(context, padding="max_length", truncation=True, return_tensors="pt", max_length=512)

        return {
            "question_input_ids": query_enc["input_ids"].squeeze(0),
            "question_attention_mask": query_enc["attention_mask"].squeeze(0),
            "context_input_ids": context_enc["input_ids"].squeeze(0),
            "context_attention_mask": context_enc["attention_mask"].squeeze(0),
        }

# 문서 임베딩을 미리 계산하여 저장
context_embeddings = {}

# 문서 임베딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder.to(device)

for docid, doc_data in corpus.items():
    doc_text = doc_data["contents"]
    doc_enc = context_tokenizer(doc_text, padding="max_length", truncation=True, return_tensors="pt", max_length=512)
    doc_input_ids = doc_enc["input_ids"].to(device)
    doc_attention_mask = doc_enc["attention_mask"].to(device)

    # 문서 임베딩 생성
    with torch.no_grad():
        doc_emb = context_encoder(doc_input_ids, attention_mask=doc_attention_mask).pooler_output
        context_embeddings[docid] = doc_emb.cpu()

# 문서 임베딩 저장
torch.save(context_embeddings, '/home/kilab_ndw/single-vector/DPR/models/context_embeddings_mrtydi.pt')

# DataLoader 생성
eval_dataset = DPRDataset(eval_data, question_tokenizer, context_tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

# 모델 평가
question_encoder.to(device)
question_encoder.eval()
context_encoder.eval()

# 평가 메트릭을 위한 리스트
labels_list = []
predictions_list = []

with torch.no_grad():
    for batch in eval_dataloader:
        q_input_ids = batch["question_input_ids"].to(device)
        q_attention_mask = batch["question_attention_mask"].to(device)

        # 질문 임베딩 생성
        q_emb = question_encoder(q_input_ids, attention_mask=q_attention_mask).pooler_output

        # 해당 문서의 임베딩을 가져와서 Cosine 유사도 계산
        for idx, docid in enumerate(batch["context_input_ids"]):
            doc_emb = context_embeddings.get(docid, None)
            if doc_emb is not None:
                cosine_similarity = torch.nn.functional.cosine_similarity(q_emb, doc_emb)
                labels_list.append(1)  # 실제로 relevant한 경우 1
                predictions_list.append(cosine_similarity.item())  # 예측값

# 평균 정밀도 (Average Precision) 계산
ap = average_precision_score(labels_list, predictions_list)

print(f"Average Precision: {ap:.4f}")
