
import json
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

# 데이터 로드 코드
# JSONL 파일 로드 함수
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {json.loads(line)["_id"]: json.loads(line) for line in f}

# QRELS 파일 로드 함수
def load_qrels(file_path):
    qrels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 헤더 스킵
        for line in f:
            qid, docid, _ = line.strip().split("\t")
            qrels[qid] = docid
    return qrels

# 데이터 로드
corpus = load_jsonl("/home/kilab_ndw/KorQuAD_2.1/corpus.jsonl")
queries = load_jsonl("/home/kilab_ndw/KorQuAD_2.1/queries.jsonl")
qrels = load_qrels("/home/kilab_ndw/KorQuAD_2.1/qrels/test.tsv")  # test 셋 사용

# 평가 데이터 준비
eval_data = []
for qid, docid in qrels.items():
    query_text = queries[qid]["text"]
    doc_text = corpus[docid]["text"]
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

# DataLoader 생성
eval_dataset = DPRDataset(eval_data, question_tokenizer, context_tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

# 모델 평가
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_encoder.to(device)
context_encoder.to(device)

question_encoder.eval()
context_encoder.eval()

# 평가 메트릭을 위한 리스트
labels_list = []
predictions_list = []

with torch.no_grad():
    for batch in eval_dataloader:
        q_input_ids = batch["question_input_ids"].to(device)
        q_attention_mask = batch["question_attention_mask"].to(device)
        c_input_ids = batch["context_input_ids"].to(device)
        c_attention_mask = batch["context_attention_mask"].to(device)

        # 인코더로부터 임베딩 추출
        q_emb = question_encoder(q_input_ids, attention_mask=q_attention_mask).pooler_output
        c_emb = context_encoder(c_input_ids, attention_mask=c_attention_mask).pooler_output

        # Cosine 유사도로 예측값 계산
        cosine_similarity = torch.nn.functional.cosine_similarity(q_emb, c_emb)

        # 라벨과 예측값 저장
        labels_list.extend([1] * len(cosine_similarity))  # 실제로 relevant한 경우 1
        predictions_list.extend(cosine_similarity.cpu().numpy())  # 예측값

# 평균 정밀도 (Average Precision) 계산
ap = average_precision_score(labels_list, predictions_list)

print(f"Average Precision: {ap:.4f}")