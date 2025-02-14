import json
from torch.utils.data import Dataset, DataLoader
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
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
qrels_train = load_qrels("/home/kilab_ndw/KorQuAD_2.1/qrels/train.tsv")
qrels_dev = load_qrels("/home/kilab_ndw/KorQuAD_2.1/qrels/dev.tsv")

# 훈련 데이터 준비
train_data = []
for qid, docid in qrels_train.items():
    query_text = queries[qid]["text"]
    doc_text = corpus[docid]["text"]
    train_data.append((query_text, doc_text))

# 검증 데이터 준비
dev_data = []
for qid, docid in qrels_dev.items():
    query_text = queries[qid]["text"]
    doc_text = corpus[docid]["text"]
    dev_data.append((query_text, doc_text))

# DPR 모델 및 데이터셋 생성
# 질문 인코더
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

# 문서 인코더
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")


class DPRDataset(Dataset):
    def __init__(self, data, question_tokenizer, context_tokenizer):
        self.data = data
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, context = self.data[idx]

        query_enc = self.question_tokenizer(query, padding="max_length", truncation=True, return_tensors="pt",
                                            max_length=512)
        context_enc = self.context_tokenizer(context, padding="max_length", truncation=True, return_tensors="pt",
                                             max_length=512)

        return {
            "question_input_ids": query_enc["input_ids"].squeeze(0),
            "question_attention_mask": query_enc["attention_mask"].squeeze(0),
            "context_input_ids": context_enc["input_ids"].squeeze(0),
            "context_attention_mask": context_enc["attention_mask"].squeeze(0),
        }


# DataLoader 생성
train_dataset = DPRDataset(train_data, question_tokenizer, context_tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

dev_dataset = DPRDataset(dev_data, question_tokenizer, context_tokenizer)
dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

# DPR 파인 튜닝
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question_encoder.to(device)
context_encoder.to(device)

optimizer = optim.AdamW(list(question_encoder.parameters()) + list(context_encoder.parameters()), lr=5e-5)
loss_fn = nn.CosineEmbeddingLoss()

num_epochs = 10  # 에폭수를 늘림
early_stop_count = 0  # early stopping 횟수
best_loss = float("inf")  # 초기 최상의 손실값


# 평가 함수 추가
def evaluate(model, dataloader):
    model.eval()
    labels_list = []
    predictions_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["question_input_ids"].to(device)
            q_attention_mask = batch["question_attention_mask"].to(device)
            c_input_ids = batch["context_input_ids"].to(device)
            c_attention_mask = batch["context_attention_mask"].to(device)

            q_emb = question_encoder(q_input_ids, attention_mask=q_attention_mask).pooler_output
            c_emb = context_encoder(c_input_ids, attention_mask=c_attention_mask).pooler_output

            cosine_similarity = torch.nn.functional.cosine_similarity(q_emb, c_emb)

            labels_list.extend([1] * len(cosine_similarity))  # relevant한 경우 1
            predictions_list.extend(cosine_similarity.cpu().numpy())  # 예측값

    ap = average_precision_score(labels_list, predictions_list)
    return ap


# 훈련 및 검증
for epoch in range(num_epochs):
    question_encoder.train()
    context_encoder.train()

    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        q_input_ids = batch["question_input_ids"].to(device)
        q_attention_mask = batch["question_attention_mask"].to(device)
        c_input_ids = batch["context_input_ids"].to(device)
        c_attention_mask = batch["context_attention_mask"].to(device)

        q_emb = question_encoder(q_input_ids, attention_mask=q_attention_mask).pooler_output
        c_emb = context_encoder(c_input_ids, attention_mask=c_attention_mask).pooler_output

        labels = torch.ones(q_emb.size(0)).to(device)
        loss = loss_fn(q_emb, c_emb, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    # 검증 데이터셋 평가
    dev_ap = evaluate(question_encoder, dev_dataloader)
    print(f"Dev Average Precision: {dev_ap:.4f}")

    # Early stopping 체크
    if total_loss < best_loss:
        best_loss = total_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 2:  # 손실이 2번 이상 증가하면 학습 멈춤
        print("Early stopping triggered!")
        break

# 모델 저장
question_encoder.save_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_question_ealry_stopping_encoder")
context_encoder.save_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_context_stopping_encoder")
question_tokenizer.save_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_question_stopping_tokenizer")
context_tokenizer.save_pretrained("/home/kilab_ndw/single-vector/DPR/models/dpr_finetuned_context_stopping_tokenizer")
