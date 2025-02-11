which nvidia-container-runtime-hook
출처: https://splendidlolli.tistory.com/648 [자꾸 생각나는 체리쥬빌레:티스토리
docker start my-server
;s
docker ps
exit
ls
cd ..
ls
cd kilab_ndw/
ls
clone https://github.com/thakur-nandan/beir-ColBERT.git
git clone https://github.com/thakur-nandan/beir-ColBERT.git
git status
ls
cd beir-ColBERT/ 
ls
groups
conda env create -f conda_env.yml
pip install conda
conda env create -f conda_env.yml\
conda env create -f conda_env.yml
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
conda env create -f conda_env.yml
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -f conda_env.yml
conda init
source ~/.bashrc
conda env create -f conda_env.yml
conda activate colbert-v0.2
ls
vim prepare_beir_data.py
python3 prepare_beir_data.py 
ls
vim check_cpu.py
python3 check_cpu.py 
python -m colbert.data_prep   --dataset nfcorpus   --split "test"   --collection ~/beir_data/nfcorpus_collection.tsv   --queries ~/beir_data/nfcorpus_queries.tsv
python -m torch.distributed.launch   --nproc_per_node=2 -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint ~/colbert_checkpoint/colbert_model.pth   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
pip install protobuf==3.20.*
python -m torch.distributed.launch   --nproc_per_node=2 -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint ~/colbert_checkpoint/colbert_model.pth   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
nvidia-smi
# 2. NCCL 관련 환경 변수 설정
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
# 3. nproc_per_node=1로 실행 (GPU 1개일 경우)
python -m torch.distributed.launch   --nproc_per_node=1 -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint ~/colbert_checkpoint/colbert_model.pth   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
# 4. OMP_NUM_THREADS 설정
export OMP_NUM_THREADS=1
# 1. 환경 변수 설정
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# 2. 분산 훈련을 비활성화하고 단일 프로세스로 실행
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint ~/colbert_checkpoint/colbert_model.pth   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
ls
cd ..
ls
cd beir-ColBERT/
ls
vim download_colbert_huggingface.py
python3 download_colbert_huggingface.py 
vim download_colbert_huggingface.py
python3 download_colbert_huggingface.py 
vim download_colbert_huggingface.py
python3 download_colbert_huggingface.py 
vim download_colbert_huggingface.py
python3 download_colbert_huggingface.py 
pip install .
python3 download_colbert_huggingface.py 
vim download_colbert_huggingface.py
python3 download_colbert_huggingface.py 
vim download_colbert_huggingface.py
python3 download_colbert_huggingface.py 
ls
git lfs clone https://huggingface.co/colbert-ir/colbertv2.0
pip install git-lfs
git lfs install
sudo apt-get install git-lfs
git lfs 
git lfs clone https://huggingface.co/colbert-ir/colbertv2.0
ls
cd colbertv2.0/
ls
cd ..
ls
dir
ls
cd colbertv2.0/
ls
cd ..
ls
ls nfcorpus
cd ..
ls
beir_da
ls beir_data/
ls
cd beir-ColBERT/
ls
cd colbertv2.0/
ls
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint ~/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/../beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
rm -rf ~/colbert_index/nfcorpus_index
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint ~/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/../beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
rm -rf ~/colbert_index/nfcorpus_index
ls /home/kilab_ndw/
ls /home/kilab_ndw/beir-ColBERT/
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/colbertv2.0/colbert_model/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
rm -rf ~/colbert_index/nfcorpus_index
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin 
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/colbert_model/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin 
rm -rf ~/colbert_index/nfcorpus_index
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
ls
vim /home/kilab_ndw/anaconda3/envs/colbert-v0.2/lib/python3.7/site-packages/colbert/evaluation/load_model.py
vim /home/kilab_ndw/anaconda3/envs/colbert-v0.2/lib/python3.7/runpy.py 
vim /home/kilab_ndw/anaconda3/envs/colbert-v0.2/lib/python3.7/site-packages/colbert/evaluation/load_model.py
vim /home/kilab_ndw/anaconda3/envs/colbert-v0.2/lib/python3.7/site-packages/colbert/utils/utils.py 
ls
cd ..
ls
vim check_checkpoint_file.py
python3 check_checkpoint_file.py 
vim check_checkpoint_file.py
vim check_checkpoint_file.py\
python3 check_checkpoint_file.py 
ls
vim download_colbert_huggingface.py 
git lfs clone https://huggingface.co/answerdotai/answerai-colbert-small-v1
ls
ls answerai-colbert-small-v1/
cd answerai-colbert-small-v1/
ls
ls onnx/
dc ..
cd ..
ls
rm -rf ~/colbert_index/nfcorpus_index
cd answerai-colbert-small-v1/
ls
cd ..
ls
ls /home/kilab_ndw/
ls /home/kilab_ndw/beir-ColBERT/
ls /home/kilab_ndw/beir-ColBERT/answerai-colbert-small-v1/model.safetensors 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/answerai-colbert-small-v1/model.safetensors   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
vim /home/kilab_ndw/anaconda3/envs/colbert-v0.2/lib/python3.7/site-packages/torch/serialization.py 
ls /home/kilab_ndw/anaconda3/envs/colbert-v0.2/lib/python3.7/site-packages/torch
ls
cd colbertv2.0/
ls
cd ..
ls
vim check_checkpoint_file.py 
python3 check_checkpoint_file.py 
vim check_checkpoint_file.py 
python3 check_checkpoint_file.py 
ls
rm -rf ~/colbert_index/nfcorpus_index
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/colbertv2.0/colbert_model/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
ls /home/kilab_ndw
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin
rm -rf ~/colbert_index/nfcorpus_index
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
vim /home/kilab_ndw/beir-ColBERT/colbert/utils/utils.py 
rm -rf ~/colbert_index/nfcorpus_index\
rm -rf ~/colbert_index/nfcorpus_index
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
rm -rf ~/colbert_index/nfcorpus_index
vim /home/kilab_ndw/beir-ColBERT/colbert/utils/utils.py 
rm -rf ~/colbert_index/nfcorpus_index
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/beir_data/nfcorpus_collection.tsv   --experiment nfcorpus
rm -rf ~/colbert_index/nfcorpus_index
cd ..
ls
cd ..
ls
cd kilab_ndw
ls
rm -rf beir-ColBERT
ls
rm -rf beir_data/
rm -rf colbert_index/
rm -rf colbert_output/
ls
git clone https://github.com/NThakur20/beir-ColBERT.git
conda env create -f conda_env.yml
ls
cd beir-ColBERT/
conda env create -f conda_env.yml
conda env remove -n colbert-v0.2
conda env exit
conda env list
conda env base
conda base
conda env update bas
conda env remove -n colbert-v0.2
conda deactivate
conda env remove -n colbert-v0.2
conda env create -f conda_env.yml
conda update -n base -c defaults conda
conda activate colbert-v0.2
ls
ls /home/kilab_ndw/
ls /home/kilab_ndw/beir-ColBERT/
exit
conda activate colbert
conda activate colbert-v0.2
ls
cd beir-ColBERT/
ls
ls colbertv2.0/
cd ..
ls
cd beir-ColBERT/
vim download_dataset_nfcorpus.py
python3 download_dataset_nfcorpus.py 
ls
mv ./nfcorpus ./datasets/
ls
ls datasets/
cd datasets/
ls
mkdir nfcorpus
ls
mv corpus.jsonl nfcorpus/
ls
ls nfcorpus/
mv qrels/ nfcorpus/
ls
mv queries.jsonl nfcorpus/
ls
ls nfcorpus/
cd ..
python -m colbert.data_prep   --dataset nfcorpus   --split "test"   --collection ~/beir-ColBERT/datasets/nfcorpus/nfcorpus_collection.tsv   --queries ~/beir-ColBERT/datasets/nfcorpus/nfcorpus_queries.tsv
ls
cd datasets/
ls
cd nfcorpus/
ls
cd ..
cd..
c d..
cd ..
ls
ls /home/kilab_ndw/beir-ColBERT
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0/
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin 
ls
ls datasets/
ls datasets/nfcorpus/
python -m torch.distributed.launch   --nproc_per_node=2 -m colbert.index   --root ~/colbert_output \       
ls
cd ..
ls
cd beir-ColBERT/
pip install protobuf==3.20.0
python -m torch.distributed.launch   --nproc_per_node=2 -m colbert.index   --root ~/colbert_output \       
python -m torch.distributed.launch   --nproc_per_node=2 -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/datasets/nfcorpus/collection.tsv   --experiment nfcorpus
rm -rf ~/colbert_index/nfcorpus_index
d
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
python -m torch.distributed.launch   --nproc_per_node=1 -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/datasets/nfcorpus/collection.tsv   --experiment nfcorpus
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
python -m colbert.index   --root ~/colbert_output   --doc_maxlen 300   --mask-punctuation   --bsize 128   --amp   --checkpoint /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin   --index_root ~/colbert_index   --index_name nfcorpus_index   --collection ~/datasets/nfcorpus/collection.tsv   --experiment nfcorpus
ls /home/kilab_ndw/beir-ColBERT/colbertv2.0/pytorch_model.bin 
ls
ls colbertv2.0/
vim /home/kilab_ndw/beir-ColBERT/colbert/utils/utils.py
ls
ls colbertv2.0/
exit
ls
cd beir-ColBERT/
ls
fmux ls
tnyx
tmux
tmux ls
ls
cd ..
ls
cd ..
ls
cd kilab_nd
cd kilab_ndw/
ls
rm -r beir-ColBERT/
ls
rm -r colbert_index/
rm -r colbert_output/
ls
mkdir single-vectro
mkdir single-vector
ls
rm -r single-vectro
ls
mrdir multi-vector
mkdir multi-vector
ls
ls single-vector/
ls
cd single-vector/
ls
cd ..
ls
cd ..
exit
ls
ls KorQuAD_2.1/
exit 
ls
exit
lls
ls
cd single-vector/
ls
dc DPR/
ls
cd d
cd DPR/
ls
python3 finetuning.py
pip install torch
ls
python3 finetuning.py 
pip install transformers
python3 finetuning.py 
vim finetuning.py 
python3 finetuning.py 
ls
vim finetuning.py 
exit
ls
cd single-vector/
ls
cd DPR/
ls
python3 evaluate_dpr.py 
rm evaluate_dpr.py 
ls
exit
ls
cd single-vector/DPR/
ls
python3 evaluate_dpr.py 
vim evaluate_dpr.py 
python3 evaluate_dpr.py 
ls
cd ..
ls
cd multi-vector/
ls
git clone https://github.com/NThakur20/beir-ColBERT.git
conda env create -f conda_env.yml
conda env delete -f conda_env.yml
conda env remove -f conda_env.yml
conda env remove colbert-v0.2
ls
cd ..
ls
cd single-vector/DPR/
ls
rm finetuning.py
ls models/
dir
ls
cd models/
rm dpr_finetuned_context_encoder/
rm -r dpr_finetuned_context_encoder/
rm -r dpr_finetuned_context_tokenizer/
rm -r dpr_finetuned_question_e
rm -r dpr_finetuned_question_encoder/
rm -r dpr_finetuned_question_tokenizer/
ls
cd ..
ls
rm -r evaluate_dpr.py 
ls
cd ..
ls
cd ..
ls
cd single-vector/
cd DPR/
exit
ls
cd KorQuAD_2.1/
ls
ls qrels
ls
cd ..
ls
cd single-vector/
ls DPR/
ls
cd DPR/
vim finetuning.py 
python3 finetuning.py 
ls
tmux new -s DPR
tmux attach -t DPR
ls
cd ..
exit
tmux ls
tmux attach -t DPR
ls
cd multi-vector/
ls
cd beir-ColBERT/
ls
conda env list
conda env delete colbert-v0.2
conda env remove colbert-v0.2
conda env remove -h colbert-v0.2
conda env remove -f colbert-v0.2
conda env list
conda remove --name colbert-v0.2 --all
ls
conda env list
tmux list
tmux
tmux ls
tmux attach -t DPR
d
conda env create -f conda_env.yml
conda activate colbert-v0.2
tmux new -s ColBERT
tmux ls
tmux attach -t DPR
tmux ls
tmux attach -t DPR
ls
cd ..
ls
cd single-vector/
ls
cd DPR/
ls
vim finetuning.py 
rm finetuning.py 
cd ..
exit
ls
cd single-vector/DPR/
vim finetuning.py 
tmux attach -t DPR
ls
exit
ls
cd multi-vector/
ls
tar -zxvf triples.train.small.tar.gz
ls
tar -zxvf top1000.dev.tar.gz
tar -zxvf collection.tar.gz
ls
dir
ls
ls /home/
ls /home/kilab_ndw/
ls /home/kilab_ndw/multi-vector/
ls /home/kilab_ndw/multi-vector/triples.train.small.tsv
ls
cd ..
ls
mkdir MSMARCO
ls
cd multi-vector/
ls
mv /home/kilab_ndw/multi-vector/triples.train.small.tsv /home/kilab_ndw/MSMARCO/
ls
mv /home/kilab_ndw/multi-vector/triples.train.small.tsv.gz /home/kilab_ndw/MSMARCO/
mv /home/kilab_ndw/multi-vector/triples.train.small.tar.gz /home/kilab_ndw/MSMARCO/
mv /home/kilab_ndw/multi-vector/collection.tar.gz /home/kilab_ndw/MSMARCO/
mv /home/kilab_ndw/multi-vector/collection.tsv /home/kilab_ndw/MSMARCO/

mv /home/kilab_ndw/multi-vector/top1000.dev.tar.gz /home/kilab_ndw/MSMARCO/
ls
mv /home/kilab_ndw/multi-vector/qrels.train.tsv /home/kilab_ndw/MSMARCO/
ls
cd ..
ls
cd MSMARCO/
ls
cd ..
exit
ls
CUDA_VISIBLE_DEVICES="1" colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /path/to/MSMARCO/triples.train.small.tsv --root /root/to/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
exit
CUDA_VISIBLE_DEVICES="0" python -m --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
CUDA_VISIBLE_DEVICES="0" python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
pip install faiss
exit
s
ls
conda activate colbert-v0.2
CUDA_VISIBLE_DEVICES="0" colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
ls
CUDA_VISIBLE_DEVICES="0" colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
exit
ls
cd MSMARCO/
cd ..
ls
cd multi-vector/
ls
cd beir-ColBERT/
ls
mkdir experiments
ls
conda activate colbert-v0.2
tmux attach -t DPR
d
tmux ls
tmux attach -t ColBERT
tmux new -s ColBERT
ls
pip install -r requirements.txt
cd ,,
cd ..
ls
cd beir-ColBERT/
ls
CUDA_VISIBLE_DEVICES="0" colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
ls
cd ..
ls
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ColBERT
pip install -r requirements.txt
ls
cd ..
ls
rm -r ColBERT/
ls
ls beir-ColBERT/
ls
pip install colbert
tmux ls
tmux attach -t ColBERT
ls
cd beir-ColBERT/
which colbert
pip show colbert
which colbert
cd ..
ls
cd multi-vector/
ls
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ColBERT
pip install -r requirements.txt
cd ..
ls
cd beir-ColBERT/
pip install torch transformers faiss-gpu numpy
pip install scikit-learn
colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
tmux attach -t ColBERT
tmux ls
tmux new -s ColBERT
tmux ls
tmux attach -t DPR
s
tmux attach -t ColBERT
ls
cd ..
ls
cd ..
exit
tmux ls
tmux attach -t DPR
tmux attach -t ColBERT
ls
cd ..
exit
python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
conda activate colbert-v0.2
python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
;s
ls
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
pip uninstall torch torchvision torchaudio
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0
python --version
conda install python=3.8
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
exit
ls
cd multi-vector/
ls
cd beir-ColBERT/
CUDA_VISIBLE_DEVICES="0" python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
pip install faiss-gpu
CUDA_VISIBLE_DEVICES="0" python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
pip install mlflow
pip install pathlib ruamel.yaml
pip install mlflow
cuda env list
conda env list
conda env activate colbert-v0.2
conda activate colbert-v0.2
CUDA_VISIBLE_DEVICES="0" python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
CUDA_VISIBLE_DEVICES="0" python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
pip install protobuf==3.20.0
CUDA_VISIBLE_DEVICES="0" python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
ls
CUDA_VISIBLE_DEVICES="0" python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 16 --accum 1 --triples /home/kilab_ndw/MSMARCO/triples.train.small.tsv --root /home/kilab_ndw/multi-vector/beir-ColBERT/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
cd ..
exit
tmux ls
tmux attach -t DPR

tmux attach -t ColBERT
tmux attach -t DPR
ls
ls single-vector/
cd single-vector/DPR/
ls
ls models/
ls
vim finetuning.py 
ls
cd models/
ls
rm -r dpr_finetuned_context_e
rm -r dpr_finetuned_context_encoder/
rm -r dpr_finetuned_context_tokenizer/
rm -r dpr_finetuned_question_encoder/
rm -r dpr_finetuned_question_tokenizer/
ls
cd ..
tmux ls
tmux attach -t DPR
vim finetuning.py 
cd ..
ls
cd ..
ls
cd KorQuAD_2.1/
ls
cd qrels/
ls
vim dev.tsv 
vim train_nr.tsv 
ls 
cd ..
ls
cd ..
ls
cd multi-vector/
ls
tmux ls
tmux attach -t DPR
tmux attach -t ColBERT
ls
cd ..
ls
dc multi-vector/
cd multi-vector/
ls
rm -r ColBERT/
y
rm -r beir-ColBERT/
ls
git clone https://github.com/NThakur20/beir-ColBERT.git
conda env remove -n colbertv0.2
conda env list
conda env remove -n colbert-v0.2
ls
conda env list
ls
cd beir-ColBERT/
conda env create -f conda_env.yml
conda activate colbert-v0.2
cd ..
sl
ls
tmux ls
tmux new -s ColBERT
ls
tmux ls
tmux attach -t DPR
tmux attach -t ColBERT
ls
tmux attach -t ColBERT
ls
tmux ls
ls
cd multi-vector/
ls
rm -r beir-ColBERT/
ls
git clone https://github.com/NThakur20/beir-ColBERT.git
conda env remove -n colbert-v0.2
conda env list
ls
exit
