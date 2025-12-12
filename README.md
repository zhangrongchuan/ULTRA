## CLASSIFICATION
### Data Preparing
```bash 
python database2graph.py
python task2dataset.py
```

### Training
```bash
python script/run.py -c config/transductive/inference.yaml --dataset RelBenchF1 --epochs 20 --bpe null --gpus "[0]" --ckpt null
  ```