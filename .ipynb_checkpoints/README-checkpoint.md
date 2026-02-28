配置项目相同环境的方法：

# 1. 创建全新的虚拟环境
python3 -m venv venv

# 2. 激活环境
source venv/bin/activate

# 3. 根据清单安装依赖
pip install -r requirements.txt

# latest error

2026-02-28 19:54:30,479 - WARNING - Retrying in 4s [Retry 3/5].
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████| 409M/409M [01:30<00:00, 4.54MB/s]
Epoch 1/10 [Train]:   0%|                                                                                              | 0/148 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/root/Master_Thesis/model_train/classifier/train.py", line 117, in <module>
    train()
  File "/root/Master_Thesis/model_train/classifier/train.py", line 58, in train
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG["epochs"]} [Train]"):
  File "/root/venv/lib/python3.12/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/root/venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 741, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/root/venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 801, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/venv/lib/python3.12/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/root/Master_Thesis/model_train/classifier/dataset.py", line 35, in __getitem__
    'rel_labels': torch.tensor(rel_label, dtype=torch.float),
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: new(): invalid data type 'str'