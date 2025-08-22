---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: group
    dtype: string
  - name: species
    dtype: string
  - name: id
    dtype: string
  - name: station
    dtype: string
  - name: year
    dtype: int16
  - name: color 
    dtype: int8
  - name: marks
    dtype: string
  - name: ymdh 
    dtype: int64
  - name: label
    dtype: int8
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train*.parquet"
  - split: test
    path: "data/test*.parquet"
---