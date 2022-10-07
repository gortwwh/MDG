# MDG: A Multi-Task Dynamic Graph Generation Framework for Multivariate Time Series Forecasting
## Data Preparation
[METR_LA](https://github.com/chnsh/DCRNN_PyTorch) [PeMS](https://github.com/divanoresia/Traffic)
```python
unzip data/metr-la.h5.zip -d data/
mkdir -p data/METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
```
## Train Model
```python
python train.py --config_filename=data/model/para_la.yaml --temperature=0.5
```
## Acknowledgments
[DCRNN-PyTorch](https://github.com/chnsh/DCRNN_PyTorch), [GTS](https://github.com/chaoshangcs/GTS)
