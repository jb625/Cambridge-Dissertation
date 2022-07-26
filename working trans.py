#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!pip install pytorch-forecasting
import warnings
import torch as pt
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

import pytorch_forecasting as ptf
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
print("worked")
data = pd.DataFrame()
data = pd.read_csv('C:/Users/james/OneDrive/Documents/QQQ.csv')
data["Close"].plot()
data["Oil_Price"].fillna(-1)


data.replace(to_replace = 0, value = -1)
print(data.isnull().values.any())
data['Oil_Price'].replace(to_replace= np.nan, method='ffill')
#data["Date"] = pd.to_datetime(data["Date"], format="%y-%m-%d")
data["Date"] = data["Date"].apply(pd.to_datetime)
data["time_idx"] = data.index
data["Group"]= 1


#data["Date"] = data["Date"].apply(pd.to_datetime)
max_prediction_length = 206
max_encoder_length = 24
training_cutoff =len(data.index) - max_prediction_length
#"Volume","Open","High","Low","Interest_Rate","Oil_Price"

#x = TimeSeriesDataSet()
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff], time_idx = "time_idx", target = "Close", 
    group_ids = ["Group"], time_varying_known_reals=["time_idx"], time_varying_unknown_reals=["Close","Volume","Open","High","Low","Interest_Rate","Oil_Price" ], min_encoder_length=0,
    max_encoder_length=max_encoder_length, min_prediction_length=1 ,min_prediction_idx = 0,
    max_prediction_length=max_prediction_length, target_normalizer=GroupNormalizer(
        groups=["Group"], transformation="softplus"),
    categorical_encoders={
        'Volume':ptf.data.encoders.NaNLabelEncoder(add_nan=True),
        'Open':ptf.data.encoders.NaNLabelEncoder(add_nan=True),
        'Low':ptf.data.encoders.NaNLabelEncoder(add_nan=True),
        'High':ptf.data.encoders.NaNLabelEncoder(add_nan=True),
        'Interest_Rate':ptf.data.encoders.NaNLabelEncoder(add_nan=True),
        'Oil_Price':ptf.data.encoders.NaNLabelEncoder(add_nan=True)
    },
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=False)
batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size *10 , num_workers=0)

print(val_dataloader)
print(train_dataloader)
actuals = pt.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()

pl.seed_everything(42)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=20,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=16,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

import pickle

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=3,
    max_epochs=20,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(2, 8),
    hidden_continuous_size_range=(2, 8),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.03, 0.09),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=8),
    reduce_on_plateau_patience=2,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
mae = (actuals - predictions).abs().mean()
mse = mae**2
print("MAE: ", mae)
print("MSE: ", mse)


raw_predictions, x = best_tft.predict(val_dataloader, mode = 'raw', return_x=True)
  
predictions, x = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, raw_predictions)
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

plt.figure(figsize=(12, 6))

plt.plot(real_stock_price, color='black', label='QQQ Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted QQQ Stock Price')

plt.title('QQQ Stock Price Prediction')
plt.xlabel('Time')
plt.xlabel('QQQ')
plt.legend()
plt.show()


# In[ ]:




