# Digit Recognizer
Kaggle Competition : https://www.kaggle.com/c/digit-recognizer

## Preparation
```
mkdir data/
```
Download [all.zip](https://www.kaggle.com/c/3004/download-all) and put files in data/

## Usage

### Hyperparameters in flags.py
`model_type` : cnn / dnn  
`units` : numbers of neuron of layers for DNN  
`filter` : numbers of filters in each CNN layer  
`kernel` : kernel size  
`dp` : drop out rate  
`batch_size` : batch size / one training step  

### Train
for DNN (96%):
```
python main.py --mode train --model_type dnn
```

for CNN (99%):
```
python main.py --mode train --model_type cnn
```

### Test
for output:
```
python main.py --mode test [--load (step)]
```
the output file would be data/prediction.csv

## Files

### Folders
`data/` : all data ( train.csv / test.csv )  
`model/` : store models 

### Files
`flags.py` : all setting  
`utils.py` : get date batch  
`main.py` : main function  
`model.py` : model structure  

