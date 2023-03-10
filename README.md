# 4th Chalearn Face Anti-spoofing Workshop and Challenge@CVPR2023 —— Team SeaRecluse

## Step
### Install dependencies:
```bash
pip install -r requirements.txt
```

### Data preprocessing:
```
If you have full content data data, it should look like this in the folder
--orig_data
    -dev *
    -test *
    -train *
    -dev.txt *
    -dev_label.txt *
    -test.txt *
    -train_label.txt *
    -data_arrange.py
```

### Divide the dataset
```
# We use all the train data and 2/3 of the val data as the final training set, and the remaining val data as the validation dataset.

python data_arrange.py
```

### Start training
```
# We have usd a pre-trained model based on ImageNet provided by timm. Please ensure that your network is accessible to download this model.
# Put the enhanced data into the folder "./orig_data" for training.
# It is recommended that the model training is not less than 300 epoch, default is 500 epoch.
# The default batch size is 64, which requires at least 24G GPU memory for training.

python main.py
```

### Test your model
```
# When the training is over, you can copy the checkpoint.pth.tar in reulst to the root directory and use it for testing.

python test.py
```

## other
The Flops requirement of this competition model is less than 100G, but this is still a huge number. If the organizer has sufficient training resources and can use a larger backbone, we believe that better results will be achieved.

