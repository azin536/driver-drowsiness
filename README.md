# driver-drowsiness

# Train
There is a cnn in the file `src.model_building`. This model will get trained to be able to predict whether an eye is open or closed. To train this model you should get the [dataset](https://data-flair.training/blogs/download-driver-drowsiness-detection-project-data/), and train the model using following command:
```
python train.py
```

# Inference
For serving the trained and saved model, following command will work:
```
python inference.py
```

# Notes
* For playing a sound, there is a .wav file in the repo
* For detecting the faces and eyes there are haar cascade file in the repo.
* Shuffling data is more important than you think.


[reference](https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/)
