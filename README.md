# PA 3: Image Captioning with CNN-LSTM Model


## Contributors
Zecheng, Wenqian, Lainey, Mingkun



## Task

This project focus on constructing an encoder-decoder neural network architecture
that generates captions for the given image. 

In our study, we use the COCO-2014 dataset, where COCO stands for "Common Objects in Contexts,"
as the training and testing dataset. Due to our GPU and time constraints, we only utilized 20% of this
dataset to build our model. To evaluate it, we employed BLEU (Bilingual Evaluation Understudy
Score), which compares the generated sentence to the reference sentence and ranges from 1.0 (exact
match) to 0.0 (no match) . Since this score is based on an n-gram model, we tested our model using
both BLEU-1 and BLEU-4.



## How to run
- Data Receiving
  - We used the COCO-2014 dataset, which can be downloaded here https://cocodataset.org/#download.  
  - The training and testing data is obtained by running `get_datasets` 
  with its corresponding configuration file from `dataset_factory.py`.
- Model Receiving
  - We constructed our CNN and LSTM model on the model_factory.py. 
  - Run get_model in model_factory.py with the input configuration 
  file and the vocabulary obtained from get_datasets to get the 
  CNN-LSTM model we used. 
- Model Training
  - One can initialize the training experiment by running the following 
  `exper = Experiment(config)` from experiment.py. 
  Then use `exper.run()` to train the CNN-LSTM model 
  and compute the validation loss. The experiment will 
  end if the epoch ends or if the experiment is stopped 
  early due to validation loss is in increasing pattern.
- Model Performance 
  - By running `exper.test()` after `exper.run()`, we can test the 
  performance of the trained model by generating words from 
  unseen images and measure its accuracy by calculating the scores
  of bleu1 and bleu4.

  


## Usage

* Define the configuration for your experiment. See `task-1-default-config.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir.
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training or evaluate performance.

## Files
- `main.py`: Main driver class
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- `dataset_factory.py`: Factory to build datasets based on config
- `model_factory.py`: Factory to build models based on config
- `file_utils.py`: utility functions for handling files
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace
