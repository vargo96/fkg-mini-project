# FKG - Mini Project

## Installation
Clone the repository:

```
git clone https://github.com/vargo96/fkg-mini-project.git
```
You might want to create a virtual environment:
```
python -m venv ~/fkg
source ~/fkg/bin/activate
# OR
conda create --name fkg
conda activate fkg
```
Then run:
```
pip install -r requirements.txt
```

## Reproduce result ttl file:
To reproduce the ```result.ttl``` with the predictions for the remaining individual for each learning problem run:
```
python run.py
```

## Miscellaneous
Some additional usage information for the train script and embedding script:

```
usage: run.py [-h] [--ontology_path ONTOLOGY_PATH] [--embeddings_path EMBEDDINGS_PATH]
              [--lps_path LPS_PATH] [--classifier CLASSIFIER] [--train_mode] [--hyper_optim]
              [--output_file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --ontology_path ONTOLOGY_PATH
                        OWL ontology file.
  --embeddings_path EMBEDDINGS_PATH
  --lps_path LPS_PATH
  --classifier CLASSIFIER
                        Available classifiers: LR, SVM, RandomForest, kNN, MLP, Perceptron
  --train_mode          Set train mode: Run 10-Fold CV on the given learning problems.
  --hyper_optim         Optimize hyperparameters before fitting.
  --output_file OUTPUT_FILE
```

```
usage: train_embeddings.py [-h] [--dataset_path DATASET_PATH]
                           [--result_path RESULT_PATH] [--model MODEL]
                           [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        3 column tsv file, one line for each triple
  --result_path RESULT_PATH
  --model MODEL         Incomplete list: TransE, DistMult, RotatE, ConvE,
                        ComplEx - See: https://pykeen.readthedocs.io/en/stable
                        /reference/models.html This script won't work for all
                        though
  --epochs EPOCHS
```
