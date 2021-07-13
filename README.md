# FKG - Mini Project

## Motivation 

Machine Learning (ML) algorithms need features to classify the instances. 
Data is stored in Knowledge Graphs (KG) in the form of triples. 
To apply ML algorithms on KG, we need to convert these triples into features. 
This is where KG embedding comes into picture. 
KG embedding gives us a low dimensional vector representation of the entities and relations in the KG.
For each learning problem, the individuals are converted into vector representations using KG embeddings and these embedded instances can be fed as an input to ML algorithms and new instances can be predicted by using these trained ML algorithms.

## Approach 

1. **Parsing the input turtle file:**
   
    To extract all the postive and negative individuals for each learning problem, the input turtle file is parsed.
   

2. **Train KG embedding models:**
   
   Five embedding models (namely TransE, DistMult, RotatE, ConvE, ComplEx) can be trained on the triples obtained from the carcinogenesis ontology. The output of the KG embedding is a CSV file containing vector representations of entities and relations.
   

3. **Conversion of input individuals into vector representations:**
   
    After training the KG embedding models, the postive and negative individuals for each learning problem are converted into X and y arrays.
   

4. **Classifier Training and Hyperparameter Optimization:**
   
    Six classifiers (namely LR, SVM, RandomForest, kNN, MLP and Perceptron) can be used on the embedded training data (X, y).
    To do hyperparameter optimization, we created an experiment of 10-fold cross-validation.
   The best set of hyperparameters is obtained on the embedded training data and the classifier using those parameters is used in the next step, i.e., Prediction.
   

5. **Prediction and writing the result file:**

   The remaining test individuals are converted into vector representations and given as an input to the chosen classifier. 
   We receive the predictions of all the individuals for each learning problem and the prediction results are written in a single turtle (link: result.ttl) file.

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
