# FKG - Mini Project

## Motivation 

Machine Learning (ML) algorithms need data in the form of feature vectors and class labels to come up with a trained model and predict new instances with that trained model. 
Data stored in Knowledge Graph(s) (KG) is in the form of triples. 
To apply ML algorithms on KG, we need to convert the individuals in the learning problem into feature vectors. 
One way to tackle this task is to use KG embedding. 
KG embedding gives us a low dimensional vector representation of the entities and the relations in the KG.
For a learning problem, the individuals are converted into vector representations using KG embeddings and these embedded individuals can be fed as an input to ML algorithms and unlabeled individuals can be predicted by using these trained ML algorithms.

## Approach 

Our approach consists of the five steps and these are as follows:

1. **Parse the input turtle file:**
    
    In this step, we parse the input turtle file (i.e. kg-mini-project-train_v2.ttl or kg-mini-project-grading.ttl) and the output contains the set of positive and negative individuals for all learning problems.
   

2. **Train KG embedding models:**
   
   Five embedding models (namely TransE, DistMult, RotatE, ConvE, ComplEx) can be trained on the triples obtained from the ontology. The output of the KG embedding is a CSV file containing vector representations of entities and relations.
   

3. **Convert input individuals into vector representations:**
   
    After training the KG embedding models, the positive and negative individuals for each learning problem are converted into feature vectors and class labels.
   

4. **Train classifiers and optimize hyperparameters:**
   
   Six classifiers (namely LR, SVM, RandomForest, kNN, MLP and Perceptron) can be used on the embedded training data (output data from Step 3).
   The best set of hyperparameters is obtained on the embedded training data and the chosen classifier using those parameters is used for prediction (Step 5).
   

5. **Predict and write the result file:**

   The remaining test individuals are converted into vector representations and given as an input to the chosen classifier. 
   We receive the predictions of all the individuals for each learning problem and the prediction results are written in a single turtle [result](result.ttl) file.

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
