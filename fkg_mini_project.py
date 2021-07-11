import numpy as np
import pandas as pd
from owlready2 import World, Thing
from rdflib import Namespace, Graph, Literal, URIRef
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class FKGMiniProject:

    param_grids = {
        'LR': {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "max_iter": [300, 400, 500]
        },
        'SVM': {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale"],
            "kernel": ["linear", "poly", "rbf"]
        },
        'RandomForest': {
            "n_estimators": [25, 50, 75, 100]
        },
        'kNN': {
            "n_neighbors": [2, 3, 4, 5, 6]
        },
        'MLP': {
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "activation": ["logistic", "tanh", "relu"],
            "max_iter": [1500],
            "hidden_layer_sizes": [(7, 5, 3), (7, 5, 3, 2), (7, 6, 5, 4, 3)]
        },
        'Perceptron': {
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "max_iter": [1000, 1500, 2000, 2500, 3000]
        }
    }

    NS_CAR = Namespace("http://dl-learner.org/carcinogenesis#")
    NS_RES = Namespace("https://lpbenchgen.org/resource/")
    NS_PROP = Namespace("https://lpbenchgen.org/property/")


    def __init__(self,
                 ontology_path,
                 embedding_path,
                 model_name = 'LR',
                 hyp_optim = False):
        self.onto = World().get_ontology(ontology_path).load()
        self.embeddings = pd.read_csv(embedding_path, index_col=0)
        self.model_name = model_name
        self.hyp_optim = hyp_optim
        self.classifier = self.__select_classifier()
        self.all_instances = self.__get_instances()

    def __select_classifier(self):
        if self.model_name == 'LR':
            classifier = LogisticRegression(class_weight='balanced')
        elif self.model_name == 'SVM':
            classifier = SVC(class_weight='balanced')
        elif self.model_name == 'RandomForest':
            classifier = RandomForestClassifier(class_weight='balanced')
        elif self.model_name == 'kNN':
            classifier = KNeighborsClassifier()
        elif self.model_name == 'MLP':
            classifier = MLPClassifier()
        elif self.model_name == 'Perceptron':
            classifier = Perceptron(class_weight='balanced')
        else:
            raise ValueError
        return classifier

    def __get_instances(self):
        instances = set()
        for c in self.onto.classes():
            instances.update([ind.iri for ind in c.instances(world=self.onto.world)])
        return instances

    def fit_and_evaluate(self, lp):
        X = self._get_X(lp['pos'] + lp['neg'])
        y = self._get_y(lp)

        param_grid = FKGMiniProject.param_grids[self.model_name]
        clf = self._optimize_hyperparameters(X, y, param_grid)
        print('F1-Score: ', clf.best_score_)

    def fit_and_predict(self, lp):
        pos_and_neg = lp['pos'] + lp['neg']
        X = self._get_X(pos_and_neg)
        y = self._get_y(lp)

        if self.hyp_optim:
            param_grid = FKGMiniProject.param_grids[self.model_name]
            clf = self._optimize_hyperparameters(X, y, param_grid)
            model = clf.best_estimator_
        else:
            model = self.classifier.fit(X, y)

        test_instances = [inst for inst in self.all_instances if inst not in pos_and_neg]
        X_test = self._get_X(test_instances)
        y_test = model.predict(X_test)

        return test_instances, y_test

    def _get_X(self, instances):
        return self.embeddings.loc[instances].to_numpy()

    def _get_y(self, lp):
        y = np.zeros(len(lp['pos']) + len(lp['neg']), dtype=int)
        y[:len(lp['pos'])] = 1
        return y

    def _optimize_hyperparameters(self, X, y, param_grid, cv=10):
        clf = GridSearchCV(self.classifier,
                                   param_grid=param_grid,
                                   cv=cv,
                                   n_jobs=-1,
                                   scoring="f1")
        clf.fit(X, y)

        print(self.model_name + " best params:" + str(clf.best_params_))
        return clf

    # TODO: Update to write all lps to same file
    def write_result_file(self, lp_name, pos, neg, result_file="result.ttl"):
        g = Graph()
        g.bind('carcinogenesis', FKGMiniProject.NS_CAR)
        g.bind('lpres', FKGMiniProject.NS_RES)
        g.bind('lpprop', FKGMiniProject.NS_PROP)

        g.add((FKGMiniProject.NS_RES.result_1pos,
               FKGMiniProject.NS_PROP.belongsToLP,
               Literal(True)))
        g.add((FKGMiniProject.NS_RES.result_1pos,
               FKGMiniProject.NS_PROP.pertainsTo,
               URIRef(FKGMiniProject.NS_RES + lp_name)))

        for p in pos:
            g.add((FKGMiniProject.NS_RES.result_1pos,
                   FKGMiniProject.NS_PROP.resource,
                   URIRef(FKGMiniProject.NS_CAR + p)))

        g.add((FKGMiniProject.NS_RES.result_1neg,
               FKGMiniProject.NS_PROP.belongsToLP,
               Literal(False)))
        g.add((FKGMiniProject.NS_RES.result_1neg,
               FKGMiniProject.NS_PROP.pertainsTo,
               URIRef(FKGMiniProject.NS_RES + lp_name)))

        for n in neg:
            g.add((FKGMiniProject.NS_RES.result_1neg,
                   FKGMiniProject.NS_PROP.resource,
                   URIRef(FKGMiniProject.NS_CAR + n)))

        g.serialize(destination=lp_name + "_"+ result_file, format='turtle')
