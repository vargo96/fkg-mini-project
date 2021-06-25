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

    def __init__(self, ontology_path, lps_path):
        self.ontology_path = ontology_path
        self.lps_path = lps_path
        self.onto = World().get_ontology(self.ontology_path).load()
        self.lps = self._read_and_parse_ttl_file(self.lps_path)
        self.NS_CAR = Namespace("http://dl-learner.org/carcinogenesis#")
        self.NS_RES = Namespace("https://lpbenchgen.org/resource/")
        self.NS_PROP = Namespace("https://lpbenchgen.org/property/")

    def _read_and_parse_ttl_file(self, lps_path):
        lp_instance_list = []
        with open(lps_path, "r") as lp_file:
            for line in lp_file:
                if line.startswith("lpres:"):
                    lp_key = line.split()[0].split(":")[1]
                elif line.strip().startswith("lpprop:excludesResource"):
                    exclude_resource_list = line.strip()[23:].split(",")
                    exclude_resource_list = [individual.replace(";", "").replace("carcinogenesis:", "").strip()
                                            for individual in exclude_resource_list]
                elif line.strip().startswith("lpprop:includesResource"):
                    include_resource_list = line.strip()[23:].split(",")
                    include_resource_list = [individual.replace(".", "").replace("carcinogenesis:", "").strip()
                                            for individual in include_resource_list]
                    lp_instance_list.append({"lp": lp_key, "pos": include_resource_list, "neg": exclude_resource_list})

        return lp_instance_list

    def get_X_y_from_lp(self, lp, emb_model_path):
        # Load Embedding Model
        embedding_model = pd.read_csv(emb_model_path, index_col=0)
        # print(embedding_model)

        X = np.zeros((len(lp['pos']) + len(lp['neg']), len(embedding_model.columns)))
        y = np.zeros(len(lp['pos']) + len(lp['neg']), dtype=int)

        index = 0

        for pos_inst in lp['pos']:
            X[index] = embedding_model.loc[str(self.NS_CAR) + pos_inst]
            y[index] = 1
            index += 1

        for neg_inst in lp['neg']:
            X[index] = embedding_model.loc[str(self.NS_CAR) + neg_inst]
            y[index] = 0
            index += 1

        return X, y

    def model_selection(self, learner, X, y, n_folds=5):
        return cross_val_score(learner, X, y, cv=n_folds, scoring='f1').mean()

    def fit(self, lp):
        # X, y = self.get_X_y_from_lp(lp, 'Shallom_entity_embeddings.csv')
        # print(X.shape)
        # print(y.shape)
        pass

    def score(self, lp):
        pass

    def write_result_file(self, lp_name, pos, neg, result_file="result.ttl"):

        g = Graph()
        g.bind('carcinogenesis', self.NS_CAR)
        g.bind('lpres', self.NS_RES)
        g.bind('lpprop', self.NS_PROP)

        g.add((self.NS_RES.result_1pos, self.NS_PROP.belongsToLP, Literal(True)))
        g.add((self.NS_RES.result_1pos, self.NS_PROP.pertainsTo, URIRef(self.NS_RES + lp_name)))
        for p in pos:
            g.add((self.NS_RES.result_1pos, self.NS_PROP.resource, URIRef(self.NS_CAR + p)))

        g.add((self.NS_RES.result_1neg, self.NS_PROP.belongsToLP, Literal(False)))
        g.add((self.NS_RES.result_1neg, self.NS_PROP.pertainsTo, URIRef(self.NS_RES + lp_name)))
        for n in neg:
            g.add((self.NS_RES.result_1neg, self.NS_PROP.resource, URIRef(self.NS_CAR + n)))

        g.serialize(destination=lp_name + "_"+ result_file, format='turtle')

    def print_infos_ontology(self):
        print("#"*50)
        print(f"Number classes: {len(list(self.onto.classes()))}")
        instances = set()
        for c in self.onto.classes():
            print(f"\t {c.name}")
            instances.update(c.instances(world=self.onto.world))
        print(f"Number object properties: {len(list(self.onto.object_properties()))}")
        for p in self.onto.object_properties():
            print(f"\t {p.name}")
        print(f"Number data properties: {len(list(self.onto.data_properties()))}")
        for p in self.onto.data_properties():
            print(f"\t {p.name}")
        print(f"Number individuals: {len(instances)}")
        print("#"*50)

    def print_infos_lps(self):
        print("#"*50)
        print(f"Number LPs: {len(self.lps)}")
        for lp in self.lps:
            print(f"LP ({lp['lp']}): PositiveEX - {len(lp['pos'])} | NegativeEX - {len(lp['neg'])} | TotalEX - {len(lp['pos']) + len(lp['neg'])}")
        print("#"*50)

    # Classifier Hyperparameter Optimization Start

    def hp_optimize_perceptron(self, X_train, y_train):
        param_grids = [
            {
                "penalty": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "max_iter": [1000, 1500, 2000, 2500, 3000]
            }
        ]

        grid_clf = Perceptron()

        grid_search = GridSearchCV(grid_clf, param_grid=param_grids, cv=5, n_jobs=-1
                                   , scoring="f1")
        grid_search.fit(X_train, y_train)

        print("Perceptron best params:" + str(grid_search.best_params_))
        return grid_search.best_params_

    def hp_optimize_kNN(self, X_train, y_train):
        param_grids = [
            {
                "n_neighbors": [2, 3, 4, 5, 6]
            }
        ]

        grid_clf = KNeighborsClassifier()

        grid_search = GridSearchCV(grid_clf, param_grid=param_grids, cv=5, n_jobs=-1
                                   , scoring="f1")
        grid_search.fit(X_train, y_train)

        print("kNN best params:" + str(grid_search.best_params_))
        return grid_search.best_params_

    def hp_optimize_neural_network(self, X_train, y_train):
        param_grids = [
            {
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "activation": ["logistic", "tanh", "relu"],
                "max_iter": [1500],
                "hidden_layer_sizes": [(7, 5, 3), (7, 5, 3, 2), (7, 6, 5, 4, 3)]
            }
        ]

        grid_clf = MLPClassifier()

        grid_search = GridSearchCV(grid_clf, param_grid=param_grids, cv=5, n_jobs=-1
                                   , scoring="f1")
        grid_search.fit(X_train, y_train)

        print("NN best params:" + str(grid_search.best_params_))
        return grid_search.best_params_

    def hp_optimize_svm(self, X_train, y_train):
        param_grids = [
            {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale"],
                "kernel": ["linear", "poly", "rbf"]
            }
        ]

        grid_clf = SVC()

        grid_search = GridSearchCV(grid_clf, param_grid=param_grids, cv=5, n_jobs=-1
                                   , scoring="f1")
        grid_search.fit(X_train, y_train)

        print("SVM best params:" + str(grid_search.best_params_))
        return grid_search.best_params_

    def hp_optimize_logistic_regression(self, X_train, y_train):
        param_grids = [
            {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "max_iter": [300, 400, 500]
            }
        ]

        grid_clf = LogisticRegression()

        grid_search = GridSearchCV(grid_clf, param_grid=param_grids, cv=5, n_jobs=-1
                                   , scoring="f1")
        grid_search.fit(X_train, y_train)

        print("Logistic Regression best params:" + str(grid_search.best_params_))
        return grid_search.best_params_

    def hp_optimize_random_forest(self, X_train, y_train):
        param_grids = [
            {
                "n_estimators": [25, 50, 75, 100]
            }
        ]

        grid_clf = RandomForestClassifier()

        grid_search = GridSearchCV(grid_clf, param_grid=param_grids, cv=5, n_jobs=-1
                                   , scoring="f1")
        grid_search.fit(X_train, y_train)

        print("Random Forest best params:" + str(grid_search.best_params_))
        return grid_search.best_params_

    # Classifier Hyperparameter Optimization End
