from fkg_mini_project import FKGMiniProject
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ontology = "carcinogenesis.owl"
lps = "kg-mini-project-grading.ttl"
project = FKGMiniProject(ontology, lps)

# print some information about the learning problems and the ontology
project.print_infos_ontology()
project.print_infos_lps()

# test printing of the result file on the first learning problem
lp = project.lps[0]
lp_name = lp['lp']
pos = lp['pos']
neg = lp['neg']

X, y = project.get_X_y_from_lp(lp, 'Shallom_entity_embeddings.csv')

# Model Selection Start

# Perceptron
perceptron_best_params_dict = project.hp_optimize_perceptron(X, y)
print("Perceptron CV Score: " + str(project.model_selection(Perceptron(**perceptron_best_params_dict), X, y)))

# Neural Networks
neural_network_best_params_dict = project.hp_optimize_neural_network(X, y)
print("Neural Network CV Score: " + str(project.model_selection(MLPClassifier(**neural_network_best_params_dict), X, y)))

# SVM
svm_best_params_dict = project.hp_optimize_svm(X, y)
print("SVM CV Score: " + str(project.model_selection(SVC(**svm_best_params_dict), X, y)))

# Logistic Regression
logistic_regression_best_params_dict = project.hp_optimize_logistic_regression(X, y)
print("Logistic Regression CV Score: " + str(project.model_selection(LogisticRegression(**logistic_regression_best_params_dict), X, y)))

# kNN
kNN_best_params_dict = project.hp_optimize_kNN(X, y)
print("kNN CV Score: " + str(project.model_selection(KNeighborsClassifier(**kNN_best_params_dict), X, y)))

# Random Forest
random_forest_params_dict = project.hp_optimize_random_forest(X, y)
print("Random Forest CV Score: " + str(project.model_selection(RandomForestClassifier(**random_forest_params_dict), X, y)))

# Model Selection End

project.write_result_file(lp_name, pos, neg)
