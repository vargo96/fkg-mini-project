import argparse
from fkg_mini_project import FKGMiniProject

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_path", type=str, default='data/carcinogenesis.owl',
                        help='OWL ontology file.')
    parser.add_argument("--embeddings_path", type=str,
                        default='embeddings/TransE_embeddings.csv')
    parser.add_argument("--lps_path", type=str, default='data/kg-mini-project-grading.ttl')
    parser.add_argument("--classifier", type=str, default='LR',
                        help='Available classifiers: LR, SVM, RandomForest, kNN, MLP, Perceptron')
    parser.add_argument("--train_mode", type=bool, default=True,
                        help='False: Train on given lps and predict remaining individuals \
                              - True: 10-Fold CV on lps')
    parser.add_argument("--hyper_optim", type=bool, default=True,
                        help='Optimize hyperparameters')
    parser.add_argument("--output_file", type=str, default='result.ttl')

    return parser.parse_args()


def parse_lps(lps_path):
    lp_instance_list = []
    with open(lps_path, "r") as lp_file:
        for line in lp_file:
            if line.startswith("lpres:"):
                lp_key = line.split()[0].split(":")[1]
            elif line.strip().startswith("lpprop:excludesResource"):
                exclude_resource_list = line.strip()[23:].split(",")
                exclude_resource_list = [individual.replace(";", "")
                                         .replace("carcinogenesis:", "").strip()
                                        for individual in exclude_resource_list]
            elif line.strip().startswith("lpprop:includesResource"):
                include_resource_list = line.strip()[23:].split(",")
                include_resource_list = [individual.replace(".", "")
                                         .replace("carcinogenesis:", "").strip()
                                        for individual in include_resource_list]
                lp_instance_list.append({"lp": lp_key,
                                         "pos": include_resource_list,
                                         "neg": exclude_resource_list})

    return lp_instance_list


def run(args):
    lps = parse_lps(args.lps_path)

    executor = FKGMiniProject(args.ontology_path,
                              args.embeddings_path,
                              model_name=args.classifier,
                              hyp_optim=args.hyper_optim)

    # TODO: Loop through lps and write the result file
    lp = lps[23]
    if args.train_mode:
        executor.fit_and_evaluate(lp)
    else:
        executor.fit_and_predict(lp)

if __name__ == '__main__':
    run(parse_args())
