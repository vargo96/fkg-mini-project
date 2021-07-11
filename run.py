import argparse
import tqdm
from fkg_mini_project import FKGMiniProject
from rdflib import Namespace, Graph, Literal, URIRef


NS_CAR = Namespace("http://dl-learner.org/carcinogenesis#")
NS_RES = Namespace("https://lpbenchgen.org/resource/")
NS_PROP = Namespace("https://lpbenchgen.org/property/")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_path", type=str, default='data/carcinogenesis.owl',
                        help='OWL ontology file.')
    parser.add_argument("--embeddings_path", type=str,
                        default='embeddings/TransE_embeddings.csv')
    parser.add_argument("--lps_path", type=str, default='data/kg-mini-project-grading.ttl')
    parser.add_argument("--classifier", type=str, default='LR',
                        help='Available classifiers: LR, SVM, RandomForest, kNN, MLP, Perceptron')
    parser.add_argument("--train_mode", action='store_true',
                        help='Set train mode: Run 10-Fold CV on the given learning problems.')
    parser.add_argument("--hyper_optim", action='store_true',
                        help='Optimize hyperparameters before fitting.')
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
                                                   .replace("carcinogenesis:",
                                                  "         http://dl-learner.org/carcinogenesis#").strip()
                        for individual in exclude_resource_list]
            elif line.strip().startswith("lpprop:includesResource"):
                include_resource_list = line.strip()[23:].split(",")
                include_resource_list = [individual.replace(".", "")
                                                   .replace("carcinogenesis:",
                                                            "http://dl-learner.org/carcinogenesis#").strip()
                        for individual in include_resource_list]
                lp_instance_list.append({"name": lp_key,
                                         "pos": include_resource_list,
                                         "neg": exclude_resource_list})

    return lp_instance_list


def add_lp_to_graph(graph, lp_name, pos, neg, index):
    current_pos = f'result_{index}pos'
    current_neg = f'result_{index}neg'
    graph.add((URIRef(NS_RES + current_pos), NS_PROP.belongsToLP, Literal(True)))
    graph.add((URIRef(NS_RES + current_pos), NS_PROP.pertainsTo, URIRef(NS_RES + lp_name)))

    for p in pos:
        graph.add((URIRef(NS_RES + current_pos), NS_PROP.resource, URIRef(p)))

    graph.add((URIRef(NS_RES + current_neg), NS_PROP.belongsToLP, Literal(False)))
    graph.add((URIRef(NS_RES + current_neg), NS_PROP.pertainsTo, URIRef(NS_RES + lp_name)))

    for n in neg:
        graph.add((URIRef(NS_RES + current_neg), NS_PROP.resource, URIRef(n)))


def run(args):
    lps = parse_lps(args.lps_path)

    project = FKGMiniProject(args.ontology_path,
                             args.embeddings_path,
                             model_name=args.classifier,
                             hyp_optim=args.hyper_optim)


    if args.train_mode:
        lp = lps[23]
        project.fit_and_evaluate(lp)
    else:
        g = Graph()
        g.bind('carcinogenesis', NS_CAR)
        g.bind('lpres', NS_RES)
        g.bind('lpprop', NS_PROP)

        progress_bar = tqdm.tqdm(total=len(lps), leave=False)
        for idx, lp in enumerate(lps):
            pos, neg = project.fit_and_predict(lp)
            add_lp_to_graph(g, lp['name'], pos, neg, idx+1)
            progress_bar.update(1)
        progress_bar.close()

        g.serialize(destination='result.ttl', format='turtle')

if __name__ == '__main__':
    run(parse_args())
