import argparse
import torch
import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Train embeddings.')
    parser.add_argument("--dataset_path", type=str, default='data/carcinogenesis.tsv',
                        help='3 column tsv file, one line for each triple')
    parser.add_argument("--result_path", type=str, default='embeddings')
    parser.add_argument("--model", type=str, default='TransE',
                        help='Incomplete list: TransE, DistMult, RotatE, ConvE, ComplEx  \
                              - See: https://pykeen.readthedocs.io/en/stable/reference/models.html \
                              This script won\'t work for all though')
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def run(args):
    tf = TriplesFactory.from_path(args.dataset_path)
    training, testing, validation = tf.split([.8, .1, .1])
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=args.model,
        stopper='early',
        training_kwargs=dict(num_epochs=args.epochs),
    )

    model_path = 'model/' + args.model
    result.save_to_directory(model_path)

    model = torch.load(model_path + '/trained_model.pkl')
    embeddings = model.entity_representations

    # Literals are not needed
    labels = [x for x in list(tf.entity_id_to_label.values()) if '^^' not in x]
    ids = torch.as_tensor(tf.entities_to_ids(labels))
    df = pd.DataFrame(embeddings[0](indices=ids).detach().numpy(),
                      index=labels,
                      columns=list(range(0, model.embedding_dim)))

    df.to_csv(args.result_path + '/' + args.model + '_embeddings.csv')

if __name__ == '__main__':
    run(parse_args())

