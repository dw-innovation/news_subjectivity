from argparse import ArgumentParser
from app.models.baseline import Baseline

MODELS = {
    'baseline': Baseline
}
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--model_path")
    parser.add_argument("--train_dataset_path")
    parser.add_argument("--dev_dataset_path")
    parser.add_argument("--test_dataset_path")
    parser.add_argument("--augment_dataset_path")
    parser.add_argument("--result_output")
    parser.add_argument("--pretrained_model")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--class_weight", action='store_true')

    args = parser.parse_args()

    model = MODELS[args.model](model_name=args.model,
                               pretrained_model=args.pretrained_model,
                               class_weight=args.class_weight)

    if args.train:
        model.train(dataset_path=args.train_dataset_path,
                    augment_dataset_path=args.augment_dataset_path,
                    model_path=args.model_path)

    if args.test:
        model.test(dataset_path=args.dev_dataset_path,
                   model_path=args.model_path,
                   result_output=args.result_output)
