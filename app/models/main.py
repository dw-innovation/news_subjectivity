from argparse import ArgumentParser
from app.models.baseline import Baseline
from app.models.transformer_models import XLMR

MODELS = {
    'baseline': Baseline,
    'xlm-roberta': XLMR
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--model_path")
    parser.add_argument("--train_dataset_path")
    parser.add_argument("--dev_dataset_path")
    parser.add_argument("--test_dataset_path")
    parser.add_argument("--test_result_output")
    parser.add_argument("--dev_result_output")
    parser.add_argument("--pretrained_model")
    parser.add_argument("--augment_dataset_path")
    parser.add_argument("--augment_type", help=["oversampling", "undersampling","no_sampling"])
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--class_weight", action='store_true')

    # transformer based models
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--train_batch", type=int)
    parser.add_argument("--eval_batch", type=int)
    parser.add_argument("--cuda_device", type=str)

    args = parser.parse_args()

    model = MODELS[args.model](model_name=args.model,
                               pretrained_model=args.pretrained_model,
                               class_weight=args.class_weight)

    if args.train:

        if model != 'baseline':
            training_args = {
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "train_batch": args.train_batch,
                "eval_batch": args.eval_batch
            }
        else:
            training_args = None

        model.train(train_dataset_path=args.train_dataset_path,
                    eval_dataset_path=args.dev_dataset_path if model!='baseline' else None,
                    augment_dataset_path=args.augment_dataset_path,
                    augment_type=args.augment_type,
                    model_path=args.model_path,
                    training_args=training_args,
                    device=args.cuda_device if args.cuda_device else None
                    )

    if args.eval:
        model.test(dataset_path=args.dev_dataset_path,
                   model_path=args.model_path,
                   result_output=args.dev_result_output)


    if args.test:
        model.test(dataset_path=args.test_dataset_path,
                   model_path=args.model_path,
                   result_output=args.test_result_output)
