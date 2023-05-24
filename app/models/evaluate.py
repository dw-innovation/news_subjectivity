import pandas, argparse, sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from loguru import logger

logger.add(f"{__name__}.log", rotation="500 MB")


def validate_files(pred_file, gold_file):
    pred_data = pandas.read_csv(pred_file, sep='\t')
    gold_data = pandas.read_csv(gold_file, sep='\t')

    if len(pred_data) != len(gold_data):
        logger.error("ERROR! Different number of instances in the files")
        return False

    if not ('label' in pred_data and
            'sentence_id' in pred_data and 'sentence_id' in gold_data):
        logger.error("ERROR! Wrong columns")
        return False

    pred_values = pred_data['label'].unique()
    gold_values = gold_data['label'].unique()

    if not ((len(pred_values) == 2 and "SUBJ" in pred_values and
             "OBJ" in pred_values) or
            (len(pred_values) == 1 and
             ("OBJ" in pred_values or "SUBJ" in pred_values))):
        logger.error("ERROR! Wrong labels")
        return False

    pred_data.rename(columns={'label': 'pred_label'}, inplace=True)

    whole_data = pandas.merge(pred_data, gold_data, on="sentence_id")

    if len(pred_data) != len(whole_data):
        logger.error("ERROR! Different ids in the two files")
        return False

    logger.info("The file is properly formatted")

    if not ('label' in gold_data):
        logger.warn("WARNING: no labels in the gold data file")
        logger.error("Impossible to proceed with evaluation")
        return False

    whole_data.rename(columns={'label': 'gold_label'}, inplace=True)

    return whole_data


def evaluate(whole_data):
    pred_values = whole_data['pred_label'].values
    gold_values = whole_data['gold_label'].values

    acc = accuracy_score(gold_values, pred_values)
    m_prec, m_rec, m_f1, m_s = precision_recall_fscore_support(gold_values, pred_values, average="macro",
                                                               zero_division=0)
    p_prec, p_rec, p_f1, p_s = precision_recall_fscore_support(gold_values, pred_values, labels=["SUBJ"],
                                                               zero_division=0)

    return {
        'macro_F1': m_f1,
        'macro_P': m_prec,
        'macro_R': m_rec,
        'SUBJ_F1': p_f1[0],
        'SUBJ_P': p_prec[0],
        'SUBJ_R': p_rec[0],
        'accuracy': acc
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to file with gold annotations.")
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to file with predict class per sentence")

    args = parser.parse_args()

    pred_file = args.pred_file_path
    gold_file = args.gold_file_path

    whole_data = validate_files(pred_file, gold_file)

    if whole_data is not False:
        logger.info("Started evaluating results for task-2...")

        logger.info(args.pred_file_path)

        scores = evaluate(whole_data)

        # logger.info('Macro F1 & Macro P & Macro R & SUBJ F1 & SUBJ P & SUBJ R & Accuracy')

        logger.info('Macro F1 Macro P Macro R SUBJ F1 SUBJ P SUBJ R Accuracy')

        scores_macro_f1 = "{:.2f}".format(scores['macro_F1'])
        scores_macro_p = "{:.2f}".format(scores['macro_P'])
        scores_macro_r = "{:.2f}".format(scores['macro_R'])
        scores_subj_f1 ="{:.2f}".format(scores['SUBJ_F1'])
        scores_subj_p = "{:.2f}".format(scores['SUBJ_P'])
        scores_subj_r = "{:.2f}".format(scores['SUBJ_R'])
        scores_accuracy = "{:.2f}".format(scores['accuracy'])

        # logger.info(f'{scores_macro_f1} & {scores_macro_p} & {scores_subj_r} & {scores_subj_p} & {scores_subj_r} & {scores_accuracy}')
        logger.info(
            f'{scores_macro_f1} {scores_macro_p} {scores_subj_r} {scores_subj_p} {scores_subj_r} {scores_accuracy}')