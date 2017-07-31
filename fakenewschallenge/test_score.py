import csv
from util import *



def calc_error(pred_file, test_file):
    with open(pred_file, "r", encoding='utf-8') as pred_table, open(test_file, "r", encoding='utf-8') as test_table:
        test_r = list(DictReader(test_table))
        pred_r = list(DictReader(pred_table))
        sum = 0
        for l_test, l_pred in zip(test_r, pred_r):
            if l_test['Stance'] == l_pred['Stance']:
                sum += 1
        print("The score is {}".format(float(sum)/len(test_r)))


if "__main__"==__name__:
    calc_error("predictions_test.csv", "test_stances.csv")
