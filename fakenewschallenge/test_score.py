import csv
from util import *
import pdb


def calc_error(pred_file, test_file):
    with open(pred_file, "r", encoding='utf-8') as pred_table, open(test_file, "r", encoding='utf-8') as test_table:
        test_r = list(DictReader(test_table))
        pred_r = list(DictReader(pred_table))
        sum = 0
        stances = set([x['Stance'] for x in test_r] + [x['Stance'] for x in pred_r])
        stances_dict = dict(list(zip(stances, list(range(len(stances))))))
        sums = len(stances)*[0]
        total = len(stances)*[0]
        for l_test, l_pred in zip(test_r, pred_r):
            try:
                total[stances_dict[l_test['Stance']]] += 1
                if l_test['Stance'] == l_pred['Stance']:
                    sums[stances_dict[l_test['Stance']]] += 1
                    sum += 1
            except:
                pdb.set_trace()
                print(l_test)
        print("The score is {}".format(float(sum)/len(test_r)))
        for s in stances:
            print("The score for {} is {}".format(s, float(sums[stances_dict[s]])/float(total[stances_dict[s]])))
            print("The precentage of {} is {}".format(s, float(total[stances_dict[s]])/len(test_r)))


if "__main__"==__name__:
    calc_error("predictions_test.csv", "test_stances.csv")
