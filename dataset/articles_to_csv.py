import csv
import glob
import random
import copy
import sys

train_files_agree = glob.glob('./train/positive/*.txt')
test_files_agree = glob.glob('./test/positive/*.txt')


train_files_discuss = glob.glob('./train/mix_within/*.txt')
test_files_discuss = glob.glob('./test/mix_within/*.txt')

train_files_disagree = []
test_files_disagree = []

def create_csv_from_files(files_agree, files_disagree, files_discuss, name):
    with open(name+'_bodies.csv', 'w', encoding="utf8") as articles_file, open(name+'_stances.csv', 'w', encoding="utf8") as stances_file, open(name+'_stances_unlabeled.csv', 'w', encoding="utf8") as unlabeled_file:
        articles_fieldnames = ['Body ID','articleBody']
        stances_fieldnames = ['Headline','Body ID','Stance']
        stances_fieldnames_unlabeled = ['Headline','Body ID']
        articles_writer = csv.DictWriter(articles_file, fieldnames=articles_fieldnames)
        stances_writer = csv.DictWriter(stances_file, fieldnames=stances_fieldnames)
        unlabeled_writer = csv.DictWriter(unlabeled_file, fieldnames=stances_fieldnames_unlabeled)
        articles_writer.writeheader()
        stances_writer.writeheader()
        unlabeled_writer.writeheader()
        stances = []
        bodies = []
        labels = []
        for f in files_disagree:
            with open(f, 'r', encoding="utf8") as file:
                try:
                    lines = file.read().splitlines()
                    if len(lines) < 3:
                        print("warning on file: {}".format(f))
                        continue
                    stances += [lines[0]]
                    bodies += [lines[2]]
                    labels += ['unrelated']
                except Exception as e:
                    print("warning on file: {}".format(f))
                    print(e)
        for f in files_agree:
            with open(f, 'r', encoding="utf8") as file:
                try:
                    lines = file.read().splitlines()
                    if len(lines) < 3:
                        print("warning on file: {}".format(f))
                        continue
                    stances += [lines[0]]
                    bodies += [lines[2]]
                    labels += ['agree']
                except Exception as e:
                    print("warning on file: {}".format(f))
                    print(e)
        for f in files_discuss:
            with open(f, 'r', encoding="utf8") as file:
                try:
                    lines = file.read().splitlines()
                    if len(lines) < 3: #ignore bad
                        print("warning on file: {}".format(f))
                        continue
                    stances += [lines[0]]
                    bodies += [lines[2]]
                    labels += ['discuss']
                except Exception as e:
                    print("warning on file: {}".format(f))
                    print(e)
        for i in range(len(stances)):
            articles_writer.writerow({'Body ID': i, 'articleBody': bodies[i]})
            stances_writer.writerow({'Headline': stances[i], 'Body ID': i, 'Stance': labels[i]})
            unlabeled_writer.writerow({'Headline': stances[i], 'Body ID': i})

if '__main__'==__name__:
    create_csv_from_files(train_files_agree, train_files_disagree, train_files_discuss, 'train')
    create_csv_from_files(test_files_agree, test_files_disagree, test_files_discuss, 'test')


