import csv
import glob
import random
import copy
import sys

files = glob.glob('./positive/*.txt')
#disagree_percentage = float(sys.argv[1])
split = float(sys.argv[1])
split_mix = 0.5
total_size = len(files)
train_size = int(split*total_size)
test_size = int(total_size - split*total_size)
random.shuffle(files)
train_files = files[:train_size]
test_files = files[train_size:]
train_files_agree = train_files[:int(train_size*0.5)]
test_files_agree = test_files[:int(test_size*0.5)]
train_files_disagree = train_files[int(train_size*0.5):]
test_files_disagree = test_files[int(test_size*0.5):]
train_files_discuss = []
test_files_discuss = []
"""
files_disagree = glob.glob('./mix_unrelated/*.txt')
#disdisagree_percentage = float(sys.argv[1])
split = float(sys.argv[1])
total_size_disagree = len(files_disagree)
train_size_disagree = int(split*total_size_disagree)
random.shuffle(files_disagree)
train_files_disagree = files_disagree[:train_size_disagree]
test_files_disagree = files_disagree[train_size_disagree:]

files_discuss = glob.glob('./mix_within/*.txt')
#disdiscuss_percentage = float(sys.argv[1])
split = float(sys.argv[1])
total_size_discuss = len(files_discuss)
train_size_discuss = int(split*total_size_discuss)
random.shuffle(files_discuss)
train_files_discuss = files_discuss[:train_size_discuss]
test_files_discuss = files_discuss[train_size_discuss:]
"""

def create_csv_from_files(files_agree, files_disagree, files_discuss, name):
    with open(name+'_bodies.csv', 'w') as articles_file, open(name+'_stances.csv', 'w') as stances_file, open(name+'_stances_unlabeled.csv', 'w') as unlabeled_file:
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
        shuffeled_files_disagree = copy.deepcopy(files_disagree)
        random.shuffle(shuffeled_files_disagree)
        for f, f_s in zip(files_disagree, shuffeled_files_disagree):
            with open(f, 'r') as file, open(f_s, 'r') as file_s:
                try:
                    lines = file.read().splitlines()
                    lines_s = file_s.read().splitlines()
                    stances += [lines_s[0]]
                    bodies += [lines[2]]
                    labels += ['unrelated']
                except:
                    print("warning on file: {}".format(f))
        for f in files_agree:
            with open(f, 'r') as file:
                try:
                    lines = file.read().splitlines()
                    stances += [lines[0]]
                    bodies += [lines[2]]
                    labels += ['agree']
                except:
                    print ("warning on file: {}".format(f))
        for f in files_discuss:
            with open(f, 'r') as file:
                try:
                    lines = file.read().splitlines()
                    stances += [lines[0]]
                    bodies += [lines[2]]
                    labels += ['discuss']
                except:
                    print ("warning on file: {}".format(f))

        for i in range(len(stances)):
            articles_writer.writerow({'Body ID': i, 'articleBody': bodies[i]})
            stances_writer.writerow({'Headline': stances[i], 'Body ID': i, 'Stance': labels[i]})
            unlabeled_writer.writerow({'Headline': stances[i], 'Body ID': i})

if '__main__'==__name__:
    create_csv_from_files(train_files_agree, train_files_disagree, train_files_discuss, 'train')
    create_csv_from_files(test_files_agree, test_files_disagree, test_files_discuss, 'test')


