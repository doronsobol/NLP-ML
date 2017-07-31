import csv
import glob
import random
import sys

files = glob.glob('*.txt')
disagree_percentage = float(sys.argv[1])
split = float(sys.argv[2])
total_size = len(files)
train_size = int(split*total_size)
random.shuffle(files)
train_files = files[:train_size]
test_files = files[train_size:]

def create_csv_from_files(files, disagree_percentage, name):
    total_size = len(files)
    num_of_disagree = int(disagree_percentage*total_size)
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
        for f in files:
            with open(f, 'r') as file:
                lines = file.read().splitlines()
                stances += [lines[0]]
                bodies += [lines[2]]
        stances_to_shuffle = stances[:num_of_disagree]
        stances_to_save = stances[num_of_disagree:]
        random.shuffle(stances_to_shuffle)
        stances = stances_to_shuffle + stances_to_save
        labels = ['unrelated']*num_of_disagree + (total_size-num_of_disagree)*['agree']
        assert len(labels) == len(stances)
        assert total_size == len(stances)
        assert total_size == len(bodies)
        for i in range(total_size):
            articles_writer.writerow({'Body ID': i, 'articleBody': bodies[i]})
            stances_writer.writerow({'Headline': stances[i], 'Body ID': i, 'Stance': labels[i]})
            unlabeled_writer.writerow({'Headline': stances[i], 'Body ID': i})

if '__main__'==__name__:
    create_csv_from_files(train_files, disagree_percentage, 'train')
    create_csv_from_files(test_files, disagree_percentage, 'test')


