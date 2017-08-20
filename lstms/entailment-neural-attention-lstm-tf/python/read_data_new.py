import os

def load_dataset(dataset_path, type_set):
    if(type_set not in ["train", "test"]):
        type_set="test"

    label_to_directory={"entailment":["positive"], "neutral":["mix_within", "mix_unrelated"], "contradiction":[]}

    premises=[]
    hypothesis=[]
    targets=[]

    dataset_dir= os.path.join(dataset_path, type_set)
    directories= os.listdir(dataset_dir)
    for directory in directories:
        my_label=""
        for label, dir_list in label_to_directory.items():
            if(directory in dir_list):
                my_label=label
                break
        if(my_label==""):
            continue

        directory_path=os.path.join(dataset_dir, directory)
        for file_path in [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path)]:
            with open(file_path, 'r', encoding="utf8") as file:
                try:
                    lines = file.read().splitlines()
                    if len(lines) < 3:
                        print("warning on file: {}".format(file_path))
                        continue
                    hypothesis += [lines[0]] #stance
                    premises += [lines[2]] #body
                    targets += [my_label] #label
                except Exception as e:
                    print("warning on file: {}".format(file_path))
                    print(e)
    return {"premises": premises, "hypothesis": hypothesis, "targets": targets}



if __name__ == "__main__":
    dirname, filename = os.path.split(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(dirname,"../../../dataset"))
    res=load_dataset(DATA_DIR, "test")
    print(res)