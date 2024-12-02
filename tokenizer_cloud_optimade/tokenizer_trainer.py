import os
import numpy as np
import yaml
# from matbench.bench import MatbenchBenchmark
import json
# from invcryrep.invcryrep import InvCryRep
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer

"""
Constructs the tokenizer trainer for future usage.
"""

class Tokenizer_Trainer(object):

    def __init__(self, file_path, file_type, save_path,
                 vocab_size=52000, min_freq=1):
        """
        file_path: pathname of dataset file or matbench task
        file_type: whether a existing file or a matbench task
        save_path: where to save trained tokenizer
        vocab_size: how many tokens in total
        min_freq: min number of appearance of token to be added to vocab
        """
        if  file_type == "matbench":
            self.data = self.load_matbench(file_path)
        else:
            # f = open(file_path, )
            # data = json.load(f)
            # self.data = list(data.values())
            self.data = np.loadtxt(file_path, dtype=str, delimiter=',')
        self.save_path = save_path
        self.tokenizer = BertWordPieceTokenizer(lowercase=False)
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = ["[CLS]", "[PAD]", "[SEP]", "[UNK]", "[MASK]"]

    def __call__(self):
        # Train model
        self.tokenizer.train_from_iterator(
            iterator=self.data, vocab_size=self.vocab_size,
            min_frequency=self.min_freq, special_tokens=self.special_tokens)
        # Create save path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # Save model
        self.tokenizer.save_model(self.save_path)

    @staticmethod
    def load_matbench(path):
        # backend = InvCryRep()
        # mb = MatbenchBenchmark(autoload=False)
        # for task in mb.tasks:
        #     if task.dataset_name == path:
        #         task.load()
        #         df_train = task.get_train_and_val_data(0, as_type="df")
        #         df_test = task.get_test_data(0,as_type="df", include_target=True)
        # df = pd.concat([df_train, df_test], axis=0)
        # for i in tqdm(range(df.shape[0])):
        #     id = df.index[i]
        #     df.loc[id, "slices"] = backend.structure2SLICES(df.loc[id, "structure"])
        # return list(df.loc[:, "slices"].values())
        pass
    
if __name__ == '__main__':
    config = yaml.load(open("config_tokenizer.yaml", "r"), Loader=yaml.FullLoader)

    file_path = config['file_path']
    file_type = config['file_type']
    save_path = config['save_path']
    vocab_size = config['vocab_size']
    min_freq = config['min_freq']

    print('Loading config from config_tokenizer.yaml')
    print(f'file_path = {file_path}')
    print(f'file_type = {file_type}')
    print(f'save_path = {save_path}')
    print(f'vocab_size = {vocab_size}')
    print(f'min_freq = {min_freq}')

    trainer = Tokenizer_Trainer(file_path=file_path,
                                file_type=file_type,
                                save_path=save_path,
                                vocab_size=vocab_size,
                                min_freq=min_freq)

    trainer()