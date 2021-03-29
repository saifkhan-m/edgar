import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from torch.utils.data import Dataset, DataLoader, random_split

class ConsumerComplaintsDataset1(Dataset):
    def __init__(self, tokenizer, max_len, chunk_len=512, overlap_len=50,
                 approach="all", max_size_dataset=None,
                 file_location="/content/drive/MyDrive/consumer_complaints.csv", min_len=249):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        self.approach = approach
        self.min_len = min_len
        self.max_size_dataset = max_size_dataset
        # self.data, self.label = self.process_data(file_location,)
        self.data, self.label = self.process_data2()

    def process_data2(self):
        train_data = pd.read_csv('/home/saif/bser/sentimentAnalysis/edgar/edgar/DowJones_10K.csv')
        train_data.drop('Unnamed: 0', axis=1, inplace=True)
        train_data = train_data.replace(np.nan, '', regex=True)
        columns = ['Business', 'Risk Factors', 'Legal Proceedings', 'MDA_FCRO', 'QQD_MR']
        train_data['Combined'] = train_data[columns].apply(lambda x: ''.join(x), axis=1)
        combined_datset = train_data.drop(columns, axis=1)
        combined_datset = combined_datset[['Combined', 'label']]
        return combined_datset['Combined'].values, combined_datset['label'].values

    def process_data(self, file_location):

        df = pd.read_csv(file_location, dtype="unicode")
        train_raw = df[df.consumer_complaint_narrative.notnull()]
        train_raw = train_raw.assign(
            len_txt=train_raw.consumer_complaint_narrative.apply(lambda x: len(x.split())))
        train_raw = train_raw[train_raw.len_txt > self.min_len]
        train_raw = train_raw[['consumer_complaint_narrative', 'product']]
        train_raw.reset_index(inplace=True, drop=True)
        train_raw.at[train_raw['product'] == 'Credit reporting',
                     'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
        train_raw.at[train_raw['product'] == 'Credit card',
                     'product'] = 'Credit card or prepaid card'
        train_raw.at[train_raw['product'] == 'Prepaid card',
                     'product'] = 'Credit card or prepaid card'
        train_raw.at[train_raw['product'] == 'Payday loan',
                     'product'] = 'Payday loan, title loan, or personal loan'
        train_raw.at[train_raw['product'] == 'Virtual currency',
                     'product'] = 'Money transfer, virtual currency, or money service'
        train_raw = train_raw.rename(
            columns={'consumer_complaint_narrative': 'text', 'product': 'label'})
        LE = LabelEncoder()
        train_raw['label'] = LE.fit_transform(train_raw['label'])
        train = train_raw.copy()
        if (self.max_size_dataset):
            train = train.loc[0:self.max_size_dataset, :]
        train = train.reindex(np.random.permutation(train.index))
        train['text'] = train.text.apply(self.clean_txt)
        return train['text'].values, train['label'].values

    def clean_txt(self, text):
        """ Remove special characters from text """

        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def long_terms_tokenizer(self, data_tokenize, targets):

        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        remain = data_tokenize.get("overflowing_tokens")
        targets = torch.tensor(targets, dtype=torch.int)

        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        token_type_ids_list.append(previous_token_type_ids)
        targets_list.append(targets)
        remain = torch.squeeze(remain)
        if remain.numpy().all() and self.approach != 'head':

            remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain) + self.chunk_len)
            idxs = idxs[(self.chunk_len - self.overlap_len - 2)
                        ::(self.chunk_len - self.overlap_len - 2)]
            input_ids_first_overlap = previous_input_ids[-(
                    self.overlap_len + 1):-1]
            start_token = torch.tensor([101], dtype=torch.long)
            end_token = torch.tensor([102], dtype=torch.long)
            # print(remain.shape)
            # print(input_ids_first_overlap.shape)
            for i, idx in enumerate(idxs):
                if i == 0:
                    # print(remain[:idx].shape)
                    input_ids = torch.cat(
                        (input_ids_first_overlap, remain[:idx]))
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx - self.overlap_len):idx]

                previous_idx = idx

                nb_token = len(input_ids) + 2
                attention_mask = torch.ones(self.chunk_len, dtype=torch.long)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = torch.zeros(self.chunk_len, dtype=torch.long)
                input_ids = torch.cat((start_token, input_ids, end_token))
                if self.chunk_len - nb_token > 0:
                    padding = torch.zeros(
                        self.chunk_len - nb_token, dtype=torch.long)
                    input_ids = torch.cat((input_ids, padding))

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                targets_list.append(targets)
            if self.approach == "tail":
                input_ids_list = [input_ids_list[-1]]
                attention_mask_list = [attention_mask_list[-1]]
                token_type_ids_list = [token_type_ids_list[-1]]
                targets_list = [targets_list[-1]]

        return ({
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            # torch.tensor(mask, dtype=torch.long),
            'mask': attention_mask_list,
            # torch.tensor(token_type_ids, dtype=torch.long),
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })

    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        consumer_complaint = str(self.data[idx])
        targets = int(self.label[idx])
        data = self.tokenizer.encode_plus(
            consumer_complaint,
            max_length=self.chunk_len,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors='pt')

        long_token = self.long_terms_tokenizer(data, targets)
        return long_token

    def __len__(self):
        """ Return data length """
        return self.label.shape[0]