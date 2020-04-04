import csv
import string

import torch


class utils:
    def __init__(self):
        self.path = "star_trek_transcripts_all_episodes_f.csv"
        self.all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
        self.n_letters = len(self.all_letters)
        self.category_lines = {}
        self.all_categories = ['st']
        self.category_lines["st"] = []
        filterwords = ['NEXTEPISODE']
        with open(self.path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar='"')
            for row in reader:
                # print(row)
                for el in row:
                    if (el not in filterwords) and (len(el) > 1):
                        v = el.strip().replace(';', '').replace('\"','')
                        self.category_lines['st'].append(v)
        self.n_categories = len(self.all_categories)

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self,line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor

    def targetTensor(self, line):
        letter_indexes = [self.all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(self.n_letters - 1)  # EOS
        return torch.LongTensor(letter_indexes)


    def batch_categoryFromOutput(self, output):
        # print(output)
        output_tensor = torch.unbind(output, dim=0)
        ret_val = []
        for outp in output_tensor:
            top_n, top_i = outp.topk(1, 0)
            category_i = top_i[0].item()
            ret_val.append(category_i)
        return ret_val