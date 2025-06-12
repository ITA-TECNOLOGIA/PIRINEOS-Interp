import os
import random
from collections import defaultdict
from torch.utils.data import Sampler

class SubjectInterleaveSampler(Sampler):
    def __init__(self, data_source, seed=42):
        """
        data_source: any dataset whose __getitem__(i)['fnames'][0] contains the path
                     like ".../subject_name/whatever.json"
        """
        self.data_source = data_source
        random.seed(seed)

        # 1) bucket indices by subject
        buckets = defaultdict(list)
        for idx in range(len(data_source)):
            fname = data_source[idx]['fnames'][0]
            subject = fname.split(os.sep)[-2]    # adjust if your path is different
            buckets[subject].append(idx)

        # 2) shuffle each bucket
        for b in buckets.values():
            random.shuffle(b)

        # 3) interleave round-robin
        interleaved = []
        i = 0
        done = False
        lists = list(buckets.values())
        while not done:
            done = True
            for bucket in lists:
                if i < len(bucket):
                    interleaved.append(bucket[i])
                    done = False
            i += 1

        self.indices = interleaved

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
