
import torchaudio
from data_loader import *
import os

def torchz(x):
    return (x-torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True)+1e-9)

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, basepath, normalize=False, augment=False):
        """
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.basepath = basepath
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, pid, audiolen = sample[0], sample[1], sample[2]
        # index 3 second window
        offset = np.random.uniform(0, float(audiolen)-3)
        offset = int(np.floor(offset*self.sample_rate))
        index = np.arange(offset, offset+self.sample_rate*3)
        spect = self.parse_audio(os.path.join(self.basepath,audio_path), index=index)
        return torchz(spect.view(1,spect.size()[0],spect.size()[1])), int(pid)

    def __len__(self):
        return self.size

class SpectrogramVerificationDataset(SpectrogramDataset):
    def __init__(self, audio_conf, manifest_filepath, basepath, normalize=False, augment=False):
        """
        """
        super(SpectrogramVerificationDataset, self).__init__(audio_conf, manifest_filepath, basepath, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path_1, audio_path_2, same = sample[0], sample[1], sample[2]
        spect_1 = self.parse_audio(os.path.join(self.basepath,audio_path_1))
        spect_2 = self.parse_audio(os.path.join(self.basepath,audio_path_2))
        spect_1 = torchz(spect_1.view(1,spect_1.size()[0],spect_1.size()[1]))
        spect_2 = torchz(spect_2.view(1,spect_2.size()[0],spect_2.size()[1]))
        return spect_1, spect_2, int(same)

    def __len__(self):
        return self.size

class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        np.random.shuffle(ids)
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
