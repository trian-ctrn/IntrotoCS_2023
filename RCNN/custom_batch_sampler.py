import numpy  as np
import random
from torch.utils.data import Sampler    
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from finetune_dataset import CustomFinetuneDataset


class CustomBatchSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        self.length = self.num_positive + self.num_negative
        self.idx_list = list(range(self.length))

        self.batch = batch_negative + batch_positive
        self.num_iter = self.length // self.batch

    def __iter__(self):
        sampler_list =[]
        for _ in range(self.num_iter):
            tmp = np.concatenate(
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter


def test():
    root_dir = 'finetune_ver13/train'
    train_data_set = CustomFinetuneDataset(root_dir)
    print(train_data_set.get_positive_num())
    print(train_data_set.get_negative_num())
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 1000, 200)

    print('sampler len: %d' % train_sampler.__len__())
    print('sampler batch num: %d' % train_sampler.get_num_batch())

    first_idx_list = [train_sampler.__iter__()][:128]
    # first_idx_list = [train_sampler.__iter__()]
    print(first_idx_list)
 
    print('positive batch: %d' % np.sum(np.array(first_idx_list) < 66517))


def test2():
    root_dir = 'finetune_ver13/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 96, 32)
    data_loader = DataLoader(train_data_set, batch_size=28, sampler=train_sampler, num_workers=2, drop_last=True)

    inputs, targets = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    test2()
