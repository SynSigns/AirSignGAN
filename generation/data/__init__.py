from .datasets import  Dataset
from torch.utils.data import DataLoader

def get_loader(args):
    # datasets
    dataset = Dataset(root=args.root, batch_size=args.batch_size, truncate_size=args.truncate_size)

    # loaders
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return loader
