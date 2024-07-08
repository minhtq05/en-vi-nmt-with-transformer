from .mtdataset import MTDataset

"""
create dataset objects for the dataset
    args: (
        en_train: List[str], English train set
        en_test: List[str], English test set
        vi_train: List[str], Vietnamese train set
        vi_test: List[str], Vietnamese test set
    )
    Return train + test Dataset object
"""
def create_dataset(en_train, en_test, vi_train, vi_test):
    train_data = MTDataset(en_train, vi_train, split='train')
    val_data = MTDataset(en_test, vi_test, split='test')

    return train_data, val_data