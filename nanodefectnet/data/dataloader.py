from torch.utils.data import DataLoader

from nanodefectnet.data.wafermap_dataset import WafermapDataset


def create_loaders(df_dataset_train, df_dataset_val, df_dataset_test, cfg):
    dataset_path = cfg["dataset"]["dataset_path"]
    batch_size = cfg["dataset"]["batch_size"]

    # 1 - Instantiate the dataset class for train, valid and test
    wafermap_dataset_train = WafermapDataset(
        df_dataset=df_dataset_train, cfg=cfg, dataset_path=dataset_path
    )
    wafermap_dataset_val = WafermapDataset(
        df_dataset=df_dataset_val, cfg=cfg, dataset_path=dataset_path
    )
    wafermap_dataset_test = WafermapDataset(
        df_dataset=df_dataset_test, cfg=cfg, dataset_path=dataset_path
    )

    # 2 - Instantiate the dataloaders
    wafermap_dataloader_train = DataLoader(
        dataset=wafermap_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    wafermap_dataloader_val = DataLoader(
        dataset=wafermap_dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    wafermap_dataloader_test = DataLoader(
        dataset=wafermap_dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    return wafermap_dataloader_train, wafermap_dataloader_val, wafermap_dataloader_test
