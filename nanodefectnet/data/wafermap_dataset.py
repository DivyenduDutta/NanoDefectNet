import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class WafermapDataset(Dataset):
    def __init__(self, df_dataset, cfg, dataset_path) -> None:
        super().__init__()

        self.df_dataset = df_dataset
        self.dataset_path = dataset_path
        self.cfg = cfg
        self.do_resize = cfg["data"].get("do_resize", 1.0) > 0.0

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.df_dataset.iloc[idx]["PATH"])
        label = self.df_dataset.iloc[idx]["LABEL"]

        if not os.path.exists(img_path):
            print("The image '{}' does not exist ".format(img_path))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.do_resize:
            img = cv2.resize(img, (self.cfg["data"]["size"], self.cfg["data"]["size"]))
        img = transforms.ToTensor()(img)

        return img, label
