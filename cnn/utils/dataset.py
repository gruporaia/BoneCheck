import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mode="train"):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

        self.crop_suffixes = [
            "_cropped.png",
        ]

        self.samples = []

        print(len(df))
        for _, row in self.df.iterrows():
            fold = row["fold"]
            file_id = os.path.splitext(row["path"])[0]
            label = int(row["class_numeric"])
            for suffix in self.crop_suffixes:
                img_path = os.path.join(self.img_dir, f"fold_{fold}", f"{file_id}{suffix}")
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                    # print('IF ', img_path,'\n')
                else:
                    print('ELSE ', img_path, '\n')
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label
