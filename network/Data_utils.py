import os
import torch
from torch.utils.data import Dataset


class DescriptionSceneDatasetV2(Dataset):
    def __init__(self, data_description_path, data_scene, data_img_path, type_model_desc):
        self.description_path = data_description_path
        self.data_scene = data_scene
        self.data_img = torch.empty(10, 1, 512)
        self.type_model_desc = type_model_desc

        img_features = os.listdir(data_img_path)
        for i, img_f in enumerate(img_features):
            f = torch.load(data_img_path + os.sep + img_f)
            self.data_img[i, :, :] = f

    def __len__(self):
        return 33840

    def __getitem__(self, index):
        data_description = torch.load(
            self.description_path + os.sep + "desc_" + str(index // 10) + '_' + str(index % 10) + ".pt")
        if self.type_model_desc == "mean":
            data_description = torch.mean(data_description, 0)
        data_scene = self.data_scene[index // 10]
        data_img = self.data_img[index % 10]
        return data_description, data_scene, data_img
