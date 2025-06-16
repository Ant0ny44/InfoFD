import os
from torch.utils.data import Dataset, DataLoader
import json
import tqdm
import pickle
import torch
from models.clip.clip import tokenize

from PIL import Image
from random import shuffle
import torchvision.transforms as transforms

from models.clip.clip import load
import numpy as np

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

class TestDataset(Dataset):

    def __init__(self, data_dict:dict[str:list[list]]|str):

        super().__init__()

        if type(data_dict)== str:
            self.real_pathes = [(path, 0) for path in self.get_list(data_dict, must_contain='0_real')]
            self.fake_pathes = [(path, 1) for path in self.get_list(data_dict, must_contain='1_fake')]
        else:

            self.real_pathes = [(path, 0) for path in data_dict['real']]
            self.fake_pathes = [(path, 1) for path in data_dict['fake']]

        self.total = self.real_pathes + self.fake_pathes
        shuffle(self.total)
        self.transform  = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN, std=STD ),
        ])

    def __getitem__(self, index):

        img_path, label = self.total[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            index += 1
            img_path, label = self.total[index]
            img = Image.open(img_path).convert("RGB")


        return self.transform(img), label
    
    def __len__(self):

        return len(self.total)
    
    def recursively_read(self,rootdir, must_contain, exts=["png", "jpg","PNG", "JPEG", "jpeg"]):

        out = [] 
        for r, d, f in os.walk(rootdir):
            for file in f:
                if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                    out.append(os.path.join(r, file))
        return out


    def get_list(self, path, must_contain=''):

        image_list = self.recursively_read(path, must_contain)
        return image_list

class TestImgDatasets():
    
    def __init__(self,real_path:str, max_sample = -1):
        
        dataset_name:str = os.listdir(real_path)
        dataset_list = [os.path.join(real_path, name) for name in dataset_name]
        self.data_pairs:dict[str:dict[str:list[str]]] = {}

        for name, path in zip(dataset_name, dataset_list):
            
            real_list = self.get_list(path, must_contain='0_real')
            fake_list = self.get_list(path, must_contain='1_fake')
            self.data_pairs.update({name:TestDataset({'real':real_list,'fake':fake_list})})

        super().__init__()

    def get_all(self)->dict[str:Dataset]:

        return self.data_pairs
    
    def recursively_read(self,rootdir, must_contain, exts=["png", "jpg","PNG", "JPEG", "jpeg"]):
        out = [] 
        for r, d, f in os.walk(rootdir):
            for file in f:
                if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                    out.append(os.path.join(r, file))
        return out


    def get_list(self, path, must_contain=''):

        image_list = self.recursively_read(path, must_contain)
        return image_list



def preprocess_test_data(batch_size:int = 64,
                          device:str = 'cuda:0', 
                          model_name:str = 'ViT-L/14', 
                          test_data_path:str = './datasets/test/real', 
                          save_path:str = './datasets/test/features/test_data_cache.pkl',
                          max_sample:int = -1):


    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Path {test_data_path} does not exist.")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datasets_dict = TestImgDatasets(test_data_path).get_all()
    print(datasets_dict)
    features_dict:dict[str:tuple[torch.Tensor, torch.Tensor]] = {}
    clip, _ = load(model_name[5:], device)

    for name, dataset in datasets_dict.items():

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=14, pin_memory=True)

        total_features:torch.Tensor = torch.zeros((len(dataset), 768))

        total_labels:torch.Tensor = torch.zeros(len(dataset))
        tbar = tqdm.tqdm(enumerate(dataloader))

        for idx, batch in tbar:

            tbar.set_description(f'Preprocess at :{name}')

            imgs, labels = batch
                
            with torch.no_grad():

                img_features, all = clip.encode_image(imgs.to(device))
                img_features = all[f'layer11'].matmul(clip.visual.proj).detach()
                if imgs.shape[0] != batch_size:

                    total_features[idx * batch_size:] = img_features
                    total_labels[idx * batch_size:] = labels

                else:
                        
                    total_features[idx * batch_size:(idx + 1) * batch_size] = img_features.detach().cpu()
                    total_labels[idx * batch_size:(idx + 1) * batch_size] = labels.detach().cpu()

            features_dict.update({name:(total_features, total_labels)})

    with open(save_path, 'wb') as f:
            
        pickle.dump(features_dict, f)

class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, train_root_path, json_path, max_samples = 1000):

        super().__init__()

        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.data = self.data[:max_samples]
        self.transform  = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN, std=STD ),
        ])
        self.train_root_path = train_root_path

    def __getitem__(self, index):

        item = self.data[index]
        img_path = os.path.join(self.train_root_path, list(item.keys())[0])
        caption = list(item.values())[0]
        if 'real' in img_path:
            label = 0
        else:
            label = 1
        try:
            img = Image.open(img_path).convert("RGB")

        except:
            return self.__getitem__((index + 1) % len(self.data))

        return caption, self.transform(img), label

    def __len__(self):
        return len(self.data)
    
def preprocess_val_data(val_data_path:str, 
                          batch_size:int = 64,
                          device:str = 'cuda:0', 
                          model_name:str = 'ViT-L/14', 
                          save_path:str = './datasets/test/features/val_data_cache.pkl',
                          max_sample:int = -1):
            

            if not os.path.exists(val_data_path):

                raise FileNotFoundError(f"Path {val_data_path} does not exist.")
            
            save_dir = os.path.dirname(save_path)

            if not os.path.exists(save_dir):
                
                os.makedirs(save_dir)


            dataset = TestImgDatasets(val_data_path,max_sample).get_all()
            raw_dataset = TestDataset(val_data_path)
            
            features_dict:dict[str:tuple[torch.Tensor, torch.Tensor]] = {}
            clip, _ = load(model_name[5:], device)



            dataloader = DataLoader(raw_dataset, batch_size=batch_size,num_workers = 14, shuffle=False)

            total_features:torch.Tensor = torch.zeros((len(raw_dataset), 768))

            total_labels:torch.Tensor = torch.zeros(len(raw_dataset))
            tbar = tqdm.tqdm(enumerate(dataloader),total=len(raw_dataset)//batch_size)

            for idx, batch in tbar:

                    tbar.set_description('Preprocess at layer 11')

                    imgs, labels = batch
                    
                    with torch.no_grad():

                        img_features, all = clip.encode_image(imgs.to(device))
                        img_features = all[f'layer11'].matmul(clip.visual.proj).detach()
                        if imgs.shape[0] != batch_size:

                            total_features[idx * batch_size:] = img_features
                            total_labels[idx * batch_size:] = labels

                        else:
                            
                            total_features[idx * batch_size:(idx + 1) * batch_size] = img_features.detach().cpu()
                            total_labels[idx * batch_size:(idx + 1) * batch_size] = labels.detach().cpu()

            features_dict = (total_features, total_labels)

            with open(save_path, 'wb') as f:
                
                pickle.dump(features_dict, f)

def preprocess_train_data(train_img_root:str, 
                          train_captions_path,
                          batch_size:int = 64,
                          device:str = 'cuda:0', 
                          model_name:str = 'ViT-L/14', 
                          save_path:str = './datasets/test/features/train_data_cache.pkl',
                          max_sample:int = -1):


    if not os.path.exists(train_captions_path):
        raise FileNotFoundError(f"Path {train_captions_path} does not exist.")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    clip, _ = load(model_name[5:], device)

    raw_dataset = CaptionDataset(train_img_root,train_captions_path, max_sample)

    dataloader = DataLoader(raw_dataset, batch_size=batch_size,num_workers=14, pin_memory=True)
    total_features:torch.Tensor = torch.zeros((len(raw_dataset),2, 768))

    total_labels:torch.Tensor = torch.zeros(len(raw_dataset))
            
    for idx, batch in enumerate(tqdm.tqdm(dataloader,total=len(raw_dataset)//batch_size)):

        prompts, imgs, labels = batch
                
        with torch.no_grad():

            prompts_token = tokenize(prompts,truncate=True)
            text_features = clip.encode_text(prompts_token.to(device))
            img_features, all = clip.encode_image(imgs.to(device))
            img_features = all[f'layer11'].matmul(clip.visual.proj).detach()
            if imgs.shape[0] != batch_size:

                total_features[idx * batch_size:] = torch.stack([text_features,img_features], dim = 1)
                total_labels[idx * batch_size:] = labels

            else:
                        
                total_features[idx * batch_size:(idx + 1) * batch_size] = torch.stack([text_features,img_features], dim = 1)
                total_labels[idx * batch_size:(idx + 1) * batch_size] = labels

    with open(save_path, 'wb') as f:
                
        pickle.dump((total_features.detach().cpu(), total_labels.detach().cpu()), f)