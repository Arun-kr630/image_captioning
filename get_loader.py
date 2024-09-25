import os 
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
import pandas as pd
import spacy
from torch.nn.utils.rnn import pad_sequence

spacy_eng=spacy.load("en_core_web_sm")
class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos={0:"<PAD>",1:"<EOS>",2:"<SOS>",3:"<UNK>"}
        self.stoi={"<PAD>":0,"<EOS>":1,"<SOS>":2,"<UNK>":3}
        self.freq_threshold=freq_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


    def build_vocabulary(self,sentence_list):
        frequencies={}
        idx=4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word]=1
                else:
                    frequencies[word] += 1
                if frequencies[word]==self.freq_threshold:
                    self.itos[idx]=word
                    self.stoi[word]=idx
                    idx+=1
    def numericalize(self,text):
        tokenized_text=self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
    

class FlickrDataset(Dataset):
    def __init__(self,root_dir,captions_file,freq_threshold=5):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.root_dir=root_dir
        self.df=pd.read_csv(captions_file)
        self.imgs=self.df["image"]
        self.captions=self.df['caption']

        self.vocab=Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self,index):
        caption=self.captions[index]
        img_id=self.imgs[index]
        img=Image.open(os.path.join(self.root_dir,img_id)).convert("RGB")
        img=self.transform(img)
        numeric_caption=[self.vocab.stoi["<SOS>"]]
        numeric_caption+=self.vocab.numericalize(caption)
        numeric_caption.append(self.vocab.stoi["<EOS>"])
        
        return img,torch.tensor(numeric_caption)
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(root_folder,captions_file,batch_size=64):
    dataset=FlickrDataset(root_folder,captions_file)
    pad_idx=dataset.vocab.stoi["<PAD>"]
    collate_fn=MyCollate(pad_idx=pad_idx)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)
    return loader,dataset