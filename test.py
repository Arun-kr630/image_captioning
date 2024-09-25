import torch
from get_loader import get_loader
from model import CNNtoRNN
from utils import load_checkpoint,print_examples,caption_image
train_loader, dataset = get_loader( root_folder="flickr8k/Images",captions_file="flickr8k/captions.txt")
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 3e-4
num_epochs = 50
train_CNN=False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device {device}')
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

checkpoint=load_checkpoint("EDchekpoints/my_checkpoint_19.pth")
model.load_state_dict=checkpoint['model_state']
optimizer.load_state_dict=checkpoint['optimizer_state']
data=iter(train_loader)
d=next(data)
img,_=d
img=img[0]
img=img.reshape(-1,3,299,299).to(device)
cap=caption_image(model,img,dataset.vocab)
print(cap)
# print_examples(model,device,dataset)