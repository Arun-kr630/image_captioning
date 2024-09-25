import torch
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import save_checkpoint
from get_loader import get_loader
from model import CNNtoRNN



def train():
    train_loader, dataset = get_loader( root_folder="flickr8k/Images",captions_file="flickr8k/captions.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 50
    train_CNN=False

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    for epoch in range(num_epochs):
        loop =tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for idx,(imgs, captions) in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            loop.set_description(f"Epoch=[{epoch}/{num_epochs}] ")
            loop.set_postfix(loss=loss.item())
        if (epoch+1)%10==0 or epoch==49:
            checkpoint={"epoch":epoch,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict()}
            save_checkpoint(checkpoint,epoch)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train()
    print("done")
