import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print("Example 1 OUTPUT: "+ " ".join(model.caption_image(test_img1.to(device), dataset.vocab)))
    test_img2 = transform(Image.open("test_examples/child.jpg").convert("RGB")).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print("Example 2 OUTPUT: "+ " ".join(model.caption_image(test_img2.to(device), dataset.vocab)))
    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(0)
    print("Example 3 CORRECT: Bus driving by parked cars")
    print("Example 3 OUTPUT: "+ " ".join(model.caption_image(test_img3.to(device), dataset.vocab)))
    test_img4 = transform(Image.open("test_examples/boat.png").convert("RGB")).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print("Example 4 OUTPUT: "+ " ".join(model.caption_image(test_img4.to(device), dataset.vocab)))
    test_img5 = transform(Image.open("test_examples/horse.png").convert("RGB")).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print("Example 5 OUTPUT: "+ " ".join(model.caption_image(test_img5.to(device), dataset.vocab)))
    model.train()

def caption_image(model, image, vocabulary, max_length=50):
    result_caption = []

    with torch.no_grad():
        x = model.encoderCNN(image).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states = model.decoderRNN.lstm(x, states)
            output = model.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result_caption.append(predicted.item())
            x = model.decoderRNN.embed(predicted).unsqueeze(0)

            if vocabulary.itos[predicted.item()] == "<EOS>":
                break

    return [vocabulary.itos[idx] for idx in result_caption]

def save_checkpoint(state,epoch ):
    filename=f"EDchekpoints/my_checkpoint_{epoch}.pth"
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    return torch.load(checkpoint)
