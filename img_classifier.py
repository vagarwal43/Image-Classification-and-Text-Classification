import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
import wandb

# logging in to Wandb
wandb_api = "56d965a22c77c00b87909fad0a629ad487f9f66e"
wandb.login(key=wandb_api)

wandb.init(project="Gen-AI_HW0",name="new_model",config={
    "epochs":8, "batch_size":16, "learning_rate":1e-4
})

img_size = (256,256)
num_labels = 3

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(batch_size):
    transform_img = T.Compose([
        T.ToTensor(), 
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256

        # T.Grayscale(num_output_channels=1),

        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
        ])
    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )
    #Validation dataset
    val_data = CsvImageDataset(
        csv_file='./data/img_val.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data,batch_size=batch_size) #validation dataloader
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, test_dataloader, val_dataloader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # old model

        # self.flatten = nn.Flatten()
        # # First layer input size must be the dimension of the image
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(img_size[0] * img_size[1] * 3, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_labels))

        # new model

        self.conv1 = nn.Conv2d(3, 128,4,4) 
        self.ln1 = nn.LayerNorm([128, 64, 64])
        self.conv2 = nn.Conv2d(128, 128, kernel_size=7, padding=3)  
        self.ln2 = nn.LayerNorm([128, 64, 64])
        
        self.fc1 = nn.Linear(128, 256)  
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(256, 128)
        self.pool = nn.AvgPool2d(2) 
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(128 * 32 * 32, 3) 
        

    def forward(self, x):

        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x) 
        x = x.permute(0, 3, 1, 2) 
        x = self.pool(x)
        x = self.flatten(x)  
        x = self.fc3(x)

        return x

def train_one_epoch(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    total_examples_seen = size*t

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item() 
        current = (batch + 1) * dataloader.batch_size
        total_examples_seen += batch_size   #total examples for part 2.1
      
        
        #Wandb logging for part 2.1, 2.3
        wandb.log({"Training_loss": loss,
                   "examples_seen": total_examples_seen})
        
        if batch % 10 == 0:
            print(f"Train loss = {loss:>7f}  [{current:>5d}/{size:>5d}]")

        
def evaluate(dataloader, dataname, model, loss_fn, t, n_epochs):
    size = len(dataloader.dataset)
    num_batches = 0
    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            num_batches+=1

            if (t == n_epochs - 1 and num_batches==1):
                log_images(X, pred, y, prefix=dataname)

    avg_loss /= size
    correct /= size
    print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    
    
    return avg_loss, correct

def log_images(X, preds, labels, prefix):
    preds = torch.argmax(preds, dim=1)
    X = X.cpu()
    preds = preds.cpu()
    labels = labels.cpu()
    
    logged_images = {}
    for i in range(X.size(0)):
        # Create the caption for each image
        caption = f"{preds[i].item()} / {labels[i].item()}"
        
        # Log each image directly as a tensor to wandb
        logged_images[f"{prefix} Image {i+1}"] = wandb.Image(X[i], caption=caption)
    
    # Log all images in the batch to wandb
    wandb.log(logged_images)
    print("Images logged to wandb")


def main(n_epochs, batch_size, learning_rate):
    print(f"Using {device} device")

    config = wandb.config
    train_dataloader, test_dataloader, val_dataloader = get_data(batch_size)
    
    model = NeuralNetwork().to(device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({'total_parameters': total_params})
    print(f"Total parameters in the model: {total_params}")
    
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_ev = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, t)

        avg_train_loss, train_acc = evaluate(train_dataloader, "Train", model, loss_fn_ev, t, n_epochs)
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": train_acc,
                   "Epochs": t})
        
        avg_val_loss, val_acc = evaluate(val_dataloader, "Val", model, loss_fn_ev, t, n_epochs)
        wandb.log({"Validation Loss": avg_val_loss, "Validation Accuracy": val_acc,
                   "Epochs": t})
        
        avg_test_loss, test_acc = evaluate(test_dataloader, "Test", model, loss_fn_ev, t, n_epochs)

    print("Done!")
    wandb.finish()

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Load the model (just for the sake of example)
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=5, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)

    args = parser.parse_args()
        
    main(args.n_epochs, args.batch_size, args.learning_rate)