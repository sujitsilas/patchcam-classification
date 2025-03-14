import torch
from data_utils import get_data_loaders, Macenko
from model import get_model
from train_val import train_model, validate_model
from grad_cam_utils import grad_cam_visualize
import matplotlib
from torchvision import transforms
import h5py
import numpy as np


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Initializing Macenko class with test image
    train_data_path="./data/pcam/training_split.h5"

    with h5py.File(train_data_path, 'r') as f:
        train_data = f['x'][:]  # shape (N, H, W, C)
        reference_image = f['x'][176298]
        #reference_image = np.array(reference_image, dtype=np.uint8)
    
    # Data transformations
    data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((299, 299)), #(299, 299) for inception
        transforms.ToTensor(),
        Macenko(reference_image=reference_image),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        
])

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=128,
        num_workers=12,
        transform = data_transform
    )


    model_type = 'inception' # inception or efficientnet
    model = get_model(model_type=model_type, device=device)
    

    epochs = 20
    trained_model = train_model(model, train_loader, val_loader, epochs=epochs, device=device)
    validate_model(trained_model, test_loader, device=device)


if __name__ == "__main__":
    main()
