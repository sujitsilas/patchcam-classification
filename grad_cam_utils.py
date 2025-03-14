import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer(model, model_type):
    """
    Returns the module (layer) to which Grad-CAM should attach
    based on the model architecture.
    """
    if model_type == 'inception':
        # For Inception v3, the final convolution block is "Mixed_7c"
        return model.Mixed_7c
    elif model_type == 'efficientnet':
        # For EfficientNet-B0, you can use the final conv ("_conv_head") or the last block in "_blocks[-1]"
        return model._conv_head
    else:
        raise ValueError(f"Unknown/unsupported model_type {model_type} for Grad-CAM.")

def grad_cam_visualize(model, loader, device='cpu', model_type='inception', save_path=None):
    """
    Applies Grad-CAM on the first batch of images from `loader`,
    then displays the heatmap overlay for the first image in that batch.
    
    Args:
        model: The PyTorch model.
        loader: A PyTorch DataLoader from which to fetch images.
        device: 'cpu' or 'cuda'.
        model_type: The type of model ('inception', 'efficientnet', etc.).
        save_path: If provided, the path where the Grad-CAM visualization will be saved.
    """
    model.eval()
    model.to(device)

    # Grab a single batch
    data_iter = iter(loader)
    images, labels = next(data_iter)

    images = images.to(device)
    labels = labels.to(device)

    # We will run Grad-CAM on the first image in the batch
    input_tensor = images[0].unsqueeze(0)

    # Identify the correct layer for Grad-CAM
    target_layer = get_target_layer(model, model_type)

    # Initialize the Grad-CAM extractor
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=(device == 'cuda'))

    # Generate the CAM
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]  # Take the CAM for the first (and only) image

    # Convert the input image to numpy for visualization
    # images[0] is [C,H,W], so permute to [H,W,C]
    rgb_img = images[0].permute(1, 2, 0).detach().cpu().numpy()

    # Normalize the image to [0,1] for display (if needed)
    rgb_min, rgb_max = rgb_img.min(), rgb_img.max()
    if (rgb_max - rgb_min) > 0:
        rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min)

    # Overlay heatmap onto the original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Display using Matplotlib
    plt.figure(figsize=(6, 6))
    plt.title("Grad-CAM Visualization")
    plt.axis('off')
    plt.imshow(visualization)

    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
