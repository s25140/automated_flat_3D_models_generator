import os
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET
from torch.utils.data import DataLoader
from skimage import io
from scipy.ndimage import sobel
from skimage.filters import gaussian

def load_model(model_path):
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def normalize_prediction(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_path, pred, output_path):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    
    # Match the original image dimensions
    image = io.imread(image_path)
    im = Image.fromarray(predict_np*255).convert('RGB')
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    
    # Save the result
    imo.save(output_path)

def calculate_metrics(pred_np, original_image):
    """Calculate metrics for segmentation quality assessment"""
    
    if len(pred_np.shape) > 2:
        pred_np = pred_np.squeeze()
    
    # Resize prediction to match original image size
    pred_image = Image.fromarray((pred_np * 255).astype(np.uint8))
    pred_np = np.array(pred_image.resize((original_image.shape[1], original_image.shape[0]), 
                                        resample=Image.BILINEAR)) / 255.0
    
    # Confidence Score (based on prediction probabilities)
    confidence_score = float(np.mean(np.abs(pred_np - 0.5) * 2))
    
    return {
        'confidence_score': confidence_score
    }

def segment_image(image_path, model_path, output_path):
    # Load model
    net = load_model(model_path)
    
    # Create dataset and dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=[image_path],
        lbl_name_list=[],
        transform=transforms.Compose([
            RescaleT(320),
            ToTensorLab(flag=0)
        ])
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    
    # Get the single image data
    data_test = next(iter(test_salobj_dataloader))
    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)
    
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)
    
    # Inference
    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
    
    # Normalize prediction
    pred = d1[:,0,:,:]
    pred = normalize_prediction(pred)
    
    # Save result
    save_output(image_path, pred, output_path)
    
    # Calculate quality metrics
    original_image = io.imread(image_path)
    pred_np = pred.cpu().data.numpy()
    metrics = calculate_metrics(pred_np, original_image)
    
    print("\nSegmentation Quality Metrics:")
    print(f"Confidence Score: {metrics['confidence_score']:.3f}")
    
    # Clean up
    del d1, d2, d3, d4, d5, d6, d7
    
    return output_path, metrics

import sys

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--output", required=True, help="Output image path")
    args = parser.parse_args()
    
    try:
        output_path, metrics = segment_image(args.input, args.model, args.output)
        print(f"Output saved to: {output_path}")
        print(f"Confidence score: {metrics['confidence_score']}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
