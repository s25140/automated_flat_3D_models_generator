import os
import sys
import argparse
from PIL import Image

# Import functions from other modules
sys.path.append('./MLFlow')
sys.path.append('./Cropping/U-2-Net')
sys.path.append('./flat_3D_models_generation(deterministic)')

from CNN_class_predict import get_model as get_classifier, predict as classify_image
from u2net_single import segment_image
from generate_flat_3D_model_with_image import generate_plane
from local_package.Cropping import crop_new_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output GLB file path")
    parser.add_argument("--height", type=float, default=1.5, help="Height in meters")
    parser.add_argument("--temp-dir", default="./temp", help="Temporary directory for intermediate files")
    args = parser.parse_args()

    try:
        os.makedirs(args.temp_dir, exist_ok=True)

        # 1. Classify image
        classifier = get_classifier("MLFlow/models/model_two_inputs_GPU.h5")
        is_cropped = classify_image(classifier, args.input)
        
        final_image = args.input
        if is_cropped < 0.5:  # If image needs cropping
            # 2. Segment the image
            mask_path = os.path.join(args.temp_dir, "mask.png")
            segmented_path, metrics = segment_image(
                args.input,
                "Cropping/U-2-Net/saved_models/u2net_finetuned/u2net_finetuned.pth",
                mask_path
            )
            
            # 3. Crop the image based on segmentation
            #if metrics['confidence_score'] > 0.5:  # Only crop if segmentation is confident
            cropped_path = os.path.join(args.temp_dir, "cropped.png")
            final_image = crop_new_image(args.input, mask_path, cropped_path)
        
        # 4. Generate 3D model
        texture = Image.open(final_image)
        height = args.height
        width = height/texture.size[1]*texture.size[0]
        
        plane = generate_plane(final_image, width, height)
        plane.export(args.output)
        
        print(f"Successfully generated 3D model: {args.output}")
        sys.exit(0)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
