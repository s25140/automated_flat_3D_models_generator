from trimesh import Trimesh
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import PBRMaterial
import numpy as np
from PIL import Image
import sys


def generate_plane(image_path, width, height):
    vertices = np.array([
        [-width / 2, -height / 2, 0],
        [ width / 2, -height / 2, 0],
        [ width / 2,  height / 2, 0],
        [-width / 2,  height / 2, 0]
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    uv = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    plane = Trimesh(vertices=vertices, faces=faces)

    texture = Image.open(image_path)
    material = PBRMaterial(
        name='WallArtMaterial',
        baseColorTexture=texture,
        metallicFactor=0.5,
        roughnessFactor=0.8
    )
    # todo: add rug material and implementation

    plane.visual = TextureVisuals(
        uv=uv,
        material=material
    )
    return plane

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output GLB file path")
    parser.add_argument("--height", type=float, default=1.5, help="Height in meters")
    args = parser.parse_args()
    
    try:
        texture = Image.open(args.input)
        height = args.height
        width = height/texture.size[1]*texture.size[0]
        
        plane = generate_plane(args.input, width, height)
        plane.export(args.output)
        print(f"3D model saved to: {args.output}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

