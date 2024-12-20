import trimesh
import numpy as np
from PIL import Image
import os
from trimesh.visual.material import PBRMaterial


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
    plane = trimesh.Trimesh(vertices=vertices, faces=faces)

    texture = Image.open(image_path)
    material = PBRMaterial(
        name='WallArtMaterial',
        baseColorTexture=texture,
        metallicFactor=0.5,
        roughnessFactor=0.8
    )
    # todo: add rug material and implementation

    plane.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=material
    )
    return plane

# Load image
current_dir = os.path.dirname(os.path.abspath(__file__))
texture_path = os.path.join(current_dir, '0_3_c.jpg')
texture = Image.open(texture_path)

# Plane dimensions in meters
#width = 2.0
height = 1.5
width = height/texture.size[1]*texture.size[0]

plane = generate_plane(texture_path, width, height)

plane.export(os.path.join(current_dir,'output.glb'))

