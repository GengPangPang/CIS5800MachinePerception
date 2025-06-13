import pyrender
from pyrender import Mesh
import numpy as np
import trimesh
import json
import glob
import pdb

class Renderer:
    """
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, intrinsics, img_w, img_h):

        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                       viewport_height=img_h,
                                       point_size=1.0)
        self.focal_x = intrinsics[0, 0]
        self.focal_y = intrinsics[1, 1]

        self.center_x = intrinsics[0, 2]
        self.center_y = intrinsics[1, 2]

        self.img_w = img_w
        self.img_h = img_h

    def render(self, meshes, R, t, img):
        # construct a scene with ambient light
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))

        for mesh in meshes:
            scene.add(mesh)

        # convert rotation from opengl to cv2
        gl2cv_rot = trimesh.transformations.rotation_matrix(
                        np.radians(180), [1, 0, 0])
        T_w_c = np.eye(4)
        T_w_c[:3, :3] = R
        T_w_c[:3, 3] = t
        T_w_gl = T_w_c @ gl2cv_rot

        # test camera coordinate
        camera = pyrender.IntrinsicsCamera(fx=self.focal_x, fy=self.focal_y,
                                                   cx=self.center_x, cy=self.center_y,
                                                    zfar=1000)
        # add camera to scene
        scene.add(camera, pose=T_w_gl)

        # render image
        renderer = pyrender.OffscreenRenderer(viewport_width=self.img_w,
                                                       viewport_height=self.img_h,
                                                       point_size=1.0)
        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.uint8)

        # some pixel values are invalid
        # paste the rendered image onto the original image based on valid depth mask
        valid_mask = (rend_depth>0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * img)

        return output_img, rend_depth
