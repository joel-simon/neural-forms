import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

class LinearNet(nn.Module):
    def __init__(self, input_size, output_shape):
        super(LinearNet, self).__init__()
        self.output_shape = output_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(input_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(output_shape))),
            # nn.Tanh()
        )
    def forward(self, z):
        out = self.model(z)
        out = out.view(out.size(0), *self.output_shape)
        return out


device = torch.device("cuda:0")
torch.cuda.set_device(device)

class Generator(nn.Module):
    def __init__(self, batch_size=100, latent_size=128, img_size=128, seed_sphere_divisions=3):
        super(Generator, self).__init__()
        src_mesh = ico_sphere(seed_sphere_divisions)
        output_shape = src_mesh.verts_packed().shape
        self.src_meshes = src_mesh.extend(batch_size).cuda()
        self.layers = LinearNet(latent_size, output_shape).cuda()
        
        

        """ Setup rendering. """
        num_views = batch_size
        self.lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)

        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=128, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=None, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        ).cuda()
        
    def forward(self, z):
        deform_verts = self.layers(z)
        new_meshes = self.src_meshes.offset_verts(deform_verts.view(-1, 3))

        # elev = torch.FloatTensor(z.shape[0]).uniform_(-5, 5)
        r = 60
        azim = torch.FloatTensor(z.shape[0]).uniform_(180-r, 180+r)

        elev = torch.FloatTensor(z.shape[0]).uniform_(0, 0)
        # azim = torch.FloatTensor(z.shape[0]).uniform_(160, 200)
        # azim = torch.FloatTensor(z.shape[0]).uniform_(-180, 180)

        R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        self.cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)

        images_predicted = self.renderer_silhouette(
            new_meshes,
            cameras=self.cameras,
            lights=self.lights
        )
        predicted_silhouette = images_predicted[..., 3]
        return new_meshes, predicted_silhouette

# class Discriminator(nn.Module):
#     def __init__(self, img_size, channels=1):
#         super(Discriminator, self).__init__()
#         self.img_size = img_size
#         self.channels = channels
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         # The height and width of downsampled image
#         ds_size = self.img_size // 2 ** 4
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)
#         return validity

class Discriminator(nn.Module):
    def __init__(self, img_size, channels=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_size*img_size*channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# class Discriminator(nn.Module):
#     def __init__(self, img_shape):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )
#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#         return validity

