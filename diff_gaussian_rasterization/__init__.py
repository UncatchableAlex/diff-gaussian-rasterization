#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import numpy as np
import torch
from . import _C
import json
from futhark_server import Server

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.array(v).astype(np.float32)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )
        # # --- Step 1: Save tensors ---
        # np.save('debug_means3D.npy', means3D.cpu().numpy())
        # np.save('debug_sh.npy', sh.cpu().numpy())
        # np.save('debug_colors_precomp.npy', colors_precomp.cpu().numpy())
        # np.save('debug_opacities.npy', opacities.cpu().numpy())
        # np.save('debug_scales.npy', scales.cpu().numpy())
        # np.save('debug_rotations.npy', rotations.cpu().numpy())
        # np.save('debug_cov3Ds_precomp.npy', cov3Ds_precomp.cpu().numpy())

        # # --- Step 2: Save scalar settings separately ---
        # scalar_settings = {
        #     'bg': raster_settings.bg.tolist(),
        #     'scale_modifier': raster_settings.scale_modifier,
        #     'viewmatrix': raster_settings.viewmatrix.tolist(),
        #     'projmatrix': raster_settings.projmatrix.tolist(),
        #     'tanfovx': raster_settings.tanfovx,
        #     'tanfovy': raster_settings.tanfovy,
        #     'image_height': raster_settings.image_height,
        #     'image_width': raster_settings.image_width,
        #     'sh_degree': raster_settings.sh_degree,
        #     'campos': raster_settings.campos.tolist(),
        #     'prefiltered': raster_settings.prefiltered,
        #     'antialiasing': raster_settings.antialiasing,
        #     'debug': raster_settings.debug
        # }
        # #print(f"Saving scalar settings: {scalar_settings}")
        # with open('debug_rasterizer_settings.json', 'w') as f:
        #     json.dump(scalar_settings, f)
        # Invoke C++/CUDA rasterizer

        # if we are supposed to use the futhark implementation of the renderer,
        # then pass our params to the futhark server through stdin. It's kind of a
        # bummer that we have to detach our tensors from the gpu to feed them to 
        # the server via stdin where they just get written to the gpu again. We basically
        # go gpu -> cpu -> gpu. Kinda sus
        if raster_settings.futhark_server:
            server = raster_settings.futhark_server
            inputs = {
                'bg':           to_numpy(raster_settings.bg),
                'means3D':      to_numpy(means3D),
                'colors':       to_numpy(colors_precomp),
                'opacities':    to_numpy(opacities),
                'scales':       to_numpy(scales),
                'rotations':    to_numpy(rotations),
                'viewmatrix':   to_numpy(raster_settings.viewmatrix).T,
                'projmatrix':   to_numpy(raster_settings.projmatrix).T,
                'tanfovx':      np.float32(raster_settings.tanfovx),
                'tanfovy':      np.float32(raster_settings.tanfovy),
                'image_height': np.int64(raster_settings.image_height),
                'image_width':  np.int64(raster_settings.image_width),
            }
            # provide all inputs to the server
            for name, value in inputs.items():
                server.put_value(name, value)

            # rasterize with the futhark server
            server.cmd_call(
                "rasterize",
                'radii',
                'color',                   
                *inputs.keys())
            radii = server.get_value('radii')
            color = np.transpose(server.get_value('color'), (2, 0, 1))

            # free variables from the server (we will replace them with new values next time we call the rasterizer)
            for name in inputs.keys():
                server.cmd_free(name)

            print(radii.shape, color.shape)

            invdepths = np.array([])
            return (torch.tensor(color, device='cuda', dtype=torch.float32), 
                    torch.tensor(radii, device='cuda', dtype=torch.int32), 
                    invdepths)


        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    futhark_server : Server

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

