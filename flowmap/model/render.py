import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ..scene.gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh
from ..scene.cameras import Camera, DifferentiableCamera_eval
from ..utils.pose_utils import get_camera_from_tensor, quadmultiply, quaternion_to_matrix, rotmat2qvec_tenosr, \
    qvec2rotmat_tensor
from ..utils.graphics_utils import getWorld2View3, getProjectionMatrix


# 这段代码是一个用于渲染场景的函数，主要是通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
# def render(viewpoint_camera: Camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
# def render(viewpoint_camera: Camera, pc : GaussianModel, xyz_flowmap : torch.Tensor, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
# def render(viewpoint_camera: Camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, xyz_flowmap=None, scaling_modifier = 1.0, override_color = None):
# render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline, self.background, intrinsics_uncropped, extrinsics, self.batch, xyz_flowmap=xyz_flowmap, camera_pose=pose)
def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, xyz_flowmap=None, scaling_modifier=1.0, override_color=None, camera_pose=None):
# def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, depth_from_flowmap, intrinsics_uncropped, extrinsics, rgb, xyz_flowmap=None, scaling_modifier=1.0, override_color=None, camera_pose=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    """
    viewpoint_camera:       # 是scene.py文件读取的sparse/cameras.bin  images.bin points3D.bin中获取相机参数
        full_proj_transform:
        image_height:1261
        image_name:
        image_width:
        original_image:
        projection_matrix:
        scale: 1.0
        training:True
        trans:array([0., 0., 0.])
        uid:31
        world_view_transform:
        zfar:100.0
        znear:0.01    
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means  #
    """# 创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度。"""
    # pc.get_xyz： 从GaussianModel，gaussian_model.py中create_from_pcd输出的xyz值
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    ## 需要根据pc.get_depth_from_flowmap 的depth结合flowmap输出的I E 反投影，得到xyz
    # depth_mixed = 0.5 * pc.get_depth_from_flowmap + 0.5 * depth_from_flowmap   # 1.0869   1.04275
    # xyz_from_flowmap = pc.get_xyz_from_depthflowmap(depth_mixed, intrinsics_uncropped, extrinsics, rgb, num_images=-1)

    # xyz_from_flowmap = pc.get_xyz_from_depthflowmap(pc.get_depth_from_flowmap, intrinsics_uncropped, extrinsics, rgb, num_images=-1)

    if xyz_flowmap is None:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        # screenspace_points = torch.zeros_like(xyz_from_flowmap, dtype=xyz_from_flowmap.dtype, requires_grad=True, device="cuda") + 0
    else:
        screenspace_points = torch.zeros_like(pc.get_xyz_from_flowmap, dtype=pc.get_xyz_from_flowmap.dtype, requires_grad=True, device="cuda") + 0
        # screenspace_points = torch.zeros_like(xyz_from_flowmap, dtype=xyz_from_flowmap.dtype, requires_grad=True, device="cuda") + 0
        # screenspace_points = torch.zeros_like(xyz_from_flowmap, dtype=pc.get_xyz_from_flowmap.dtype, requires_grad=True, device="cuda") + 0
        # screenspace_points = torch.zeros_like(xyz_flowmap, dtype=xyz_flowmap.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # print("pipe: ", pipe)   # <arguments.GroupParams object at 0x7fe065fc76d0> [19/06 02:56:58]
    # Set up rasterization configuration
    # 计算视场的 tan 值，这将用于设置光栅化配置。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # print("tanfovx: ", tanfovx)      # 0.3833445449846394
    # print("tanfovy: ", tanfovy)      # 0.7085160148877734

    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵、球面谐波、相机中心等。
    if camera_pose is not None:
        w2c = torch.eye(4).cuda()
        projmatrix = (w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))).squeeze(0)    # 执行批矩阵乘法（batch matrix multiplication）
        camera_pos = w2c.inverse()[3, :3]

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            # viewmatrix=viewpoint_camera.world_view_transform,
            # projmatrix=viewpoint_camera.full_proj_transform,
            viewmatrix=w2c,
            projmatrix=projmatrix,
            sh_degree=pc.active_sh_degree,
            # campos=viewpoint_camera.camera_center,                  # self.world_view_transform.inverse()[3, :3]
            campos=camera_pos,
            prefiltered=False,
            debug=False
            # debug=pipe.debug
        )
    else:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,  # self.world_view_transform.inverse()[3, :3]
            prefiltered=False,
            debug=False
            # debug=pipe.debug
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上。

    # 获取高斯分布的三维坐标、屏幕空间坐标和透明度。
    if xyz_flowmap is None:
        if camera_pose is not None:
            rel_w2c = get_camera_from_tensor(camera_pose)
            # rel_w2c = get_camera_from_tensor(quaternion, T)  # quaternion: [1,0,0,0]  T:[0,0,0]
            # Transform mean and rot of Gaussians to camera frame
            gaussians_xyz = pc._xyz.clone()
            gaussians_rot = pc._rotation.clone()
            xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
            xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)

            # 将高斯模型的均值和旋转从世界坐标系转换到相机坐标系下的坐标值
            gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]  # w2c * 三维坐标的齐次坐标
            gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
            means3D = gaussians_xyz_trans
        else:
            means3D = pc.get_xyz
    else:
        if camera_pose is None:
            means3D = xyz_flowmap
        else:
            rel_w2c = get_camera_from_tensor(camera_pose)
            # rel_w2c = get_camera_from_tensor(quaternion, T)  # quaternion: [1,0,0,0]  T:[0,0,0]
            # Transform mean and rot of Gaussians to camera frame
            gaussians_xyz = pc._xyz_from_flowmap.clone()
            # gaussians_xyz = xyz_from_flowmap.clone()
            gaussians_rot = pc._rotation.clone()
            xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
            xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)

            # 将高斯模型的均值和旋转从世界坐标系转换到相机坐标系下的坐标值
            gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]  # w2c * 三维坐标的齐次坐标
            gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
            means3D = gaussians_xyz_trans

    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则，它将由光栅化器根据尺度和旋转进行计算。
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:   # false
    #     # print("666666666666")       # 不读取
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)  # 获取预计算的三维协方差矩阵。
    # else:       # 获取缩放和旋转信息。（对应的就是3D高斯的协方差矩阵了）
    #     # print("7777777777777")
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    scales = pc.get_scaling
    if camera_pose is not None:
        rotations = gaussians_rot_trans
    else:
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预先计算的颜色，则使用它们。否则，如果希望在Python中从球谐函数中预计算颜色，请执行此操作。如果没有，则颜色将通过光栅化器进行从球谐函数到RGB的转换。
    shs = None
    colors_precomp = None
    shs = pc.get_features
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)  # 将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）。
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # 计算相机中心到每个点的方向向量，并归一化。
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)                     # 计算相机中心到每个点的方向向量，并归一化。
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)              # 使用SH特征将方向向量转换为RGB颜色。
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)                             # 将RGB颜色的范围限制在0到1之间。
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径。
    # print("means3D:", means3D)   # [[-1.8414, -1.9516, 18.6913], ..., [13.7768,  5.6580, 15.5191]]
    # print("means2D:", means2D)   # [[0., 0., 0.], ..., [0., 0., 0.]]
    # print("opacity:", opacity)   # [[0.1000]]     # 随着训练优化，会发生变化
    # print("shs:", shs)           # [[-0.7159,  0.1460,  1.1191], [ 0.0000,  0.0000,  0.0000], [ 0.0000,  0.0000,  0.0000], ...
    # print("colors_precomp:", colors_precomp)    #  None
    # print("scales:", scales)                    # tensor([[0.1693, 0.1693, 0.1693], ...,  
    # print("rotations:", rotations)              # tensor([[1., 0., 0., 0.]
    # print("cov3D_precomp:", cov3D_precomp)      # None
    rendered_image, radii = rasterizer(
        means3D=means3D,  # 从gaussian_model的create_from_pcd&pc.get_xy函数提取
        means2D=means2D,  # pc.get_xy创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度
        shs=shs,  # pc.get_features
        colors_precomp=colors_precomp,
        opacities=opacity,  # pc.get_opacity
        scales=scales,  # pc.get_scaling
        rotations=rotations,  # pc.get_rotation
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 返回一个字典，包含渲染的图像、屏幕空间坐标、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径。
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


# def render_eval(viewpoint_camera: DifferentiableCamera_eval, pc: GaussianModel, pipe, bg_color: torch.Tensor, xyz_flowmap=None, scaling_modifier=1.0, override_color=None, camera_pose=None):
def render_eval(viewpoint_camera: DifferentiableCamera_eval, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                xyz_flowmap=None, scaling_modifier=1.0, override_color=None, quaternion=None, T=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    """
    viewpoint_camera:       # 是scene.py文件读取的sparse/cameras.bin  images.bin points3D.bin中获取相机参数
        full_proj_transform:
        image_height:1261
        image_name:
        image_width:
        original_image:
        projection_matrix:
        scale: 1.0
        training:True
        trans:array([0., 0., 0.])
        uid:31
        world_view_transform:
        zfar:100.0
        znear:0.01    
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means  #
    """# 创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度。"""
    # pc.get_xyz： 从GaussianModel，gaussian_model.py中create_from_pcd输出的xyz值
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    if xyz_flowmap is None:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    else:
        screenspace_points = torch.zeros_like(xyz_flowmap, dtype=xyz_flowmap.dtype, requires_grad=True,
                                              device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # print("pipe: ", pipe)   # <arguments.GroupParams object at 0x7fe065fc76d0> [19/06 02:56:58]
    # Set up rasterization configuration
    # 计算视场的 tan 值，这将用于设置光栅化配置。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # print("tanfovx: ", tanfovx)      # 0.3833445449846394
    # print("tanfovy: ", tanfovy)      # 0.7085160148877734

    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵、球面谐波、相机中心等。

    w2c = torch.eye(4).cuda()
    projmatrix = (w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]

    ### add0908                                         # 需要把R: [1,0,0,0]转换成[[1,0,0], [0,1,0], [0,0,1]]
    # R = qvec2rotmat_tensor(quaternion)
    # # rel_w2c = get_camera_from_tensor(quaternion, T)    # quar → matrix
    # # R = rel_w2c[:3]
    # # T = rel_w2c[-1:]
    # world_view_transform = torch.tensor(getWorld2View3(R, T, translate=np.array([.0, .0, .0]), scale=1.0)).transpose(0, 1).cuda()
    # # world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
    # projmatrix = viewpoint_camera.projection_matrix
    # full_proj_transform = world_view_transform.unsqueeze(0).bmm(projmatrix.unsqueeze(0)).squeeze(0)
    # camera_center = world_view_transform.inverse()[3, :3]
    ###

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,   # 这三个参数没过来
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,

        ### add0908
        # viewmatrix=world_view_transform,
        # projmatrix=full_proj_transform,

        sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,  # self.world_view_transform.inverse()[3, :3]
        campos=camera_pos,
        # campos=camera_center,
        prefiltered=False,
        debug=False
        # debug=pipe.debug
    )
    # print("raster_settings:", raster_settings)
    # image_height=1261, image_width=711, tanfovx=0.3833445449846394, tanfovy=0.7085160148877734, bg=tensor([0., 0., 0.], device='cuda:0'), scale_modifier=1.0,
    # viewmatrix=tensor([[ 0.7182,  0.3005, -0.6276,  0.0000], ..., [ 3.6417,  1.8329, -2.1219,  1.0000]], device='cuda:0')
    # projmatrix=tensor([[ 1.8736,  0.4241, -0.6276, -0.6276], ..., [ 9.4998,  2.5870, -2.1321, -2.1219]], device='cuda:0')
    # sh_degree=0, campos=tensor([-4.4980, -0.4152, -0.8483], device='cuda:0'), prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上。

    # 获取高斯分布的三维坐标、屏幕空间坐标和透明度。
    if xyz_flowmap is None:
        # means3D = pc.get_xyz

        rel_w2c = get_camera_from_tensor(quaternion, T)  # quaternion: [1,0,0,0]  T:[0,0,0]
        # rel_w2c = get_camera_from_tensor(camera_pose)

        gaussians_xyz = pc._xyz.clone()  # 求高斯函数 xyz 位置
        gaussians_rot = pc._rotation.clone()  # 求高斯函数的旋转 rotation

        xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
        xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)  # xyz 齐次坐标

        # 将高斯模型的均值和旋转从世界坐标系转换到相机坐标系下的坐标值
        gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]  # w2c * 三维坐标的齐次坐标
        # gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
        gaussians_rot_trans = quadmultiply(quaternion, gaussians_rot)
        means3D = gaussians_xyz_trans

    else:
        means3D = xyz_flowmap
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则，它将由光栅化器根据尺度和旋转进行计算。
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:   # false
    #     # print("666666666666")       # 不读取
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)  # 获取预计算的三维协方差矩阵。
    # else:       # 获取缩放和旋转信息。（对应的就是3D高斯的协方差矩阵了）
    #     # print("7777777777777")
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    scales = pc.get_scaling
    # rotations = pc.get_rotation
    rotations = gaussians_rot_trans

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预先计算的颜色，则使用它们。否则，如果希望在Python中从球谐函数中预计算颜色，请执行此操作。如果没有，则颜色将通过光栅化器进行从球谐函数到RGB的转换。
    shs = None
    colors_precomp = None
    shs = pc.get_features
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)  # 将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）。
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # 计算相机中心到每个点的方向向量，并归一化。
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)                     # 计算相机中心到每个点的方向向量，并归一化。
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)              # 使用SH特征将方向向量转换为RGB颜色。
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)                             # 将RGB颜色的范围限制在0到1之间。
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径。
    # print("means3D:", means3D)   # [[-1.8414, -1.9516, 18.6913], ..., [13.7768,  5.6580, 15.5191]]
    # print("means2D:", means2D)   # [[0., 0., 0.], ..., [0., 0., 0.]]
    # print("opacity:", opacity)   # [[0.1000]]     # 随着训练优化，会发生变化
    # print("shs:", shs)           # [[-0.7159,  0.1460,  1.1191], [ 0.0000,  0.0000,  0.0000], [ 0.0000,  0.0000,  0.0000], ...
    # print("colors_precomp:", colors_precomp)    #  None
    # print("scales:", scales)                    # tensor([[0.1693, 0.1693, 0.1693], ...,
    # print("rotations:", rotations)              # tensor([[1., 0., 0., 0.]
    # print("cov3D_precomp:", cov3D_precomp)      # None
    rendered_image, radii = rasterizer(
        # viewmatrix=viewpoint_camera.world_view_transform,  # 这三个参数没过来
        # projmatrix=viewpoint_camera.full_proj_transform,
        # campos=viewpoint_camera.camera_center,  # self.world_view_transform.inverse()[3, :3]
        means3D=means3D,  # 从gaussian_model的create_from_pcd&pc.get_xy函数提取
        means2D=means2D,  # pc.get_xy创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度
        shs=shs,  # pc.get_features
        colors_precomp=colors_precomp,
        opacities=opacity,  # pc.get_opacity
        scales=scales,  # pc.get_scaling
        rotations=rotations,  # pc.get_rotation
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 返回一个字典，包含渲染的图像、屏幕空间坐标、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径。
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}