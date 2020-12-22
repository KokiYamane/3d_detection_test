import copy
import numpy as np
from numpy.lib.utils import source
import open3d as o3d
import matplotlib.pyplot as plt


class ModelBaseMatching():
    def __init__(self) -> None:
        super().__init__()

    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size {:.3}.".format(voxel_size))
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(
            ":: Estimate normal with search radius {:.3}.".format(radius_normal))
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(
            ":: Compute FPFH feature with search radius {:.3}.".format(radius_feature))
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print(
            "   Since the downsampling voxel size is {:.3},".format(voxel_size))
        print("   we use a liberal distance threshold {:.3}.".format(
            distance_threshold))
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
        return result

    def refine_registration(self, source, target, voxel_size, result_ransac):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold {:.3}.".format(distance_threshold))
        result = o3d.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.registration.TransformationEstimationPointToPlane())
        return result

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        # target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def calc(self, source, target, voxel_size=0.01):
        # Global Registration by RANSAC
        source_down, source_fpfh = self.preprocess_point_cloud(
            source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(
            target, voxel_size)
        result_ransac = self.execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size)

        # refine by ICP
        return self.refine_registration(
            source_down, target_down, voxel_size, result_ransac)


if __name__ == '__main__':
    mbm = ModelBaseMatching()

    # load data
    mesh_arrow = o3d.io.read_triangle_mesh('arrow.obj')
    source = mesh_arrow.sample_points_uniformly(
        number_of_points=5000)

    # target = copy.deepcopy(source).translate((2, 0, 0))
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # target.transform(trans_init)

    color_raw = o3d.io.read_image(
        'data/20201219_224429/color/color000000000200.png')
    depth_raw = o3d.io.read_image(
        'data/20201219_224429/depth/depth000000000200.png')
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    print(rgbd_image)

    target = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    o3d.visualization.draw_geometries([target])

    mbm.draw_registration_result(source, target, np.eye(4))

    # model base matching
    result = mbm.calc(source, target)
    mbm.draw_registration_result(source, target, result.transformation)
    print(result.transformation)
