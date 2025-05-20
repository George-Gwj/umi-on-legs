import urdfpy
import numpy as np
from scipy.spatial.transform import Rotation as R

def adjust_inertial_to_principal(urdf_path, output_path):
    urdf = urdfpy.URDF.load(urdf_path)
    for link in urdf.links:
        if link.inertial:
            I = link.inertial.inertia.to_matrix()
            # 特征值分解
            eig_vals, eig_vecs = np.linalg.eigh(I)
            # 构建旋转矩阵
            rotation = eig_vecs.T  # 转置后为从原坐标系到主轴的旋转
            # 调整inertial的origin
            original_origin = link.inertial.origin
            new_rotation = R.from_matrix(rotation).as_euler('xyz')
            # 更新origin的旋转
            adjusted_origin = urdfpy.Origin(
                xyz=original_origin.xyz,
                rpy=new_rotation
            )
            link.inertial.origin = adjusted_origin
            # 更新惯性矩阵为对角
            link.inertial.inertia = urdfpy.Inertia(
                eig_vals[0], eig_vals[1], eig_vals[2],
                0, 0, 0
            )
            # 调整子元素的origin（示例仅处理视觉）
            for visual in link.visuals:
                # 应用逆旋转
                visual_rot = R.from_euler('xyz', visual.origin.rpy)
                adjusted_visual_rot = visual_rot * R.from_matrix(rotation.T)
                visual.origin.rpy = adjusted_visual_rot.as_euler('xyz')
    urdf.save(output_path)

if __name__=='__main__':
    urdfpath = "mani-centric-wbc/resources/robots/aliengoLerobot/aliengoPiper_copy.urdf"
    outputpath = "mani-centric-wbc/resources/robots/aliengoLerobot/aliengoPiper_adjusted.urdf"
    adjust_inertial_to_principal(urdfpath, outputpath)