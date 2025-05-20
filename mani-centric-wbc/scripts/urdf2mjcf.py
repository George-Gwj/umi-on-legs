from urdf2mjcf import run

run(
    urdf_path="/home/george/code/umi-on-legs/mani-centric-wbc/resources/robots/aliengoLerobot/aliengoPiper.urdf",
    mjcf_path="/home/george/code/umi-on-legs/mani-centric-wbc/resources/robots/aliengoLerobot/aliengoPiper.urdf.mjcf",
    copy_meshes=True,
)
