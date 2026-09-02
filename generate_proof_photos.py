import os
import mujoco
from PIL import Image

def generate_all_proof_photos():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "davinci_dvrk", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    pic_dir = os.path.join(work_dir, "pic")
    os.makedirs(pic_dir, exist_ok=True)

    # Arm poses: reaching down into surgical tasks
    data.ctrl[0] = 0.08     # Left Yaw
    data.ctrl[1] = 0.20     # Left Pitch
    data.ctrl[2] = 0.22     # Left Insertion (deep over suture pad)
    data.ctrl[3] = 0.50     # Left Roll
    data.ctrl[4] = 0.35     # Left Wrist Pitch
    data.ctrl[5] = 0.10     # Left Wrist Yaw
    data.ctrl[6] = 0.30     # Left Jaw 1
    data.ctrl[7] = 0.30     # Left Jaw 2

    data.ctrl[8] = -0.08    # Right Yaw
    data.ctrl[9] = 0.20     # Right Pitch
    data.ctrl[10] = 0.22    # Right Insertion (deep over pegboard)
    data.ctrl[11] = -0.50   # Right Roll
    data.ctrl[12] = 0.35    # Right Wrist Pitch
    data.ctrl[13] = -0.10   # Right Wrist Yaw
    data.ctrl[14] = 0.30    # Right Jaw 1
    data.ctrl[15] = 0.30    # Right Jaw 2

    for _ in range(80):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=1080, width=1920)

    shots = [
        ("proof_overview.png", [0.0, -0.20, 0.52], 2.15, -24.0, 142.0),
        ("proof_left_arm_suture.png", [-0.05, -0.12, 0.44], 0.75, -28.0, 155.0),
        ("proof_right_arm_pegtransfer.png", [0.05, 0.06, 0.44], 0.75, -28.0, 125.0),
        ("proof_topdown_surgical_field.png", [0.0, -0.05, 0.42], 1.20, -75.0, 90.0)
    ]

    for filename, lookat, distance, elevation, azimuth in shots:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        out_file = os.path.join(pic_dir, filename)
        Image.fromarray(pixels).save(out_file)
        print(f"[PROOF CREATED] {out_file}")

if __name__ == "__main__":
    generate_all_proof_photos()
