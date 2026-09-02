import os
import mujoco
from PIL import Image

def capture_sim_frame(output_path=None):
    if output_path is None:
        output_path = r"C:\Users\Kirito\.gemini\antigravity\brain\5d36091f-5f83-4ef1-804a-212e8ac791da\dual_psm_render.png"

    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "davinci_dvrk", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Left Arm PSM1 Reaching down to suture pad:
    data.ctrl[0] = 0.05     # Yaw
    data.ctrl[1] = 0.16     # Pitch
    data.ctrl[2] = 0.18     # Insertion
    data.ctrl[3] = 0.20     # Roll
    data.ctrl[4] = 0.28     # Wrist pitch
    data.ctrl[5] = 0.0      # Wrist yaw
    data.ctrl[6] = 0.35     # Jaw 1
    data.ctrl[7] = 0.35     # Jaw 2

    # Right Arm PSM2 Reaching down to peg board:
    data.ctrl[8] = -0.05    # Yaw
    data.ctrl[9] = 0.16     # Pitch
    data.ctrl[10] = 0.18    # Insertion
    data.ctrl[11] = -0.20   # Roll
    data.ctrl[12] = 0.28    # Wrist pitch
    data.ctrl[13] = 0.0     # Wrist yaw
    data.ctrl[14] = 0.35    # Jaw 1
    data.ctrl[15] = 0.35    # Jaw 2

    for _ in range(80):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=900, width=1200)
    
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, -0.25, 0.55]
    camera.distance = 2.1
    camera.elevation = -24.0
    camera.azimuth = 140.0

    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"[SUCCESS] Snapshot saved: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path

if __name__ == "__main__":
    capture_sim_frame()
