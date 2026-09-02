import os
import mujoco
from PIL import Image

def render_apollo_hand():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    for _ in range(50):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)

    # Super close-up on the left hand showing all 5 fingers
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.03, -0.22, 0.68]
    cam.distance = 0.45
    cam.elevation = -15.0
    cam.azimuth = 75.0
    renderer.update_scene(data, camera=cam)
    
    p = os.path.join(work_dir, "pic", "apptronik_apollo_5fingers_closeup.png")
    Image.fromarray(renderer.render()).save(p)
    print(f"[APOLLO 5-FINGER HAND SAVED] {p}")

if __name__ == "__main__":
    render_apollo_hand()
