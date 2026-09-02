import os
import sys
import json
import time
import subprocess
import mujoco
from PIL import Image

def take_desktop_screenshot(output_path=None):
    if output_path is None:
        output_path = r"C:\Users\Kirito\.gemini\antigravity\brain\5d36091f-5f83-4ef1-804a-212e8ac791da\screen_view.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab(all_screens=True)
        img.save(output_path, "PNG")
        return {"status": "success", "file_path": output_path, "size_bytes": os.path.getsize(output_path)}
    except Exception:
        # Fallback to PowerShell .NET screen capture
        ps_cmd = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$Screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$Bitmap = New-Object System.Drawing.Bitmap $Screen.Width, $Screen.Height
$Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
$Graphics.CopyFromScreen($Screen.Left, $Screen.Top, 0, 0, $Bitmap.Size)
$Bitmap.Save('{output_path}', [System.Drawing.Imaging.ImageFormat]::Png)
$Graphics.Dispose()
$Bitmap.Dispose()
"""
        subprocess.run(["powershell", "-Command", ps_cmd], check=True)
        return {"status": "success", "file_path": output_path, "size_bytes": os.path.getsize(output_path)}

def render_simulation(output_path=None, azimuth=140.0, elevation=-20.0, distance=1.95):
    if output_path is None:
        output_path = r"C:\Users\Kirito\.gemini\antigravity\brain\5d36091f-5f83-4ef1-804a-212e8ac791da\davinci_render.png"

    work_dir = r"d:\GitHub\medical-science"
    model_path = os.path.join(work_dir, "davinci_dvrk", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    data.ctrl[0] = 0.0
    data.ctrl[1] = 0.15
    data.ctrl[2] = 0.12
    data.ctrl[3] = 0.0
    data.ctrl[4] = 0.25
    data.ctrl[5] = 0.0
    data.ctrl[6] = 0.3
    
    for _ in range(100):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=900, width=1200)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.50]
    camera.distance = distance
    camera.elevation = elevation
    camera.azimuth = azimuth

    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return {"status": "success", "file_path": output_path, "size_bytes": os.path.getsize(output_path)}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        res = render_simulation()
        print(json.dumps(res))
    else:
        res = take_desktop_screenshot()
        print(json.dumps(res))
