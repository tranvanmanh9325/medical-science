import os
import sys
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glfw
import OpenGL.GL as gl
import mujoco

from main import BlenderMuJoCoViewer

def capture_viewer_frame(output_path=None):
    if output_path is None:
        output_path = r"C:\Users\Kirito\.gemini\antigravity\brain\1e6d4b70-36b5-47e6-b83c-d2785737a999\apollo_scientific_suite_preview.png"

    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    
    viewer = BlenderMuJoCoViewer(model_path, width=1600, height=900)
    
    # Step physics for 2.0 seconds (400 steps) so telemetry settles and buffers fill
    print("[TEST] Running 400 physics steps with active standing balance...")
    for _ in range(400):
        viewer._step_physics_with_balance()
        telem = viewer.telemetry.update(viewer.data)
        viewer.oscilloscope.append(
            viewer.data.time,
            telem['pelvis_z'],
            telem['fz_left'],
            telem['fz_right'],
            telem['com_vel'][1]
        )

    # Render single frame with HUD
    w, h = viewer.width, viewer.height
    viewport = mujoco.MjrRect(0, 0, w, h)

    mujoco.mjv_updateScene(viewer.model, viewer.data, viewer.opt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.scn)
    viewer._inject_3d_scientific_overlays(telem)
    mujoco.mjr_render(viewport, viewer.scn, viewer.con)

    # 2D HUD Pass
    gl.glUseProgram(0)
    gl.glBindVertexArray(0)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
    gl.glDisable(gl.GL_LIGHTING)
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDepthMask(gl.GL_FALSE)
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glOrtho(0, viewer.width, viewer.height, 0, -1, 1)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glPushMatrix()
    gl.glLoadIdentity()

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Render All HUD Panels (Unified)
    if viewer.show_hud:
        viewer._draw_top_scientific_ribbon(telem)
        viewer._draw_left_diagnostic_dashboard(telem)
        
        osc_w = min(420, viewer.width - 370)
        viewer.oscilloscope.draw(viewer.width - osc_w - 16, 125, osc_w, 240, viewer.font_renderer)

        viewer._draw_bottom_controls_dock()
        viewer._draw_gizmo_overlay()



    gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthMask(gl.GL_TRUE)

    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # Read back framebuffer BEFORE swap or from front
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    pixels = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    img = Image.frombytes('RGBA', (w, h), pixels).transpose(Image.FLIP_TOP_BOTTOM)

    glfw.swap_buffers(viewer.window)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    print(f"[SUCCESS] High-fidelity scientific capture saved to: {output_path} ({os.path.getsize(output_path)} bytes)")

    # Cleanup
    glfw.destroy_window(viewer.window)
    glfw.terminate()
    return output_path

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    capture_viewer_frame(out)

