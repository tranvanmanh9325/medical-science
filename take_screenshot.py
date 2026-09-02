import os
import sys
import time

def capture_screen(output_path=None):
    if output_path is None:
        output_path = r"C:\Users\Kirito\.gemini\antigravity\brain\5d36091f-5f83-4ef1-804a-212e8ac791da\screen_view.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab(all_screens=True)
        img.save(output_path, "PNG")
        print(f"[SUCCESS] Screenshot captured via PIL: {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path
    except Exception as e:
        # Fallback to PowerShell .NET Graphics screenshot
        print(f"[INFO] Falling back to .NET screen capture: {e}")
        import subprocess
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
        print(f"[SUCCESS] Screenshot captured via .NET: {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    capture_screen(out)
