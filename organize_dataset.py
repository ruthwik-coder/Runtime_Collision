"""
organize_dataset.py
-------------------
Converts n2.avi ... n6.avi to .mp4 using FFmpeg
and organizes s1.mp4 ... s8.mp4 and n1.mp4 ... n6.mp4 into:
- Test_Dataset/s_collisions/
- Test_Dataset/n_normal/
"""

import os
import shutil
import subprocess

BASE_DIR = r"d:\c_files4\Runtime"
S_DIR = os.path.join(BASE_DIR, "Test_Dataset", "s_collisions")
N_DIR = os.path.join(BASE_DIR, "Test_Dataset", "n_normal")

os.makedirs(S_DIR, exist_ok=True)
os.makedirs(N_DIR, exist_ok=True)

print("[*] Organizing s1.mp4 ... s8.mp4 (Accidents)...")
for i in range(1, 9):
    src = os.path.join(BASE_DIR, f"s{i}.mp4")
    dst = os.path.join(S_DIR, f"s{i}.mp4")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  [+] Copied {src} -> {dst}")

print("\n[*] Organizing n1.mp4 (Normal Traffic)...")
n1_src = os.path.join(BASE_DIR, "n1.mp4")
n1_dst = os.path.join(N_DIR, "n1.mp4")
if os.path.exists(n1_src):
    shutil.copy2(n1_src, n1_dst)
    print(f"  [+] Copied {n1_src} -> {n1_dst}")

print("\n[*] Converting n2.avi ... n6.avi to .mp4 via FFmpeg...")
for i in range(2, 7):
    avi_src = os.path.join(BASE_DIR, f"n{i}.avi")
    mp4_dst = os.path.join(N_DIR, f"n{i}.mp4")
    if os.path.exists(avi_src):
        cmd = [
            "ffmpeg", "-y", "-i", avi_src,
            "-vcodec", "libx264", "-acodec", "aac",
            "-movflags", "+faststart", mp4_dst
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(mp4_dst):
            print(f"  [+] Converted {avi_src} -> {mp4_dst}")
        else:
            print(f"  [!] Failed to convert {avi_src}")

print("\n[+] Dataset Organization Complete!")
print(f"    Collision Videos (s1..s8) in : {S_DIR} ({len(os.listdir(S_DIR))} files)")
print(f"    Normal Videos (n1..n6) in    : {N_DIR} ({len(os.listdir(N_DIR))} files)")
