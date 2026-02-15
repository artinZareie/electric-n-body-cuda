#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import struct

vtk_files = sorted([f"vtk/{f}" for f in os.listdir("vtk") if f.endswith(".vtk")])

print(f"Found {len(vtk_files)} VTK files")

def read_binary_vtk(filename):
    with open(filename, "rb") as f:
        lines = []
        while True:
            line = f.readline()
            if not line:
                break
            lines.append(line.decode('ascii', errors='ignore'))
            if "POINTS" in line.decode('ascii', errors='ignore'):
                n_points = int(line.decode('ascii').split()[1])
                break
        
        points = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            for j in range(3):
                data = f.read(4)
                points[i, j] = struct.unpack('>f', data)[0]
        
        while True:
            line = f.readline()
            if not line or b"LOOKUP_TABLE" in line:
                break
        
        charges = np.zeros(n_points, dtype=np.float32)
        for i in range(n_points):
            data = f.read(4)
            charges[i] = struct.unpack('>f', data)[0]
    
    return points, charges

x_min, y_min, z_min = float("inf"), float("inf"), float("inf")
x_max, y_max, z_max = float("-inf"), float("-inf"), float("-inf")
q_min, q_max = float("inf"), float("-inf")

print("Scanning bounds...")
for vtk_file in vtk_files:
    points, charges = read_binary_vtk(vtk_file)
    x_min = min(x_min, points[:, 0].min())
    x_max = max(x_max, points[:, 0].max())
    y_min = min(y_min, points[:, 1].min())
    y_max = max(y_max, points[:, 1].max())
    z_min = min(z_min, points[:, 2].min())
    z_max = max(z_max, points[:, 2].max())
    q_min = min(q_min, charges.min())
    q_max = max(q_max, charges.max())

margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
x_min -= margin
x_max += margin
y_min -= margin
y_max += margin
z_min -= margin
z_max += margin

print(
    f"Bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]"
)
print(f"Charge range: [{q_min:.4f}, {q_max:.4f}]")

os.makedirs("frames", exist_ok=True)

for i, vtk_file in enumerate(vtk_files):
    points, charges = read_binary_vtk(vtk_file)
    
    max_abs_q = max(abs(q_min), abs(q_max)) if max(abs(q_min), abs(q_max)) > 0 else 1.0
    MIN_INTENSITY = 0.15

    data_file = f"frames/data_{i:06d}.txt"
    with open(data_file, "w") as f:
        for j in range(len(points)):
            q = charges[j]
            intensity = MIN_INTENSITY + (1.0 - MIN_INTENSITY) * abs(q) / max_abs_q
            if q > 0:
                r, g, b = int(intensity * 255), 0, 0
            elif q < 0:
                r, g, b = 0, 0, int(intensity * 255)
            else:
                r, g, b = int(MIN_INTENSITY * 255), int(MIN_INTENSITY * 255), int(MIN_INTENSITY * 255)
            color_int = (r << 16) | (g << 8) | b
            f.write(f"{points[j, 0]} {points[j, 1]} {points[j, 2]} {color_int}\n")
    
    output_png = f"frames/frame_{i:06d}.png"
    
    gnuplot_cmd = f"""
set terminal pngcairo size 1920,1080 background rgb "black"
set output '{output_png}'
set view 60,30,1,1
set xlabel "X" textcolor rgb "white"
set ylabel "Y" textcolor rgb "white"
set zlabel "Z" textcolor rgb "white"
set border lc rgb "white"
set ticslevel 0
set xtics textcolor rgb "white"
set ytics textcolor rgb "white"
set ztics textcolor rgb "white"
set xrange [{x_min}:{x_max}]
set yrange [{y_min}:{y_max}]
set zrange [{z_min}:{z_max}]
set title "Frame {i}" textcolor rgb "white"
set grid lc rgb "#333333"
unset colorbox
splot '{data_file}' using 1:2:3:4 with points pt 7 ps 2.5 lc rgb variable notitle
"""
    
    with open("temp_plot.gnu", "w") as f:
        f.write(gnuplot_cmd)
    
    result = subprocess.run(["gnuplot", "temp_plot.gnu"], capture_output=True)
    
    if result.returncode != 0:
        print(f"Error rendering frame {i}: {result.stderr.decode()}")
        continue
    
    if i % 10 == 0:
        print(f"Rendered frame {i}/{len(vtk_files)}")

os.remove("temp_plot.gnu")

print(f"\nAll {len(vtk_files)} frames rendered")
print("\nCreating video...")

result = subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        "frames/frame_%06d.png",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "simulation.mp4",
    ],
    capture_output=True,
)

if result.returncode == 0:
    print("Video created: simulation.mp4")
else:
    print(f"Error creating video: {result.stderr.decode()}")
