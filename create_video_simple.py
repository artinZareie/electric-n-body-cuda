#!/usr/bin/env python3
import os
import subprocess
import numpy as np

os.makedirs("frames", exist_ok=True)

vtk_files = sorted([f"vtk/{f}" for f in os.listdir("vtk") if f.endswith(".vtk")])

print(f"Found {len(vtk_files)} VTK files")

x_min, y_min, z_min = float("inf"), float("inf"), float("inf")
x_max, y_max, z_max = float("-inf"), float("-inf"), float("-inf")
q_min, q_max = float("inf"), float("-inf")

for vtk_file in vtk_files:
    with open(vtk_file, "r") as f:
        lines = f.readlines()

    in_points = False
    in_charges = False
    n_points = 0

    for j, line in enumerate(lines):
        if line.startswith("POINTS"):
            n_points = int(line.split()[1])
            in_points = True
            continue

        if in_points and n_points > 0:
            coords = list(map(float, line.split()))
            if len(coords) >= 3:
                x_min = min(x_min, coords[0])
                x_max = max(x_max, coords[0])
                y_min = min(y_min, coords[1])
                y_max = max(y_max, coords[1])
                z_min = min(z_min, coords[2])
                z_max = max(z_max, coords[2])
                n_points -= 1
                if n_points == 0:
                    in_points = False

        if line.startswith("LOOKUP_TABLE"):
            in_charges = True
            continue

        if in_charges:
            try:
                charge = float(line.strip())
                q_min = min(q_min, charge)
                q_max = max(q_max, charge)
            except ValueError:
                pass

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

for i, vtk_file in enumerate(vtk_files):
    with open(vtk_file, "r") as f:
        lines = f.readlines()

    points = []
    charges = []

    in_points = False
    in_charges = False
    n_points = 0

    for j, line in enumerate(lines):
        if line.startswith("POINTS"):
            n_points = int(line.split()[1])
            in_points = True
            continue

        if in_points and n_points > 0:
            coords = line.strip()
            if coords:
                points.append(coords)
                n_points -= 1
                if n_points == 0:
                    in_points = False

        if line.startswith("LOOKUP_TABLE"):
            in_charges = True
            continue

        if in_charges and line.strip():
            try:
                charges.append(float(line.strip()))
            except ValueError:
                pass

    if not points or not charges:
        continue

    data_file = f"frames/data_{i:06d}.txt"
    with open(data_file, "w") as f:
        for pt, q in zip(points, charges):
            q_norm = (q - q_min) / (q_max - q_min) if q_max > q_min else 0.5
            f.write(f"{pt} {q_norm}\n")

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
set palette defined (0 "red", 0.5 "yellow", 1 "cyan")
set cbrange [0:1]
unset colorbox
splot '{data_file}' using 1:2:3:4 with points pt 7 ps 2.5 lc palette notitle
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
