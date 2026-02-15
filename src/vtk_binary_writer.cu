#include "vtk_binary_writer.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

static constexpr float MIN_INTENSITY = 0.15f;

static inline float swap_float_to_big_endian(float value)
{
    float result;
    const char *src = reinterpret_cast<const char *>(&value);
    char *dst = reinterpret_cast<char *>(&result);
    dst[0] = src[3];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[0];
    return result;
}

void write_vtk_frame_binary(const Particles &particles, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    auto cpu_view = const_cast<Particles &>(particles).cpu_view();
    size_t n = particles.size();

    float max_abs_q = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        max_abs_q = std::max(max_abs_q, std::fabs(cpu_view.q[i]));
    }
    if (max_abs_q == 0.0f)
        max_abs_q = 1.0f;

    file << "# vtk DataFile Version 3.0\n";
    file << "N-Body Simulation\n";
    file << "BINARY\n";
    file << "DATASET POLYDATA\n";
    file << "POINTS " << n << " float\n";

    for (size_t i = 0; i < n; ++i)
    {
        float coords[3] = {
            swap_float_to_big_endian(cpu_view.x[i]),
            swap_float_to_big_endian(cpu_view.y[i]),
            swap_float_to_big_endian(cpu_view.z[i]),
        };
        file.write(reinterpret_cast<const char *>(coords), sizeof(coords));
    }
    file << "\n";

    file << "POINT_DATA " << n << "\n";
    file << "SCALARS charge float 1\n";
    file << "LOOKUP_TABLE default\n";

    for (size_t i = 0; i < n; ++i)
    {
        float q = swap_float_to_big_endian(cpu_view.q[i]);
        file.write(reinterpret_cast<const char *>(&q), sizeof(q));
    }
    file << "\n";

    // Embed explicit RGB colors: negative=blue, positive=red, brightness=magnitude
    // In VTK BINARY mode, COLOR_SCALARS are written as unsigned char (0-255)
    file << "COLOR_SCALARS rgb 3\n";
    for (size_t i = 0; i < n; ++i)
    {
        float q = cpu_view.q[i];
        float intensity = MIN_INTENSITY + (1.0f - MIN_INTENSITY) * (std::fabs(q) / max_abs_q);
        unsigned char r = 0, g = 0, b = 0;
        if (q > 0.0f)
            r = static_cast<unsigned char>(intensity * 255.0f);
        else if (q < 0.0f)
            b = static_cast<unsigned char>(intensity * 255.0f);
        else
        {
            r = g = b = static_cast<unsigned char>(MIN_INTENSITY * 255.0f);
        }
        unsigned char rgb[3] = {r, g, b};
        file.write(reinterpret_cast<const char *>(rgb), 3);
    }
    file << "\n";

    file.close();
}
