#include "vtk_binary_writer.cuh"
#include <cstring>
#include <fstream>
#include <stdexcept>

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

    file.close();
}
