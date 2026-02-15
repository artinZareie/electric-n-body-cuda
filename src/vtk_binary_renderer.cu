#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vtk_binary_renderer.cuh>
#include <vtk_binary_writer.cuh>

void VtkBinaryRenderer::initialize(const std::string &output_path)
{
    m_output_path = output_path;
    mkdir(output_path.c_str(), 0755);
    m_initialized = true;
}

void VtkBinaryRenderer::render(const Particles &particles, size_t frame_index)
{
    if (!m_initialized)
        return;

    std::ostringstream filename;
    filename << m_output_path << "/output_" << std::setfill('0') << std::setw(6) << frame_index << ".vtk";

    write_vtk_frame_binary(particles, filename.str());
}

void VtkBinaryRenderer::shutdown()
{
    m_initialized = false;
}

std::unique_ptr<IRenderer> VtkBinaryRendererFactory::create()
{
    return std::make_unique<VtkBinaryRenderer>();
}
