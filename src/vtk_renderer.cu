#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vtk_renderer.cuh>
#include <vtk_writer.cuh>

void VtkRenderer::initialize(const std::string &output_path)
{
    m_output_path = output_path;
    mkdir(output_path.c_str(), 0755);
    m_initialized = true;
}

void VtkRenderer::render(const Particles &particles, size_t frame_index)
{
    if (!m_initialized)
        return;

    std::ostringstream filename;
    filename << m_output_path << "/output_" << std::setfill('0') << std::setw(6) << frame_index << ".vtk";

    write_vtk_frame(particles, filename.str());
}

void VtkRenderer::shutdown()
{
    m_initialized = false;
}

std::unique_ptr<IRenderer> VtkRendererFactory::create()
{
    return std::make_unique<VtkRenderer>();
}
