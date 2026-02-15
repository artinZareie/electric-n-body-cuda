project('nbody-simulation', ['cpp', 'cuda'],
  version: '1.0.0',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'werror=false',
    'optimization=3',
    'debug=false',
  ]
)

include_dir = include_directories('include')

source_files = files(
  'src/main.cu',
  'src/particles.cu',
  'src/compute.cu',
  'src/vtk_writer.cu',
  'src/vtk_binary_writer.cu',
  'src/simulation_engine.cu',
  'src/physics_system.cu',
  'src/vtk_renderer.cu',
  'src/vtk_binary_renderer.cu'
)

cc = meson.get_compiler('cpp')
m_dep = cc.find_library('m', required: true) 

gpu_flags = [
  '--generate-code=arch=compute_80,code=sm_80',
  '-Xcompiler=-O3',
  '--use_fast_math',
]

# Tihs project is meant to be compiled using nvcc.
nbody_exe = executable('nbody',
    sources: source_files,
    include_directories: include_dir,
    dependencies: [m_dep],
    cpp_args: gpu_flags,
    link_args: gpu_flags, 
    install: true
  )

  if get_option('debug')
    add_project_arguments('-g', '-O0', language: 'cpp')
  endif
