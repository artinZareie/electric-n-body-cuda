project('nbody-simulation', 'cuda',
  version: '1.0.0',
  default_options: [
    'cuda_std=c++20',
    'warning_level=3',
    'werror=false',
    'optimization=3',
    'debug=false',
  ]
)

include_dir = include_directories('include')

sources = files(
  'src/main.cu',
  'src/particles.cu',
  'src/compute.cu',
  'src/vtk_writer.cu',
  'src/simulation_engine.cu',
  'src/physics_system.cu',
  'src/vtk_renderer.cu',
  'src/root_renderer.cu',
)

cc = meson.get_compiler('cuda')
m_dep = cc.find_library('m', required: true)

root_cflags_raw = run_command('root-config', '--cflags', check: true).stdout().strip().split()
root_glibs_raw = run_command('root-config', '--glibs', check: true).stdout().strip().split()

root_cflags = []
foreach f : root_cflags_raw
  root_cflags += ['-Xcompiler=' + f]
endforeach

root_dep = declare_dependency(
  compile_args: root_cflags,
  link_args: root_glibs_raw,
)

gpu_flags = [
  '--generate-code=arch=compute_80,code=sm_80',
  '-Xcompiler=-O3',
]

executable('nbody',
  sources: sources,
  include_directories: include_dir,
  dependencies: [m_dep, root_dep],
  cuda_args: gpu_flags,
  link_args: gpu_flags,
  install: true,
)
