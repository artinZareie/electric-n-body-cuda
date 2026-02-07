project('nbody-simulation', 'cpp',
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
  'src/main.cpp',
)

cc = meson.get_compiler('cpp')
m_dep = cc.find_library('m', required: true) 

# Tihs project is meant to be compiled using nvc++.
nbody_exe = executable('nbody',
    sources: source_files,
    include_directories: include_dir,
    dependencies: [m_dep],
    cpp_args: [
        '-fast',
        '-Minfo=accel',
    ],
    install: true
)

if get_option('debug')
  add_project_arguments('-g', '-O0', language: 'cpp')
endif
