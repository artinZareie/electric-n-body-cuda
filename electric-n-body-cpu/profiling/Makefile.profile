# Makefile.profile — Build the profiling binary
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fopenmp -Wall -Wextra -Iinclude
LDFLAGS  := -fopenmp -lm

PROFILE_SRC := profiling/profile_main.cpp src/particles.cpp src/vtk_writer.cpp
PROFILE_OBJ := build/profile_main.o build/prof_particles.o build/prof_vtk_writer.o
TARGET      := nbody_profile

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(PROFILE_OBJ)
	$(CXX) $(PROFILE_OBJ) -o $@ $(LDFLAGS)

build/profile_main.o: profiling/profile_main.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/prof_particles.o: src/particles.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/prof_vtk_writer.o: src/vtk_writer.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build:
	mkdir -p build

clean:
	rm -f $(PROFILE_OBJ) $(TARGET)
