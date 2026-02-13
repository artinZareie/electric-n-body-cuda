.PHONY: all build clean run format

all: build

setup:
	meson setup --reconfigure build

build: setup
	meson compile -C build

clean:
	rm -rf build

run: build
	./build/nbody

format:
	find src include -type f \( -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +
