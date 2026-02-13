.PHONY: all build clean run format

all: build

setup:
	meson setup build --wipe --ini-file native-file.ini

build:
	meson compile -C build

clean:
	rm -rf build

run: build
	./build/nbody

format:
	find src include -type f \( -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +
