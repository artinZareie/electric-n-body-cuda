# =============================================
# N-Body Simulation Makefile with nvc++ support
# =============================================

# Configuration
# ------------
BUILD_DIR ?= build
CXX ?= nvc++
MESON ?= meson
NINJA ?= ninja
PROJECT_NAME = nbody-simulation
EXECUTABLE = nbody

# GPU Architecture (adjust for your hardware)
# cc70: Volta, cc75: Turing, cc80: Ampere, cc90: Ada Lovelace
GPU_ARCH ?= cc80

# Optimization level
# 0: debug, 2: moderate, 3: aggressive
OPT_LEVEL ?= 2

# C++ Standard
CPP_STD ?= c++17

# Default target
.PHONY: all
all: release

# Color output for better readability
ifneq ($(findstring $(shell tput colors), 256 88 16 8),)
    COLOR_GREEN = \033[0;32m
    COLOR_YELLOW = \033[0;33m
    COLOR_CYAN = \033[0;36m
    COLOR_RESET = \033[0m
else
    COLOR_GREEN =
    COLOR_YELLOW =
    COLOR_CYAN =
    COLOR_RESET =
endif

# =============================================
# BUILD TARGETS
# =============================================

# Release build (optimized)
.PHONY: release
release:
	@echo "$(COLOR_GREEN)Building release version with nvc++...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)
	@CXX="$(CXX)" CPPFLAGS="" \
	$(MESON) setup $(BUILD_DIR) \
		--buildtype=release \
		--optimization=$(OPT_LEVEL) \
		-Dcpp_std=$(CPP_STD) \
		-Dwarning_level=1 \
		-Db_lto=true \
		-Db_ndebug=true
	@cd $(BUILD_DIR) && $(NINJA)
	@echo "$(COLOR_GREEN)Build complete! Executable: $(BUILD_DIR)/$(EXECUTABLE)$(COLOR_RESET)"

# Debug build (with symbols, no optimization)
.PHONY: debug
debug:
	@echo "$(COLOR_YELLOW)Building debug version with nvc++...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)-debug
	@CXX="$(CXX)" CPPFLAGS="" \
	$(MESON) setup $(BUILD_DIR)-debug \
		--buildtype=debug \
		--optimization=0 \
		-Dcpp_std=$(CPP_STD) \
		-Dwarning_level=3 \
		-Db_sanitize=address \
		-Ddebug=true
	@cd $(BUILD_DIR)-debug && $(NINJA)
	@echo "$(COLOR_YELLOW)Debug build complete! Executable: $(BUILD_DIR)-debug/$(EXECUTABLE)$(COLOR_RESET)"

# GPU-accelerated build with specific architecture
.PHONY: gpu
gpu:
	@echo "$(COLOR_CYAN)Building GPU-accelerated version for $(GPU_ARCH)...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)-gpu
	@CXX="$(CXX)" CPPFLAGS="" \
	$(MESON) setup $(BUILD_DIR)-gpu \
		--buildtype=release \
		--optimization=3 \
		-Dcpp_std=$(CPP_STD) \
		-Dwarning_level=1 \
		-Dcpp_args="-gpu=$(GPU_ARCH) -fast -Minfo=all"
	@cd $(BUILD_DIR)-gpu && $(NINJA)
	@echo "$(COLOR_CYAN)GPU build complete! Executable: $(BUILD_DIR)-gpu/$(EXECUTABLE)$(COLOR_RESET)"

# Profile-guided optimization build (two-phase)
.PHONY: pgo
pgo:
	@echo "$(COLOR_GREEN)Building with Profile-Guided Optimization...$(COLOR_RESET)"
	@echo "Phase 1: Instrumented build"
	@mkdir -p $(BUILD_DIR)-pgo
	@CXX="$(CXX)" CPPFLAGS="" \
	$(MESON) setup $(BUILD_DIR)-pgo \
		--buildtype=release \
		--optimization=2 \
		-Dcpp_std=$(CPP_STD) \
		-Dcpp_args="-Mpgi -Mpfi"
	@cd $(BUILD_DIR)-pgo && $(NINJA)
	@echo "Phase 2: Run with training data (create profile)"
	@cd $(BUILD_DIR)-pgo && ./$(EXECUTABLE) --training || true
	@echo "Phase 3: Build with profile feedback"
	@cd $(BUILD_DIR)-pgo && $(NINJA) clean && $(NINJA)
	@echo "$(COLOR_GREEN)PGO build complete!$(COLOR_RESET)"

# =============================================
# DEVELOPMENT TARGETS
# =============================================

# Configure only (no build)
.PHONY: configure
configure:
	@echo "$(COLOR_YELLOW)Configuring project with nvc++...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)
	@CXX="$(CXX)" $(MESON) setup $(BUILD_DIR) \
		--buildtype=release \
		--optimization=$(OPT_LEVEL) \
		-Dcpp_std=$(CPP_STD)

# Build only (assumes already configured)
.PHONY: build
build:
	@echo "$(COLOR_YELLOW)Building project...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && $(NINJA)

# Run the simulation
.PHONY: run
run: release
	@echo "$(COLOR_GREEN)Running simulation...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && ./$(EXECUTABLE)

# Run debug version
.PHONY: rundebug
rundebug: debug
	@echo "$(COLOR_YELLOW)Running debug version...$(COLOR_RESET)"
	@cd $(BUILD_DIR)-debug && ./$(EXECUTABLE)

# Run GPU version
.PHONY: rungpu
rungpu: gpu
	@echo "$(COLOR_CYAN)Running GPU-accelerated version...$(COLOR_RESET)"
	@cd $(BUILD_DIR)-gpu && ./$(EXECUTABLE)

# =============================================
# UTILITY TARGETS
# =============================================

# Clean build artifacts
.PHONY: clean
clean:
	@echo "$(COLOR_YELLOW)Cleaning build directories...$(COLOR_RESET)"
	@rm -rf $(BUILD_DIR) $(BUILD_DIR)-debug $(BUILD_DIR)-gpu $(BUILD_DIR)-pgo
	@find . -name "*.o" -delete
	@find . -name "*.prof" -delete
	@find . -name "*.profraw" -delete
	@echo "Clean complete!"

# Reconfigure from scratch
.PHONY: reconfigure
reconfigure: clean configure

# Run tests (if you have them)
.PHONY: test
test: debug
	@echo "$(COLOR_YELLOW)Running tests...$(COLOR_RESET)"
	@cd $(BUILD_DIR)-debug && $(MESON) test -v

# Generate compile_commands.json for editors
.PHONY: compile-commands
compile-commands:
	@CXX="$(CXX)" $(MESON) setup $(BUILD_DIR) \
		--buildtype=debug \
		-Dcpp_std=$(CPP_STD)
	@ln -sf $(BUILD_DIR)/compile_commands.json .

# Install to system (requires permissions)
.PHONY: install
install: release
	@echo "$(COLOR_GREEN)Installing to system...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && sudo $(NINJA) install

# Uninstall from system
.PHONY: uninstall
uninstall:
	@echo "$(COLOR_YELLOW)Uninstalling...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && sudo $(NINJA) uninstall || true

# =============================================
# ANALYSIS TARGETS
# =============================================

# Show build configuration
.PHONY: config
config:
	@if [ -d "$(BUILD_DIR)" ]; then \
		echo "$(COLOR_CYAN)Current configuration:$(COLOR_RESET)"; \
		cd $(BUILD_DIR) && $(MESON) configure; \
	else \
		echo "Project not configured. Run 'make configure' first."; \
	fi

# Generate optimization report
.PHONY: opt-report
opt-report:
	@mkdir -p $(BUILD_DIR)-report
	@CXX="$(CXX)" CPPFLAGS="" \
	$(MESON) setup $(BUILD_DIR)-report \
		--buildtype=release \
		--optimization=2 \
		-Dcpp_std=$(CPP_STD) \
		-Dcpp_args="-Minfo=all -Mneginfo -Minfo=inline,loop,vect"
	@cd $(BUILD_DIR)-report && $(NINJA) 2>&1 | grep -A5 -B5 "optimization"

# =============================================
# QUICK COMMANDS FOR N-BODY SIMULATION
# =============================================

# Quick build with common nvc++ flags for HPC
.PHONY: fast
fast:
	@echo "$(COLOR_GREEN)Quick optimized build for N-Body simulation...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)-fast
	@CXX="$(CXX)" CPPFLAGS="" \
	$(MESON) setup $(BUILD_DIR)-fast \
		--buildtype=release \
		--optimization=3 \
		-Dcpp_std=$(CPP_STD) \
		-Dcpp_args="-fast -Mvect=simd:256 -Mconcur -Munroll -Mipa=fast"
	@cd $(BUILD_DIR)-fast && $(NINJA)
	@echo "$(COLOR_GREEN)Fast build complete!$(COLOR_RESET)"

# Build with OpenMP for parallel CPU execution
.PHONY: openmp
openmp:
	@echo "$(COLOR_CYAN)Building with OpenMP support...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)-openmp
	@CXX="$(CXX)" CPPFLAGS="" \
	$(MESON) setup $(BUILD_DIR)-openmp \
		--buildtype=release \
		--optimization=2 \
		-Dcpp_std=$(CPP_STD) \
		-Dcpp_args="-mp -Minfo=mp"
	@cd $(BUILD_DIR)-openmp && $(NINJA)

# =============================================
# HELP
# =============================================

.PHONY: help
help:
	@echo "$(COLOR_CYAN)N-Body Simulation Build System$(COLOR_RESET)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Main targets:"
	@echo "  all/release   - Build optimized version (default)"
	@echo "  debug         - Build with debug symbols"
	@echo "  gpu           - Build for GPU acceleration (arch: $(GPU_ARCH))"
	@echo "  fast          - Quick build with HPC optimizations"
	@echo "  openmp        - Build with OpenMP parallelism"
	@echo ""
	@echo "Development:"
	@echo "  run           - Build release and execute"
	@echo "  rundebug      - Build debug and execute"
	@echo "  rungpu        - Build GPU version and execute"
	@echo "  test          - Run tests"
	@echo ""
	@echo "Utility:"
	@echo "  clean         - Remove all build artifacts"
	@echo "  reconfigure   - Clean and reconfigure"
	@echo "  config        - Show current configuration"
	@echo "  install       - Install to system"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make gpu GPU_ARCH=cc90        # Build for Ada Lovelace GPU"
	@echo "  make fast OPT_LEVEL=3         # Maximum optimization"
	@echo "  make run                      # Build and run immediately"
	@echo ""
	@echo "Compiler: $(CXX)"
	@echo "C++ Standard: $(CPP_STD)"
	@echo "Default GPU Arch: $(GPU_ARCH)"

# =============================================
# AUTO-DEPENDENCIES FOR SOURCE FILES
# =============================================

# Generate list of source files automatically
SRC_FILES := $(wildcard src/*.cpp)
HEADER_FILES := $(wildcard include/*.hpp)

# Print source file info
.PHONY: info
info:
	@echo "$(COLOR_CYAN)Project Info:$(COLOR_RESET)"
	@echo "Source files: $(words $(SRC_FILES))"
	@echo "  $(SRC_FILES)"
	@echo "Header files: $(words $(HEADER_FILES))"
	@echo "  $(HEADER_FILES)"
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Executable: $(EXECUTABLE)"

# Default target when just typing 'make'
.DEFAULT_GOAL := help