# Makefile for heat_diffusion_mpi Fortran code
#
# Usage:
#   make              # Build with default compiler (mpif90)
#   make clean        # Remove built files
#   make FC=mpifort   # Build with specific compiler

# Compiler and flags
FC = mpif90
FFLAGS = -O2 -march=core-avx2 
# For debugging, use instead:
# FFLAGS = -O0 -g -Wall -Wextra -fcheck=all -fbacktrace

# Target executable
TARGET = heat_diffusion_mpi

# Source files
SOURCES = heat_diffusion_mpi.f90

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES)
	$(FC) $(FFLAGS) -o $(TARGET) $(SOURCES)

# Clean up
clean:
	rm -f $(TARGET) *.o *.mod

# Phony targets
.PHONY: all clean

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Build the executable (default)"
	@echo "  make clean    - Remove built files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make"
	@echo "  mpirun -n 4 ./$(TARGET) --size 512 --steps 1000"
