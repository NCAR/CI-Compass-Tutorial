#!/usr/bin/env python3
"""
2D Heat Diffusion with JAX

JAX-accelerated version of the heat diffusion simulation, demonstrating:
- GPU/TPU acceleration via JAX
- Just-in-time compilation with @jit
- Automatic vectorization
- Optional multi-device parallelism with pmap

This version showcases the performance benefits of JAX compared to the pure
Python implementation, while maintaining similar readability.

Example usage:
    # Run with default settings (512x512 grid, 1000 steps)
    python heat_diffusion_jax.py

    # Larger problem
    python heat_diffusion_jax.py --size 2048 --steps 5000

    # CPU-only mode (disable GPU)
    JAX_PLATFORMS=cpu python heat_diffusion_jax.py --size 1024 --steps 1000

    # Check available devices
    python heat_diffusion_jax.py --list-devices
"""

import os
import logging

# Configure JAX before import to prevent CUDA initialization attempts
# This must be set before importing jax to take effect
if os.environ.get('JAX_PLATFORMS') == 'cpu':
    os.environ['JAX_PLUGINS'] = ''
    # Suppress JAX CUDA plugin errors when running CPU-only
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)

import jax
import jax.numpy as jnp
from jax import jit, vmap
import argparse
import time
from functools import partial


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

# Thermal diffusivity
ALPHA = 0.01

# Heat source parameters
SOURCE_CENTER_X = 0.5
SOURCE_CENTER_Y = 0.5
SIGMA = 0.15            # Radial decay length scale
OMEGA = 2.0 * jnp.pi    # One oscillation per unit time


# ---------------------------------------------------------------------------
# Command-line argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="2D Heat Diffusion with JAX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Grid size
    parser.add_argument("--size", type=int, default=512,
                        help="Grid size (square grid, default: 512)")

    # Simulation length
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of timesteps (default: 1000)")

    # Device listing
    parser.add_argument("--list-devices", action="store_true",
                        help="List available JAX devices and exit")

    # Optimization level
    parser.add_argument("--optimized", action="store_true",
                        help="Use optimized version (precomputed terms, no trig)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Domain setup
# ---------------------------------------------------------------------------

def setup_domain(args):
    """
    Set up the computational domain and coordinate arrays.

    Returns a dictionary with grid information and coordinate meshes.
    """
    nx = args.size
    ny = args.size

    # Physical domain is [0, 1] x [0, 1]
    Lx, Ly = 1.0, 1.0
    dx = Lx / nx
    dy = Ly / ny

    # Create cell-centered coordinate arrays using JAX meshgrid
    x = jnp.linspace(dx/2, Lx - dx/2, nx)
    y = jnp.linspace(dy/2, Ly - dy/2, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    return {
        'nx': nx,
        'ny': ny,
        'dx': dx,
        'dy': dy,
        'Lx': Lx,
        'Ly': Ly,
        'X': X,
        'Y': Y,
    }


# ---------------------------------------------------------------------------
# Source term computation (unoptimized version)
# ---------------------------------------------------------------------------

@jit
def compute_distance(x, y, cx, cy):
    """
    Compute Euclidean distance from point (x, y) to center (cx, cy).

    Uses sqrt -- expensive operation that could be optimized.
    JAX will automatically vectorize this over arrays.
    """
    delta_x = x - cx
    delta_y = y - cy
    return jnp.sqrt(delta_x * delta_x + delta_y * delta_y)


@jit
def compute_angle(x, y, cx, cy):
    """
    Compute angle (in radians) from center (cx, cy) to point (x, y).

    Uses atan2 -- expensive operation that could be avoided.
    """
    delta_y = y - cy
    delta_x = x - cx
    return jnp.arctan2(delta_y, delta_x)


@jit
def compute_source_term_unopt(X, Y, Lx, Ly, t):
    """
    Compute the heat source term Q(x, y, t) - UNOPTIMIZED VERSION.

    Q(x, y, t) = sin²(pi*x/Lx) * sin²(pi*y/Ly)     (spatial modes, always ≥ 0)
               * exp(-r / sigma)                    (radial decay)
               * (1 + 0.5 * sin(omega * t))         (time oscillation)
               * (1 + 0.1 * cos(3 * theta))         (angular variation)

    where:
        r     = distance from center
        theta = angle from center

    *** THIS VERSION IS DELIBERATELY UNOPTIMIZED ***

    It recomputes expensive operations on every call for demonstration.

    Expensive operations used (per grid point, per call):
        sqrt    x1  (in compute_distance)
        atan2   x1  (in compute_angle)
        sin     x3
        cos     x2
        exp     x1
    """
    pi = jnp.pi

    # Distance from center -- uses sqrt
    r = compute_distance(X, Y, SOURCE_CENTER_X, SOURCE_CENTER_Y)

    # Angle from center -- uses atan2
    theta = compute_angle(X, Y, SOURCE_CENTER_X, SOURCE_CENTER_Y)

    # Assemble source -- uses sin, cos, exp
    # Using sin² and sin² to ensure Q >= 0 (prevents negative temperatures)
    spatial_x = jnp.sin(pi * X / Lx)
    spatial_y = jnp.sin(pi * Y / Ly)

    Q = (spatial_x * spatial_x
         * spatial_y * spatial_y
         * jnp.exp(-r / SIGMA)
         * (1.0 + 0.5 * jnp.sin(OMEGA * t))
         * (1.0 + 0.1 * jnp.cos(3.0 * theta)))

    return Q


# ---------------------------------------------------------------------------
# Source term computation (optimized version)
# ---------------------------------------------------------------------------

def precompute_source_terms(X, Y, Lx, Ly):
    """
    Precompute time-independent parts of the source term.

    This demonstrates a key optimization: separating spatial terms (which
    don't change during the simulation) from temporal terms.

    Returns:
        spatial_static: The spatial pattern that gets modulated by time
        theta: Precomputed angles (avoids atan2 in timestep loop)
    """
    pi = jnp.pi

    # Distance from center - using sqrt
    dx = X - SOURCE_CENTER_X
    dy = Y - SOURCE_CENTER_Y
    r = jnp.sqrt(dx * dx + dy * dy)

    # Angle from center - uses atan2
    theta = jnp.arctan2(dy, dx)

    # Time-independent spatial pattern (using sin² to ensure non-negative)
    spatial_x = jnp.sin(pi * X / Lx)
    spatial_y = jnp.sin(pi * Y / Ly)
    spatial_static = (spatial_x * spatial_x
                     * spatial_y * spatial_y
                     * jnp.exp(-r / SIGMA))

    return spatial_static, theta


@jit
def compute_source_term_opt(spatial_static, theta, t):
    """
    Compute the heat source term Q(x, y, t) - OPTIMIZED VERSION.

    Uses precomputed spatial terms and angles to avoid sqrt, atan2, and
    some trig functions in the timestep loop.

    Expensive operations reduced to:
        sin     x1  (temporal oscillation only)
        cos     x1  (angular variation, precomputed angle)

    This is much faster than the unoptimized version!
    """
    Q = (spatial_static
         * (1.0 + 0.5 * jnp.sin(OMEGA * t))
         * (1.0 + 0.1 * jnp.cos(3.0 * theta)))

    return Q


# ---------------------------------------------------------------------------
# Stencil update
# ---------------------------------------------------------------------------

@jit
def update_stencil(T, Q, alpha, dx, dy, dt):
    """
    Apply the 5-point stencil to advance temperature one timestep.

    T_new[i,j] = T[i,j] + dt * (alpha * laplacian(T) + Q)

    where laplacian is the standard second-order finite difference:

        laplacian = (T[i+1,j] + T[i-1,j] - 2*T[i,j]) / dx^2
                  + (T[i,j+1] + T[i,j-1] - 2*T[i,j]) / dy^2

    Uses array slicing to avoid explicit loops -- JAX compiles this
    efficiently and can run it on GPU.

    Note: This implementation uses Dirichlet boundary conditions (T=0 at edges)
    by default through array slicing.
    """
    dx_sq = dx * dx
    dy_sq = dy * dy

    # Extract interior points and neighbors using slicing
    T_center = T[1:-1, 1:-1]
    T_left   = T[:-2,  1:-1]
    T_right  = T[2:,   1:-1]
    T_down   = T[1:-1, :-2]
    T_up     = T[1:-1, 2:]

    # Compute Laplacian
    d2Tdx2 = (T_right + T_left - 2.0 * T_center) / dx_sq
    d2Tdy2 = (T_up + T_down - 2.0 * T_center) / dy_sq
    laplacian = d2Tdx2 + d2Tdy2

    # Forward Euler update for interior points
    T_interior_new = T_center + dt * (alpha * laplacian + Q)

    # Create new temperature array with updated interior
    # Boundaries remain zero (Dirichlet boundary conditions)
    T_new = jnp.zeros_like(T)
    T_new = T_new.at[1:-1, 1:-1].set(T_interior_new)

    return T_new


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # List devices if requested
    if args.list_devices:
        print("=" * 62)
        print("  JAX Device Information")
        print("=" * 62)
        print(f"  JAX version:      {jax.__version__}")
        print(f"  Available devices: {jax.devices()}")
        print(f"  Default backend:  {jax.default_backend()}")
        print("=" * 62)
        return

    # ---- Set up the domain ----
    domain = setup_domain(args)
    nx = domain['nx']
    ny = domain['ny']
    dx = domain['dx']
    dy = domain['dy']
    X = domain['X']
    Y = domain['Y']
    Lx = domain['Lx']
    Ly = domain['Ly']

    # Timestep from CFL stability criterion:
    #   dt <= 1 / (2 * alpha * (1/dx^2 + 1/dy^2))
    # We use 80% of the maximum stable timestep.
    dt_max = 1.0 / (2.0 * ALPHA * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    dt = 0.8 * dt_max

    # ---- Print run info ----
    print("=" * 62)
    print("  2D Heat Diffusion with JAX")
    print("=" * 62)
    print(f"  Grid size:        {nx} x {ny}")
    print(f"  Timesteps:        {args.steps}")
    print(f"  JAX backend:      {jax.default_backend()}")
    print(f"  JAX devices:      {len(jax.devices())} x {jax.devices()[0].device_kind}")
    print(f"  Optimization:     {'ON' if args.optimized else 'OFF'}")
    print("=" * 62)

    # ---- Initialize temperature field ----
    # Create array with ghost cells: (nx+2) x (ny+2)
    # Initial condition: Gaussian pulse at center
    T = jnp.zeros((nx + 2, ny + 2))
    gaussian = jnp.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)
    T = T.at[1:-1, 1:-1].set(gaussian)

    # ---- Choose optimized or unoptimized version ----
    if args.optimized:
        # Precompute time-independent source terms
        spatial_static, theta = precompute_source_terms(X, Y, Lx, Ly)
        compute_source = lambda t: compute_source_term_opt(spatial_static, theta, t)
        opt_label = "optimized"
    else:
        # Use unoptimized version that recomputes everything
        compute_source = lambda t: compute_source_term_unopt(X, Y, Lx, Ly, t)
        opt_label = "unoptimized"

    # ---- Warm-up: compile functions and warm up caches ----
    # JAX uses JIT compilation, so the first call triggers compilation.
    # We run a few iterations here so they don't affect our timing.
    print(f"Warming up ({opt_label} version)...")
    for step in range(5):
        t_phys = step * dt
        Q = compute_source(t_phys)
        T = update_stencil(T, Q, ALPHA, dx, dy, dt)

    # Block until all computations are done (important for GPU timing)
    T.block_until_ready()
    print("Warm-up complete, starting timed run...")

    # ---- Time-stepping loop ----
    wall_start = time.time()
    time_source = 0.0
    time_stencil = 0.0

    for step in range(args.steps):
        # Compute source term
        t0 = time.time()
        t_phys = step * dt
        Q = compute_source(t_phys)
        Q.block_until_ready()  # Ensure computation is done before timing
        time_source += time.time() - t0

        # Update temperature with 5-point stencil
        t0 = time.time()
        T = update_stencil(T, Q, ALPHA, dx, dy, dt)
        T.block_until_ready()  # Ensure computation is done before timing
        time_stencil += time.time() - t0

    # ---- Final timing report ----
    wall_end = time.time()
    elapsed = wall_end - wall_start

    pts_per_sec = (nx * ny * args.steps) / elapsed

    print("=" * 62)
    print("  Timing breakdown:")
    print(f"    Source term:    {time_source:6.3f} s  ({time_source/elapsed*100:5.1f}%)")
    print(f"    Stencil:        {time_stencil:6.3f} s  ({time_stencil/elapsed*100:5.1f}%)")
    print(f"    Total:          {elapsed:6.3f} s")


if __name__ == "__main__":
    main()
