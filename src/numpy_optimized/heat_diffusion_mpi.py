#!/usr/bin/env python3
"""
2D Heat Diffusion with MPI (NumPy version - SOURCE TERM OPTIMIZED)

Performance optimization applied:
- Precomputed time-independent source term components (major speedup)

This version isolates only the source term optimization to measure its impact
separately from other optimizations.

Demonstrates strong and weak scaling of an MPI parallel finite difference
simulation using NumPy arrays for efficient vectorized operations. This version
uses uppercase MPI communication methods (Sendrecv) for efficient buffer-based
communication.

Example usage:
    # Run on 4 processes with a 512x512 grid for 1000 timesteps
    mpirun -n 4 python heat_diffusion_mpi_optimized_source_term.py --size 512 --steps 1000

    # Strong scaling test: fixed problem size, varying process count
    mpirun -n 1  python heat_diffusion_mpi_optimized_source_term.py --size 1024 --steps 500
    mpirun -n 4  python heat_diffusion_mpi_optimized_source_term.py --size 1024 --steps 500
    mpirun -n 16 python heat_diffusion_mpi_optimized_source_term.py --size 1024 --steps 500

    # Weak scaling test: fixed local size, varying process count
    mpirun -n 1  python heat_diffusion_mpi_optimized_source_term.py --size 256  --steps 500
    mpirun -n 4  python heat_diffusion_mpi_optimized_source_term.py --size 512  --steps 500
    mpirun -n 16 python heat_diffusion_mpi_optimized_source_term.py --size 1024 --steps 500
"""

import numpy as np
from mpi4py import MPI
import argparse
from collections import namedtuple


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

# Thermal diffusivity
ALPHA = 0.01

# Heat source parameters
SOURCE_CENTER_X = 0.5
SOURCE_CENTER_Y = 0.5
SIGMA = 0.15            # Radial decay length scale
OMEGA = 2.0 * np.pi     # One oscillation per unit time


# ---------------------------------------------------------------------------
# Domain data structure (OPTIMIZED: added Q_static)
# ---------------------------------------------------------------------------

Domain = namedtuple('Domain', [
    'cart_comm', 'rank', 'size', 'dims', 'coords',
    'nx_global', 'ny_global', 'local_nx', 'local_ny',
    'dx', 'dy', 'Lx', 'Ly', 'X', 'Y', 'neighbors', 'Q_static'
])


# ---------------------------------------------------------------------------
# Command-line argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="2D Heat Diffusion with MPI (NumPy version - SOURCE TERM OPTIMIZED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Grid size
    parser.add_argument("--size", type=int, default=512,
                        help="Global grid size (square grid, default: 512)")

    # Simulation length
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of timesteps (default: 1000)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Domain setup (OPTIMIZED: precompute source term components)
# ---------------------------------------------------------------------------

def setup_domain(comm, args):
    """
    Create a 2D Cartesian MPI topology and set up the local grid.

    Returns a Domain namedtuple with everything each rank needs to know about
    its piece of the domain.

    OPTIMIZATION: Precomputes time-independent source term components during
    initialization to avoid expensive transcendental function calls every timestep.
    """
    size = comm.Get_size()

    # Let MPI choose a 2D process layout (e.g. 64 ranks -> 8x8)
    dims = MPI.Compute_dims(size, 2)

    # Create the Cartesian communicator (non-periodic boundaries)
    cart_comm = comm.Create_cart(dims, periods=[False, False], reorder=True)
    rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(rank)

    # ---- Determine grid sizes ----
    nx_global = args.size
    ny_global = args.size

    if nx_global % dims[0] != 0 or ny_global % dims[1] != 0:
        if rank == 0:
            print(f"ERROR: Grid {nx_global}x{ny_global} is not evenly "
                  f"divisible by process grid {dims[0]}x{dims[1]}")
            print(f"  {nx_global} / {dims[0]} = {nx_global/dims[0]:.1f}")
            print(f"  {ny_global} / {dims[1]} = {ny_global/dims[1]:.1f}")
        MPI.Finalize()
        exit(1)

    local_nx = nx_global // dims[0]
    local_ny = ny_global // dims[1]

    # Physical domain is [0, 1] x [0, 1]
    Lx, Ly = 1.0, 1.0
    dx = Lx / nx_global
    dy = Ly / ny_global

    # This rank's position in physical space
    x_start = coords[0] * local_nx * dx
    y_start = coords[1] * local_ny * dy

    # Cell-centered coordinate arrays (NumPy 2D arrays)
    # Create 1D coordinate arrays first, then mesh them into 2D
    x_coords = x_start + (np.arange(local_nx) + 0.5) * dx
    y_coords = y_start + (np.arange(local_ny) + 0.5) * dy
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Neighbor ranks for halo exchange
    # Shift returns (source, dest) -- MPI_PROC_NULL at boundaries
    lo_x, hi_x = cart_comm.Shift(0, 1)   # neighbors in x-direction
    lo_y, hi_y = cart_comm.Shift(1, 1)   # neighbors in y-direction

    neighbors = {'lo_x': lo_x, 'hi_x': hi_x, 'lo_y': lo_y, 'hi_y': hi_y}

    # ---- OPTIMIZATION: Precompute time-independent source term components ----
    # These expensive operations (sqrt, atan2, sin, cos, exp) only need to be
    # computed once instead of every timestep.

    # Distance from center (uses sqrt - expensive!)
    r = np.sqrt((X - SOURCE_CENTER_X)**2 + (Y - SOURCE_CENTER_Y)**2)

    # Angle from center (uses atan2 - expensive!)
    theta = np.arctan2(Y - SOURCE_CENTER_Y, X - SOURCE_CENTER_X)

    # Combine all time-independent terms into one precomputed array
    Q_static = (np.sin(np.pi * X / Lx)
                * np.cos(np.pi * Y / Ly)
                * np.exp(-r / SIGMA)
                * (1.0 + 0.1 * np.cos(3.0 * theta)))

    return Domain(
        cart_comm=cart_comm,
        rank=rank,
        size=size,
        dims=dims,
        coords=coords,
        nx_global=nx_global,
        ny_global=ny_global,
        local_nx=local_nx,
        local_ny=local_ny,
        dx=dx,
        dy=dy,
        Lx=Lx,
        Ly=Ly,
        X=X,
        Y=Y,
        neighbors=neighbors,
        Q_static=Q_static
    )


# ---------------------------------------------------------------------------
# Halo exchange (point-to-point communication)
# ---------------------------------------------------------------------------

def exchange_halos(T, domain):
    """
    Exchange ghost-cell data with neighboring MPI ranks.

    The local array T is a (local_nx + 2) x (local_ny + 2) NumPy array:
      - Interior data lives in T[1:nx+1, 1:ny+1]
      - Ghost cells are the outer ring: row 0, row nx+1, col 0, col ny+1

    Neighbor layout in 2D Cartesian topology:

              [lo_y neighbor]
                    |
        [lo_x] -- [rank] -- [hi_x]
                    |
              [hi_y neighbor]

    Uses mpi4py's uppercase Sendrecv, which handles contiguous NumPy buffers
    directly -- much faster than lowercase sendrecv which uses pickle
    serialization for Python objects.
    """
    cart_comm = domain.cart_comm
    nx = domain.local_nx
    ny = domain.local_ny
    nb = domain.neighbors

    # --- X-direction exchange (sending/receiving rows) ---
    # Rows are contiguous in memory with C-order arrays

    # Send our high-x boundary row to our hi_x neighbor;
    # receive our lo_x neighbor's high-x row into our low ghost row.
    cart_comm.Sendrecv(T[nx, 1:ny+1].copy(), dest=nb['hi_x'], sendtag=0,
                       recvbuf=T[0, 1:ny+1], source=nb['lo_x'], recvtag=0)

    # Send our low-x boundary row to our lo_x neighbor;
    # receive our hi_x neighbor's low-x row into our high ghost row.
    cart_comm.Sendrecv(T[1, 1:ny+1].copy(), dest=nb['lo_x'], sendtag=1,
                       recvbuf=T[nx+1, 1:ny+1], source=nb['hi_x'], recvtag=1)

    # --- Y-direction exchange (sending/receiving columns) ---
    # Columns are not contiguous in C-order arrays, so we must copy them
    # into contiguous buffers for efficient MPI communication.

    # Send high-y column to hi_y; receive from lo_y into low ghost column.
    send_col = T[1:nx+1, ny].copy()
    recv_col = np.empty(nx, dtype=T.dtype)
    cart_comm.Sendrecv(send_col, dest=nb['hi_y'], sendtag=2,
                       recvbuf=recv_col, source=nb['lo_y'], recvtag=2)
    T[1:nx+1, 0] = recv_col

    # Send low-y column to lo_y; receive from hi_y into high ghost column.
    send_col = T[1:nx+1, 1].copy()
    recv_col = np.empty(nx, dtype=T.dtype)
    cart_comm.Sendrecv(send_col, dest=nb['lo_y'], sendtag=3,
                       recvbuf=recv_col, source=nb['hi_y'], recvtag=3)
    T[1:nx+1, ny+1] = recv_col


# ---------------------------------------------------------------------------
# Source term computation (OPTIMIZED)
# ---------------------------------------------------------------------------

def compute_source_term(Q_static, t):
    """
    Compute the heat source term Q(x, y, t) using precomputed static components.

    Q(x, y, t) = Q_static * (1 + 0.5 * sin(omega * t))

    where Q_static was precomputed during initialization and contains:
        sin(pi*x/Lx) * cos(pi*y/Ly)     (spatial modes)
      * exp(-r / sigma)                  (radial decay)
      * (1 + 0.1 * cos(3 * theta))       (angular variation)

    OPTIMIZATION: Instead of recomputing expensive operations (sqrt, atan2,
    sin, cos, exp) every timestep, we use precomputed values and only apply
    the time-dependent modulation.

    Original version computed per timestep:
        sqrt    x1  (in compute_distance)
        atan2   x1  (in compute_angle)
        sin     x3
        cos     x2
        exp     x1

    Optimized version computes per timestep:
        sin     x1

    Expected speedup: 5-10x for source term computation alone.
    """
    return Q_static * (1.0 + 0.5 * np.sin(OMEGA * t))


# ---------------------------------------------------------------------------
# Stencil update
# ---------------------------------------------------------------------------

def update_stencil(T, T_new, Q, domain, alpha, dt):
    """
    Apply the 5-point stencil to advance temperature one timestep using NumPy.

    T_new[i,j] = T[i,j] + dt * (alpha * laplacian(T) + Q)

    where laplacian is the standard second-order finite difference:

        laplacian = (T[i+1,j] + T[i-1,j] - 2*T[i,j]) / dx^2
                  + (T[i,j+1] + T[i,j-1] - 2*T[i,j]) / dy^2

    Uses vectorized NumPy array slicing instead of explicit loops.
    """
    nx = domain.local_nx
    ny = domain.local_ny
    dx_sq = domain.dx * domain.dx
    dy_sq = domain.dy * domain.dy

    # Compute laplacian using array slicing (no explicit loops)
    # Interior points: T[1:nx+1, 1:ny+1]
    d2Tdx2 = (T[2:nx+2, 1:ny+1] + T[0:nx, 1:ny+1] - 2.0 * T[1:nx+1, 1:ny+1]) / dx_sq
    d2Tdy2 = (T[1:nx+1, 2:ny+2] + T[1:nx+1, 0:ny] - 2.0 * T[1:nx+1, 1:ny+1]) / dy_sq

    # Forward Euler update (vectorized)
    T_new[1:nx+1, 1:ny+1] = T[1:nx+1, 1:ny+1] + dt * (
        alpha * (d2Tdx2 + d2Tdy2) + Q
    )


# ---------------------------------------------------------------------------
# Timing statistics computation
# ---------------------------------------------------------------------------

def compute_timing_stats(all_times, size):
    """
    Compute min/median/max statistics for timing data across all ranks.

    Args:
        all_times: List of [halo_time, compute_time, elapsed_time] for each rank
        size: Number of MPI ranks

    Returns:
        Tuple of (times_min, times_med, times_max, slowest_rank_breakdown) where:
        - times_min/med/max are lists of [halo, computation, elapsed, other] statistics
        - slowest_rank_breakdown is [halo, computation, other] for the slowest rank
          (used for percentages that add up to 100%)
    """
    # Compute "other" time for each rank (overhead not in halo or computation)
    time_other_all = []
    for r in range(size):
        time_other = all_times[r][2] - (all_times[r][0] + all_times[r][1])
        time_other_all.append(time_other)

    # Find the slowest rank (max elapsed time)
    slowest_rank = max(range(size), key=lambda r: all_times[r][2])
    slowest_rank_breakdown = [
        all_times[slowest_rank][0],  # halo
        all_times[slowest_rank][1],  # computation
        time_other_all[slowest_rank] # other
    ]

    # Convert to NumPy arrays for easier statistics computation
    all_times_array = np.array(all_times)
    time_other_array = np.array(time_other_all)

    # Compute min/median/max for each metric
    # Index 0=halo, 1=computation, 2=elapsed, 3=other
    times_min = [
        np.min(all_times_array[:, 0]),
        np.min(all_times_array[:, 1]),
        np.min(all_times_array[:, 2]),
        np.min(time_other_array)
    ]
    times_max = [
        np.max(all_times_array[:, 0]),
        np.max(all_times_array[:, 1]),
        np.max(all_times_array[:, 2]),
        np.max(time_other_array)
    ]
    times_med = [
        np.median(all_times_array[:, 0]),
        np.median(all_times_array[:, 1]),
        np.median(all_times_array[:, 2]),
        np.median(time_other_array)
    ]

    return times_min, times_med, times_max, slowest_rank_breakdown


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    args = parse_args()

    # ---- Set up the 2D domain decomposition ----
    # OPTIMIZATION: This now precomputes Q_static
    domain = setup_domain(comm, args)
    cart_comm = domain.cart_comm
    rank = domain.rank
    nx = domain.local_nx
    ny = domain.local_ny
    dx = domain.dx
    dy = domain.dy

    # Timestep from CFL stability criterion:
    #   dt <= 1 / (2 * alpha * (1/dx^2 + 1/dy^2))
    # We use 80% of the maximum stable timestep.
    dt_max = 1.0 / (2.0 * ALPHA * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    dt = 0.8 * dt_max

    # ---- Allocate arrays ----
    # Each is a (nx+2) x (ny+2) NumPy array to include ghost cells.
    # Use C-order (row-major) for contiguous row access in halo exchange.
    T     = np.zeros((nx + 2, ny + 2), dtype=np.float64, order='C')
    T_new = np.zeros((nx + 2, ny + 2), dtype=np.float64, order='C')

    # Initial condition: a Gaussian temperature pulse at the center
    X = domain.X
    Y = domain.Y
    T[1:nx+1, 1:ny+1] = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)

    # ---- Print run info ----
    if rank == 0:
        print("=" * 62)
        print("  2D Heat Diffusion with MPI (NumPy - SOURCE TERM OPTIMIZED)")
        print("=" * 62)
        print(f"  Global grid:      {domain.nx_global} x {domain.ny_global}")
        print(f"  Process grid:     {domain.dims[0]} x {domain.dims[1]}"
              f" = {domain.size} ranks")
        print(f"  Local grid:       {nx} x {ny} per rank")
        print(f"  Timesteps:        {args.steps}")
        print("=" * 62)

    # ---- Warm-up: a few iterations to initialize MPI communication ----
    # This helps establish MPI buffers, trigger any JIT compilation, and warm
    # up caches before we start timing. Without this, the first few measured
    # iterations would show artificially high communication costs.
    for _ in range(5):
        exchange_halos(T, domain)
        t_phys = 0.0
        Q = compute_source_term(domain.Q_static, t_phys)  # OPTIMIZED: use Q_static
        update_stencil(T, T_new, Q, domain, ALPHA, dt)
        T, T_new = T_new, T

    # ---- Time-stepping loop ----
    cart_comm.Barrier()
    wall_start = MPI.Wtime()

    # Initialize timing accumulators
    time_halo_exchange = 0.0
    time_computation = 0.0

    for step in range(args.steps):

        # 1) Exchange ghost cells with neighbors (point-to-point)
        t0 = MPI.Wtime()
        exchange_halos(T, domain)
        time_halo_exchange += MPI.Wtime() - t0

        # 2) Compute the source term
        # OPTIMIZED: Much faster - only computes time-dependent modulation
        t0 = MPI.Wtime()
        t_phys = step * dt
        Q = compute_source_term(domain.Q_static, t_phys)

        # 3) Update temperature with the 5-point stencil
        update_stencil(T, T_new, Q, domain, ALPHA, dt)
        time_computation += MPI.Wtime() - t0

        # 4) Swap arrays for next step
        T, T_new = T_new, T

    # ---- Final timing report ----
    cart_comm.Barrier()
    wall_end = MPI.Wtime()
    elapsed = wall_end - wall_start

    # Gather all timing data to rank 0 for statistics
    local_times = [time_halo_exchange, time_computation, elapsed]
    all_times = cart_comm.gather(local_times, root=0)

    if rank == 0:
        # Compute timing statistics across all ranks
        times_min, times_med, times_max, slowest = compute_timing_stats(all_times, domain.size)

        med_elapsed = times_med[2]
        max_elapsed = times_max[2]

        pts_per_sec = (domain.nx_global * domain.ny_global * args.steps / med_elapsed)

        print("=" * 62)
        print("  Timing breakdown (min / median / max across ranks):")
        print(f"    Halo exchange:  {times_min[0]:6.3f} / {times_med[0]:6.3f} / {times_max[0]:6.3f} s  ({slowest[0]/max_elapsed*100:5.1f}%)")
        print(f"    Computation:    {times_min[1]:6.3f} / {times_med[1]:6.3f} / {times_max[1]:6.3f} s  ({slowest[1]/max_elapsed*100:5.1f}%)")
        print(f"    Other:          {times_min[3]:6.3f} / {times_med[3]:6.3f} / {times_max[3]:6.3f} s  ({slowest[2]/max_elapsed*100:5.1f}%)")
        print(f"    Total:          {max_elapsed:6.3f} s")
        print("=" * 62)


if __name__ == "__main__":
    main()
