#!/usr/bin/env python3
"""
2D Heat Diffusion with MPI (pure Python, no NumPy)

Demonstrates strong and weak scaling of an MPI parallel finite difference
simulation. The code intentionally includes unoptimized mathematical operations
to show performance tuning opportunities.

Example usage:
    # Run on 4 processes with a 512x512 grid for 1000 timesteps
    mpirun -n 4 python heat_diffusion_mpi.py --size 512 --steps 1000

    # Strong scaling test: fixed problem size, varying process count
    mpirun -n 1  python heat_diffusion_mpi.py --size 1024 --steps 500
    mpirun -n 4  python heat_diffusion_mpi.py --size 1024 --steps 500
    mpirun -n 16 python heat_diffusion_mpi.py --size 1024 --steps 500

    # Weak scaling test: fixed local size, varying process count
    mpirun -n 1  python heat_diffusion_mpi.py --size 256  --steps 500
    mpirun -n 4  python heat_diffusion_mpi.py --size 512  --steps 500
    mpirun -n 16 python heat_diffusion_mpi.py --size 1024 --steps 500
"""

import math
from mpi4py import MPI
import argparse
from collections import namedtuple
from statistics import median


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

# Thermal diffusivity
ALPHA = 0.01

# Heat source parameters
SOURCE_CENTER_X = 0.5
SOURCE_CENTER_Y = 0.5
SIGMA = 0.15            # Radial decay length scale
OMEGA = 2.0 * math.pi   # One oscillation per unit time


# ---------------------------------------------------------------------------
# Domain data structure
# ---------------------------------------------------------------------------

Domain = namedtuple('Domain', [
    'cart_comm', 'rank', 'size', 'dims', 'coords',
    'nx_global', 'ny_global', 'local_nx', 'local_ny',
    'dx', 'dy', 'Lx', 'Ly', 'X', 'Y', 'neighbors'
])


# ---------------------------------------------------------------------------
# Command-line argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="2D Heat Diffusion with MPI (pure Python, no NumPy)",
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
# Domain setup
# ---------------------------------------------------------------------------

def setup_domain(comm, args):
    """
    Create a 2D Cartesian MPI topology and set up the local grid.

    Returns a dictionary with everything each rank needs to know about
    its piece of the domain.
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

    # Cell-centered coordinate arrays (plain Python 2D lists)
    # X[i][j] = x-coordinate, Y[i][j] = y-coordinate
    X = [[0.0] * local_ny for _ in range(local_nx)]
    Y = [[0.0] * local_ny for _ in range(local_nx)]
    for i in range(local_nx):
        for j in range(local_ny):
            X[i][j] = x_start + (i + 0.5) * dx
            Y[i][j] = y_start + (j + 0.5) * dy

    # Neighbor ranks for halo exchange
    # Shift returns (source, dest) -- MPI_PROC_NULL at boundaries
    lo_x, hi_x = cart_comm.Shift(0, 1)   # neighbors in x-direction
    lo_y, hi_y = cart_comm.Shift(1, 1)   # neighbors in y-direction

    neighbors = {'lo_x': lo_x, 'hi_x': hi_x, 'lo_y': lo_y, 'hi_y': hi_y}

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
        neighbors=neighbors
    )


# ---------------------------------------------------------------------------
# Halo exchange (point-to-point communication)
# ---------------------------------------------------------------------------

def exchange_halos(T, domain):
    """
    Exchange ghost-cell data with neighboring MPI ranks.

    The local array T is a (local_nx + 2) x (local_ny + 2) list-of-lists:
      - Interior data lives in T[1:nx+1][1:ny+1]
      - Ghost cells are the outer ring: row 0, row nx+1, col 0, col ny+1

    Neighbor layout in 2D Cartesian topology:

              [lo_y neighbor]
                    |
        [lo_x] -- [rank] -- [hi_x]
                    |
              [hi_y neighbor]

    Uses mpi4py's lowercase sendrecv, which handles Python lists directly
    via pickle serialization.  (The NumPy version uses uppercase Sendrecv
    with buffer arrays instead -- faster, but requires contiguous arrays.)
    """
    cart_comm = domain.cart_comm
    nx = domain.local_nx
    ny = domain.local_ny
    nb = domain.neighbors

    # --- X-direction exchange (sending/receiving rows) ---

    # Send our high-x boundary row to our hi_x neighbor;
    # receive our lo_x neighbor's high-x row into our low ghost row.
    send_row = T[nx][1:ny+1]    # list slice (makes a copy)
    recv_row = cart_comm.sendrecv(send_row, dest=nb['hi_x'], sendtag=0,
                                  source=nb['lo_x'], recvtag=0)
    if recv_row is not None:
        T[0][1:ny+1] = recv_row

    # Send our low-x boundary row to our lo_x neighbor;
    # receive our hi_x neighbor's low-x row into our high ghost row.
    send_row = T[1][1:ny+1]
    recv_row = cart_comm.sendrecv(send_row, dest=nb['lo_x'], sendtag=1,
                                  source=nb['hi_x'], recvtag=1)
    if recv_row is not None:
        T[nx+1][1:ny+1] = recv_row

    # --- Y-direction exchange (sending/receiving columns) ---
    # Columns are not contiguous in a list-of-lists, so we extract them
    # into a plain list for sending.

    # Send high-y column to hi_y; receive from lo_y into low ghost column.
    send_col = [T[i][ny] for i in range(1, nx+1)]
    recv_col = cart_comm.sendrecv(send_col, dest=nb['hi_y'], sendtag=2,
                                  source=nb['lo_y'], recvtag=2)
    if recv_col is not None:
        for i in range(nx):
            T[i+1][0] = recv_col[i]

    # Send low-y column to lo_y; receive from hi_y into high ghost column.
    send_col = [T[i][1] for i in range(1, nx+1)]
    recv_col = cart_comm.sendrecv(send_col, dest=nb['lo_y'], sendtag=3,
                                  source=nb['hi_y'], recvtag=3)
    if recv_col is not None:
        for i in range(nx):
            T[i+1][ny+1] = recv_col[i]


# ---------------------------------------------------------------------------
# Source term computation (unoptimized)
# ---------------------------------------------------------------------------

def compute_distance(x, y, cx, cy):
    """
    Compute Euclidean distance from point (x, y) to center (cx, cy).

    Uses sqrt -- expensive operation that could be optimized in some cases.
    """
    delta_x = x - cx
    delta_y = y - cy
    return math.sqrt(delta_x * delta_x + delta_y * delta_y)


def compute_angle(x, y, cx, cy):
    """
    Compute angle (in radians) from center (cx, cy) to point (x, y).

    Uses atan2 -- expensive operation that could be precomputed or avoided.
    """
    delta_y = y - cy
    delta_x = x - cx
    return math.atan2(delta_y, delta_x)


def compute_source_term(X, Y, Lx, Ly, t, local_nx, local_ny):
    """
    Compute the heat source term Q(x, y, t) using explicit Python loops.

    Q(x, y, t) = sin(pi*x/Lx) * cos(pi*y/Ly)       (spatial modes)
               * exp(-r / sigma)                      (radial decay)
               * (1 + 0.5 * sin(omega * t))           (time oscillation)
               * (1 + 0.1 * cos(3 * theta))           (angular variation)

    where:
        r     = distance from center
        theta = angle from center

    *** THIS FUNCTION IS DELIBERATELY UNOPTIMIZED ***

    It recomputes every expensive math operation from scratch on every
    call, using explicit Python loops.  In a real application you would
    look for ways to avoid this redundant work -- and that is exactly
    part of the exercise!

    Expensive operations used (per grid point, per call):
        sqrt    x1  (in compute_distance)
        atan2   x1  (in compute_angle)
        sin     x3
        cos     x2
        exp     x1

    Optimization opportunities:
        - Precompute spatial terms that don't depend on time
        - Precompute angles (theta) once during initialization
        - Use lookup tables for trig functions
        - Avoid sqrt when only r² is needed
    """
    pi = math.pi

    Q = [[0.0] * local_ny for _ in range(local_nx)]

    for i in range(local_nx):
        for j in range(local_ny):
            x = X[i][j]
            y = Y[i][j]

            # Distance from center -- uses sqrt
            r = compute_distance(x, y, SOURCE_CENTER_X, SOURCE_CENTER_Y)

            # Angle from center -- uses atan2
            theta = compute_angle(x, y, SOURCE_CENTER_X, SOURCE_CENTER_Y)

            # Assemble source -- uses sin, cos, exp
            Q[i][j] = (math.sin(pi * x / Lx)
                        * math.cos(pi * y / Ly)
                        * math.exp(-r / SIGMA)
                        * (1.0 + 0.5 * math.sin(OMEGA * t))
                        * (1.0 + 0.1 * math.cos(3.0 * theta)))

    return Q


# ---------------------------------------------------------------------------
# Stencil update
# ---------------------------------------------------------------------------

def update_stencil(T, T_new, Q, domain, alpha, dt):
    """
    Apply the 5-point stencil to advance temperature one timestep.

    T_new[i][j] = T[i][j] + dt * (alpha * laplacian(T) + Q)

    where laplacian is the standard second-order finite difference:

        laplacian = (T[i+1][j] + T[i-1][j] - 2*T[i][j]) / dx^2
                  + (T[i][j+1] + T[i][j-1] - 2*T[i][j]) / dy^2
    """
    nx = domain.local_nx
    ny = domain.local_ny
    dx_sq = domain.dx * domain.dx
    dy_sq = domain.dy * domain.dy

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            # Second derivatives from finite differences
            d2Tdx2 = (T[i+1][j] + T[i-1][j] - 2.0 * T[i][j]) / dx_sq
            d2Tdy2 = (T[i][j+1] + T[i][j-1] - 2.0 * T[i][j]) / dy_sq

            # Forward Euler update
            # Q is indexed [i-1][j-1] because it has no ghost cells
            T_new[i][j] = T[i][j] + dt * (
                alpha * (d2Tdx2 + d2Tdy2) + Q[i-1][j-1]
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

    # Compute min/median/max for each metric
    # Index 0=halo, 1=computation, 2=elapsed, 3=other
    times_min = [
        min(all_times[r][0] for r in range(size)),
        min(all_times[r][1] for r in range(size)),
        min(all_times[r][2] for r in range(size)),
        min(time_other_all)
    ]
    times_max = [
        max(all_times[r][0] for r in range(size)),
        max(all_times[r][1] for r in range(size)),
        max(all_times[r][2] for r in range(size)),
        max(time_other_all)
    ]
    times_med = [
        median([all_times[r][0] for r in range(size)]),
        median([all_times[r][1] for r in range(size)]),
        median([all_times[r][2] for r in range(size)]),
        median(time_other_all)
    ]

    return times_min, times_med, times_max, slowest_rank_breakdown


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    args = parse_args()

    # ---- Set up the 2D domain decomposition ----
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
    # Each is a (nx+2) x (ny+2) list-of-lists to include ghost cells.
    T     = [[0.0] * (ny + 2) for _ in range(nx + 2)]
    T_new = [[0.0] * (ny + 2) for _ in range(nx + 2)]

    # Initial condition: a Gaussian temperature pulse at the center
    X = domain.X
    Y = domain.Y
    for i in range(nx):
        for j in range(ny):
            x = X[i][j]
            y = Y[i][j]
            T[i+1][j+1] = math.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.02)

    cell_area = dx * dy   # needed for energy integral

    # ---- Print run info ----
    if rank == 0:
        print("=" * 62)
        print("  2D Heat Diffusion with MPI (pure Python, no NumPy)")
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
        Q = compute_source_term(X, Y, domain.Lx, domain.Ly, t_phys, nx, ny)
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
        t0 = MPI.Wtime()
        t_phys = step * dt
        Q = compute_source_term(X, Y, domain.Lx, domain.Ly, t_phys, nx, ny)

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
