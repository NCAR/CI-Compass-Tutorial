!===============================================================================
! 2D Heat Diffusion with MPI (Fortran version)
!
! Demonstrates strong and weak scaling of an MPI parallel finite difference
! simulation using Fortran. The code intentionally includes unoptimized
! mathematical operations to show performance tuning opportunities.
!
! Example usage:
!   # Run on 4 processes with a 512x512 grid for 1000 timesteps
!   mpirun -n 4 ./heat_diffusion_mpi --size 512 --steps 1000
!
!   # Strong scaling test: fixed problem size, varying process count
!   mpirun -n 1  ./heat_diffusion_mpi --size 1024 --steps 500
!   mpirun -n 4  ./heat_diffusion_mpi --size 1024 --steps 500
!   mpirun -n 16 ./heat_diffusion_mpi --size 1024 --steps 500
!
!   # Weak scaling test: fixed local size, varying process count
!   mpirun -n 1  ./heat_diffusion_mpi --size 256  --steps 500
!   mpirun -n 4  ./heat_diffusion_mpi --size 512  --steps 500
!   mpirun -n 16 ./heat_diffusion_mpi --size 1024 --steps 500
!===============================================================================

program heat_diffusion_mpi
  use mpi
  implicit none

  ! Physical parameters
  real(8), parameter :: ALPHA = 0.01d0           ! Thermal diffusivity
  real(8), parameter :: SOURCE_CENTER_X = 0.5d0
  real(8), parameter :: SOURCE_CENTER_Y = 0.5d0
  real(8), parameter :: SIGMA = 0.15d0           ! Radial decay length
  real(8), parameter :: OMEGA = 2.0d0 * 3.141592653589793d0  ! Angular frequency
  real(8), parameter :: PI = 3.141592653589793d0

  ! MPI variables
  integer :: comm, cart_comm, rank, nprocs, ierr
  integer :: dims(2), coords(2), periods(2)
  logical :: reorder
  integer :: lo_x, hi_x, lo_y, hi_y

  ! Grid parameters
  integer :: nx_global, ny_global
  integer :: local_nx, local_ny
  real(8) :: Lx, Ly, dx, dy
  real(8) :: x_start, y_start

  ! Arrays (allocatable)
  real(8), allocatable :: T(:,:), T_new(:,:), Q(:,:)
  real(8), allocatable :: X(:,:), Y(:,:)

  ! Time stepping
  real(8) :: dt, dt_max, t_phys
  integer :: nsteps, step

  ! Timing
  real(8) :: wall_start, wall_end, elapsed
  real(8) :: time_halo_exchange, time_computation
  real(8) :: t0
  real(8), allocatable :: all_times(:,:)
  real(8) :: local_times(3)

  ! Other variables
  integer :: i, j, n
  integer :: grid_size, num_steps

  ! Initialize MPI
  call MPI_Init(ierr)
  comm = MPI_COMM_WORLD
  call MPI_Comm_size(comm, nprocs, ierr)

  ! Parse command-line arguments
  call parse_args(grid_size, num_steps)
  nx_global = grid_size
  ny_global = grid_size
  nsteps = num_steps

  ! Create 2D Cartesian topology
  dims = 0
  call MPI_Dims_create(nprocs, 2, dims, ierr)
  periods = (/ .false., .false. /)
  reorder = .true.
  call MPI_Cart_create(comm, 2, dims, periods, reorder, cart_comm, ierr)
  call MPI_Comm_rank(cart_comm, rank, ierr)
  call MPI_Cart_coords(cart_comm, rank, 2, coords, ierr)

  ! Check if grid is evenly divisible
  if (mod(nx_global, dims(1)) /= 0 .or. mod(ny_global, dims(2)) /= 0) then
    if (rank == 0) then
      print *, 'ERROR: Grid ', nx_global, 'x', ny_global, ' is not evenly divisible'
      print *, '       by process grid ', dims(1), 'x', dims(2)
    end if
    call MPI_Finalize(ierr)
    stop
  end if

  ! Local grid dimensions
  local_nx = nx_global / dims(1)
  local_ny = ny_global / dims(2)

  ! Physical domain and grid spacing
  Lx = 1.0d0
  Ly = 1.0d0
  dx = Lx / dble(nx_global)
  dy = Ly / dble(ny_global)

  ! This rank's position in physical space
  x_start = dble(coords(1)) * dble(local_nx) * dx
  y_start = dble(coords(2)) * dble(local_ny) * dy

  ! Find neighbor ranks for halo exchange
  call MPI_Cart_shift(cart_comm, 0, 1, lo_x, hi_x, ierr)
  call MPI_Cart_shift(cart_comm, 1, 1, lo_y, hi_y, ierr)

  ! Allocate arrays with ghost cells
  ! T and T_new have shape (0:local_nx+1, 0:local_ny+1)
  ! Interior is (1:local_nx, 1:local_ny)
  allocate(T(0:local_nx+1, 0:local_ny+1))
  allocate(T_new(0:local_nx+1, 0:local_ny+1))
  allocate(Q(local_nx, local_ny))
  allocate(X(local_nx, local_ny))
  allocate(Y(local_nx, local_ny))

  ! Initialize arrays
  T = 0.0d0
  T_new = 0.0d0

  ! Set up coordinate arrays (cell-centered)
  do j = 1, local_ny
    do i = 1, local_nx
      X(i,j) = x_start + (dble(i) - 0.5d0) * dx
      Y(i,j) = y_start + (dble(j) - 0.5d0) * dy
    end do
  end do

  ! Initial condition: Gaussian temperature pulse at center
  do j = 1, local_ny
    do i = 1, local_nx
      T(i,j) = exp(-((X(i,j) - 0.5d0)**2 + (Y(i,j) - 0.5d0)**2) / 0.02d0)
    end do
  end do

  ! Compute timestep from CFL stability criterion
  dt_max = 1.0d0 / (2.0d0 * ALPHA * (1.0d0/(dx*dx) + 1.0d0/(dy*dy)))
  dt = 0.8d0 * dt_max

  ! Print run info
  if (rank == 0) then
    print *, '=============================================================='
    print *, '  2D Heat Diffusion with MPI (Fortran version)'
    print *, '=============================================================='
    print '(A,I6,A,I6)', '  Global grid:      ', nx_global, ' x ', ny_global
    print '(A,I4,A,I4,A,I6,A)', '  Process grid:     ', dims(1), ' x ', dims(2), &
                                ' = ', nprocs, ' ranks'
    print '(A,I6,A,I6,A)', '  Local grid:       ', local_nx, ' x ', local_ny, ' per rank'
    print '(A,I6)', '  Timesteps:        ', nsteps
    print *, '=============================================================='
  end if

  ! Warm-up: a few iterations to initialize MPI communication
  ! This helps establish MPI buffers and warm up caches before timing.
  do n = 1, 5
    call exchange_halos(T, local_nx, local_ny, cart_comm, lo_x, hi_x, lo_y, hi_y)
    t_phys = 0.0d0
    call compute_source_term(Q, X, Y, Lx, Ly, t_phys, local_nx, local_ny)
    call update_stencil(T, T_new, Q, local_nx, local_ny, dx, dy, ALPHA, dt)
    ! Swap arrays
    call swap_arrays(T, T_new)
  end do

  ! Time-stepping loop
  call MPI_Barrier(cart_comm, ierr)
  wall_start = MPI_Wtime()

  time_halo_exchange = 0.0d0
  time_computation = 0.0d0

  do step = 1, nsteps
    ! 1) Exchange ghost cells
    t0 = MPI_Wtime()
    call exchange_halos(T, local_nx, local_ny, cart_comm, lo_x, hi_x, lo_y, hi_y)
    time_halo_exchange = time_halo_exchange + (MPI_Wtime() - t0)

    ! 2) Compute source term and update stencil
    t0 = MPI_Wtime()
    t_phys = dble(step) * dt
    call compute_source_term(Q, X, Y, Lx, Ly, t_phys, local_nx, local_ny)
    call update_stencil(T, T_new, Q, local_nx, local_ny, dx, dy, ALPHA, dt)
    time_computation = time_computation + (MPI_Wtime() - t0)

    ! 3) Swap arrays
    call swap_arrays(T, T_new)
  end do

  call MPI_Barrier(cart_comm, ierr)
  wall_end = MPI_Wtime()
  elapsed = wall_end - wall_start

  ! Gather timing data to rank 0
  local_times(1) = time_halo_exchange
  local_times(2) = time_computation
  local_times(3) = elapsed

  if (rank == 0) then
    allocate(all_times(3, nprocs))
  else
    allocate(all_times(1, 1))  ! Dummy allocation
  end if

  call MPI_Gather(local_times, 3, MPI_DOUBLE_PRECISION, &
                  all_times, 3, MPI_DOUBLE_PRECISION, 0, cart_comm, ierr)

  ! Print timing statistics
  if (rank == 0) then
    call print_timing_stats(all_times, nprocs, nx_global, ny_global, nsteps)
  end if

  ! Clean up
  deallocate(T, T_new, Q, X, Y, all_times)
  call MPI_Finalize(ierr)

contains


!===============================================================================
! Subroutine: parse_args
! Parse command-line arguments for grid size and number of steps
!===============================================================================
subroutine parse_args(grid_size, num_steps)
  implicit none
  integer, intent(out) :: grid_size, num_steps
  integer :: i, nargs
  character(len=100) :: arg

  ! Default values
  grid_size = 512
  num_steps = 1000

  nargs = command_argument_count()
  i = 1
  do while (i <= nargs)
    call get_command_argument(i, arg)
    if (trim(arg) == '--size') then
      i = i + 1
      call get_command_argument(i, arg)
      read(arg, *) grid_size
    else if (trim(arg) == '--steps') then
      i = i + 1
      call get_command_argument(i, arg)
      read(arg, *) num_steps
    end if
    i = i + 1
  end do

end subroutine parse_args


!===============================================================================
! Subroutine: exchange_halos
! Exchange ghost-cell data with neighboring MPI ranks
!
! Neighbor layout in 2D Cartesian topology:
!
!           [lo_y neighbor]
!                 |
!     [lo_x] -- [rank] -- [hi_x]
!                 |
!           [hi_y neighbor]
!===============================================================================
subroutine exchange_halos(T, nx, ny, cart_comm, lo_x, hi_x, lo_y, hi_y)
  use mpi
  implicit none
  integer, intent(in) :: nx, ny, cart_comm
  integer, intent(in) :: lo_x, hi_x, lo_y, hi_y
  real(8), intent(inout) :: T(0:nx+1, 0:ny+1)

  integer :: ierr
  real(8) :: send_buf(ny), recv_buf(ny)
  real(8) :: send_col(nx), recv_col(nx)

  ! X-direction exchange (rows in Fortran are not contiguous, so we copy)
  ! Actually in Fortran column-major, columns are contiguous, but our first
  ! index iterates over x, so we need to be careful

  ! Send high-x boundary to hi_x; receive from lo_x into low ghost
  send_buf(1:ny) = T(nx, 1:ny)
  call MPI_Sendrecv(send_buf, ny, MPI_DOUBLE_PRECISION, hi_x, 0, &
                    recv_buf, ny, MPI_DOUBLE_PRECISION, lo_x, 0, &
                    cart_comm, MPI_STATUS_IGNORE, ierr)
  if (lo_x /= MPI_PROC_NULL) T(0, 1:ny) = recv_buf(1:ny)

  ! Send low-x boundary to lo_x; receive from hi_x into high ghost
  send_buf(1:ny) = T(1, 1:ny)
  call MPI_Sendrecv(send_buf, ny, MPI_DOUBLE_PRECISION, lo_x, 1, &
                    recv_buf, ny, MPI_DOUBLE_PRECISION, hi_x, 1, &
                    cart_comm, MPI_STATUS_IGNORE, ierr)
  if (hi_x /= MPI_PROC_NULL) T(nx+1, 1:ny) = recv_buf(1:ny)

  ! Y-direction exchange (columns)
  ! Send high-y column to hi_y; receive from lo_y into low ghost
  send_col(1:nx) = T(1:nx, ny)
  call MPI_Sendrecv(send_col, nx, MPI_DOUBLE_PRECISION, hi_y, 2, &
                    recv_col, nx, MPI_DOUBLE_PRECISION, lo_y, 2, &
                    cart_comm, MPI_STATUS_IGNORE, ierr)
  if (lo_y /= MPI_PROC_NULL) T(1:nx, 0) = recv_col(1:nx)

  ! Send low-y column to lo_y; receive from hi_y into high ghost
  send_col(1:nx) = T(1:nx, 1)
  call MPI_Sendrecv(send_col, nx, MPI_DOUBLE_PRECISION, lo_y, 3, &
                    recv_col, nx, MPI_DOUBLE_PRECISION, hi_y, 3, &
                    cart_comm, MPI_STATUS_IGNORE, ierr)
  if (hi_y /= MPI_PROC_NULL) T(1:nx, ny+1) = recv_col(1:nx)

end subroutine exchange_halos




!===============================================================================
! Subroutine: compute_source_term
! Compute the heat source term Q(x,y,t)
!
! *** THIS FUNCTION IS DELIBERATELY UNOPTIMIZED ***
!
! It recomputes every expensive math operation from scratch on every call.
! In a real application you would look for ways to avoid this redundant work.
!
! Expensive operations used (per grid point, per call):
!   sqrt    x1  (in compute_distance)
!   atan2   x1  (in compute_angle)
!   sin     x3
!   cos     x2
!   exp     x1
!
! Optimization opportunities:
!   - Precompute spatial terms that don't depend on time
!   - Precompute angles (theta) once during initialization
!   - Use lookup tables for trig functions
!   - Avoid sqrt when only r² is needed
!===============================================================================
subroutine compute_source_term(Q, X, Y, Lx, Ly, t, nx, ny)
  implicit none
  integer, intent(in) :: nx, ny
  real(8), intent(in) :: X(nx,ny), Y(nx,ny), Lx, Ly, t
  real(8), intent(out) :: Q(nx,ny)

  real(8), parameter :: SOURCE_CENTER_X = 0.5d0
  real(8), parameter :: SOURCE_CENTER_Y = 0.5d0
  real(8), parameter :: SIGMA = 0.15d0
  real(8), parameter :: OMEGA = 2.0d0 * 3.141592653589793d0
  real(8), parameter :: PI = 3.141592653589793d0

  integer :: i, j
  real(8) :: xi, yj, r, theta
  real(8) :: delta_x, delta_y

  do j = 1, ny
    do i = 1, nx
      xi = X(i,j)
      yj = Y(i,j)

      ! Distance from center - uses sqrt
      delta_x = xi - SOURCE_CENTER_X
      delta_y = yj - SOURCE_CENTER_Y
      r = sqrt(delta_x**2 + delta_y**2)

      ! Angle from center - uses atan2
      theta = atan2(delta_y, delta_x)

      ! Assemble source - uses sin, cos, exp
      Q(i,j) = sin(PI * xi / Lx) &
             * cos(PI * yj / Ly) &
             * exp(-r / SIGMA) &
             * (1.0d0 + 0.5d0 * sin(OMEGA * t)) &
             * (1.0d0 + 0.1d0 * cos(3.0d0 * theta))
    end do
  end do

end subroutine compute_source_term


!===============================================================================
! Subroutine: update_stencil
! Apply 5-point stencil to advance temperature one timestep
!
! T_new(i,j) = T(i,j) + dt * (alpha * laplacian(T) + Q)
!
! where laplacian is the standard second-order finite difference
!===============================================================================
subroutine update_stencil(T, T_new, Q, nx, ny, dx, dy, alpha, dt)
  implicit none
  integer, intent(in) :: nx, ny
  real(8), intent(in) :: T(0:nx+1, 0:ny+1), Q(nx,ny)
  real(8), intent(in) :: dx, dy, alpha, dt
  real(8), intent(out) :: T_new(0:nx+1, 0:ny+1)

  integer :: i, j
  real(8) :: dx_sq, dy_sq, d2Tdx2, d2Tdy2

  dx_sq = dx * dx
  dy_sq = dy * dy

  do j = 1, ny
    do i = 1, nx
      ! Second derivatives from finite differences
      d2Tdx2 = (T(i+1,j) + T(i-1,j) - 2.0d0 * T(i,j)) / dx_sq
      d2Tdy2 = (T(i,j+1) + T(i,j-1) - 2.0d0 * T(i,j)) / dy_sq

      ! Forward Euler update
      T_new(i,j) = T(i,j) + dt * (alpha * (d2Tdx2 + d2Tdy2) + Q(i,j))
    end do
  end do

end subroutine update_stencil


!===============================================================================
! Subroutine: swap_arrays
! Swap two arrays by pointer reassignment (Fortran 90+ feature)
!===============================================================================
subroutine swap_arrays(A, B)
  implicit none
  real(8), intent(inout) :: A(:,:), B(:,:)
  real(8), allocatable :: temp(:,:)

  ! Fortran doesn't have pointer swap like C, so we do a simple swap
  allocate(temp(lbound(A,1):ubound(A,1), lbound(A,2):ubound(A,2)))
  temp = A
  A = B
  B = temp
  deallocate(temp)

end subroutine swap_arrays


!===============================================================================
! Subroutine: sort_array
! Simple bubble sort for small arrays
!===============================================================================
subroutine sort_array(arr, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: arr(n)
  integer :: i, j
  real(8) :: temp

  do i = 1, n-1
    do j = i+1, n
      if (arr(j) < arr(i)) then
        temp = arr(i)
        arr(i) = arr(j)
        arr(j) = temp
      end if
    end do
  end do

end subroutine sort_array


!===============================================================================
! Function: compute_median
! Compute median from an already-sorted array
!===============================================================================
function compute_median(arr, n) result(med)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8) :: med

  if (mod(n, 2) == 0) then
    med = (arr(n/2) + arr(n/2 + 1)) / 2.0d0
  else
    med = arr(n/2 + 1)
  end if

end function compute_median


!===============================================================================
! Subroutine: print_timing_stats
! Compute and print timing statistics across all ranks
!===============================================================================
subroutine print_timing_stats(all_times, nprocs, nx_global, ny_global, nsteps)
  implicit none
  integer, intent(in) :: nprocs, nx_global, ny_global, nsteps
  real(8), intent(in) :: all_times(3, nprocs)

  real(8) :: time_other(nprocs)
  real(8) :: times_min(4), times_max(4), times_med(4)
  real(8) :: slowest(3), max_elapsed
  real(8) :: pts_per_sec
  integer :: r, slowest_rank
  real(8), allocatable :: sorted(:)

  ! Compute "other" time for each rank
  do r = 1, nprocs
    time_other(r) = all_times(3,r) - (all_times(1,r) + all_times(2,r))
  end do

  ! Find slowest rank
  slowest_rank = 1
  do r = 2, nprocs
    if (all_times(3,r) > all_times(3,slowest_rank)) then
      slowest_rank = r
    end if
  end do

  slowest(1) = all_times(1, slowest_rank)
  slowest(2) = all_times(2, slowest_rank)
  slowest(3) = time_other(slowest_rank)

  ! Compute min/max statistics
  times_min(1) = minval(all_times(1,:))
  times_min(2) = minval(all_times(2,:))
  times_min(3) = minval(all_times(3,:))
  times_min(4) = minval(time_other)

  times_max(1) = maxval(all_times(1,:))
  times_max(2) = maxval(all_times(2,:))
  times_max(3) = maxval(all_times(3,:))
  times_max(4) = maxval(time_other)

  ! Compute median statistics
  allocate(sorted(nprocs))

  sorted = all_times(1,:)
  call sort_array(sorted, nprocs)
  times_med(1) = compute_median(sorted, nprocs)

  sorted = all_times(2,:)
  call sort_array(sorted, nprocs)
  times_med(2) = compute_median(sorted, nprocs)

  sorted = all_times(3,:)
  call sort_array(sorted, nprocs)
  times_med(3) = compute_median(sorted, nprocs)

  sorted = time_other
  call sort_array(sorted, nprocs)
  times_med(4) = compute_median(sorted, nprocs)

  deallocate(sorted)

  max_elapsed = times_max(3)
  pts_per_sec = dble(nx_global) * dble(ny_global) * dble(nsteps) / times_med(3)

  ! Print results
  print *, '=============================================================='
  print *, '  Timing breakdown (min / median / max across ranks):'
  write(*, '(A,F7.3,A,F7.3,A,F7.3,A,F6.1,A)') &
    '    Halo exchange:  ', times_min(1), ' / ', times_med(1), ' / ', times_max(1), &
    ' s  (', slowest(1)/max_elapsed*100.0d0, '%)'
  write(*, '(A,F7.3,A,F7.3,A,F7.3,A,F6.1,A)') &
    '    Computation:    ', times_min(2), ' / ', times_med(2), ' / ', times_max(2), &
    ' s  (', slowest(2)/max_elapsed*100.0d0, '%)'
  write(*, '(A,F7.3,A,F7.3,A,F7.3,A,F6.1,A)') &
    '    Other:          ', times_min(4), ' / ', times_med(4), ' / ', times_max(4), &
    ' s  (', slowest(3)/max_elapsed*100.0d0, '%)'
  write(*, '(A,F7.3,A)') '    Total:          ', max_elapsed, ' s'
  print *, '=============================================================='

end subroutine print_timing_stats

end program heat_diffusion_mpi
