module oas_main

  implicit none

contains

  subroutine mult(nx, ny, x, y)

    implicit none

    integer, intent(in) :: nx, ny
    real*8, intent(in) :: x(nx)
    real*8, intent(out) :: y(ny)

    integer :: i, j

    y(:) = 0.

    do j=1,ny
      do i=1,nx
        y(j) = y(j) + x(i)**2
      end do
    end do

  end subroutine mult


  subroutine assemblestructmtx_main(n, tot_n_fem, size, nodes, A, J, Iy, Iz, & ! 6
    K_a, K_t, K_y, K_z, & ! 4
    elem_IDs, cons, & ! 3
    E, G, x_gl, T, & ! 3
    K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, & ! 6
    const2, const_y, const_z, rhs, K, x) ! 7

    use solveRoutines, only: solve
    implicit none

    ! Input
    integer, intent(in) :: n, size, cons, tot_n_fem
    integer, intent(inout) :: elem_IDs(n-1, 2)
    real(kind=8), intent(in) :: nodes(tot_n_fem, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
    real(kind=8), intent(in) :: E(n-1), G(n-1), x_gl(3)
    real(kind=8), intent(inout) :: K_a(2, 2), K_t(2, 2), K_y(4, 4), K_z(4, 4)
    real(kind=8), intent(inout) :: T(3, 3), K_elem(12, 12), T_elem(12, 12)
    real(kind=8), intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12), Pelem_y(4, 12), Pelem_z(4, 12)
    real(kind=8), intent(in) :: const2(2, 2), const_y(4, 4), const_z(4, 4), rhs(size)

    ! Output
    real(kind=8), intent(out) :: x(size), K(size, size)

    ! Working
    real(kind=8) :: P0(3), P1(3), x_loc(3), y_loc(3), z_loc(3), x_cross(3), y_cross(3)
    real(kind=8) :: L, EA_L, GJ_L, EIy_L3, EIz_L3, res(12, 12)
    real(kind=8) :: mat12x12(12, 12), mat12x4(12, 4), mat12x2(12, 2)
    integer ::  num_elems, num_nodes, num_cons, ielem, in0, in1, ind, i
    real(kind=8) :: Pelem_a_T(12, 2), Pelem_t_T(12, 2), K_(size, size)
    real(kind=8) :: Pelem_y_T(12, 4), Pelem_z_T(12, 4), T_elem_T(12, 12), b(size)
    integer :: ipiv(size), n_solve


    num_elems = n - 1
    num_nodes = n
    num_cons = 1 ! only 1 con in current spatialbeam code

    K(:, :) = 0.
    do ielem = 1, num_elems ! loop over num elements
      P0 = nodes(elem_IDs(ielem, 1), :)
      P1 = nodes(elem_IDs(ielem, 2), :)

      call unit(P1 - P0, x_loc)
      call cross(x_loc, x_gl, x_cross)
      call unit(x_cross, y_loc)
      call cross(x_loc, y_loc, y_cross)
      call unit(y_cross, z_loc)

      T(1, :) = x_loc
      T(2, :) = y_loc
      T(3, :) = z_loc

      do ind = 1, 4
        T_elem(3*(ind-1)+1:3*(ind-1)+3, 3*(ind-1)+1:3*(ind-1)+3) = T
      end do

      call norm(P1 - P0, L)
      EA_L = E(ielem) * A(ielem) / L
      GJ_L = G(ielem) * J(ielem) / L
      EIy_L3 = E(ielem) * Iy(ielem) / L**3
      EIz_L3 = E(ielem) * Iz(ielem) / L**3

      K_a(:, :) = EA_L * const2
      K_t(:, :) = GJ_L * const2

      K_y(:, :) = EIy_L3 * const_y
      K_y(2, :) = K_y(2, :) * L
      K_y(4, :) = K_y(4, :) * L
      K_y(:, 2) = K_y(:, 2) * L
      K_y(:, 4) = K_y(:, 4) * L

      K_z(:, :) = EIz_L3 * const_z
      K_z(2, :) = K_z(2, :) * L
      K_z(4, :) = K_z(4, :) * L
      K_z(:, 2) = K_z(:, 2) * L
      K_z(:, 4) = K_z(:, 4) * L

      K_elem(:, :) = 0.
      call transpose2(2, 12, Pelem_a, Pelem_a_T)
      call matmul2(12, 2, 2, Pelem_a_T, K_a, mat12x2)
      call matmul2(12, 2, 12, mat12x2, Pelem_a, res)
      K_elem = K_elem + res

      call transpose2(2, 12, Pelem_t, Pelem_t_T)
      call matmul2(12, 2, 2, Pelem_t_T, K_t, mat12x2)
      call matmul2(12, 2, 12, mat12x2, Pelem_t, res)
      K_elem = K_elem + res

      call transpose2(4, 12, Pelem_y, Pelem_y_T)
      call matmul2(12, 4, 4, Pelem_y_T, K_y, mat12x4)
      call matmul2(12, 4, 12, mat12x4, Pelem_y, res)
      K_elem = K_elem + res

      call transpose2(4, 12, Pelem_z, Pelem_z_T)
      call matmul2(12, 4, 4, Pelem_z_T, K_z, mat12x4)
      call matmul2(12, 4, 12, mat12x4, Pelem_z, res)
      K_elem = K_elem + res

      call transpose2(12, 12, T_elem, T_elem_T)
      call matmul2(12, 12, 12, T_elem_T, K_elem, mat12x12)
      call matmul2(12, 12, 12, mat12x12, T_elem, res)

      in0 = elem_IDs(ielem, 1)
      in1 = elem_IDs(ielem, 2)

      K(6*(in0-1)+1:6*(in0-1)+6, 6*(in0-1)+1:6*(in0-1)+6) = &
      K(6*(in0-1)+1:6*(in0-1)+6, 6*(in0-1)+1:6*(in0-1)+6) + res(:6, :6)

      K(6*(in1-1)+1:6*(in1-1)+6, 6*(in0-1)+1:6*(in0-1)+6) = &
      K(6*(in1-1)+1:6*(in1-1)+6, 6*(in0-1)+1:6*(in0-1)+6) + res(7:, :6)

      K(6*(in0-1)+1:6*(in0-1)+6, 6*(in1-1)+1:6*(in1-1)+6) = &
      K(6*(in0-1)+1:6*(in0-1)+6, 6*(in1-1)+1:6*(in1-1)+6) + res(:6, 7:)

      K(6*(in1-1)+1:6*(in1-1)+6, 6*(in1-1)+1:6*(in1-1)+6) = &
      K(6*(in1-1)+1:6*(in1-1)+6, 6*(in1-1)+1:6*(in1-1)+6) + res(7:, 7:)

    end do

    do i = 1, 6
      K(6*num_nodes+i, 6*cons+i) = 10**9.
      K(6*cons+i, 6*num_nodes+i) = 10**9.
    end do

    n_solve = size
    b = rhs
    K_ = K
    call solve(K_, x, b, n_solve, ipiv)

  end subroutine assemblestructmtx_main

  subroutine transpose2(m, n, mtx, new_mtx)

    implicit none

    integer, intent(in) :: m, n
    real(kind=8), intent(in) :: mtx(m, n)
    real(kind=8), intent(out) :: new_mtx(n, m)

    integer :: i, j

    do i=1,m
      do j=1,n
        new_mtx(j, i) = mtx(i, j)
      end do
    end do

  end subroutine

  subroutine matmul2(m, n, p, A, B, C)

    implicit none

    integer, intent(in) :: m, n, p
    real(kind=8), intent(in) :: A(m, n), B(n, p)
    real(kind=8), intent(out) :: C(m, p)

    integer :: i, j, k

    C(:, :) = 0.

    do i=1,m
      do j=1,p
        do k=1,n
          C(i, j) = C(i, j) + A(i, k) * B(k, j)
        end do
      end do
    end do

  end subroutine

  subroutine matmul2c(m, n, p, A, B, C)

    implicit none

    integer, intent(in) :: m, n, p
    complex(kind=8), intent(in) :: A(m, n), B(n, p)
    complex(kind=8), intent(out) :: C(m, p)

    integer :: i, j, k

    C(:, :) = 0.

    do i=1,m
      do j=1,p
        do k=1,n
          C(i, j) = C(i, j) + A(i, k) * B(k, j)
        end do
      end do
    end do

  end subroutine

  subroutine assembleaeromtx_main(ny, nx, ny_, nx_, alpha, points, bpts, mesh, skip, symmetry, mtx)

    implicit none

    ! Input
    integer, intent(in) :: ny, nx, ny_, nx_
    complex(kind=8), intent(in) :: alpha, mesh(nx_, ny_, 3)
    complex(kind=8), intent(in) :: points(nx-1, ny-1, 3), bpts(nx_-1, ny_, 3)
    logical, intent(in) :: skip, symmetry

    ! Output
    complex(kind=8), intent(out) :: mtx((nx-1)*(ny-1), (nx_-1)*(ny_-1), 3)

    ! Working
    integer :: el_j, el_i, cp_j, cp_i, el_loc_j, el_loc, cp_loc_j, cp_loc
    complex(kind=8) :: pi, P(3), A(3), B(3), u(3), C(3), D(3)
    complex(kind=8) :: A_sym(3), B_sym(3), C_sym(3), D_sym(3)
    complex(kind=8) :: ur2(3), r1(3), r2(3), r1_mag, r2_mag
    complex(kind=8) :: ur1(3), bound(3), dot_ur2, dot_ur1
    complex(kind=8) :: edges(3), C_te(3), D_te(3), C_te_sym(3), D_te_sym(3)

    pi = 4.d0*atan(1.d0)

    ! Trailing vortices in AVL follow the x-axis; no cos or sin
    u(1) = cos(alpha * pi / 180.)
    u(2) = 0.
    u(3) = sin(alpha * pi / 180.)

    mtx(:, :, :) = 0.

    do el_j = 1, ny_-1 ! spanwise loop through horseshoe elements
      el_loc_j = (el_j - 1) * (nx_ - 1)
      C_te = mesh(nx_, el_j + 1, :)
      D_te = mesh(nx_, el_j + 0, :)

      if (symmetry) then
        C_te_sym = C_te
        D_te_sym = D_te
        C_te_sym(2) = -C_te_sym(2)
        D_te_sym(2) = -D_te_sym(2)
      end if

      do cp_j = 1, ny-1 ! spanwise loop through control points
        cp_loc_j = (cp_j - 1) * (nx - 1)

        do cp_i = 1, nx-1 ! chordwise loop through control points
          cp_loc = cp_i + cp_loc_j
          P = points(cp_i, cp_j, :)

          r1 = P - D_te
          r2 = P - C_te
          call normc(r1, r1_mag)
          call normc(r2, r2_mag)

          call crossc(u, r2, ur2)
          call crossc(u, r1, ur1)

          edges(:) = 0.
          call dotc(u, r2, dot_ur2)
          call dotc(u, r1, dot_ur1)
          edges = ur2 / (r2_mag * (r2_mag - dot_ur2))
          edges = edges - ur1 / (r1_mag * (r1_mag - dot_ur1))

          if (symmetry) then
            r1 = P - D_te_sym
            r2 = P - C_te_sym
            call normc(r1, r1_mag)
            call normc(r2, r2_mag)

            call crossc(u, r2, ur2)
            call crossc(u, r1, ur1)

            call dotc(u, r2, dot_ur2)
            call dotc(u, r1, dot_ur1)

            edges = edges - ur2 / (r2_mag * (r2_mag - dot_ur2))
            edges = edges + ur1 / (r1_mag * (r1_mag - dot_ur1))
          end if

          do el_i = nx_-1, 1, -1 ! chordwise loop through horseshoe elements
            el_loc = el_i + el_loc_j

            A = bpts(el_i + 0, el_j + 0, :)
            B = bpts(el_i + 0, el_j + 1, :)

            if (el_i .EQ. nx_ - 1) then
              C = C_te
              D = D_te
            else
              C = bpts(el_i + 1, el_j + 1, :)
              D = bpts(el_i + 1, el_j + 0, :)
            end if

            call calc_vorticity(B, C, P, edges)
            call calc_vorticity(D, A, P, edges)

            if (symmetry) then
              A_sym = A
              B_sym = B
              C_sym = C
              D_sym = D
              A_sym(2) = -A_sym(2)
              B_sym(2) = -B_sym(2)
              C_sym(2) = -C_sym(2)
              D_sym(2) = -D_sym(2)

              call calc_vorticity(C_sym, B_sym, P, edges)
              call calc_vorticity(A_sym, D_sym, P, edges)
            end if

            if ((skip)  .and. (cp_loc .EQ. el_loc)) then
              bound(:) = 0.
              if (symmetry) then
                call calc_vorticity(B_sym, A_sym, P, bound)
              end if
              mtx(cp_loc, el_loc, :) = edges + bound
            else
              bound(:) = 0.
              call calc_vorticity(A, B, P, bound)
              if (symmetry) then
                call calc_vorticity(B_sym, A_sym, P, bound)
              end if
              mtx(cp_loc, el_loc, :) = edges + bound
            end if

          end do
        end do

       end do
    end do

  end subroutine assembleaeromtx_main



  subroutine calc_vorticity(A, B, P, out)

    implicit none

    ! Input
    complex(kind=8), intent(in) :: A(3), B(3), P(3)

    ! Output
    complex(kind=8), intent(inout) :: out(3)

    ! Working
    complex(kind=8) :: r1(3), r2(3), r1_mag, r2_mag, r1r2(3), mag_mult, dot_r1r2

    r1 = P - A
    r2 = P - B

    call normc(r1, r1_mag)
    call normc(r2, r2_mag)

    call crossc(r1, r2, r1r2)
    mag_mult = r1_mag * r2_mag

    call dotc(r1, r2, dot_r1r2)
    out = out + (r1_mag + r2_mag) * r1r2 / (mag_mult * (mag_mult + dot_r1r2))

  end subroutine calc_vorticity



  subroutine biotsavart(A, B, P, inf, rev, out)

    implicit none

    ! Input
    complex(kind=8), intent(in) :: A(3), B(3), P(3)
    logical, intent(in) :: inf, rev

    ! Output
    complex(kind=8), intent(inout) :: out(3)

    ! Working
    complex(kind=8) :: rPA, rPB, rAB, rH
    complex(kind=8) :: cosA, cosB, C(3)
    complex(kind=8) :: eps, tmp(3), dot_BAPA, dot_BABA, dot_PBAB

    eps = 1e-5

    call normc(A - P, rPA)
    call normc(B - P, rPB)
    call normc(B - A, rAB)

    call dotc(B - A, P - A, dot_BAPA)
    call dotc(B - A, B - A, dot_BABA)
    call dotc(P - B, A - B, dot_PBAB)

    call normc(P - A - dot_BAPA / dot_BABA * (B - A), rH)
    rH = rH + eps
    cosA = dot_BAPA / (rPA * rAB)
    cosB = dot_PBAB / (rPB * rAB)
    call crossc(B - P, A - P, C)
    call unitc(C, C)

    if (inf) then
       tmp = -C / rH * (cosA + 1)
    else
       tmp = -C / rH * (cosA + cosB)
    end if

    if (rev) then
       tmp = -tmp
    end if

    out = out + tmp

  end subroutine biotsavart


! COMPLEX FUNCTIONS

  subroutine unitc(v, U)

    implicit none

    complex(kind=8), intent(in) :: v(3)
    complex(kind=8), intent(out) :: U(3)
    complex(kind=8) :: nm

    call normc(v, nm)
    U(1) = v(1) / nm
    U(2) = v(2) / nm
    U(3) = v(3) / nm

  end subroutine unitc

  subroutine normc(v, norm_output)

    implicit none

    complex(kind=8), intent(in) :: v(3)
    complex(kind=8), intent(out) :: norm_output
    complex(kind=8) :: dot_prod

    !norm = sqrt(dot_product(v, v))
    call dotc(v, v, dot_prod)
    norm_output = dot_prod ** 0.5

  end subroutine normc



  subroutine dotc(a, b, dot_prod)

    implicit none

    complex(kind=8), intent(in) :: a(3), b(3)
    complex(kind=8), intent(out) :: dot_prod

    dot_prod = a(1) * b(1) + a(2) * b(2) + a(3) * b(3)

  end subroutine dotc



  subroutine crossc(A, B, C)

    implicit none

    complex(kind=8), intent(in) :: A(3), B(3)
    complex(kind=8), intent(out) :: C(3)

    C(1) = A(2) * B(3) - A(3) * B(2)
    C(2) = A(3) * B(1) - A(1) * B(3)
    C(3) = A(1) * B(2) - A(2) * B(1)

  end subroutine crossc

! REAL FUNCTIONS

  subroutine unit(v, U)

    implicit none

    real(kind=8), intent(in) :: v(3)
    real(kind=8), intent(out) :: U(3)
    real(kind=8) :: nm

    call norm(v, nm)
    U(1) = v(1) / nm
    U(2) = v(2) / nm
    U(3) = v(3) / nm

  end subroutine unit

  subroutine norm(v, norm_output)

    implicit none

    real(kind=8), intent(in) :: v(3)
    real(kind=8), intent(out) :: norm_output
    real(kind=8) :: dot_prod

    !norm = sqrt(dot_product(v, v))
    call dot(v, v, dot_prod)
    norm_output = dot_prod ** 0.5

  end subroutine norm



  subroutine dot(a, b, dot_prod)

    implicit none

    real(kind=8), intent(in) :: a(3), b(3)
    real(kind=8), intent(out) :: dot_prod

    dot_prod = a(1) * b(1) + a(2) * b(2) + a(3) * b(3)

  end subroutine dot



  subroutine cross(A, B, C)

    implicit none

    real(kind=8), intent(in) :: A(3), B(3)
    real(kind=8), intent(out) :: C(3)

    C(1) = A(2) * B(3) - A(3) * B(2)
    C(2) = A(3) * B(1) - A(1) * B(3)
    C(3) = A(1) * B(2) - A(2) * B(1)

  end subroutine cross



end module
