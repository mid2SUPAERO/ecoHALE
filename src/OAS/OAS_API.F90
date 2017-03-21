module oas_api

  use OAS_main
  implicit none

contains

  subroutine assemblestructmtx(n, tot_n_fem, nodes, A, J, Iy, Iz, &
    K_a, K_t, K_y, K_z, &
    cons, E, G, x_gl, T, &
    K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, &
    const2, const_y, const_z, K)

    implicit none

    !f2py intent(in)   n, tot_n_fem, cons, nodes, A, J, Iy, Iz, E, G, x_gl, K_a, K_t, K_y, K_z, T, K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, const2, const_y, const_z
    !f2py intent(out) K
    !f2py depends(tot_n_fem) nodes
    !f2py depends(n) elem_IDs, nodes, A, J, Iy, Iz, K

    ! Input
    integer, intent(in) :: n, cons, tot_n_fem
    real(kind=8), intent(in) :: nodes(tot_n_fem, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
    real(kind=8), intent(in) :: E, G, x_gl(3)
    real(kind=8), intent(inout) :: K_a(2, 2), K_t(2, 2), K_y(4, 4), K_z(4, 4)
    real(kind=8), intent(inout) :: T(3, 3), K_elem(12, 12), T_elem(12, 12)
    real(kind=8), intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12), Pelem_y(4, 12), Pelem_z(4, 12)
    real(kind=8), intent(in) :: const2(2, 2), const_y(4, 4), const_z(4, 4)

    ! Output
    real(kind=8), intent(out) :: K(6*n+6, 6*n+6)

    call assemblestructmtx_main(n, tot_n_fem, nodes, A, J, Iy, Iz, &
      K_a, K_t, K_y, K_z, &
      cons, E, G, x_gl, T, &
      K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, &
      const2, const_y, const_z, K)

  end subroutine assemblestructmtx



  subroutine assemblestructmtx_d(n, tot_n_fem, nodes, nodesd&
&   , a, ad, j, jd, iy, iyd, iz, izd, k_a, k_t, k_y, k_z, cons&
&   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
&   , const2, const_y, const_z, k, kd)

    use oas_main_d, only: assemblestructmtx_main_d
    implicit none

    !f2py intent(in)   n, tot_n_fem, cons, nodes, A, J, Iy, Iz, E, G, x_gl, K_a, K_t, K_y, K_z, T, K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, const2, const_y, const_z
    !f2py intent(out) K, Kd
    !f2py depends(tot_n_fem) nodes
    !f2py depends(n) elem_IDs, nodes, A, J, Iy, Iz, K, Kd

    ! Input
    integer, intent(in) :: n, cons, tot_n_fem
    real(kind=8), intent(in) :: nodes(tot_n_fem, 3), nodesd(tot_n_fem, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
    real(kind=8), intent(in) :: Ad(n-1), Jd(n-1), Iyd(n-1), Izd(n-1)
    real(kind=8), intent(in) :: E, G, x_gl(3)
    real(kind=8), intent(inout) :: K_a(2, 2), K_t(2, 2), K_y(4, 4), K_z(4, 4)
    real(kind=8), intent(inout) :: T(3, 3), K_elem(12, 12), T_elem(12, 12)
    real(kind=8), intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12), Pelem_y(4, 12), Pelem_z(4, 12)
    real(kind=8), intent(in) :: const2(2, 2), const_y(4, 4), const_z(4, 4)

    ! Output
    real(kind=8), intent(out) :: K(6*n+6, 6*n+6), Kd(6*n+6, 6*n+6)

    call assemblestructmtx_main_d(n, tot_n_fem, nodes, nodesd&
  &   , a, ad, j, jd, iy, iyd, iz, izd, k_a, k_t, k_y, k_z, cons&
  &   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
  &   , const2, const_y, const_z, k, kd)

  end subroutine assemblestructmtx_d

  subroutine assemblestructmtx_b(n, tot_n_fem, nodes, nodesb&
&   , a, ab, j, jb, iy, iyb, iz, izb, k_a, k_t, k_y, k_z, cons&
&   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
&   , const2, const_y, const_z, k, kb)

    use oas_main_b, only: assemblestructmtx_main_b
    implicit none

    !f2py intent(in)   n, tot_n_fem, cons, nodes, A, J, Iy, Iz, E, G, x_gl, K_a, K_t, K_y, K_z, T, K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, const2, const_y, const_z, K, Kb
    !f2py intent(out) Ab, Jb, Iyb, Izb, nodesb
    !f2py depends(tot_n_fem) nodes, nodesb
    !f2py depends(n) elem_IDs, A, J, Iy, Iz, Ab, Jb, Iyb, Izb, K, Kb

    ! Input
    integer, intent(in) :: n, cons, tot_n_fem
    real(kind=8), intent(in) :: nodes(tot_n_fem, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
    real(kind=8), intent(in) :: E, G, x_gl(3)
    real(kind=8), intent(inout) :: K_a(2, 2), K_t(2, 2), K_y(4, 4), K_z(4, 4)
    real(kind=8), intent(inout) :: T(3, 3), K_elem(12, 12), T_elem(12, 12)
    real(kind=8), intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12), Pelem_y(4, 12), Pelem_z(4, 12)
    real(kind=8), intent(in) :: const2(2, 2), const_y(4, 4), const_z(4, 4)
    real(kind=8), intent(in) :: Kb(6*n+6, 6*n+6)
    real(kind=8), intent(in) :: K(6*n+6, 6*n+6)

    ! Output
    real(kind=8), intent(out) :: Ab(n-1), Jb(n-1), Iyb(n-1), Izb(n-1), nodesb(tot_n_fem, 3)

    call assemblestructmtx_main_b(n, tot_n_fem, nodes, nodesb&
  &   , a, ab, j, jb, iy, iyb, iz, izb, k_a, k_t, k_y, k_z, cons&
  &   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
  &   , const2, const_y, const_z, k, kb)

  end subroutine assemblestructmtx_b

  subroutine assembleaeromtx(ny, nx, ny_, nx_, alpha, points, bpts, mesh, skip, symmetry, mtx)

    implicit none

    ! Input
    integer, intent(in) :: ny, nx, ny_, nx_
    real(kind=8), intent(in) :: alpha, mesh(nx_, ny_, 3)
    real(kind=8), intent(in) :: points(nx-1, ny-1, 3), bpts(nx_-1, ny_, 3)
    logical, intent(in) :: skip, symmetry

    ! Output
    real(kind=8), intent(out) :: mtx((nx-1)*(ny-1), (nx_-1)*(ny_-1), 3)

    call assembleaeromtx_main(ny, nx, ny_, nx_, alpha, points, bpts, mesh, skip, symmetry, mtx)

  end subroutine assembleaeromtx

  subroutine assembleaeromtx_d(ny, nx, ny_, nx_, alpha, alphad, points, pointsd, &
    bpts, bptsd, mesh, meshd, skip, symmetry, mtx, mtxd)

    use oas_main_d, only: assembleaeromtx_main_d
    implicit none

    ! Input
    integer, intent(in) :: ny, nx, ny_, nx_
    real(kind=8), intent(in) :: alpha, alphad
    real(kind=8), intent(in) :: mesh(nx_, ny_, 3), meshd(nx_, ny_, 3)
    real(kind=8), intent(in) :: points(nx-1, ny-1, 3), pointsd(nx-1, ny-1, 3)
    real(kind=8), intent(in) :: bpts(nx_-1, ny_, 3), bptsd(nx_-1, ny_, 3)
    logical, intent(in) :: skip, symmetry

    ! Output
    real(kind=8), intent(out) :: mtx((nx-1)*(ny-1), (nx_-1)*(ny_-1), 3)
    real(kind=8), intent(out) :: mtxd((nx-1)*(ny-1), (nx_-1)*(ny_-1), 3)

    call assembleaeromtx_main_d(ny, nx, ny_, nx_, alpha, alphad, &
  &   points, pointsd, bpts, bptsd, mesh, meshd, skip, symmetry, mtx, mtxd)

  end subroutine assembleaeromtx_d

  subroutine assembleaeromtx_b(ny, nx, ny_, nx_, alpha, alphab, points, pointsb, &
    bpts, bptsb, mesh, meshb, skip, symmetry, mtx, mtxb)

    use oas_main_b, only: assembleaeromtx_main_b
    implicit none

    ! Input
    integer, intent(in) :: ny, nx, ny_, nx_
    real(kind=8), intent(in) :: alpha, mesh(nx_, ny_, 3)
    real(kind=8), intent(in) :: points(nx-1, ny-1, 3), bpts(nx_-1, ny_, 3)
    logical, intent(in) :: skip, symmetry
    real(kind=8), intent(in) :: mtxb((nx-1)*(ny-1), (nx_-1)*(ny_-1), 3)

    ! Output
    real(kind=8), intent(out) :: mtx((nx-1)*(ny-1), (nx_-1)*(ny_-1), 3)
    real(kind=8), intent(out) :: alphab, meshb(nx_, ny_, 3)
    real(kind=8), intent(out) :: pointsb(nx-1, ny-1, 3), bptsb(nx_-1, ny_, 3)

    call assembleaeromtx_main_b(ny, nx, ny_, nx_, alpha, alphab, &
  &   points, pointsb, bpts, bptsb, mesh, meshb, skip, symmetry, mtx, mtxb)

  end subroutine assembleaeromtx_b

  subroutine calc_vonmises(nodes, r, disp, E, G, x_gl, n, vonmises)

    implicit none

    ! Input
    integer, intent(in) :: n
    real(kind=8), intent(in) :: nodes(n, 3), r(n-1), disp(n, 6)
    real(kind=8), intent(in) :: E, G, x_gl(3)

    ! Output
    real(kind=8), intent(out) :: vonmises(n-1, 2)

    call calc_vonmises_main(nodes, r, disp, E, G, x_gl, n, vonmises)

  end subroutine

  subroutine calc_vonmises_b(nodes, nodesb, r, rb, disp, &
&   dispb, e, g, x_gl, n, vonmises, vonmisesb)

    use oas_main_b, only: calc_vonmises_main_b
    implicit none

    ! Input
    integer, intent(in) :: n
    real(kind=8), intent(in) :: nodes(n, 3), r(n-1), disp(n, 6)
    real(kind=8), intent(in) :: E, G, x_gl(3)
    real(kind=8), intent(in) :: vonmises(n-1, 2), vonmisesb(n-1, 2)

    ! Output
    real(kind=8), intent(out) :: nodesb(n, 3), rb(n-1), dispb(n, 6)

    nodesb(:, :) = 0.
    rb(:) = 0.
    dispb(:, :) = 0.

    call calc_vonmises_main_b(nodes, nodesb, r, rb, disp, &
  &   dispb, e, g, x_gl, n, vonmises, vonmisesb)

  end subroutine

  subroutine calc_vonmises_d(nodes, nodesd, r, rd, disp, &
&   dispd, e, g, x_gl, n, vonmises, vonmisesd)

    use oas_main_d, only: calc_vonmises_main_d
    implicit none

    ! Input
    integer, intent(in) :: n
    real(kind=8), intent(in) :: nodes(n, 3), nodesd(n, 3), r(n-1), rd(n-1)
    real(kind=8), intent(in) :: disp(n, 6), dispd(n, 6)
    real(kind=8), intent(in) :: E, G, x_gl(3)

    ! Output
    real(kind=8), intent(out) :: vonmises(n-1, 2), vonmisesd(n-1, 2)

    vonmisesd(:, :) = 0.

    call calc_vonmises_main_d(nodes, nodesd, r, rd, disp, &
  &   dispd, e, g, x_gl, n, vonmises, vonmisesd)

  end subroutine

  subroutine transferdisplacements(nx, ny, mesh, disp, w, def_mesh)

    implicit none

    ! Input
    integer, intent(in) :: nx, ny
    real(kind=8), intent(in) :: mesh(nx, ny, 3), disp(ny, 6), w

    ! Output
    real(kind=8), intent(out) :: def_mesh(nx, ny, 3)

    call transferdisplacements_main(nx, ny, mesh, disp, w, def_mesh)

  end subroutine transferdisplacements

  subroutine transferdisplacements_d(nx, ny, mesh, meshd, disp, dispd, w, def_mesh, def_meshd)

    use oas_main_d, only: transferdisplacements_main_d
    implicit none

    ! Input
    integer, intent(in) :: nx, ny
    real(kind=8), intent(in) :: mesh(nx, ny, 3), disp(ny, 6), w
    real(kind=8), intent(in) :: meshd(nx, ny, 3), dispd(ny, 6)

    ! Output
    real(kind=8), intent(out) :: def_mesh(nx, ny, 3), def_meshd(nx, ny, 3)

    call transferdisplacements_main_d(nx, ny, mesh, meshd, disp, dispd, w, def_mesh, def_meshd)

  end subroutine transferdisplacements_d

  subroutine transferdisplacements_b(nx, ny, mesh, meshb, disp, dispb, w, def_mesh, def_meshb)

    use oas_main_b, only: transferdisplacements_main_b
    implicit none

    ! Input
    integer, intent(in) :: nx, ny
    real(kind=8), intent(in) :: mesh(nx, ny, 3), disp(ny, 6), w
    real(kind=8), intent(in) :: def_meshb(nx, ny, 3), def_mesh(nx, ny, 3)

    ! Output
    real(kind=8), intent(out) :: meshb(nx, ny, 3), dispb(ny, 6)

    call transferdisplacements_main_b(nx, ny, mesh, meshb, disp, dispb, w, def_mesh, def_meshb)

  end subroutine transferdisplacements_b

end module
