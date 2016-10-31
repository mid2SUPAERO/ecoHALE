module oas_api

  use OAS_main
  implicit none

contains

  subroutine assemblesparsemtx(num_elems, tot_n_fem, nnz, x_gl, &
      E, G, A, J, Iy, Iz, nodes, elems, &
      coeff_at, coeff_y, coeff_z, &
      Pelem_a, Pelem_t, Pelem_y, Pelem_z, &
      data, rows, cols)

      implicit none

      ! Input
      integer, intent(in) :: tot_n_fem, num_elems, nnz
      real(kind=8), intent(in) :: x_gl(3)
      real(kind=8), intent(in) :: E(num_elems), G(num_elems)
      real(kind=8), intent(in) :: A(num_elems), J(num_elems)
      real(kind=8), intent(in) :: Iy(num_elems), Iz(num_elems)
      real(kind=8), intent(in) :: nodes(tot_n_fem, 3)
      integer, intent(in) :: elems(num_elems, 2)
      ! Local stiffness matrix coefficients
      real(kind=8), intent(in) :: coeff_at(2, 2), coeff_y(4, 4), coeff_z(4, 4)
      ! Local permutation matrices to map to list of dofs for local element
      real(kind=8), intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12)
      real(kind=8), intent(in) :: Pelem_y(4, 12), Pelem_z(4, 12)

      ! Output
      real(kind=8), intent(out) :: data(nnz)
      integer, intent(out) :: rows(nnz), cols(nnz)

      integer :: i, k1, k2, ind, ind1, ind2, ielem

      call assemblesparsemtx_main(num_elems, tot_n_fem, nnz, x_gl, &
          E, G, A, J, Iy, Iz, nodes, elems, &
          coeff_at, coeff_y, coeff_z, &
          Pelem_a, Pelem_t, Pelem_y, Pelem_z, &
          data, rows, cols)


  end subroutine assemblesparsemtx


  subroutine assemblestructmtx(n, tot_n_fem, size, nodes, A, J, Iy, Iz, & ! 6
    K_a, K_t, K_y, K_z, & ! 4
    elem_IDs, cons, & ! 3
    E, G, x_gl, T, & ! 3
    K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, & ! 6
    const2, const_y, const_z, rhs, K, x) ! 7

    implicit none

    !f2py intent(in)   n, tot_n_fem, size, elem_IDs, cons, nodes, A, J, Iy, Iz, E, G, x_gl, K_a, K_t, K_y, K_z, T, K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, const2, const_y, const_z
    !f2py intent(out) K, x
    !f2py depends(tot_n_fem) nodes
    !f2py depends(n) elem_IDs, nodes, A, J, Iy, Iz, E, G
    !f2py depends(size) K, x

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

    call assemblestructmtx_main(n, tot_n_fem, size, nodes, A, J, Iy, Iz, &
      K_a, K_t, K_y, K_z, &
      elem_IDs, cons, &
      E, G, x_gl, T, &
      K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, &
      const2, const_y, const_z, rhs, K, x)

  end subroutine assemblestructmtx



  subroutine assemblestructmtx_d(n, tot_n_fem, size, nodes, nodesd&
&   , a, ad, j, jd, iy, iyd, iz, izd, k_a, k_t, k_y, k_z, elem_ids, cons&
&   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
&   , const2, const_y, const_z, rhs, rhsd, k, kd, x, xd)

    use oas_main_d, only: assemblestructmtx_main_d
    implicit none

    !f2py intent(in)   n, tot_n_fem, size, elem_IDs, cons, nodes, A, J, Iy, Iz, E, G, x_gl, K_a, K_t, K_y, K_z, T, K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, const2, const_y, const_z
    !f2py intent(out) K, x, Kd, xd
    !f2py depends(tot_n_fem) nodes
    !f2py depends(n) elem_IDs, nodes, A, J, Iy, Iz, E, G
    !f2py depends(size) K, x, Kd, xd

    ! Input
    integer, intent(in) :: n, size, cons, tot_n_fem
    integer, intent(inout) :: elem_IDs(n-1, 2)
    real(kind=8), intent(in) :: nodes(tot_n_fem, 3), nodesd(tot_n_fem, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
    real(kind=8), intent(in) :: Ad(n-1), Jd(n-1), Iyd(n-1), Izd(n-1)
    real(kind=8), intent(in) :: E(n-1), G(n-1), x_gl(3)
    real(kind=8), intent(inout) :: K_a(2, 2), K_t(2, 2), K_y(4, 4), K_z(4, 4)
    real(kind=8), intent(inout) :: T(3, 3), K_elem(12, 12), T_elem(12, 12)
    real(kind=8), intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12), Pelem_y(4, 12), Pelem_z(4, 12)
    real(kind=8), intent(in) :: const2(2, 2), const_y(4, 4), const_z(4, 4), rhs(size)
    real(kind=8), intent(out) :: rhsd(size)

    ! Output
    real(kind=8), intent(out) :: K(size, size), Kd(size, size), x(size), xd(size)

    call assemblestructmtx_main_d(n, tot_n_fem, size, nodes, nodesd&
  &   , a, ad, j, jd, iy, iyd, iz, izd, k_a, k_t, k_y, k_z, elem_ids, cons&
  &   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
  &   , const2, const_y, const_z, rhs, rhsd, k, kd, x, xd)

  end subroutine assemblestructmtx_d

  subroutine assemblestructmtx_b(n, tot_n_fem, size, nodes, nodesb&
&   , a, ab, j, jb, iy, iyb, iz, izb, k_a, k_t, k_y, k_z, elem_ids, cons&
&   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
&   , const2, const_y, const_z, rhs, rhsb, k, kb, x, xb)

    use oas_main_b, only: assemblestructmtx_main_b
    implicit none

    !f2py intent(in)   n, tot_n_fem, size, elem_IDs, cons, nodes, A, J, Iy, Iz, E, G, x_gl, K_a, K_t, K_y, K_z, T, K_elem, Pelem_a, Pelem_t, Pelem_y, Pelem_z, T_elem, const2, const_y, const_z, K, x, Kb, xb
    !f2py intent(out) Ab, Jb, Iyb, Izb, nodesb, rhsb
    !f2py depends(tot_n_fem) nodes, nodesb
    !f2py depends(n) elem_IDs, A, J, Iy, Iz, Ab, Jb, Iyb, Izb, E, G
    !f2py depends(size) K, x, Kb, xb

    ! Input
    integer, intent(in) :: n, size, cons, tot_n_fem
    integer, intent(inout) :: elem_IDs(n-1, 2)
    real(kind=8), intent(in) :: nodes(tot_n_fem, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
    real(kind=8), intent(in) :: E(n-1), G(n-1), x_gl(3)
    real(kind=8), intent(inout) :: K_a(2, 2), K_t(2, 2), K_y(4, 4), K_z(4, 4)
    real(kind=8), intent(inout) :: T(3, 3), K_elem(12, 12), T_elem(12, 12)
    real(kind=8), intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12), Pelem_y(4, 12), Pelem_z(4, 12)
    real(kind=8), intent(in) :: const2(2, 2), const_y(4, 4), const_z(4, 4), rhs(size)
    real(kind=8), intent(in) :: Kb(size, size), xb(size)
    real(kind=8), intent(in) :: K(size, size), x(size)

    ! Output
    real(kind=8), intent(out) :: Ab(n-1), Jb(n-1), Iyb(n-1), Izb(n-1), nodesb(tot_n_fem, 3), rhsb(size)

    call assemblestructmtx_main_b(n, tot_n_fem, size, nodes, nodesb&
  &   , a, ab, j, jb, iy, iyb, iz, izb, k_a, k_t, k_y, k_z, elem_ids, cons&
  &   , e, g, x_gl, t, k_elem, pelem_a, pelem_t, pelem_y, pelem_z, t_elem&
  &   , const2, const_y, const_z, rhs, rhsb, k, kb, x, xb)

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



end module
