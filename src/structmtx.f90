subroutine assemblestructmtx(mesh, A, J, Iy, Iz, loads, & ! 6
  M_a, M_t, M_y, M_z, & ! 4
  elem_IDs, cons, fem_origin, & ! 3
  E_py, G_py, x_gl, T, & ! 3
  K_elem, S_a, S_t, S_y, S_z, T_elem, & ! 6
  const2, const_y, const_z, n, size, mtx, rhs) ! 7

  implicit none

  !f2py intent(in)   n, size, elem_IDs, cons, mesh, A, J, Iy, Iz, loads, fem_origin, E_py, G_py, x_gl, M_a, M_t, M_y, M_z, T, K_elem, S_a, S_t, S_y, S_z, T_elem, const2, const_y, const_z
  !f2py intent(out) mtx, rhs
  !f2py depend(n) elem_IDs, mesh, A, J, Iy, Iz, loads
  !f2py depend(size) mtx, rhs

  ! Input
  integer, intent(in) :: n, size, cons
  integer, intent(inout) :: elem_IDs(n-1, 2)
  complex*16, intent(in) :: mesh(2, n, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
  complex*16, intent(in) :: loads(n, 6), fem_origin, E_py, G_py, x_gl(3)
  complex*16, intent(inout) :: M_a(2, 2), M_t(2, 2), M_y(4, 4), M_z(4, 4)
  complex*16, intent(inout) :: T(3, 3), K_elem(12, 12), T_elem(12, 12)
  complex*16, intent(in) :: S_a(2, 12), S_t(2, 12), S_y(4, 12), S_z(4, 12)
  complex*16, intent(in) :: const2(2, 2), const_y(4, 4), const_z(4, 4)

  ! Output
  complex*16, intent(out) :: mtx(size, size), rhs(size)

  ! Working
  complex*16 :: nodes(n, 3), elem_nodes(n-1, 2, 3), E(n-1), G(n-1)
  complex*16 :: P0(3), P1(3), x_loc(3), y_loc(3), z_loc(3), x_cross(3), y_cross(3)
  complex*16 :: L, EA_L, GJ_L, EIy_L3, EIz_L3, norm, res(12, 12), loads_C(6, n)
  integer ::  num_elems, num_nodes, num_cons, ielem, in0, in1, ind, k


  nodes = (1-fem_origin) * mesh(1, :, :) + fem_origin * mesh(2, :, :)

  num_elems = n - 1
  num_nodes = n
  num_cons = 1 ! this may change? only 1 con in current spatialbeam code

  elem_IDs(:, :) = elem_IDs(:, :) + 1 ! account for 1-indexing in Fortran vs 0-indexing in Python

  do ielem = 1, num_elems ! loop over num elements
    in0 = elem_IDs(ielem, 1)
    in1 = elem_IDs(ielem, 2)

    elem_nodes(ielem, 1, :) = nodes(in0, :)
    elem_nodes(ielem, 2, :) = nodes(in1, :)
  end do

  E(:) = E_py * 1.0d0
  G(:) = G_py * 1.0d0

  mtx(:, :) = 0.
  do ielem = 1, num_elems ! loop over num elements
    P0 = elem_nodes(ielem, 1, :)
    P1 = elem_nodes(ielem, 2, :)

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

    L = norm(P1 - P0)
    EA_L = E(ielem) * A(ielem) / L
    GJ_L = G(ielem) * J(ielem) / L
    EIy_L3 = E(ielem) * Iy(ielem) / L**3
    EIz_L3 = E(ielem) * Iz(ielem) / L**3

    M_a(:, :) = EA_L * const2
    M_t(:, :) = GJ_L * const2

    M_y(:, :) = EIy_L3 * const_y
    M_y(2, :) = M_y(2, :) * L
    M_y(4, :) = M_y(4, :) * L
    M_y(:, 2) = M_y(:, 2) * L
    M_y(:, 4) = M_y(:, 4) * L

    M_z(:, :) = EIz_L3 * const_z
    M_z(2, :) = M_z(2, :) * L
    M_z(4, :) = M_z(4, :) * L
    M_z(:, 2) = M_z(:, 2) * L
    M_z(:, 4) = M_z(:, 4) * L

    K_elem(:, :) = 0.
    K_elem = K_elem + matmul(matmul(transpose(S_a), M_a), S_a)
    K_elem = K_elem + matmul(matmul(transpose(S_t), M_t), S_t)
    K_elem = K_elem + matmul(matmul(transpose(S_y), M_y), S_y)
    K_elem = K_elem + matmul(matmul(transpose(S_z), M_z), S_z)

    res = matmul(matmul(transpose(T_elem), K_elem), T_elem)

    in0 = elem_IDs(ielem, 1)
    in1 = elem_IDs(ielem, 2)

    mtx(6*(in0-1)+1:6*(in0-1)+6, 6*(in0-1)+1:6*(in0-1)+6) = &
    mtx(6*(in0-1)+1:6*(in0-1)+6, 6*(in0-1)+1:6*(in0-1)+6) + res(:6, :6)

    mtx(6*(in1-1)+1:6*(in1-1)+6, 6*(in0-1)+1:6*(in0-1)+6) = &
    mtx(6*(in1-1)+1:6*(in1-1)+6, 6*(in0-1)+1:6*(in0-1)+6) + res(7:, :6)

    mtx(6*(in0-1)+1:6*(in0-1)+6, 6*(in1-1)+1:6*(in1-1)+6) = &
    mtx(6*(in0-1)+1:6*(in0-1)+6, 6*(in1-1)+1:6*(in1-1)+6) + res(:6, 7:)

    mtx(6*(in1-1)+1:6*(in1-1)+6, 6*(in1-1)+1:6*(in1-1)+6) = &
    mtx(6*(in1-1)+1:6*(in1-1)+6, 6*(in1-1)+1:6*(in1-1)+6) + res(7:, 7:)

  end do


  do k = 1, 6
    mtx(6*num_nodes+k, 6*cons+k) = 1.
    mtx(6*cons+k, 6*num_nodes+k) = 1.
  end do

  rhs(:) = 0.0
  ! change ordering from Fortran to C
  loads_C = reshape(loads, shape(loads_C), order=(/2, 1/))
  rhs(:6*num_nodes) = reshape(loads_C, (/6*num_nodes/))


end subroutine assemblestructmtx

subroutine unit(v, U)

  implicit none

  complex*16, intent(in) :: v(3)
  complex*16, intent(out) :: U(3)
  complex*16 :: norm, nm

  nm = norm(v)
  U(1) = v(1) / nm
  U(2) = v(2) / nm
  U(3) = v(3) / nm

end subroutine unit
