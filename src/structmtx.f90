subroutine assemblestructmtx(mesh, A, J, Iy, Iz, loads, & ! 6
  M_a, M_t, M_y, M_z, & ! 4
  elem_IDs, cons, fem_origin, & ! 3
  E, G, x_gl, T, & ! 3
  K_elem, S_a, S_t, S_y, S_z, T_elem, & ! 6
  const2, const_y, const_z, n, size, mtx, rhs) ! 7

  implicit none

  !f2py intent(in)   n, size, elem_IDs, cons, mesh, A, J, Iy, Iz, loads, fem_origin, E, G, x_gl, M_a, M_t, M_y, M_z, T, K_elem, S_a, S_t, S_y, S_z, T_elem, const2, const_y, const_z
  !f2py intent(out) mtx, rhs
  !f2py depend(n) elem_IDs, mesh, A, J, Iy, Iz, loads
  !f2py depend(size) mtx, rhs

  ! Input
  integer, intent(in) :: n, size, elem_IDs(n-1, 2), cons
  complex*16, intent(in) :: mesh(2, n, 3), A(n-1), J(n-1), Iy(n-1), Iz(n-1)
  complex*16, intent(in) :: loads(n, 6), fem_origin, E, G, x_gl(3)
  complex*16, intent(in) :: M_a(2, 2), M_t(2, 2), M_y(4, 4), M_z(4, 4)
  complex*16, intent(in) :: T(3, 3), K_elem(12, 12)
  complex*16, intent(in) :: S_a(2, 12), S_t(2, 12), S_y(4, 12), S_z(4, 12)
  complex*16, intent(in) :: T_elem(12, 12), const2(2, 2), const_y(4, 4), const_z(4, 4)

  ! Output
  complex*16, intent(out) :: mtx(size, size), rhs(size)

  ! Working
  complex*16 :: nodes(n, 3), num_elems(2), num_nodes(2), num_cons

  nodes = (1-fem_origin) * mesh(1, :, :) + fem_origin * mesh(2, :, :)

  num_elems = n - 1
  num_nodes = n
  num_cons = 1

  ! print *, num_elems
  print *, n



end subroutine assemblestructmtx



complex*16 function unit(v, U)

  implicit none

  complex*16, intent(in) :: v(3)
  complex*16, intent(out) :: U(3)
  complex*16 :: norm, nm

  nm = norm(v)
  U(1) = v(1) / nm
  U(2) = v(2) / nm
  U(3) = v(3) / nm

  return

end function unit
