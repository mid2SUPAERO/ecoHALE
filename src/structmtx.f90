subroutine assemblesparsemtx(num_nodes, num_elems, nnz, x_gl, &
    E, G, A, J, Iy, Iz, nodes, elems, &
    coeff_at, coeff_y, coeff_z, &
    Pelem_a, Pelem_t, Pelem_y, Pelem_z, &
    data, rows, cols)

    implicit none
    !f2py intent(in) num_nodes, num_elems, nnz, x_gl, E, G, A, J, Iy, Iz, nodes, elems, coeff_at, coeff_y, coeff_z, Pelem_a, Pelem_t, Pelem_y, Pelem_z
    !f2py intent(out) data, rows, cols
    !f2py depend(num_nodes) nodes
    !f2py depend(num_elems) E, G, A, J, Iy, Iz, elems
    !f2py depend(nnz) data, rows, cols

    ! Input
    integer, intent(in) :: num_nodes, num_elems, nnz
    complex*16, intent(in) :: x_gl(3)
    complex*16, intent(in) :: E(num_elems), G(num_elems)
    complex*16, intent(in) :: A(num_elems), J(num_elems)
    complex*16, intent(in) :: Iy(num_elems), Iz(num_elems)
    complex*16, intent(in) :: nodes(num_nodes, 3)
    integer, intent(in) :: elems(num_elems, 2)
    ! Local stiffness matrix coefficients
    complex*16, intent(in) :: coeff_at(2, 2), coeff_y(4, 4), coeff_z(4, 4)
    ! Local permutation matrices to map to list of dofs for local element
    complex*16, intent(in) :: Pelem_a(2, 12), Pelem_t(2, 12)
    complex*16, intent(in) :: Pelem_y(4, 12), Pelem_z(4, 12)

    ! Output
    complex*16, intent(out) :: data(nnz)
    integer, intent(out) :: rows(nnz), cols(nnz)

    ! Local stiffness matrices for axial, torsion, bending (y,z)
    complex*16 :: Kelem_a(2, 2), Kelem_t(2, 2)
    complex*16 :: Kelem_y(4, 4), Kelem_z(4, 4)
    ! Local transformation matrix (12,12) to map from local to global frame
    complex*16 :: Telem(12, 12), T(3, 3)
    ! Arrays that help in mapping from local element ordering to global ordering
    integer :: rows_elem(12, 12), cols_elem(12, 12)
    integer :: ones11(12, 12), ones12(12, 12)
    integer :: ones21(12, 12), ones22(12, 12)
    ! Local stiffness matrix in global frame
    complex*16 :: Kelem(12, 12)

    ! Miscellaneous
    complex*16 :: L, xyz1(3), xyz2(3)
    complex*16 :: x_loc(3), y_loc(3), z_loc(3), x_cross(3), y_cross(3)
    integer :: k, k1, k2, ind, ind1, ind2, ielem
    complex*16 :: norm

    do k1 = 1, 12
        do k2 = 1, 12
            rows_elem(k1, k2) = mod(k1-1, 6)
            cols_elem(k1, k2) = mod(k2-1, 6)
        end do
    end do

    ones11(:, :) = 0
    ones12(:, :) = 0
    ones21(:, :) = 0
    ones22(:, :) = 0

    ones11( 1:6 , 1:6 ) = 1
    ones12( 1:6 , 7:12) = 1
    ones21( 7:12, 1:6 ) = 1
    ones22( 7:12, 7:12) = 1

    Telem(:, :) = 0.

    data(:) = 0.
    rows(:) = 0
    cols(:) = 0

    ind = 0
    do ielem = 1, num_elems
        xyz1 = nodes(elems(ielem, 1), :)
        xyz2 = nodes(elems(ielem, 2), :)
        L = norm(xyz2 - xyz1)

        x_loc = (xyz2 - xyz1) / norm(xyz2 - xyz1)
        call cross(x_loc, x_gl, x_cross)
        y_loc = x_cross / norm(x_cross)
        call cross(x_loc, y_loc, y_cross)
        z_loc = y_cross / norm(y_cross)

        T(1, :) = x_loc
        T(2, :) = y_loc
        T(3, :) = z_loc

        do k = 1, 4
            Telem(3*(k-1)+1:3*(k-1)+3, 3*(k-1)+1:3*(k-1)+3) = T
        end do

        Kelem_a = coeff_at * E(ielem) * A(ielem) / L
        Kelem_t = coeff_at * G(ielem) * J(ielem) / L
        Kelem_y = coeff_y * E(ielem) * Iy(ielem) / L**3
        Kelem_y(2:4:2, :) = Kelem_y(2:4:2, :) * L
        Kelem_y(:, 2:4:2) = Kelem_y(:, 2:4:2) * L
        Kelem_z = coeff_z * E(ielem) * Iz(ielem) / L**3
        Kelem_z(2:4:2, :) = Kelem_z(2:4:2, :) * L
        Kelem_z(:, 2:4:2) = Kelem_z(:, 2:4:2) * L

        Kelem(:, :) = &
          matmul(matmul(transpose(Pelem_a), Kelem_a), Pelem_a) + &
          matmul(matmul(transpose(Pelem_t), Kelem_t), Pelem_t) + &
          matmul(matmul(transpose(Pelem_y), Kelem_y), Pelem_y) + &
          matmul(matmul(transpose(Pelem_z), Kelem_z), Pelem_z)
        Kelem = matmul(matmul(transpose(Telem), Kelem), Telem)

        ind1 = 6 * (elems(ielem, 1)-1)
        ind2 = 6 * (elems(ielem, 2)-1)

        do k1 = 1, 12
            do k2 = 1, 12
                ind = ind + 1
                data(ind) = data(ind) + Kelem(k1, k2)
                rows(ind) = rows(ind) + rows_elem(k1, k2) + 1
                cols(ind) = cols(ind) + cols_elem(k1, k2) + 1

                rows(ind) = rows(ind) + ones11(k1, k2) * ind1
                cols(ind) = cols(ind) + ones11(k1, k2) * ind1

                rows(ind) = rows(ind) + ones12(k1, k2) * ind1
                cols(ind) = cols(ind) + ones12(k1, k2) * ind2

                rows(ind) = rows(ind) + ones21(k1, k2) * ind2
                cols(ind) = cols(ind) + ones21(k1, k2) * ind1

                rows(ind) = rows(ind) + ones22(k1, k2) * ind2
                cols(ind) = cols(ind) + ones22(k1, k2) * ind2
            end do
        end do
    end do

    if (ind .ne. nnz) then
        print *, 'Error in assemblesparsemtx: did not reach end of nnz vectors'
    end if

    rows(:) = rows(:) - 1
    cols(:) = cols(:) - 1

end subroutine assemblesparsemtx


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


  nodes = (1-fem_origin) * mesh(1, :, :) + fem_origin * mesh(n, :, :)

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
