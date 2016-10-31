subroutine assembleaeromtx(ny, nx, ny_, nx_, alpha, points, bpts, mesh, skip, symmetry, mtx)

  implicit none

  !f2py intent(in) ny, nx, ny_, nx_, alpha, points, bpts, mesh
  !f2py intent(out) mtx
  !f2py depends(ny) points, mtx
  !f2py depends(nx) points, mtx
  !f2py depends(ny_) bpts, mtx, mesh
  !f2py depends(nx_) bpts, mtx, mesh


  ! Input
  integer, intent(in) :: ny, nx, ny_, nx_
  complex*16, intent(in) :: alpha, mesh(nx_, ny_, 3)
  complex*16, intent(in) :: points(nx-1, ny-1, 3), bpts(nx_-1, ny_, 3)
  logical, intent(in) :: skip, symmetry

  ! Output
  complex*16, intent(out) :: mtx((nx-1)*(ny-1), (nx_-1)*(ny_-1), 3)

  ! Working
  integer :: el_j, el_i, cp_j, cp_i, el_loc_j, el_loc, cp_loc_j, cp_loc
  complex*16 :: pi, P(3), A(3), B(3), u(3), C(3), D(3)
  complex*16 :: A_sym(3), B_sym(3), C_sym(3), D_sym(3)
  complex*16 :: norm, ur2(3), r1(3), r2(3), r1_mag, r2_mag
  complex*16 :: ur1(3), dot, bound(3)
  complex*16 :: edges(3), C_te(3), D_te(3), C_te_sym(3), D_te_sym(3)

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
        r1_mag = norm(r1)
        r2_mag = norm(r2)

        call cross(u, r2, ur2)
        call cross(u, r1, ur1)

        edges(:) = 0.
        edges = ur2 / (r2_mag * (r2_mag - dot(u, r2)))
        edges = edges - ur1 / (r1_mag * (r1_mag - dot(u, r1)))

        if (symmetry) then
          r1 = P - D_te_sym
          r2 = P - C_te_sym
          r1_mag = norm(r1)
          r2_mag = norm(r2)

          call cross(u, r2, ur2)
          call cross(u, r1, ur1)

          edges = edges - ur2 / (r2_mag * (r2_mag - dot(u, r2)))
          edges = edges + ur1 / (r1_mag * (r1_mag - dot(u, r1)))
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

end subroutine assembleaeromtx



subroutine calc_vorticity(A, B, P, out)

  implicit none

  ! Input
  complex*16, intent(in) :: A(3), B(3), P(3)

  ! Output
  complex*16, intent(inout) :: out(3)

  ! Working
  complex*16 :: r1(3), r2(3), r1_mag, r2_mag, norm, dot, r1r2(3), mag_mult

  r1 = P - A
  r2 = P - B

  r1_mag = norm(r1)
  r2_mag = norm(r2)

  call cross(r1, r2, r1r2)
  mag_mult = r1_mag * r2_mag

  out = out + (r1_mag + r2_mag) * r1r2 / (mag_mult * (mag_mult + dot(r1, r2)))

end subroutine calc_vorticity



subroutine biotsavart(A, B, P, inf, rev, out)

  implicit none

  ! Input
  complex*16, intent(in) :: A(3), B(3), P(3)
  logical, intent(in) :: inf, rev

  ! Output
  complex*16, intent(inout) :: out(3)

  ! Working
  complex*16 :: rPA, rPB, rAB, rH
  complex*16 :: cosA, cosB, C(3)
  complex*16 :: norm, dot, eps, tmp(3)

  eps = 1e-5

  rPA = norm(A - P)
  rPB = norm(B - P)
  rAB = norm(B - A)
  rH = norm(P - A - dot(B - A, P - A) / &
       dot(B - A, B - A) * (B - A)) + eps
  cosA = dot(P - A, B - A) / (rPA * rAB)
  cosB = dot(P - B, A - B) / (rPB * rAB)
  call cross(B - P, A - P, C)
  C(:) = C(:) / norm(C)

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



complex*16 function norm(v)

  implicit none

  complex*16, intent(in) :: v(3)
  complex*16 :: dot

  !norm = sqrt(dot_product(v, v))
  norm = dot(v, v) ** 0.5

  return

end function norm



complex*16 function dot(a, b)

  implicit none

  complex*16, intent(in) :: a(3), b(3)

  dot = a(1) * b(1) + a(2) * b(2) + a(3) * b(3)

  return

end function dot



subroutine cross(A, B, C)

  implicit none

  complex*16, intent(in) :: A(3), B(3)
  complex*16, intent(out) :: C(3)

  C(1) = A(2) * B(3) - A(3) * B(2)
  C(2) = A(3) * B(1) - A(1) * B(3)
  C(3) = A(1) * B(2) - A(2) * B(1)

end subroutine cross
