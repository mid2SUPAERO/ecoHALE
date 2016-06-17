subroutine assembleaeromtx_kink(n, nx, alpha, mesh, points, bpts, mtx)

  implicit none

  !f2py intent(in) n, nx, alpha, mesh, points, bpts
  !f2py intent(out) mtx
  !f2py depend(n) mesh, points, bpts, mtx
  !f2py depend(nx) mesh, points, bpts, mtx


  ! Input
  integer, intent(in) :: n, nx
  complex*16, intent(in) :: alpha, mesh(nx, n, 3)
  complex*16, intent(in) :: points(nx-1, n-1, 3), bpts(nx-1, n, 3)

  ! Output
  complex*16, intent(out) :: mtx((nx-1)*(n-1), (nx-1)*(n-1), 3)

  ! Working
  integer :: el_j, el_i, cp_j, cp_i, el_loc_j, el_loc, cp_loc_j, cp_loc
  complex*16 :: pi, P(3), A(3), B(3), D(3), E(3), F(3), G(3), Vinf(3)
  complex*16 :: chk(3)
  real*16 :: infinity

  pi = 4.d0*atan(1.d0)
  infinity = huge(infinity)

  Vinf(1) = cos(alpha * pi / 180.)
  Vinf(2) = 0.
  Vinf(3) = sin(alpha * pi / 180.)

  mtx(:, :, :) = 0.

  do el_j = 1, n-1 ! spanwise loop through horseshoe elements
    el_loc_j = (el_j - 1) * (nx - 1)

    do el_i = 1, nx-1 ! chordwise loop through horseshoe elements
      el_loc = el_i + el_loc_j

      A = bpts(el_i, el_j + 0, :)
      B = bpts(el_i, el_j + 1, :)
      D = mesh(el_i + 1, el_j + 0, :)
      E = mesh(el_i + 1, el_j + 1, :)
      F = D + Vinf
      G = E + Vinf

      do cp_j = 1, n-1 ! spanwise loop through control points
        cp_loc_j = (cp_j - 1) * (nx - 1)

        do cp_i = 1, nx-1 ! chordwise loop through control points
          cp_loc = cp_i + cp_loc_j
          P = points(cp_i, cp_j, :)
          chk(:) = 0.

          call biotsavart(A, B, P, .False., .False., chk)

          if ((4. .GE. infinity) .or. (chk(1) .NE. chk(1))) then
          else
            mtx(cp_loc, el_loc, :) = mtx(cp_loc, el_loc, :) + chk
          end if

          call biotsavart(B, E, P, .False., .False., mtx(cp_loc, el_loc, :))
          call biotsavart(A, D, P, .False., .True.,  mtx(cp_loc, el_loc, :))
          call biotsavart(E, G, P, .True.,  .False., mtx(cp_loc, el_loc, :))
          call biotsavart(D, F, P, .True.,  .True.,  mtx(cp_loc, el_loc, :))
        end do
      end do

     end do
  end do

  mtx = mtx / (4. * pi)

end subroutine assembleaeromtx_kink

subroutine assembleaeromtx_freestream(n, alpha, points, bpts, mtx)

  implicit none

  !f2py intent(in) n, alpha, points, bpts
  !f2py intent(out) mtx
  !f2py depend(n) mesh, points, bpts, mtx

  ! Input
  integer, intent(in) :: n
  complex*16, intent(in) :: alpha
  complex*16, intent(in) :: points(n-1, 3), bpts(n, 3)

  ! Output
  complex*16, intent(out) :: mtx(n-1, n-1, 3)

  ! Working
  integer :: i, j
  complex*16 :: pi, P(3), A(3), B(3), F(3), G(3), Vinf(3)

  pi = 4.d0*atan(1.d0)

  Vinf(1) = cos(alpha * pi / 180.)
  Vinf(2) = 0.
  Vinf(3) = sin(alpha * pi / 180.)

  mtx(:, :, :) = 0.

  do i = 1, n-1 ! Loop over control points
     P = points(i, :)

     do j = 1, n-1 ! Loop over elements
        A = bpts(j + 0, :)
        B = bpts(j + 1, :)
        F = A + Vinf
        G = B + Vinf

        call biotsavart(A, B, P, .False., .False., mtx(i, j, :))
        call biotsavart(B, G, P, .True.,  .False., mtx(i, j, :))
        call biotsavart(A, F, P, .True.,  .True.,  mtx(i, j, :))

     end do
  end do

  mtx = mtx / (4. * pi)

end subroutine assembleaeromtx_freestream

subroutine assembleaeromtx_paper(n, alpha, points, bpts, skip, mtx)

  implicit none

  !f2py intent(in) n, alpha, points, bpts
  !f2py intent(out) mtx
  !f2py depend(n) mesh, points, bpts, mtx

  ! Input
  integer, intent(in) :: n
  complex*16, intent(in) :: alpha
  complex*16, intent(in) :: points(n-1, 3), bpts(n, 3)
  logical, intent(in) :: skip

  ! Output
  complex*16, intent(out) :: mtx(n-1, n-1, 3)

  ! Working
  integer :: i, j
  complex*16 :: pi, P(3), A(3), B(3), u(3)
  complex*16 :: norm, ur2(3), r0(3), r1(3), r2(3), r0_mag, r1_mag, r2_mag
  complex*16 :: r1r2(3), ur1(3), dot, t1(3), t2(3), t3(3)

  pi = 4.d0*atan(1.d0)

  u(1) = cos(alpha * pi / 180.)
  u(2) = 0.
  u(3) = sin(alpha * pi / 180.)

  mtx(:, :, :) = 0.

  do i = 1, n-1 ! Loop over control points
     P = points(i, :)

     do j = 1, n-1 ! Loop over elements
        A = bpts(j + 0, :)
        B = bpts(j + 1, :)

        r0 = B - A
        r1 = P - A
        r2 = P - B

        r0_mag = norm(r0)
        r1_mag = norm(r1)
        r2_mag = norm(r2)

        call cross(u, r2, ur2)
        call cross(r1, r2, r1r2)
        call cross(u, r1, ur1)

        t1 = ur2 / (r2_mag * (r2_mag - dot(u, r2)))
        t3 = ur1 / (r1_mag * (r1_mag - dot(u, r1)))

        if ((skip)  .and. (i .EQ. j)) then
          mtx(i, j, :) = t1 - t3
        else
          t2 = (r1_mag + r2_mag) * r1r2 / &
               (r1_mag * r2_mag * (r1_mag * r2_mag + dot(r1, r2)))
          mtx(i, j, :) = t1 + t2 - t3
        end if

     end do
  end do

  mtx = mtx / (4. * pi)

end subroutine assembleaeromtx_paper

subroutine assembleaeromtx_planform(n, alpha, points, bpts, skip, mtx)

  implicit none

  !f2py intent(in) n, alpha, points, bpts
  !f2py intent(out) mtx
  !f2py depend(n) mesh, points, bpts, mtx

  ! Input
  integer, intent(in) :: n
  complex*16, intent(in) :: alpha
  complex*16, intent(in) :: points(n-1, 3), bpts(n, 3)
  logical, intent(in) :: skip

  ! Output
  complex*16, intent(out) :: mtx(n-1, n-1, 3)

  ! Working
  integer :: i, j
  complex*16 :: pi, P(3), A(3), B(3), u(3)
  complex*16 :: norm, ur2(3), r0(3), r1(3), r2(3), r0_mag, r1_mag, r2_mag
  complex*16 :: r1r2(3), ur1(3), dot, t1(3), t2(3), t3(3)

  pi = 4.d0*atan(1.d0)

  u(1) =1.

  mtx(:, :, :) = 0.

  do i = 1, n-1 ! Loop over control points
     P = points(i, :)

     do j = 1, n-1 ! Loop over elements
        A = bpts(j + 0, :)
        B = bpts(j + 1, :)

        r0 = B - A
        r1 = P - A
        r2 = P - B

        r0_mag = norm(r0)
        r1_mag = norm(r1)
        r2_mag = norm(r2)

        call cross(u, r2, ur2)
        call cross(r1, r2, r1r2)
        call cross(u, r1, ur1)

        t1 = ur2 / (r2_mag * (r2_mag - dot(u, r2)))
        t3 = ur1 / (r1_mag * (r1_mag - dot(u, r1)))

        if ((skip)  .and. (i .EQ. j)) then
          mtx(i, j, :) = t1 - t3
        else
          t2 = (r1_mag + r2_mag) * r1r2 / &
               (r1_mag * r2_mag * (r1_mag * r2_mag + dot(r1, r2)))
          mtx(i, j, :) = t1 + t2 - t3
        end if

     end do
  end do

  mtx = mtx / (4. * pi)

end subroutine assembleaeromtx_planform




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
