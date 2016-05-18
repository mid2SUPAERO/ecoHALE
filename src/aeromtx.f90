subroutine assembleaeromtx(n, alpha, mesh, points, bpts, mtx)

  implicit none

  !f2py intent(in) n, alpha, mesh, points, bpts
  !f2py intent(out) mtx
  !f2py depend(n) mesh, points, bpts, mtx

  ! Input
  integer, intent(in) :: n
  complex*16, intent(in) :: alpha, mesh(2, n, 3)
  complex*16, intent(in) :: points(n-1, 3), bpts(n, 3)

  ! Output
  complex*16, intent(out) :: mtx(n-1, n-1, 3)

  ! Working
  integer :: i, j
  complex*16 :: pi, P(3), A(3), B(3), D(3), E(3), F(3), G(3), Vinf(3)

  pi = 4.*atan(1.)

  Vinf(1) = cos(alpha * pi / 180.)
  Vinf(2) = 0.
  Vinf(3) = sin(alpha * pi / 180.)

  mtx(:, :, :) = 0.

  do i = 1, n-1 ! Loop over control points
     P = points(i, :)
     
     do j = 1, n-1 ! Loop over elements
        A = bpts(j + 0, :)
        B = bpts(j + 1, :)
        D = mesh(2, j + 0, :)
        E = mesh(2, j + 1, :)
        F = D + Vinf
        G = E + Vinf

        call biotsavart(A, B, P, .False., .False., mtx(i, j, :))
        call biotsavart(B, E, P, .False., .False., mtx(i, j, :))
        call biotsavart(A, D, P, .False., .True.,  mtx(i, j, :))
        call biotsavart(E, G, P, .True.,  .False., mtx(i, j, :))
        call biotsavart(D, F, P, .True.,  .True.,  mtx(i, j, :))

     end do
  end do

end subroutine assembleaeromtx




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
  complex*16 :: norm, pi, eps, tmp(3)

  pi = 4.*atan(1.)
  eps = 1e-5
  
  rPA = norm(A - P)
  rPB = norm(B - P)
  rAB = norm(B - A)
  rH = norm(P - A - dot_product(B - A, P - A) / &
       dot_product(B - A, B - A) * (B - A)) + eps
  cosA = dot_product(P - A, B - A) / (rPA * rAB)
  cosB = dot_product(P - B, A - B) / (rPB * rAB)
  call cross(B - P, A - P, C)
  C(:) = C(:) / norm(C)

  if (inf) then
     tmp = -C / rH * (cosA + 1) / (4 * pi)
  else
     tmp = -C / rH * (cosA + cosB) / (4 * pi)
  end if
  
  if (rev) then
     tmp = -tmp
  end if

  out = out + tmp

end subroutine biotsavart



complex*16 function norm(v)

  implicit none

  complex*16, intent(in) :: v(3)

  norm = sqrt(dot_product(v, v))

end function norm



subroutine cross(A, B, C)

  implicit none

  complex*16, intent(in) :: A(3), B(3)
  complex*16, intent(out) :: C(3)

  C(1) = A(2) * B(3) - A(3) * B(2)
  C(2) = A(3) * B(1) - A(1) * B(3)
  C(3) = A(1) * B(2) - A(2) * B(1)

end subroutine cross
