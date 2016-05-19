#/bin/sh

SRC_FILES = src/aeromtx.f90
# SRC_FILES = src/aeromtx_b-all.f90
# SRC_FILES = src/assembleaeromtx_b.f90
# SRC_FILES = src/biotsavart_b.f90
# SRC_FILES = src/cros_b.f90
# SRC_FILES = src/dot_b.f90
# SRC_FILES = src/norm_b.f90


default:
	f2py --fcompiler=gnu95 -c ${SRC_FILES} -m lib
