#/bin/sh

SRC_FILES = src/aeromtx.f90

default:	
	f2py --fcompiler=gnu95 -c ${SRC_FILES} -m lib
