Program MakeCompositeF90
   Implicit NONE
   
#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscda.h"
#include "include/finclude/petscvec.h90"
#include "include/finclude/petscda.h90"

   DA                                       :: da
   Vec                                      :: phi, psi
   PetscScalar, Dimension(:), Pointer       :: phi_array, psi_array
   PetscTruth                               :: saveTXT, saveVTK, saveEnsight, showU, showPhi
   
   Integer                                  :: N, nx, ny, nz, level, numprocs, i, ierr
   Integer                                  :: ndim
   PetscScalar                              :: scaling
   
   Character (len=128)                      :: filename, CharBuffer
   
   Call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
   
   Call PetscOptionsGetInt(PETSC_NULL_CHARACTER, "-level", level, PETSC_NULL_INTEGER, ierr)
   Call PetscOptionsGetInt(PETSC_NULL_CHARACTER, "-n", numprocs, PETSC_NULL_INTEGER, ierr)
   Call PetscOptionsGetInt(PETSC_NULL_CHARACTER, "-nx", nx, PETSC_NULL_INTEGER, ierr)
   Call PetscOptionsGetInt(PETSC_NULL_CHARACTER, "-ny", ny, PETSC_NULL_INTEGER, ierr)
   nz = 1
   Call PetscOptionsGetInt(PETSC_NULL_CHARACTER, "-nz", nz, PETSC_NULL_INTEGER, ierr)
    
   N= nx * ny * nz
   
   Allocate(phi_array(N))
   Allocate(psi_array(N))

   !!! PHI
   phi_array = 0.0
   psi_array = 0.0      
   Do i = 0, numprocs-1
      Write(filename, 100) i, level
      scaling = i + 1.0
      
      Call PetscPrintf(PETSC_COMM_WORLD, 'Reading ' // filename, ierr)
      Call PetscPrintf(PETSC_COMM_WORLD, '\n'c, ierr)
      
      Open(Unit = 99, File = filename, status = 'Old')
      Rewind(99)
      Read(99,*) CharBuffer
      Read(99,*) CharBuffer
      Read(99,*) CharBuffer
      Read(99,*) CharBuffer
      
      Read(99, *) phi_array
      psi_array = psi_array + scaling * phi_array
      Close(99)
   End Do

   Write(filename, 110) level
   Open(Unit = 99, File = filename, status = 'Unknown')
   Rewind(99)
   Write(99, 200) 'Vec_0'
   Write(99, 200) 'part'
   Write(99, 200) '         1'  
   Write(99, 200) 'block'
   Do i = 1, N
      Write(99,300) psi_array(i)
   End Do
   Close(99)

   !!! U
   phi_array = 0.0
   psi_array = 0.0      
   Do i = 0, numprocs-1
      Write(filename, 101) i, level
      scaling = i + 1.0
      
      Call PetscPrintf(PETSC_COMM_WORLD, 'Reading ' // filename, ierr)
      Call PetscPrintf(PETSC_COMM_WORLD, '\n'c, ierr)
      
      Open(Unit = 99, File = filename, status = 'Old')
      Rewind(99)
      Read(99,*) CharBuffer
      Read(99,*) CharBuffer
      Read(99,*) CharBuffer
      Read(99,*) CharBuffer
      
      Read(99, *) phi_array
      psi_array = psi_array + phi_array
      Close(99)
   End Do

   Write(filename, 111) level
   Open(Unit = 99, File = filename, status = 'Unknown')
   Rewind(99)
   Write(99, 200) 'Vec_0'
   Write(99, 200) 'part'
   Write(99, 200) '         1'  
   Write(99, 200) 'block'
   Do i = 1, N
      Write(99,300) psi_array(i)
   End Do
   Close(99)

   !!! Case file
   Write(filename,131) level
   Open(File = filename, Unit=99, status = 'Unknown')
   Rewind(99)
   Write(99,200) 'FORMAT'
   Write(99,200) 'type:  ensight gold'
   Write(99,200) 'GEOMETRY'
   Write(99,130) level
   Write(99,200) 'VARIABLE'
   Write(99,120) level
   Write(99,121) level
   Close(99)

   Call PetscFinalize(ierr)

   
100 Format('Partition_Phi-',I3.3, '-level', I1, '.res')
101 Format('Partition_U-',I3.3, '-level', I1, '.res')
110 Format('Partition_Phi_all-level', I1, '.res')
111 Format('Partition_U_all-level', I1, '.res')
120 Format('scalar per node: U Partition_Phi_all-level', I1, '.res')
121 Format('scalar per node: PHI Partition_U_all-level', I1, '.res')
130 Format('model: Partition-level', I1, '.geo')
131 Format('Partition_all-level', I1, '.case')
200 Format(A)
300 Format(e12.5)
End Program MakeCompositeF90