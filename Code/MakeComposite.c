static char help[] = "Create a Composite Phi and U out of the individual fields, save it or display it\n\n";


#include <stdio.h>
#include <stdlib.h>

#include "petscvec.h"
#include "petscda.h"


extern PetscErrorCode ShowComposite_Phi(Vec phi, const char filename[]);
extern PetscErrorCode SaveComposite_Phi_TXT(Vec phi, const char filename[]);
extern PetscErrorCode SaveComposite_U_TXT(Vec u, const char filename[]);
extern PetscErrorCode SaveComposite_Phi_Ensight(Vec phi, const char filename[]);
extern PetscErrorCode SaveComposite_U_Ensight(Vec u, const char filename[]);

extern PetscErrorCode VecView_TXT(Vec x, const char filename[]);
extern PetscErrorCode VecView_VTKASCII(Vec x, const char filename[]);
extern PetscErrorCode VecView_EnsightASCII(Vec x, const char filename[]);
extern PetscErrorCode VecRead_TXT(Vec x, const char filename[]);


#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char ** argv) {
    PetscErrorCode	ierr;

    char			filename [ FILENAME_MAX ];
    const char		txt_sfx[] = ".txt";
    const char		raw_sfx[] = ".raw";
    const char		vtk_sfx[] = ".vtk";
    const char		geo_sfx[] = ".geo";
    const char		res_sfx[] = ".res";

    Vec             phi, psi;

    PetscTruth      SaveTXT     = PETSC_TRUE;
    PetscTruth      SaveVTK     = PETSC_FALSE;
    PetscTruth      SaveEnsight = PETSC_FALSE;

    int             i, N, numprocs, level, numlevels;
    int             nx, ny, nz, ndim=2;
    DA              da;
    
    PetscFunctionBegin;
    ierr = PetscInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);


    ierr = PetscOptionsGetTruth(PETSC_NULL, "-saveTXT", &SaveTXT, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-saveVTK", &SaveVTK, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-saveEnsight", &SaveEnsight, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL, "-level", &level, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL, "-n", &numprocs, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL, "-nx", &nx, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL, "-ny", &ny, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL, "-nz", &nz, PETSC_NULL); CHKERRQ(ierr);
    
    N = nx * ny * nz;
    if (nz>0) ndim=3;
    ierr = DACreate(PETSC_COMM_SELF, ndim, DA_NONPERIODIC, DA_STENCIL_STAR, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da); CHKERRQ(ierr);
    ierr = DACreateGlobalVector(da, &phi); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);

    ierr = VecZeroEntries(phi); CHKERRQ(ierr);
    ierr = VecZeroEntries(psi); CHKERRQ(ierr);    
    for (i=0; i<numprocs; i++){
        sprintf(filename, "Partition_U-%.3d-level%d.txt", i, level);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Reading %s\n", filename); CHKERRQ(ierr);
        ierr = VecRead_TXT(phi, filename); CHKERRQ(ierr);
//        ierr = VecAXPY(psi, 1.0, phi); CHKERRQ(ierr);
    }
    
    PetscFinalize();
    return 0;
    
    
    if (SaveTXT == PETSC_TRUE){
        sprintf(filename, "Partition_U_all-level%d.txt", level);
        ierr = VecView_TXT(psi, filename);
    }
    if (SaveEnsight == PETSC_TRUE){
        sprintf(filename, "Partition_U_all-level%d.res", level);
        ierr = VecView_EnsightASCII(psi, filename);
    }
    if (SaveVTK == PETSC_TRUE){
        sprintf(filename, "Partition_U_all-level%d.vtk", level);
        ierr = VecView_VTKASCII(psi, filename);
    }
    
    ierr = VecZeroEntries(phi); CHKERRQ(ierr);
    ierr = VecZeroEntries(psi); CHKERRQ(ierr);    
    for (i=0; i<numprocs; i++){
        sprintf(filename, "Partition_Phi-%.3d-level%d.txt", i, level);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Reading %s\n", filename); CHKERRQ(ierr);
        ierr = VecRead_TXT(phi, filename); CHKERRQ(ierr);
        ierr = VecAXPY(psi, (PetscScalar)i+1.0, phi); CHKERRQ(ierr);
    }
    if (SaveTXT == PETSC_TRUE){
        sprintf(filename, "Partition_Phi_all-level%d.txt", level);
        ierr = VecView_TXT(psi, filename);
    }
    if (SaveEnsight == PETSC_TRUE){
        sprintf(filename, "Partition_Phi_all-level%d.res", level);
        ierr = VecView_EnsightASCII(psi, filename);
    }
    if (SaveVTK == PETSC_TRUE){
        sprintf(filename, "Partition_Phi_all-level%d.vtk", level);
        ierr = VecView_VTKASCII(psi, filename);
    }
    PetscFinalize();
    return 0;
    
}





#undef __FUNCT__
#define __FUNCT__ "VecViewTXT"
PetscErrorCode VecView_TXT(Vec x, const char filename[]){
    Vec				 natural, io;
    VecScatter		 tozero;
    PetscMPIInt	     rank, size;
    int				 N;
    PetscScalar	     *io_array;
    DA				 da;
    PetscViewer	     viewer;
    int				 i, j, k, mx, my, mz;
    MPI_Comm         comm;
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;    
    ierr = PetscObjectGetComm((PetscObject) x, &comm); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
    
    ierr = VecGetSize(x, &N); CHKERRQ(ierr);
    
    ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da); CHKERRQ(ierr);
    if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
    ierr = DAGetInfo(da, 0, &mx, &my, &mz,0,0,0, 0,0,0,0); CHKERRQ(ierr);
    
    ierr = DACreateNaturalVector(da, &natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(da, x, INSERT_VALUES, natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(da, x, INSERT_VALUES, natural); CHKERRQ(ierr);
    
    ierr = VecScatterCreateToZero(natural, &tozero, &io); CHKERRQ(ierr);
    ierr = VecScatterBegin(tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterDestroy(tozero); CHKERRQ(ierr);
    ierr = VecDestroy(natural); CHKERRQ(ierr);
    

    ierr = VecGetArray(io, &io_array); CHKERRQ(ierr);	  
    if (!rank){
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
        for (k=0; k<mz; k++){
            for(i=0; i<mx; i++){
                for (j=0; j<my; j++){
                    ierr = PetscViewerASCIIPrintf(viewer, "%G	  ", PetscRealPart(io_array[k*my*mx + j*mx + i])); CHKERRQ(ierr);
                }
                ierr = PetscViewerASCIIPrintf(viewer, "\n"); CHKERRQ(ierr);
            }
            ierr =  PetscViewerASCIIPrintf(viewer, "\n"); CHKERRQ(ierr);
        }
        ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(io, &io_array); CHKERRQ(ierr);		

    /*
    if (!rank){
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
        ierr = VecView(io, viewer); CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    }
    */
    ierr = VecDestroy(io); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRead_TXT"
PetscErrorCode VecRead_TXT(Vec x, const char filename[]){
    Vec				 natural, io;
    VecScatter		 tozero;
    PetscMPIInt	     rank, size;
    int				 N;
    PetscScalar	     *io_array;
    DA				 da;
    FILE             *fp;
    int				 i, j, k, mx, my, mz;
    MPI_Comm         comm;
    PetscErrorCode   ierr;
    float            TmpBuf;
    
    
    PetscFunctionBegin;
    ierr = PetscObjectGetComm((PetscObject) x, &comm); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
    
    ierr = VecGetSize(x, &N); CHKERRQ(ierr);
    
    ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da); CHKERRQ(ierr);
    if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
    ierr = DAGetInfo(da, 0, &mx, &my, &mz,0,0,0, 0,0,0,0); CHKERRQ(ierr);
    
    ierr = DACreateNaturalVector(da, &natural); CHKERRQ(ierr);
    
    ierr = VecScatterCreateToZero(natural, &tozero, &io); CHKERRQ(ierr);
    ierr = VecGetArray(io, &io_array); CHKERRQ(ierr);	  
    
    ierr = PetscFOpen(comm, filename, "r", &fp); CHKERRQ(ierr);
    if (!rank){
        for (k=0; k<mz; k++){
            for(i=0; i<mx; i++){
                for (j =0; j<my; j++){
                    fscanf(fp, "%f	  ", &TmpBuf);
                    io_array[k*my*mx + j*mx + i] = (PetscScalar) TmpBuf;
                }
            }
        }
    }
    ierr = VecRestoreArray(io, &io_array); CHKERRQ(ierr);		
    
    ierr = VecScatterBegin(tozero, io, natural, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd  (tozero, io, natural, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterDestroy(tozero); CHKERRQ(ierr);
    
    ierr = DANaturalToGlobalBegin(da, natural, INSERT_VALUES, x); CHKERRQ(ierr);
    ierr = DANaturalToGlobalEnd  (da, natural, INSERT_VALUES, x); CHKERRQ(ierr);
    ierr = VecDestroy(natural); CHKERRQ(ierr);
    
    ierr = PetscFClose(comm, fp); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecView_VTKASCII"
PetscErrorCode VecView_VTKASCII(Vec x, const char filename[])
{
    MPI_Comm           comm;
    DA                 da;
    Vec                natural, master;
    PetscViewer        viewer;
    PetscScalar        *array, *values;
    PetscInt           n, N, maxn, mx, my, mz, dof;
    MPI_Status         status;
    PetscMPIInt        rank, size, tag;
    PetscErrorCode     ierr;
    VecScatter         ScatterToZero;
    const char         *name;

    PetscFunctionBegin;
    ierr = PetscObjectGetComm((PetscObject) x, &comm); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
    
    ierr = VecGetSize(x, &N); CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &n); CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da); CHKERRQ(ierr);
    if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
    
    ierr = DAGetInfo(da, 0, &mx, &my, &mz,0,0,0, &dof,0,0,0); CHKERRQ(ierr);
    if (dof!=1) SETERRQ(PETSC_ERR_ARG_WRONG,"dof>1 not implemented yet");
    
    
    ierr = PetscObjectGetName((PetscObject)x,&name); CHKERRQ(ierr);
    ierr = VecGetArray(x, &array); CHKERRQ(ierr);
    ierr = DACreateNaturalVector(da,&natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(da,x,INSERT_VALUES,natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(da,x,INSERT_VALUES,natural); CHKERRQ(ierr);
    ierr = VecScatterCreateToZero(natural, &ScatterToZero, &master); CHKERRQ(ierr);
    ierr = VecScatterBegin(ScatterToZero,natural,master,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(ScatterToZero,natural,master,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
    
    if (!rank) {
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "# vtk DataFile Version 2.0\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%s\n",name); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "ASCII\n"); CHKERRQ(ierr);
        
        /* Todo: get coordinates of nodes */
        ierr = PetscViewerASCIIPrintf(viewer, "DATASET STRUCTURED_POINTS\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "DIMENSIONS %d %d %d\n", mx, my, mz); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", N); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "SCALARS VecView_VTK float 1\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n"); CHKERRQ(ierr);
        
        ierr = VecView(master, viewer); CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    }  
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecView_EnsightASCII"
PetscErrorCode VecView_EnsightASCII(Vec x, const char filename[]){
    Vec               natural, io;
    VecScatter        tozero;
    PetscMPIInt       rank, size;
    int               N;
    PetscScalar       *io_array;
    DA                da;
    PetscViewer       viewer;
    PetscInt          i, j, k, mx, my, mz, dof;
    MPI_Comm          comm;
    PetscErrorCode    ierr;
    const char        *name;
   
    PetscFunctionBegin;
    ierr = PetscObjectGetComm((PetscObject) x, &comm); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
    
    ierr = VecGetSize(x, &N); CHKERRQ(ierr);
    
    ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da); CHKERRQ(ierr);
    if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
    ierr = DAGetInfo(da, 0, 0,0,0, 0,0,0, &dof, 0,0,0); CHKERRQ(ierr);
    if (dof!=1) SETERRQ(PETSC_ERR_ARG_WRONG,"dof>1 not implemented yet");
    
    ierr = PetscObjectGetName((PetscObject)x,&name);
    
    
    ierr = DACreateNaturalVector(da, &natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(da, x, INSERT_VALUES, natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(da, x, INSERT_VALUES, natural); CHKERRQ(ierr);
    
    ierr = VecScatterCreateToZero(natural, &tozero, &io); CHKERRQ(ierr);
    ierr = VecScatterBegin(tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterDestroy(tozero); CHKERRQ(ierr);
    ierr = VecDestroy(natural); CHKERRQ(ierr);
    
    ierr = VecGetArray(io, &io_array); CHKERRQ(ierr);	  
    
    if (!rank){
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
        
        ierr = PetscViewerASCIIPrintf(viewer, "%s\n",name); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "part\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%10d\n", 1); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "block\n"); CHKERRQ(ierr);
        
        for(i=0; i<N; i++){
            ierr = PetscViewerASCIIPrintf(viewer, "%12.5e\n", PetscRealPart(io_array[i])); CHKERRQ(ierr);
        }
        ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(io, &io_array); CHKERRQ(ierr);		
    ierr = VecDestroy(io); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

