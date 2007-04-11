static char help[] = "Compute a partition minimizing the sum of the first eigenvalue of each component in 2D\n\n";

/*
The matrix is scaled by a factor 2*hx*hy
*/

#include <stdio.h>

#include "petscksp.h"
#include "petscvec.h"
#include "petscda.h"
#include "slepceps.h"

typedef struct {
    PetscInt      nx, ny;
    PetscScalar   mu;
    DA            da;
} AppCtx;

extern PetscErrorCode ComputeK(AppCtx user, Mat K);
int main (int argc, char ** argv) {
    PetscErrorCode   ierr;
    AppCtx           user;   
    Mat              K;
    
    int              N;
    PetscTruth       flag;
    
    PetscMPIInt      numprocs, myrank;
    
    /* Eigenvalue solver stuff */
    EPS              eps;
    EPSType          type;
    ST               st;
    PetscScalar      st_shift = 0.0;
    STType           st_type  = STSINV; 
    int              its, maxit, nconv, nev;
    PetscScalar      tol, eigr, eigi;
    Vec              ur, ui;
    
    PetscLogDouble   eps_ts, eps_tf, eps_t;
    
    
    SlepcInitialize(&argc, &argv, (char*)0, help);

    user.nx = 10;
    PetscOptionsGetInt(PETSC_NULL, "-nx", &user.nx, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-ny", &user.ny, &flag);  
    if( flag==PETSC_FALSE ) user.ny=user.nx;
    N = user.nx*user.ny;
    PetscOptionsGetScalar(PETSC_NULL, "-mu", &user.mu, PETSC_NULL);
    
    PetscPrintf(PETSC_COMM_WORLD, "\nOptimal Partition problem, N=%d (%dx%d grid)\n\n", 
                N, user.nx, user.ny);

    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    
    DACreate2d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_STAR, user.nx, user.ny, PETSC_DECIDE,
               PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &user.da);
    
    DAGetMatrix(user.da, MATMPIAIJ, &K);
    
    ComputeK(user, K);
//    MatView(K, PETSC_VIEWER_DRAW_WORLD);
    
    /* Create the eigensolver context */
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, K, PETSC_NULL);
    EPSSetProblemType(eps, EPS_HEP);
    EPSGetST(eps, &st);
    
    STSetType(st, st_type);
    STSetShift(st, st_shift);
    
    STSetFromOptions(st);
    EPSSetFromOptions(eps);
    
    /* solve the eigenvalue problem */
    PetscGetTime(&eps_ts);
    EPSSolve(eps);
    PetscGetTime(&eps_tf);
    
    eps_t = eps_tf - eps_ts;
    PetscPrintf(PETSC_COMM_WORLD, " Time spent on EPSSolve : %f sec\n", eps_t);
    EPSGetIterationNumber(eps, &its);
    PetscPrintf(PETSC_COMM_WORLD," Number of iterations of eigenvalue solver: %d\n",its);
    
    /* Display some information obout the EPS solver */
    EPSGetType(eps,&type);
    PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);
    EPSGetDimensions(eps,&nev,PETSC_NULL);
    PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %d\n",nev);
    EPSGetTolerances(eps,&tol,&maxit);
    PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%d\n",tol,maxit);
    EPSGetConverged(eps,&nconv);
    PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %d\n\n",nconv);

    for (its=0; its<nconv; its++){
        EPSGetValue(eps, its, &eigr, &eigi);
        PetscPrintf(PETSC_COMM_WORLD, "[%d] eigenvalue: %9f + %9f i\n", its, eigr, eigi); 
    }
    
/*    MatGetVecs(K, PETSC_NULL, &ur);
    MatGetVecs(K, PETSC_NULL, &ui);
    
    EPSGetEigenpair(eps, nconv-1 , &eigr, &eigi, ur, ui);
    VecView(ur, PETSC_VIEWER_DRAW_WORLD);
*/    
    eigr = eigr * (PetscReal)(user.nx-1) * (PetscReal)(user.ny-1) / 2.0; 
    eigi = eigi * (PetscReal)(user.nx-1) * (PetscReal)(user.ny-1) / 2.0; 

    PetscPrintf(PETSC_COMM_WORLD, "Smallest computed eigenvalue: %9f + %9f i\n", eigr, eigi); 


    MatDestroy(K);
    DADestroy(user.da);           
    SlepcFinalize();
}


PetscErrorCode ComputeK(AppCtx user, Mat K)
{
    DA             da = user.da;
    PetscErrorCode ierr;
    PetscInt       i, j, mx, my, xm, ym, xs, ys;
    PetscScalar    v[5],Hx,Hy,HxdHy,HydHx;
    MatStencil     row,col[5];
    
    Hx = 1.0 / (PetscReal)(user.nx-1); 
    Hy = 1.0 / (PetscReal)(user.ny-1);
    HxdHy = Hx/Hy; HydHx = Hy/Hx;
    DAGetCorners(da, &xs, &ys, PETSC_NULL, &xm, &ym,PETSC_NULL);
    PetscPrintf(PETSC_COMM_SELF, "xs=%d xm=%d ys=%d ym=%d\n", xs, xm, ys, ym);

    for (j=ys; j<ys+ym; j++){
        for(i=xs; i<xs+xm; i++){
            row.i = i; row.j = j;
	       if (i==0 || j==0 || i==user.nx-1 || j==user.ny-1){
                v[0] = 2.0*(HxdHy + HydHx);
                MatSetValuesStencil(K,1,&row,1,&row,v,INSERT_VALUES);
            } else {
                v[0] = -HxdHy; col[0].i = i;   col[0].j = j-1;
                v[1] = -HydHx; col[1].i = i-1; col[1].j = j;
                v[2] = 2.0*(HxdHy + HydHx); col[2].i = row.i; col[2].j = row.j;
                v[3] = -HydHx; col[3].i = i+1; col[3].j = j;
                v[4] = -HxdHy; col[4].i = i;   col[4].j = j+1;
                MatSetValuesStencil(K, 1, &row, 5, col, v, INSERT_VALUES);
        }
      }
    }
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    return 0;
}
