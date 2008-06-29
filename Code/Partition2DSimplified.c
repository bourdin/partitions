static char help[] = "Compute a partition minimizing the sum of the first eigenvalue of each component in 2D\n\n";

/*
The matrix is scaled by a factor 2*hx*hy
*/

#include <stdio.h>
#include <stdlib.h>

#include "petscksp.h"
#include "petscvec.h"
#include "petscda.h"
#include "slepceps.h"

typedef struct {
	 PetscInt		nx, ny; /* Dimension of the discretized domain */
	 PetscScalar	mu;	  /* Penalization factor for the computation of the eigenvalue */
	 PetscTruth		per;	  /* true for periodic boundary conditions, false otherwise */
	 DA				da;	  /* Information about the distributed layout */
	 PetscScalar	step;	  /* Initial step of the steepest descent methods */
	 EPS				eps;	  /* Eigenvalue solver context */
	 Mat				K;		  /* Matrix for the Laplacian */
	 PetscInt      epsnum; /* which eigenvalues are we optimizing */
} AppCtx;

extern PetscErrorCode DistanceFromSimplex(PetscScalar *dist, Vec phi); 
extern PetscErrorCode ComputeK(AppCtx user, Vec phi);
extern PetscErrorCode ComputeLambdaU(AppCtx user, Vec phi, PetscScalar *lambda, Vec u);
extern PetscErrorCode ComputeG(AppCtx user, Vec G, Vec u);
extern PetscErrorCode InitPhiQuarter(AppCtx user, Vec phi);
extern PetscErrorCode InitPhiRandom(AppCtx user, Vec phi);
extern PetscErrorCode VecView_TXT(Vec x, const char filename[]);
extern PetscErrorCode VecView_RAW(Vec x, const char filename[]);
extern PetscErrorCode VecView_VTKASCII(Vec x, const char filename[]);
extern PetscErrorCode DAView_GEOASCII(DA da, const char filename []);
extern PetscErrorCode VecView_EnsightASCII(Vec x, const char filename[]);
extern PetscErrorCode SimplexProjection(AppCtx user, Vec x);
extern PetscErrorCode SimplexInteriorProjection(AppCtx user, Vec x);
extern PetscErrorCode SimplexProjection2(AppCtx use, Vec x);
extern PetscErrorCode VecRead_TXT(Vec x, const char filename[]);
      


#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char ** argv) {
    PetscErrorCode	ierr;
    AppCtx			user;	  
    Vec				phi;
    Vec				u, G, Gproj, psi, vec_one, phi2, phi2sum;
    PetscScalar      *phi2_array, *phi2sum_array;
    
    PetscScalar		lambda, F, Fold;
    PetscScalar		stepmax = 1.0e4;
    PetscScalar		stepmin = 1.0e-5;
    PetscScalar		error, tol = 1.0e-3;
    const char		u_prfx[] = "Partition_U-";
    const char		phi_prfx[] = "Partition_Phi-";
    char			filename [ FILENAME_MAX ];
    const char		txtsfx[] = ".txt";
    const char		rawsfx[] = ".raw";
    const char		vtksfx[] = ".vtk";
    const char		geosfx[] = ".geo";
    const char		ressfx[] = ".res";
    
    PetscScalar     *phi_array, *psi_array;
    PetscScalar     muinit, mufinal;
    PetscScalar     GNorm;
        
    int				N, i, it;
    PetscInt        maxit = 1000;
    PetscTruth		flag;
    
    PetscMPIInt		numprocs, myrank;
    PetscViewer     viewer;
    
    /* Eigenvalue solver stuff */
    EPSType			type;
    ST				st;
    PetscScalar		st_shift = 0.0;
    STType			st_type	= STSINV; 
    int				its;
    KSP				eps_ksp;
    PC				eps_pc;
    PetscTruth      printhelp, restart;
    
    PetscLogDouble	eps_ts, eps_tf, eps_t;
    PetscReal       dist;
    
    
    SlepcInitialize(&argc, &argv, (char*)0, help);
    
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    
    
    user.epsnum = 1;
    PetscOptionsGetInt(PETSC_NULL, "-epsnum", &user.epsnum, PETSC_NULL);
    
    PetscOptionsGetInt(PETSC_NULL, "-maxit", &maxit, PETSC_NULL);
    
    PetscOptionsGetScalar(PETSC_NULL, "-tol", &tol, PETSC_NULL);
    
    user.nx = 10;
    PetscOptionsGetInt(PETSC_NULL, "-nx", &user.nx, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-ny", &user.ny, &flag);	 
    if( flag==PETSC_FALSE ) user.ny=user.nx;
    N = user.nx*user.ny;
    PetscOptionsGetScalar(PETSC_NULL, "-muinit", &muinit, PETSC_NULL);
    mufinal = muinit;
    
    PetscOptionsGetScalar(PETSC_NULL, "-mufinal", &mufinal, PETSC_NULL);
    user.mu = muinit;
    
    user.step = 10.0;
    stepmin   = user.step;
    PetscOptionsGetScalar(PETSC_NULL, "-step", &user.step, PETSC_NULL);
   
    if (numprocs==1) {
        PetscPrintf(PETSC_COMM_WORLD, "\nCannot partition in less than 2 subsets! ");
        PetscPrintf(PETSC_COMM_WORLD, "\nRestart on more than 1 cpu");
        SlepcFinalize();
        return -1;
    }		
   
    user.per = PETSC_FALSE;
    PetscOptionsGetTruth(PETSC_NULL, "-periodic", &user.per, PETSC_NULL);
    
    restart = PETSC_FALSE;
    PetscOptionsGetTruth(PETSC_NULL, "-restart", &restart, PETSC_NULL);
    
    PetscPrintf(PETSC_COMM_WORLD, "\nOptimal Partition problem, N=%d (%dx%d grid)\n\n", 
      			 N, user.nx, user.ny);
    PetscLogPrintSummary(MPI_COMM_WORLD,"petsc_log_summary.log");	  
    
    if (user.per) {
        PetscPrintf(PETSC_COMM_WORLD, "Using periodic boundary conditions\n");
        DACreate2d(PETSC_COMM_SELF, DA_XYPERIODIC, DA_STENCIL_STAR, user.nx, user.ny,
     				PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &user.da);
    }
    else {
        PetscPrintf(PETSC_COMM_WORLD, "Using non-periodic boundary conditions\n");
        DACreate2d(PETSC_COMM_SELF, DA_NONPERIODIC, DA_STENCIL_STAR, user.nx, user.ny,
                     PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &user.da);
    }
      
    DAGetMatrix(user.da, MATSEQAIJ, &user.K);
    DACreateGlobalVector(user.da, &phi);
    VecDuplicate(phi, &u);
    VecDuplicate(phi, &G);
    VecDuplicate(phi, &psi);
    VecDuplicate(phi, &vec_one);
    VecDuplicate(phi, &phi2);
    
    VecDuplicate(phi, &phi2sum);
    
    VecSet(vec_one, (PetscScalar) 1.0);
    
    /* Create the eigensolver context */
    EPSCreate(PETSC_COMM_SELF, &user.eps);
    
    EPSSetOperators(user.eps, user.K, PETSC_NULL);
    EPSSetProblemType(user.eps, EPS_HEP);
    EPSGetST(user.eps, &st);
    EPSSetDimensions(user.eps, user.epsnum, 5*user.epsnum);
    
    STSetType(st, st_type);
    STSetShift(st, st_shift);
    
    STGetKSP(st, &eps_ksp);
    KSPGetPC(eps_ksp, &eps_pc);
    
    PCSetType(eps_pc, PCICC);
    KSPSetType(eps_ksp, KSPCG);
    
    STSetFromOptions(st);
    EPSSetFromOptions(user.eps);
    
    if (!restart){
        InitPhiRandom(user, phi);
        VecScale(phi, (PetscScalar) 1.0 / (PetscScalar) numprocs);
//        SimplexInteriorProjection(user, phi);
        SimplexProjection2(user, phi);
//        sprintf(filename, "%s%.3d%s", phi_prfx, myrank, txtsfx);
//        VecView_TXT(phi, filename);
    } else {
        sprintf(filename, "%s%.3d%s", phi_prfx, myrank, txtsfx);
        VecRead_TXT(phi, filename);
    }
    
    
	// Save .geo and .case file
	/*
    if (!myrank){
        DAView_GEOASCII(user.da, "Partition.geo");	 
        PetscViewerASCIIOpen(PETSC_COMM_SELF, "Partition.case", &viewer);
        PetscViewerASCIIPrintf(viewer, "FORMAT\n");
        PetscViewerASCIIPrintf(viewer, "type:  ensight gold\n");
        PetscViewerASCIIPrintf(viewer, "GEOMETRY\n");
        PetscViewerASCIIPrintf(viewer, "model: Partition.geo\n");
        PetscViewerASCIIPrintf(viewer, "VARIABLE\n");
        for (i=0; i<numprocs; i++){
            PetscViewerASCIIPrintf(viewer, "scalar per node: U%i %s%.3d%s\n", i, u_prfx, i, ressfx);
        }
        for (i=0; i<numprocs; i++){
            PetscViewerASCIIPrintf(viewer, "scalar per node: PHI%i %s%.3d%s\n", i, phi_prfx, i, ressfx);
        }
        PetscViewerFlush(viewer);
        PetscViewerDestroy(viewer);
    }
    */
    
    F = 0.0;
    Fold = 0.0;
    it = 0;
    PetscPrintf(PETSC_COMM_WORLD, "Iteration %d:\n", it);
    ComputeLambdaU(user, phi, &lambda, u);
    
    MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    
    error = tol + 1.0;
    
    it = 0.0;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "Partition.log", &viewer);
    PetscViewerASCIIPrintf(viewer, "%d   %e   ", it, F);
    PetscViewerASCIISynchronizedPrintf(viewer, "%e   ", lambda);
    PetscViewerFlush(viewer);
    PetscViewerASCIIPrintf(viewer, "%e \n", tol, it);
    
    do { 
        it++;
        
        // Update mu linearly between muinit and mufinal in maxit/2 steps, starting from iteration 0
        //user.mu = PetscMin(muinit + (PetscScalar) (2*it) / (PetscScalar) maxit * (mufinal - muinit), mufinal);
        		
        Fold = F;
        F = 0.0;
        PetscPrintf(PETSC_COMM_WORLD, "Iteration %d:\n", it);
        
        // Compute the gradient of the objective function w.r.t. u
        ComputeG(user, G, u);
        
        // Update phi
        VecAXPY(phi, user.step, G);
        
        
        // Project phi onto the simplex \sum_k \phi^k_i=1 for i = 0-nx*ny-1
        SimplexProjection2(user, phi);
 
        // Compute the distance the simplex:
        DistanceFromSimplex(&dist, phi);
        PetscPrintf(PETSC_COMM_WORLD, "   Distance from simplex: %f\n", dist);

        //Compute the eigenvalues u associated to the new phi
        ComputeLambdaU(user, phi, &lambda, u);
        
        //compute F= \sum_k lambda^k_epsnum
        MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
        
        
        // Update the step

        if (F<=Fold) {
            user.step = user.step * 1.2;
            user.step = PetscMin(user.step, stepmax);
        } else {
            user.step = user.step / 2.0;
            user.step = PetscMax(user.step, stepmin);
        }

        //Display some stuff
		PetscPrintf(PETSC_COMM_WORLD, "F = %e, step = %e, error = %e, mu = %e\n\n", F, user.step, error, user.mu);

        //Save the same stuff in "Partition.log"
        PetscViewerASCIIPrintf(viewer, "%d   %e   ", it, F);
        PetscViewerASCIISynchronizedPrintf(viewer, "%e   ", lambda);
        PetscViewerFlush(viewer);
        PetscViewerASCIIPrintf(viewer, "%e \n", error, it);
      
      
      
        // Saves the results
        if (it%25 == 0){
            // Save into a new file
            //sprintf(filename, "%s%.3d-%.5d%s", u_prfx, myrank, it, txtsfx);
            			 
            // Reuse the same file over and over
//            sprintf(filename, "%s%.3d%s", u_prfx, myrank, txtsfx);
//            VecView_TXT(u, filename);
               		
            // Save in VTK format
            // sprintf(filename, "%s%.3d%s", u_prfx, myrank, vtksfx);
            // VecView_VTKASCII(u, filename);
            
            // Save in ensight gold ASCII format
//            sprintf(filename, "%s%.3d%s", u_prfx, myrank, ressfx);
//            VecView_EnsightASCII(u, filename);
            
            //sprintf(filename, "%s%.3d-%.5d%s", phi_prfx, myrank, it, txtsfx);
//            sprintf(filename, "%s%.3d%s", phi_prfx, myrank, txtsfx);
//            VecView_TXT(phi, filename);
               		
            // sprintf(filename, "%s%.3d%s", phi_prfx, myrank, vtksfx);
            // VecView_VTKASCII(phi, filename);
            
            // Save in ensight gold ASCII format
//            sprintf(filename, "%s%.3d%s", phi_prfx, myrank, ressfx);
//            VecView_EnsightASCII(phi, filename);

            // Save a composite of all PHI U
            sprintf(filename, "Partition_Phi_all.txt");
            VecCopy(phi, phi2);
            VecScale(phi2, (PetscScalar) myrank+1.0);
            VecGetArray(phi2, &phi_array);
            VecGetArray(psi,  &psi_array);            
            MPI_Allreduce(phi_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
            VecRestoreArray(phi2, &phi_array);
            VecRestoreArray(psi,  &psi_array);
            VecView_TXT(psi, filename);

            sprintf(filename, "Partition_U_all.txt");
            VecGetArray(u, &phi_array);
            VecGetArray(psi,  &psi_array);            
            MPI_Allreduce(phi_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
            VecRestoreArray(phi2, &phi_array);
            VecRestoreArray(psi,  &psi_array);
            VecView_TXT(psi, filename);
        }
    } while ( (it < 20 ) || ( ( it < maxit ) && (error > tol) ) );
   
    // sprintf(filename, "%s%.3d%s", u_prfx, myrank, vtksfx);
    // VecView_VTKASCII(u, filename);
    sprintf(filename, "%s%.3d%s", u_prfx, myrank, txtsfx);
    VecView_TXT(u, filename);
    sprintf(filename, "%s%.3d%s", u_prfx, myrank, ressfx);
    VecView_EnsightASCII(u, filename);
    
    // sprintf(filename, "%s%.3d%s", phi_prfx, myrank, vtksfx);
    // VecView_VTKASCII(phi, filename);
    sprintf(filename, "%s%.3d%s", phi_prfx, myrank, txtsfx);
    VecView_TXT(phi, filename);
    sprintf(filename, "%s%.3d%s", phi_prfx, myrank, ressfx);
    VecView_EnsightASCII(phi, filename);
      
    // Be nice and deallocate
    VecDestroy(phi2);
    VecDestroy(phi2sum);
    VecDestroy(phi);
    VecDestroy(psi);
    VecDestroy(u);
    VecDestroy(G);
    MatDestroy(user.K);
    DADestroy(user.da);	 
    EPSDestroy(user.eps);
    
    // Same informations on the run (including command line options)
    PetscLogPrintSummary(MPI_COMM_WORLD,"petsc_log_summary.log");		
    PetscViewerDestroy(viewer);
    
    SlepcFinalize();
}


#undef __FUNCT__
#define __FUNCT__ "ComputeK"
PetscErrorCode ComputeK(AppCtx user, Vec phi)
{
    Mat				 K	 = user.K;
    DA				 da = user.da;
    PetscErrorCode   ierr;
    PetscInt		 i, j, xm, ym, xs, ys;
    PetscScalar	     v[5],Hx,Hy,HxdHy,HydHx;
    MatStencil		 row,col[5];
    PetscScalar	     **local_phi;
    
    PetscFunctionBegin;
    ierr = DAVecGetArray(user.da, phi, &local_phi); CHKERRQ(ierr);
    
    Hx = 1.0 / (PetscReal)(user.nx-1); 
    Hy = 1.0 / (PetscReal)(user.ny-1);
    HxdHy = Hx/Hy; HydHx = Hy/Hx;
    ierr = DAGetCorners(da, &xs, &ys, PETSC_NULL, &xm, &ym,PETSC_NULL); CHKERRQ(ierr);

    for (j=ys; j<ys+ym; j++){
        for(i=xs; i<xs+xm; i++){
            row.i = i; row.j = j;
            if ( (!user.per) && (i==0 || j==0 || i==user.nx-1 || j==user.ny-1)){
                v[0] = 2.0*(HxdHy + HydHx) + user.mu * (1.0 - local_phi[j][i]);
                ierr = MatSetValuesStencil(K,1,&row,1,&row,v,INSERT_VALUES); CHKERRQ(ierr);
            } else {
                v[0] = -HxdHy; col[0].i = i;	  col[0].j = j-1;
                v[1] = -HydHx; col[1].i = i-1; col[1].j = j;
                v[2] = 2.0*(HxdHy + HydHx); col[2].i = row.i; col[2].j = row.j;
                v[2] += user.mu * (1.0 - local_phi[j][i])*Hx*Hy;
                v[3] = -HydHx; col[3].i = i+1; col[3].j = j;
                v[4] = -HxdHy; col[4].i = i;	  col[4].j = j+1;
                ierr = MatSetValuesStencil(K, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }
    ierr = DAVecRestoreArray(user.da, phi, &local_phi); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeLambdaU"
PetscErrorCode ComputeLambdaU(AppCtx user, Vec phi, PetscScalar *lambda, Vec u){
    PetscLogDouble	  eps_ts, eps_tf, eps_t;
    int              its;
    Vec              ui;
    PetscScalar      eigi, normu;
    int              nconv;
    PetscInt         myrank;
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    
    ierr = ComputeK(user, phi); CHKERRQ(ierr);
    ierr = EPSSetOperators(user.eps, user.K, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscGetTime(&eps_ts); CHKERRQ(ierr);
    ierr = EPSSolve(user.eps); CHKERRQ(ierr);
    ierr = PetscGetTime(&eps_tf); CHKERRQ(ierr);
    eps_t = eps_tf - eps_ts;
    ierr = EPSGetIterationNumber(user.eps, &its); CHKERRQ(ierr);
    
    
    ierr = VecDuplicate(u, &ui); CHKERRQ(ierr);
    ierr = EPSGetConverged(user.eps, &nconv); CHKERRQ(ierr);
    
    ierr = EPSGetEigenpair(user.eps, nconv-user.epsnum , lambda, &eigi, u, ui); CHKERRQ(ierr);
    ierr = VecNorm(u, NORM_2, &normu); CHKERRQ(ierr);
    normu = 1.0 / normu;
    ierr = VecScale(u, normu); CHKERRQ(ierr);
    
    ierr = VecDestroy(ui); CHKERRQ(ierr);
    
    *lambda = *lambda * (PetscReal)(user.nx-1) * (PetscReal)(user.ny-1) / 2.0; 
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "        lambda[%d] = %e    EPSSolve converged in %f s for %d iterations\n", myrank, *lambda, eps_t, its); CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitPhiQuarter"
extern PetscErrorCode InitPhiQuarter(AppCtx user, Vec phi){
    PetscScalar     **local_phi;
    PetscInt        i, j, mx, my, xm, ym, xs, ys;
    PetscScalar     zero = 0.0, one	= 1.0;
    PetscErrorCode  ierr;
    
    PetscFunctionBegin;
    ierr = DAGetCorners(user.da, &xs, &ys, PETSC_NULL, &xm, &ym,PETSC_NULL); CHKERRQ(ierr);

    ierr = VecSet(phi, zero); CHKERRQ(ierr);
    ierr = DAVecGetArray(user.da, phi, &local_phi); CHKERRQ(ierr);
    for (j=ys; j<ys+ym; j++){
        for(i=xs; i<xs+xm; i++){
            if ( (i < user.nx/2) || (j < user.ny/2) ){
                local_phi[j][i] = one;
            }
        }
    }
    ierr = DAVecRestoreArray(user.da, phi, &local_phi); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitPhiRandom"
extern PetscErrorCode InitPhiRandom(AppCtx user, Vec phi){
    PetscRandom	   rndm;
    PetscLogDouble tim;
    PetscMPIInt    rank;
    MPI_Comm       comm;
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = PetscObjectGetComm((PetscObject) phi, &comm); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    
    ierr = PetscRandomCreate(comm, &rndm); CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rndm); CHKERRQ(ierr);
    ierr = PetscGetTime(&tim); CHKERRQ(ierr);
//    ierr = PetscPrintf(PETSC_COMM_SELF, "Seed is %d\n", (unsigned long) rank*tim); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Seed is %d\n", (unsigned long) rank*1213585940); CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(rndm, (unsigned long) tim*rank); CHKERRQ(ierr);
    ierr = PetscRandomSeed(rndm); CHKERRQ(ierr);
    
    ierr = VecSetRandom(phi, rndm); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(rndm); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Computeg"
extern PetscErrorCode ComputeG(AppCtx user, Vec G, Vec u){
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;
    ierr = VecPointwiseMult(G, u, u); CHKERRQ(ierr);
    PetscFunctionReturn(0);
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
#define __FUNCT__ "VecView_Raw"
PetscErrorCode VecView_RAW(Vec x, const char filename[]){
    Vec             natural, io;
    VecScatter      tozero;
    PetscMPIInt	    myrank;
    int				N;
    DA				da;
    PetscViewer	    viewer;
    int				i, j, mx, my;
    PetscErrorCode  ierr;
    
    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    ierr = VecGetSize(x, &N); CHKERRQ(ierr);
    
    ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da); CHKERRQ(ierr);
    if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
    ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0, 0,0,0,0); CHKERRQ(ierr);
    
    ierr = DACreateNaturalVector(da, &natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(da, x, INSERT_VALUES, natural); CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd  (da, x, INSERT_VALUES, natural); CHKERRQ(ierr);
    
    ierr = VecScatterCreateToZero(natural, &tozero, &io); CHKERRQ(ierr);
    ierr = VecScatterBegin(tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd  (tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterDestroy(tozero); CHKERRQ(ierr);
    ierr = VecDestroy(natural); CHKERRQ(ierr);
    
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = VecView(io,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    ierr = VecDestroy(io); CHKERRQ(ierr);
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
    PetscInt           xs, xm, ys, ym;
    PetscInt           i, p;
    MPI_Status         status;
    PetscMPIInt        rank, size, tag;
    PetscErrorCode     ierr;
    VecScatter         ScatterToZero;
    const char         *name;

    PetscFunctionBegin;
    ierr = PetscObjectGetComm((PetscObject) x, &comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    
    ierr = VecGetSize(x, &N); CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &n); CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da);CHKERRQ(ierr);
    if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
    
    ierr = DAGetInfo(da, 0, &mx, &my, &mz,0,0,0, &dof,0,0,0);CHKERRQ(ierr);
    if (dof!=1) SETERRQ(PETSC_ERR_ARG_WRONG,"dof>1 not implemented yet");
    
    
    ierr = PetscObjectGetName((PetscObject)x,&name);
    ierr = VecGetArray(x, &array);CHKERRQ(ierr);
    ierr = DACreateNaturalVector(da,&natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(da,x,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(da,x,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = VecScatterCreateToZero(natural, &ScatterToZero, &master);CHKERRQ(ierr);
    ierr = VecScatterBegin(ScatterToZero,natural,master,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ScatterToZero,natural,master,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    
    if (!rank) {
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%s\n",name);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "ASCII\n");CHKERRQ(ierr);
        
        /* Todo: get coordinates of nodes */
        ierr = PetscViewerASCIIPrintf(viewer, "DATASET STRUCTURED_POINTS\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "DIMENSIONS %d %d %d\n", mx, my, mz);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", N);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "SCALARS VecView_VTK float 1\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
        
        ierr = VecView(master, viewer);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }  
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DAView_GEOASCII"
PetscErrorCode DAView_GEOASCII(DA da, const char filename []){
    PetscInt          mx, my, mz;
    PetscMPIInt       rank, size;
    PetscViewer       viewer;
    MPI_Comm          comm;
    PetscErrorCode    ierr;
    const char        *name;
   
    PetscFunctionBegin
    ierr = PetscObjectGetComm((PetscObject) da, &comm); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
    
    ierr = DAGetInfo(da, 0, &mx, &my, &mz,0,0,0, 0,0,0,0); CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) da, &name); CHKERRQ(ierr);
    if (!rank){
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%s\n", name); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "Generated by DAView_GEOASCII\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "node id given\nelement id given\nextents\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e%12.5e\n", (PetscScalar) 0, (PetscScalar) mx-1); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e%12.5e\n", (PetscScalar) 0, (PetscScalar) my-1); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e%12.5e\n", (PetscScalar) 0, (PetscScalar) mz-1); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "part\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%10d\n", 1); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%s\n", name); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "block uniform\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%10d%10d%10d\n", mx, my, mz); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e\n", 0.0); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e\n", 0.0); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e\n", 0.0); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e\n", 1.0); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e\n", 1.0); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%12.5e\n", 1.0); CHKERRQ(ierr);
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
        
        ierr = PetscViewerASCIIPrintf(viewer, "%s\n",name);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "part\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%10d\n", 1);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "block\n");CHKERRQ(ierr);
        
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

#undef __FUNCT__
#define __FUNCT__ "DIstanceFromSimplex"
PetscErrorCode DistanceFromSimplex(PetscScalar *dist, Vec phi)
{
    PetscMPIInt      myrank, numprocs;
    PetscScalar      *phi_array, *psi_array;
    Vec              psi, one;
    PetscInt         N;
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;    
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &one); CHKERRQ(ierr);
    ierr = VecSet(one, (PetscScalar) 1.0); CHKERRQ(ierr);
    ierr = VecGetArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecGetArray(phi, &phi_array); CHKERRQ(ierr);
    ierr = VecGetSize(psi, &N); CHKERRQ(ierr);
    
    MPI_Allreduce(phi_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    
    ierr = VecRestoreArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(phi, &phi_array); CHKERRQ(ierr);

    ierr = VecAXPY(psi, -1.0, one); CHKERRQ(ierr);
    
    ierr = VecScale(psi, (PetscReal) 1.0/N); CHKERRQ(ierr);
    ierr = VecNorm(psi, NORM_2, dist); CHKERRQ(ierr);
    ierr = VecDestroy(one); CHKERRQ(ierr);
    ierr = VecDestroy(psi); CHKERRQ(ierr);

    PetscFunctionReturn(0);    
}    

#undef __FUNCT__
#define __FUNCT__ "SimplexProjection2"
PetscErrorCode SimplexProjection2(AppCtx user, Vec phi)
{
    PetscMPIInt      myrank, numprocs;
    PetscScalar      *phi_array, *psi_array;
    Vec              psi, one;
    PetscInt         N;
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;    
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &one); CHKERRQ(ierr);
    ierr = VecSet(one, (PetscScalar) 1.0); CHKERRQ(ierr);
    ierr = VecGetArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecGetArray(phi, &phi_array); CHKERRQ(ierr);
    ierr = VecGetSize(psi, &N); CHKERRQ(ierr);
    
    MPI_Allreduce(phi_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    
    ierr = VecRestoreArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(phi, &phi_array); CHKERRQ(ierr);
    
    ierr = VecPointwiseMax(psi, psi, one); CHKERRQ(ierr);
    ierr = VecPointwiseDivide(phi, phi, psi); CHKERRQ(ierr);
    
    ierr = VecDestroy(psi); CHKERRQ(ierr);
    ierr = VecDestroy(one); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SimplexProjection"
PetscErrorCode SimplexProjection(AppCtx user, Vec phi)
{
    PetscMPIInt       *I, *n;
    Vec               psi;
    PetscScalar       *psi_array, *phi_array;
    PetscInt          l, i;
    PetscMPIInt		  myrank, numprocs;
    PetscErrorCode    ierr;
    
    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);
    ierr = VecGetArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecGetArray(phi, &phi_array); CHKERRQ(ierr);
    
    ierr = PetscMalloc(user.nx*user.ny*sizeof(PetscMPIInt), &I); CHKERRQ(ierr);
    ierr = PetscMalloc(user.nx*user.ny*sizeof(PetscMPIInt), &n); CHKERRQ(ierr);
    for (i=0; i<user.nx*user.ny; i++) I[i] = 0;
    
    for (l=0; l<numprocs; l++){
        for (i=0; i<user.nx*user.ny; i++){
            if (I[i]) phi_array[i] = 0.0;
        }
        MPI_Allreduce(phi_array, psi_array, user.nx * user.ny, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
        MPI_Allreduce(I, n, user.nx * user.ny, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
        for (i=0; i<user.nx*user.ny; i++){
            if (numprocs - n[i]) phi_array[i] = phi_array[i]- (PetscScalar) (1-I[i]) * (psi_array[i]-1.0) / (PetscScalar) (numprocs - n[i]);     
        }
        for (i=0; i<user.nx*user.ny; i++){
            if (phi_array[i] <0.0) {
            I[i]  = 1;
                phi_array[i]= (PetscScalar) 0.0;
            }
        }
    }
    ierr = VecRestoreArray(phi, &phi_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecDestroy(psi); CHKERRQ(ierr);
    ierr = PetscFree(I); CHKERRQ(ierr);
    ierr = PetscFree(n); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SimplexInteriorProjection"
PetscErrorCode SimplexInteriorProjection(AppCtx user, Vec phi)
{
    PetscMPIInt       *I, *n;
    Vec               psi, phi_in;
    PetscScalar       *psi_array, *phi_array, *phi_in_array;
    PetscInt          l, i, N;
    PetscMPIInt		  myrank, numprocs;
    PetscErrorCode    ierr;
    
    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    ierr = VecGetSize(phi, &N); CHKERRQ(ierr);
    
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &phi_in); CHKERRQ(ierr);
    ierr = VecCopy(phi, phi_in); CHKERRQ(ierr);
    
    ierr = VecGetArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecGetArray(phi, &phi_array); CHKERRQ(ierr);
    ierr = VecGetArray(phi_in, &phi_in_array); CHKERRQ(ierr);
    
    ierr = PetscMalloc(N*sizeof(PetscMPIInt), &I); CHKERRQ(ierr);
    ierr = PetscMalloc(N*sizeof(PetscMPIInt), &n); CHKERRQ(ierr);
    for (i=0; i<N; i++) I[i] = 0;
    
    for (l=0; l<numprocs; l++){
        for (i=0; i<N; i++){
            if (I[i]) phi_array[i] = 0.0;
        }
        MPI_Allreduce(phi_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
        MPI_Allreduce(I, n, N, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
        for (i=0; i<N; i++){
            if (numprocs - n[i]) phi_array[i] = phi_array[i]- (PetscScalar) (1-I[i]) * (psi_array[i]-1.0) / (PetscScalar) (numprocs - n[i]);     
        }
        for (i=0; i<N; i++){
            if (phi_array[i] <0.0) {
                I[i]  = 1;
                phi_array[i]= (PetscScalar) 0.0;
            }
        }
    }
    MPI_Allreduce(phi_in_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    for (i=10; i<N; i++) {
        if ( psi_array[i] < 1.0){ 
            phi_array[i] = phi_in_array[i];
        }
    }
    ierr = VecRestoreArray(phi, &phi_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(phi_in, &phi_in_array); CHKERRQ(ierr);
    ierr = VecDestroy(psi); CHKERRQ(ierr);
    ierr = VecDestroy(phi_in); CHKERRQ(ierr);
    ierr = PetscFree(I); CHKERRQ(ierr);
    ierr = PetscFree(n); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
      
      
      
      
      
      
      
