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
extern PetscErrorCode SimplexProjection2(AppCtx user, Vec x);

      


int main (int argc, char ** argv) {
    PetscErrorCode	ierr;
    AppCtx          user;	  
    Vec             phi;
    Vec             u, G, Gproj, psi, vec_one, phi2, phi2sum;
    PetscScalar     *phi2_array, *phi2sum_array;
    
    PetscScalar		lambda, F, Fold, Fbest;
    PetscScalar		stepmax = 1.0e+6;
    PetscScalar		stepmin = 1.0e-5;
    PetscScalar		error, tol = 1.0e-3;
    const char		u_prfx[] = "Partition_U-";
    const char		phi_prfx[] = "Partition_Phi-";
    const char		ubest_prfx[] = "Partition_U_Best-";
    const char		phibest_prfx[] = "Partition_Phi_Best-";
    char			filename [ FILENAME_MAX ];
    char            logfile  [ FILENAME_MAX ];
    const char		txtsfx[] = ".txt";
    const char		rawsfx[] = ".raw";
    const char		vtksfx[] = ".vtk";
    const char		geosfx[] = ".geo";
    const char		ressfx[] = ".res";
    
    PetscScalar     *phi_array, *psi_array;
    PetscScalar     muinit, mufinal;
    PetscScalar     GNorm;
        
    int				N, i, it, numruns, irun, ibest;
    PetscInt        maxit = 1000;
    PetscTruth		flag;
    
    PetscMPIInt		numprocs, myrank;
    PetscViewer     viewer, viewerbest;
    
    /* Eigenvalue solver stuff */
    EPSType			type;
    ST				st;
    PetscScalar		st_shift = 0.0;
    STType			st_type	= STSINV; 
    int				its;
    KSP				eps_ksp;
    PC				eps_pc;
    PetscTruth      printhelp;
    
    PetscLogDouble	eps_ts, eps_tf, eps_t;
    
    
    SlepcInitialize(&argc, &argv, (char*)0, help);
    
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    
    numruns = 1;
    PetscOptionsGetInt(PETSC_NULL, "-numruns", &numruns, PETSC_NULL);

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
    
    Fbest = 1.0e+300;
    ibest = 0;
    for (irun=0; irun<numruns; irun++){
        InitPhiRandom(user, phi);
        VecScale(phi, (PetscScalar) 1.0 / (PetscScalar) numprocs);
        SimplexProjection(user, phi);
        
	//        sprintf(filename, "%s%.3d%s", phi_prfx, myrank, txtsfx);
	//        PetscPrintf(PETSC_COMM_SELF, "[%d] Saving %s\n", myrank, filename);
	//        VecView_TXT(phi, filename);
    	 
    	    // Save .geo and .case file
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
    
        F = 0.0;
        Fold = 0.0;
        it = 0;
        PetscPrintf(PETSC_COMM_WORLD, "Iteration %d:\n", it);
        ComputeLambdaU(user, phi, &lambda, u);
        
        MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
        
        error = tol + 1.0;
        
        it = 0.0;

        PetscViewerASCIIOpen(PETSC_COMM_WORLD, "Partition.log", &viewerbest);


        sprintf(logfile, "Partition-%.4i.log", irun);
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, logfile, &viewer);
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
            
            // Compute Projection(phi + G)		
            VecCopy(phi, phi2);
            VecAXPY(phi2, (PetscScalar) 1.0, G);
            SimplexProjection(user, phi2);
            VecAXPY(phi2, (PetscScalar) -1.0, phi);
            
            // Compute L2(<G^k,phi2^k>) 
            VecPointwiseMult(phi2, phi2, G);
            
            /*
            VecGetArray(phi2, &phi2_array);
            if (myrank == 0) VecGetArray(phi2sum, &phi2sum_array);
            MPI_Reduce(phi2_array, phi2sum_array, user.nx * user.ny, MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD);
            VecRestoreArray(phi2, &phi2_array);
            if (myrank == 0) {
               VecRestoreArray(phi2sum, &phi2sum_array);
               VecNorm(phi2sum, NORM_2, &error);
            }
            MPI_Bcast(&error, 1, MPIU_SCALAR, 0, PETSC_COMM_WORLD);
            */
    
            VecGetArray(phi2, &phi2_array);
            VecGetArray(phi2sum, &phi2sum_array);
            MPI_Allreduce(phi2_array, phi2sum_array, user.nx * user.ny, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
            VecRestoreArray(phi2, &phi2_array);
            VecRestoreArray(phi2sum, &phi2sum_array);
            VecNorm(phi2sum, NORM_2, &error);
          
    		// Update phi
    		VecAXPY(phi, user.step, G);
    
            // Project phi onto the simplex \sum_k \phi^k_i=1 for i = 0-nx*ny-1
    		SimplexProjection(user, phi);
    		
            // Compute the eigenvalues u associated to the new phi
    		ComputeLambdaU(user, phi, &lambda, u);
    		
    		// Compute F= \sum_k lambda^k_epsnum
            MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    
    
            // Update the step
    		if (F<=Fold) {
                user.step = user.step * 1.05;
    			user.step = PetscMin(user.step, stepmax);
            }
    		else {
                user.step = user.step * 0.75;
                user.step = PetscMax(user.step, stepmin);
            }
            // Display some stuff
    		PetscPrintf(PETSC_COMM_WORLD, "F = %e, step = %e, error = %e, mu = %e\n\n", F, user.step, error, user.mu);
    
            // Save the same stuff in "Partition.log"
            PetscViewerASCIIPrintf(viewer, "%d   %e   ", it, F);
            PetscViewerASCIISynchronizedPrintf(viewer, "%e   ", lambda);
            PetscViewerFlush(viewer);
            PetscViewerASCIIPrintf(viewer, "%e \n", error);
          
          
          
	    if (it%10 == 0){
	      sprintf(filename, "%s%.3d-%.3d%s", u_prfx, myrank, irun, txtsfx);
	      VecView_TXT(u, filename);
	      
	      // Save in ensight gold ASCII format
	      sprintf(filename, "%s%.3d-%.3d%s", u_prfx, myrank, irun, ressfx);
	      VecView_EnsightASCII(u, filename);
	      
	      sprintf(filename, "%s%.3d-%.3d%s", phi_prfx, myrank, irun, txtsfx);
	      VecView_TXT(phi, filename);
	      
	      // Save in ensight gold ASCII format
	      sprintf(filename, "%s%.3d-%.3d%s", phi_prfx, myrank, irun, ressfx);
	      VecView_EnsightASCII(phi, filename);
	    }

        } while ( (it < 20 ) || ( ( it < maxit ) && (error > tol) ) );
        
	// Save the results
	sprintf(filename, "%s%.3d-%.3d%s", u_prfx, myrank, irun, txtsfx);
	VecView_TXT(u, filename);
	
	// Save in ensight gold ASCII format
	sprintf(filename, "%s%.3d-%.3d%s", u_prfx, myrank, irun, ressfx);
	VecView_EnsightASCII(u, filename);
	
	sprintf(filename, "%s%.3d-%.3d%s", phi_prfx, myrank, irun, txtsfx);
	VecView_TXT(phi, filename);
	
	// Save in ensight gold ASCII format
	sprintf(filename, "%s%.3d-%.3d%s", phi_prfx, myrank, irun, ressfx);
	VecView_EnsightASCII(phi, filename);
	
        PetscViewerDestroy(viewer);

        PetscViewerASCIIPrintf(viewerbest, "%d   %e   ", irun, F);
        PetscViewerASCIISynchronizedPrintf(viewerbest, "%e   ", lambda);
        PetscViewerFlush(viewerbest);
        PetscViewerASCIIPrintf(viewerbest, "\n");

        if (F<Fbest){
        
            Fbest = F;
            ibest = irun;

            sprintf(filename, "%s%.3d%s", ubest_prfx, myrank, txtsfx);
            VecView_TXT(u, filename);

            sprintf(filename, "%s%.3d%s", ubest_prfx, myrank, ressfx);
            VecView_EnsightASCII(u, filename);
            
            sprintf(filename, "%s%.3d%s", phibest_prfx, myrank, txtsfx);
            VecView_TXT(phi, filename);

            sprintf(filename, "%s%.3d%s", phibest_prfx, myrank, ressfx);
            VecView_EnsightASCII(phi, filename);
        }
        PetscPrintf(PETSC_COMM_WORLD, "\n\nBEST COMPUTATION IS NOW #%i\n\n", ibest);
    }
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
//    PetscViewerDestroy(viewer);
    PetscViewerDestroy(viewerbest);

    SlepcFinalize();
}


PetscErrorCode ComputeK(AppCtx user, Vec phi)
{
    Mat             K  = user.K;
    DA              da = user.da;
    PetscErrorCode  ierr;
    PetscInt        i, j, xm, ym, xs, ys;
    PetscScalar     v[5],Hx,Hy,HxdHy,HydHx;
    MatStencil      row,col[5];
    PetscScalar     **local_phi;
    
    DAVecGetArray(user.da, phi, &local_phi);
    
    Hx = 1.0 / (PetscReal)(user.nx-1); 
    Hy = 1.0 / (PetscReal)(user.ny-1);
    HxdHy = Hx/Hy; HydHx = Hy/Hx;
    DAGetCorners(da, &xs, &ys, PETSC_NULL, &xm, &ym,PETSC_NULL);

    for (j=ys; j<ys+ym; j++){
        for(i=xs; i<xs+xm; i++){
            row.i = i; row.j = j;
            if ( (!user.per) && (i==0 || j==0 || i==user.nx-1 || j==user.ny-1)){
                v[0] = 2.0*(HxdHy + HydHx) + user.mu * (1.0 - local_phi[j][i]);
                MatSetValuesStencil(K,1,&row,1,&row,v,INSERT_VALUES);
            } else {
                v[0] = -HxdHy; col[0].i = i;	  col[0].j = j-1;
                v[1] = -HydHx; col[1].i = i-1; col[1].j = j;
                v[2] = 2.0*(HxdHy + HydHx); col[2].i = row.i; col[2].j = row.j;
                v[2] += user.mu * (1.0 - local_phi[j][i])*Hx*Hy;
                v[3] = -HydHx; col[3].i = i+1; col[3].j = j;
                v[4] = -HxdHy; col[4].i = i;	  col[4].j = j+1;
                MatSetValuesStencil(K, 1, &row, 5, col, v, INSERT_VALUES);
            }
        }
    }
    DAVecRestoreArray(user.da, phi, &local_phi);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    return 0;
}

PetscErrorCode ComputeLambdaU(AppCtx user, Vec phi, PetscScalar *lambda, Vec u){
    PetscLogDouble  eps_ts, eps_tf, eps_t;
    int             its;
    Vec             ui;
    PetscScalar		eigi, normu;
    int             nconv;
    PetscInt        myrank;
    
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    
    
    ComputeK(user, phi);
    EPSSetOperators(user.eps, user.K, PETSC_NULL);
    PetscGetTime(&eps_ts);
    EPSSolve(user.eps);
    PetscGetTime(&eps_tf);
    eps_t = eps_tf - eps_ts;
    EPSGetIterationNumber(user.eps, &its);
    
    
    VecDuplicate(u, &ui);
    EPSGetConverged(user.eps, &nconv);
    
    EPSGetEigenpair(user.eps, nconv-user.epsnum , lambda, &eigi, u, ui);
    VecNorm(u, NORM_2, &normu);
    normu = 1.0 / normu;
    VecScale(u, normu);
    
    VecDestroy(ui);
         
    *lambda = *lambda * (PetscReal)(user.nx-1) * (PetscReal)(user.ny-1) / 2.0; 
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "        lambda[%d] = %e    EPSSolve converged in %f s for %d iterations\n", myrank, *lambda, eps_t, its);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
    return 0;
}

extern PetscErrorCode InitPhiQuarter(AppCtx user, Vec phi){
    PetscScalar	    **local_phi;
    PetscInt        i, j, mx, my, xm, ym, xs, ys;
    PetscScalar     zero = 0.0, one	= 1.0;

    DAGetCorners(user.da, &xs, &ys, PETSC_NULL, &xm, &ym,PETSC_NULL);
    
    VecSet(phi, zero);
    DAVecGetArray(user.da, phi, &local_phi);
    for (j=ys; j<ys+ym; j++){
        for(i=xs; i<xs+xm; i++){
            if ( (i < user.nx/2) || (j < user.ny/2) ){
                local_phi[j][i] = one;
       		}
         }
    }
    DAVecRestoreArray(user.da, phi, &local_phi);
    return 0;
}

extern PetscErrorCode InitPhiRandom(AppCtx user, Vec phi){
    PetscRandom     rndm;
    PetscLogDouble  tim;
    PetscMPIInt     rank;
    MPI_Comm        comm;
    
    PetscObjectGetComm((PetscObject) phi, &comm);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    PetscRandomCreate(comm, &rndm);
    PetscRandomSetFromOptions(rndm);
    PetscGetTime(&tim);
    PetscRandomSetSeed(rndm, (unsigned long) tim*rank);

    PetscRandomSetSeed(rndm, (unsigned long) rank);
	 
    PetscRandomSeed(rndm);
    VecSetRandom(phi, rndm);
    PetscRandomDestroy(rndm);
    
    return 0;
}

extern PetscErrorCode ComputeG(AppCtx user, Vec G, Vec u){
	 
    VecPointwiseMult(G, u, u);
    return 0;
}

PetscErrorCode VecView_TXT(Vec x, const char filename[]){
    Vec             natural, io;
    VecScatter      tozero;
    PetscMPIInt     rank, size;
    int             N;
    PetscScalar     *io_array;
    DA              da;
    PetscViewer     viewer;
    int             i, j, k, mx, my, mz;
    MPI_Comm        comm;
    PetscErrorCode  ierr;
    
    
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

PetscErrorCode VecView_RAW(Vec x, const char filename[]){
    Vec				natural, io;
    VecScatter		tozero;
    PetscMPIInt	    myrank;
    int				N;
    DA				da;
    PetscViewer	    viewer;
    int				i, j, mx, my;
    
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    VecGetSize(x, &N);
    
    PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da);
    if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
    DAGetInfo(da, 0, &mx, &my, 0,0,0,0, 0,0,0,0);
    
    DACreateNaturalVector(da, &natural);
    DAGlobalToNaturalBegin(da, x, INSERT_VALUES, natural);
    DAGlobalToNaturalEnd(da, x, INSERT_VALUES, natural);
    
    VecScatterCreateToZero(natural, &tozero, &io);
    VecScatterBegin(tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(tozero, natural, io, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterDestroy(tozero);
    VecDestroy(natural);
    
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(io,viewer);
    PetscViewerDestroy(viewer);
    VecDestroy(io);
    return 0;
}

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


PetscErrorCode DAView_GEOASCII(DA da, const char filename []){
    PetscInt          mx, my, mz;
    PetscMPIInt       rank, size;
    PetscViewer       viewer;
    MPI_Comm          comm;
    PetscErrorCode    ierr;
    const char        *name;
    
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


PetscErrorCode SimplexProjection2(AppCtx user, Vec phi)
{
    PetscMPIInt     *I, *n;
    Vec             psi;
    PetscScalar     *psi_array, *phi_array;
    PetscInt        l, i;
    PetscMPIInt     myrank, numprocs;
    
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    
    VecDuplicate(phi, &psi);
    VecGetArray(psi, &psi_array);
    VecGetArray(phi, &phi_array);
    
    PetscMalloc(user.nx*user.ny*sizeof(PetscMPIInt), &I);
    PetscMalloc(user.nx*user.ny*sizeof(PetscMPIInt), &n);
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
    VecRestoreArray(phi, &phi_array);
    VecRestoreArray(psi, &psi_array);   
    VecDestroy(psi);
    PetscFree(I);
    PetscFree(n);
    
    PetscFunctionReturn(0);
}

PetscErrorCode SimplexProjection(AppCtx user, Vec phi)
{
	PetscMPIInt     *I, *n;
	Vec             psi;
	PetscScalar     *psi_array, *phi_array;
	PetscInt        l, i;
	PetscMPIInt     myrank, numprocs;
	
	MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
	MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
	
	VecDuplicate(phi, &psi);
	VecGetArray(psi, &psi_array);
	VecGetArray(phi, &phi_array);
	
	PetscMalloc(user.nx*user.ny*sizeof(PetscMPIInt), &I);
	PetscMalloc(user.nx*user.ny*sizeof(PetscMPIInt), &n);
	for (i=0; i<user.nx*user.ny; i++) I[i] = 0;
	for (l=0; l<numprocs; l++){
		for (i=0; i<user.nx*user.ny; i++){
			if (I[i]) phi_array[i] = 0.0;
		}
		MPI_Allreduce(phi_array, psi_array, user.nx * user.ny, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce(I, n, user.nx * user.ny, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		for (i=0; i<user.nx*user.ny; i++){
			if ((psi_array[i] > 1.0) && (numprocs - n[i])) phi_array[i] = phi_array[i]- (PetscScalar) (1-I[i]) * (psi_array[i]-1.0) / (PetscScalar) (numprocs - n[i]);     
			if (phi_array[i] < 0.0) {
				I[i]  = 1;
				phi_array[i]= (PetscScalar) 0.0;
			}
		}
	}
	VecRestoreArray(phi, &phi_array);
	VecRestoreArray(psi, &psi_array);   
	VecDestroy(psi);
	PetscFree(I);
	PetscFree(n);
	
	PetscFunctionReturn(0);
}
