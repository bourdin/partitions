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
//	 int				k;		  /* Number of domain to partition into */
//  k willbe the number of cpus
	 PetscTruth		per;	  /* true for periodic boundary conditions, false otherwise */
	 DA				da;	  /* Information about the distributed layout */
	 PetscScalar	step;	  /* Initial step of the steepest descent methods */
	 EPS				eps;	  /* Eigenvalue solver context */
	 Mat				K;		  /* Matrix for the Laplacian */
} AppCtx;

extern PetscErrorCode ComputeK(AppCtx user, Vec phi);
extern PetscErrorCode ComputeLambdaU(AppCtx user, Vec phi, PetscScalar *lambda, Vec u);
extern PetscErrorCode ComputeG(AppCtx user, Vec G, Vec u);
extern PetscErrorCode InitPhiQuarter(AppCtx user, Vec phi);
extern PetscErrorCode InitPhiRandom(AppCtx user, Vec phi);
extern PetscErrorCode VecView_TXT(Vec x, const char filename[]);
extern PetscErrorCode VecView_RAW(Vec x, const char filename[]);
extern PetscErrorCode VecView_VTKASCII(Vec x, const char filename[]);


int main (int argc, char ** argv) {
	 PetscErrorCode	ierr;
	 AppCtx				user;	  
	 Vec					phi;
	 Vec					u, G, psi, vec_one;
	 PetscScalar		lambda, F, Fold;
	 PetscScalar		stepmax = 1.0e+6;
	 PetscScalar		error, tol = 1.0e-3;
	 const char			u_prfx[] = "Partition_U-";
	 const char			phi_prfx[] = "Partition_Phi-";
	 char				   filename [ FILENAME_MAX ];
	 const char			txtsfx[] = ".txt";
	 const char			rawsfx[] = ".raw";
	 const char			vtksfx[] = ".vtk";
	 PetscScalar      *phi_array, *psi_array;

		  
	 int				   N, i, it, maxit = 5000;
	 PetscTruth			flag;
	 
	 PetscMPIInt		numprocs, myrank;
	 
	 /* Eigenvalue solver stuff */
	 EPSType			   type;
	 ST					st;
	 PetscScalar		st_shift = 0.0;
	 STType				st_type	= STSINV; 
	 int				   its;
	 KSP				   eps_ksp;
	 PC					eps_pc;
	 PetscTruth       printhelp;
	 
	 PetscLogDouble	eps_ts, eps_tf, eps_t;
	 
	 
	 SlepcInitialize(&argc, &argv, (char*)0, help);

	 MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
	 MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
//	 PetscOptionsGetTruth(PETSC_NULL, "-h", &printhelp, 0);
//    if (printhelp) {
//      PetscPrintf(PETSC_COMM_WORLD, help);
//      PetscFinalize();
//      return -1;
//      }

	 user.nx = 10;
	 PetscOptionsGetInt(PETSC_NULL, "-nx", &user.nx, PETSC_NULL);
	 PetscOptionsGetInt(PETSC_NULL, "-ny", &user.ny, &flag);	 
	 if( flag==PETSC_FALSE ) user.ny=user.nx;
	 N = user.nx*user.ny;
	 PetscOptionsGetScalar(PETSC_NULL, "-mu", &user.mu, PETSC_NULL);
	 user.step = 10.0;
	 PetscOptionsGetScalar(PETSC_NULL, "-step", &user.step, PETSC_NULL);
	 
	 if (numprocs==1) {
		  PetscPrintf(PETSC_COMM_WORLD, "\nCannot partition in less than 2 subsets! ");
		  PetscPrintf(PETSC_COMM_WORLD, "\nRestart on more than 1 cpu");
		  return -1;
	 }		
	 PetscOptionsGetTruth(PETSC_NULL, "-periodic", &user.per, 0);
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
    VecSet(vec_one, (PetscScalar) 1.0);
	 
	 /* Create the eigensolver context */
	 EPSCreate(PETSC_COMM_SELF, &user.eps);

	 EPSSetOperators(user.eps, user.K, PETSC_NULL);
	 EPSSetProblemType(user.eps, EPS_HEP);
	 EPSGetST(user.eps, &st);
	 
	 STSetType(st, st_type);
	 STSetShift(st, st_shift);
	 
	 STGetKSP(st, &eps_ksp);
	 KSPGetPC(eps_ksp, &eps_pc);
	 PCSetType(eps_pc, PCCHOLESKY);
	 KSPSetType(eps_ksp, KSPPREONLY);

	 STSetFromOptions(st);
	 EPSSetFromOptions(user.eps);
	 
    InitPhiRandom(user, phi);
//	 VecScale(phi, (PetscScalar) .1);

	 sprintf(filename, "%s%.3d%s", phi_prfx, myrank, txtsfx);
	 PetscPrintf(PETSC_COMM_SELF, "[%d] Saving %s\n", myrank, filename);
	 VecView_TXT(phi, filename);
	 
	 F = 0.0;
	 Fold = 0.0;
	 it = 0;
	 PetscPrintf(PETSC_COMM_WORLD, "Iteration %d:\n", it);
    ComputeLambdaU(user, phi, &lambda, u);
   
    MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

	 error = tol + 1.0;
	 
	 while ( it < maxit ){ 
		it++;
		Fold = F;
		F = 0.0;
		PetscPrintf(PETSC_COMM_WORLD, "Iteration %d:\n", it);
		ComputeG(user, G, u);
		VecAXPY(phi, user.step, G);
      VecGetArray(psi, &psi_array);
      VecGetArray(phi, &phi_array);
      MPI_Allreduce(phi_array, psi_array, user.nx * user.ny, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
      VecRestoreArray(phi, &phi_array);
      VecRestoreArray(psi, &psi_array);


		  // truncation
      VecPointwiseMax(psi, psi, vec_one);
		VecPointwiseDivide(phi, phi, psi);
		F = 0.0;
		ComputeLambdaU(user, phi, &lambda, u);
//		PetscSynchronizedPrintf(PETSC_COMM_WORLD, "		 lambda[%d] = %f\n", myrank, lambda);
//      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

		if (F<=Fold) {
         user.step = user.step * 1.2;
			user.step = PetscMin(user.step, stepmax);
      }
		else {
		   user.step = user.step / 2.0;
      }
		error = (Fold - F) / F;
		PetscPrintf(PETSC_COMM_WORLD, "F = %f, step = %f, error = %f\n\n", F, user.step, error);

////*********************************************************************************************
//      //THIS STILL NEEDS TO BE UPDATED
		  if (it%10 == 0){
////				  for (i=0; i<user.k; i++){
////						VecView(phi[i], PETSC_VIEWER_DRAW_WORLD);
////				  }				
//				
//				
//				for (i=0; i<user.k; i++){
  					 sprintf(filename, "%s%.3d-%.5d%s", u_prfx, myrank, it,txtsfx);
					 //sprintf(filename, "%s%.3d%s", u_prfx, myrank, txtsfx);
					 PetscPrintf(PETSC_COMM_SELF, "[%d] Saving %s\n", myrank, filename);
					 VecView_TXT(u, filename);
//
//					/*
//					 sprintf(filename, "%s%.3d-%.5d%s", u_prfx, myrank, it, vtksfx);
//					 PetscPrintf(PETSC_COMM_SELF, "[%d] Saving %s\n", myrank, filename);
//					 VecView_VTKASCII(u, filename);
//					*/
//
						sprintf(filename, "%s%.3d-%.5d%s", phi_prfx, myrank, it, txtsfx);
//					 sprintf(filename, "%s%.3d%s", phi_prfx, myrank, txtsfx);
					 PetscPrintf(PETSC_COMM_SELF, "[%d] Saving %s\n", myrank, filename);
					 VecView_TXT(phi, filename);
//
//					/*
//					 sprintf(filename, "%s%.3d-%.5d%s", phi_prfx, myrank, it, vtksfx);
//					 PetscPrintf(PETSC_COMM_SELF, "[%d] Saving %s\n", myrank, filename);
//					 VecView_VTKASCII(phi, filename);
//					 */
//				}
		  }
	 }
//	 
	 VecDestroy(phi);
	 VecDestroy(psi);
	 VecDestroy(u);
	 VecDestroy(G);
	 MatDestroy(user.K);
	 DADestroy(user.da);	 
	 EPSDestroy(user.eps);
	 PetscLogPrintSummary(MPI_COMM_WORLD,"petsc_log_summary.log");		

	 SlepcFinalize();
}


PetscErrorCode ComputeK(AppCtx user, Vec phi)
{
	 Mat				 K	 = user.K;
	 DA				 da = user.da;
	 PetscErrorCode ierr;
	 PetscInt		 i, j, mx, my, xm, ym, xs, ys;
	 PetscScalar	 v[5],Hx,Hy,HxdHy,HydHx;
	 MatStencil		 row,col[5];
	 PetscScalar	 **local_phi;
	 
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
	 PetscLogDouble	eps_ts, eps_tf, eps_t;
	 int					its;
	 Vec					ui;
	 PetscScalar		eigi, normu;
	 int					nconv;
	 PetscInt         myrank;
	 
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
	 EPSGetEigenpair(user.eps, nconv-1 , lambda, &eigi, u, ui);
	 VecNorm(u, NORM_2, &normu);
	 normu = 1.0 / normu;
	 VecScale(u, normu);
	 
	 VecDestroy(ui);
		  
	 *lambda = *lambda * (PetscReal)(user.nx-1) * (PetscReal)(user.ny-1) / 2.0; 
	 PetscSynchronizedPrintf(PETSC_COMM_WORLD, "        lambda[%d] = %f    EPSSolve converged in %f s for %d iterations\n", myrank, *lambda, eps_t, its);
	 PetscSynchronizedFlush(PETSC_COMM_WORLD);
	 return 0;
}

extern PetscErrorCode InitPhiQuarter(AppCtx user, Vec phi){
	 PetscScalar	 **local_phi;
	 PetscInt		 i, j, mx, my, xm, ym, xs, ys;
	 PetscScalar	 zero = 0.0, one	= 1.0;
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
	 PetscRandom	 rndm;
	 PetscLogDouble tim;
	 PetscMPIInt    rank;
	 MPI_Comm       comm;
	 
    PetscObjectGetComm((PetscObject) phi, &comm);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	 PetscRandomCreate(comm, &rndm);
	 PetscRandomSetFromOptions(rndm);
	 PetscGetTime(&tim);
	 PetscRandomSetSeed(rndm, (unsigned long) tim*rank);
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
	 Vec				natural, io;
	 VecScatter		tozero;
	 PetscMPIInt	rank, size;
	 int				N;
	 PetscScalar	*io_array;
	 DA				da;
	 PetscViewer	viewer;
	 int				i, j, mx, my;
	 MPI_Comm      comm;
	 
    PetscObjectGetComm((PetscObject) x, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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

	 VecGetArray(io, &io_array);	  

	 if (rank ==	 0){
		  PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer);
		  for (j=0; j<my; j++){
				for(i=0; i<mx; i++){
					 PetscViewerASCIIPrintf(viewer, "%G	  ", PetscRealPart(io_array[j*mx+i]));
				}
				PetscViewerASCIIPrintf(viewer, "\n");
		  }
		  PetscViewerFlush(viewer);
		  PetscViewerDestroy(viewer);
	 }
	 VecRestoreArray(io, &io_array);		
	 VecDestroy(io);
	 return 0;
}

PetscErrorCode VecView_RAW(Vec x, const char filename[]){
	 Vec				natural, io;
	 VecScatter		tozero;
	 PetscMPIInt	myrank;
	 int				N;
	 DA				da;
	 PetscViewer	viewer;
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
  MPI_Comm				comm;
  DA						da;
  Vec						natural, master;
  PetscViewer			viewer;
  PetscScalar		  *array, *values;
  PetscInt				n, N, maxn, mx, my, mz, dof;
  PetscInt				xs, xm, ys, ym;
  PetscInt				i, p;
  MPI_Status			status;
  PetscMPIInt			rank, size, tag;
  PetscErrorCode		ierr;
  VecScatter			ScatterToZero;
  const char					*name;

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
