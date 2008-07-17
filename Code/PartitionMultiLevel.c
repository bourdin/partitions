static char help[] = "Compute a partition minimizing the sum of the first eigenvalue of each component in 2 or 3D\n\n";

/*
The matrix is scaled by a factor 2*hx*hy
The 2D txt files can be plotted using gnuplot with 
set size square
splot 'Partition_Phi_all.txt' matrix with pm3d
*/

#include <stdio.h>
#include <stdlib.h>

#include "petscksp.h"
#include "petscvec.h"
#include "petscda.h"
#include "slepceps.h"

typedef struct {
    PetscInt    ndim;
    PetscInt	nx, ny, nz;   /* Dimension of the discretized domain */
    PetscScalar	mu;	      /* Penalization factor for the computation of the eigenvalue */
    PetscTruth	per;	  /* true for periodic boundary conditions, false otherwise */
    DA			da;	      /* Information about the distributed layout */
    PetscScalar	step;	  /* Initial step of the steepest descent methods */
    PetscScalar stepmin;
    PetscScalar stepmax;
    EPS			eps;      /* Eigenvalue solver context */
    Mat			K;        /* Matrix for the Laplacian */
    PetscInt    epsnum;   /* which eigenvalues are we optimizing */
    PetscInt    numlevels; /* number of mesh refinements */
} AppCtx;

extern PetscErrorCode DistanceFromSimplex(PetscScalar *dist, Vec phi); 
extern PetscErrorCode ShowComposite_Phi(Vec phi, const char filename[]);
extern PetscErrorCode SaveComposite_Phi(Vec phi, const char filename[]);
extern PetscErrorCode SaveComposite_U(Vec u, const char filename[]);
extern PetscErrorCode ComputeK2d(AppCtx user, Vec phi);
extern PetscErrorCode ComputeK3d(AppCtx user, Vec phi);
extern PetscErrorCode ComputeLambdaU(AppCtx user, Vec phi, PetscScalar *lambda, Vec u);
extern PetscErrorCode ComputeG(AppCtx user, Vec G, Vec u);
extern PetscErrorCode InitPhiQuarter(AppCtx user, Vec phi);
extern PetscErrorCode InitPhiRandom(AppCtx user, Vec phi);
extern PetscErrorCode InitEnsight(AppCtx user, const char u_prfx[], const char phi_prfx[], const char res_sfx[], int level);
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
    Vec				phi, phi_err;
    Vec				u, G;
    
    PetscScalar		lambda, F, Fold;
    PetscScalar		error, myerror, tol;
    const char		u_prfx[] = "Partition_U-";
    const char		phi_prfx[] = "Partition_Phi-";
    char			filename [ FILENAME_MAX ];
    const char		txt_sfx[] = ".txt";
    const char		raw_sfx[] = ".raw";
    const char		vtk_sfx[] = ".vtk";
    const char		geo_sfx[] = ".geo";
    const char		res_sfx[] = ".res";
    
    PetscTruth      SaveTXT;
    PetscTruth      SaveVTK;
    PetscTruth      SaveEnsight;
    PetscTruth      SaveComposite;
    int             modsave;
    PetscTruth      showphi = PETSC_FALSE;
            
    int				N, i, it;
    PetscInt        maxit;
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
    PetscTruth      printhelp;
    
    PetscLogDouble	eps_ts, eps_tf, eps_t;
    PetscReal       dist;
    PetscTruth      TwoPass;
    PetscTruth      FinalPass=PETSC_TRUE;
    PetscTruth      FirstPass=PETSC_TRUE;
    PetscTruth      is3d     =PETSC_FALSE;
    int             level = 0;
    DA              dac;
    PetscReal       step;
    int            stages[2];
    
    PetscFunctionBegin;
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    
    ierr = PetscLogStageRegister(&stages[0], "Eigensolver"); CHKERRQ(ierr)
    ierr = PetscLogStageRegister(&stages[1], "I/O"); CHKERRQ(ierr)
    
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    if (numprocs==1) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCannot partition in less than 2 subsets! "); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\nRestart on more than 1 cpu"); CHKERRQ(ierr);
        SlepcFinalize();
        return -1;
    }		
    
    
    // GET PARAMETERS FROM THE COMMAND LINE
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-showphi", &showphi, PETSC_NULL);    CHKERRQ(ierr); 
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-3d", &is3d, PETSC_NULL);    CHKERRQ(ierr); 
    if (is3d == PETSC_TRUE){
        user.ndim = 3;
    } else {
        user.ndim = 2;
    }
    
    user.epsnum = 1;
    ierr = PetscOptionsGetInt(PETSC_NULL, "-epsnum", &user.epsnum, PETSC_NULL);    CHKERRQ(ierr); 
    maxit = 1000;
    ierr = PetscOptionsGetInt(PETSC_NULL, "-maxit", &maxit, PETSC_NULL); CHKERRQ(ierr);
    tol = 1.0e-3;
    ierr = PetscOptionsGetScalar(PETSC_NULL, "-tol", &tol, PETSC_NULL); CHKERRQ(ierr);
    
    user.nx = 10;
    ierr = PetscOptionsGetInt(PETSC_NULL, "-nx", &user.nx, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL, "-ny", &user.ny, &flag); CHKERRQ(ierr);
    if (flag==PETSC_FALSE) user.ny=user.nx; 
    
    if (user.ndim==3){
        ierr = PetscOptionsGetInt(PETSC_NULL, "-nz", &user.nz, &flag); CHKERRQ(ierr);
        if (flag==PETSC_FALSE) user.nz=user.nx; 
        N = user.nx*user.ny*user.nz;
    } else {
        user.nz = 1;
        N = user.nx*user.ny;
    }
    
    user.mu = 1.0e3;
    ierr = PetscOptionsGetScalar(PETSC_NULL, "-mu", &user.mu, PETSC_NULL); CHKERRQ(ierr);
    
    user.step = 10.0;
    ierr = PetscOptionsGetScalar(PETSC_NULL, "-step", &user.step, PETSC_NULL); CHKERRQ(ierr);
    user.stepmin = user.step;
    user.stepmax = 1.0e4;
    ierr = PetscOptionsGetScalar(PETSC_NULL, "-stepmin", &user.stepmin, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULL, "-stepmax", &user.stepmax, PETSC_NULL); CHKERRQ(ierr);
    step = user.step;
    
    user.per = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-periodic", &user.per, PETSC_NULL); CHKERRQ(ierr);
    
    SaveTXT = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-saveTXT", &SaveTXT, PETSC_NULL); CHKERRQ(ierr);
    SaveVTK = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-saveVTK", &SaveVTK, PETSC_NULL); CHKERRQ(ierr);
    SaveEnsight = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-saveEnsight", &SaveEnsight, PETSC_NULL); CHKERRQ(ierr);
    SaveComposite = PETSC_TRUE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-saveComposite", &SaveComposite, PETSC_NULL); CHKERRQ(ierr);

    modsave = 25;
    ierr = PetscOptionsGetInt(PETSC_NULL,   "-modsave", &modsave, PETSC_NULL); CHKERRQ(ierr);

    TwoPass = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-twopass", &TwoPass, PETSC_NULL); CHKERRQ(ierr);
    if (TwoPass == PETSC_TRUE) FinalPass = PETSC_FALSE;
    
    user.numlevels = 1;
    ierr = PetscOptionsGetInt(PETSC_NULL, "-levels", & user.numlevels, PETSC_NULL); CHKERRQ(ierr);

    // SOME INITIALIZATIONS
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n%dD Optimal Partition problem, N=%d (%dx%dx%d grid)\n\n", user.ndim, N, user.nx, user.ny, user.nz); CHKERRQ(ierr);
    ierr = PetscLogPrintSummary(MPI_COMM_WORLD,"petsc_log_summary.log"); CHKERRQ(ierr);
    
    if (user.per) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Using periodic boundary conditions\n"); CHKERRQ(ierr);
	if (user.ndim == 2) {
            ierr = DACreate2d(PETSC_COMM_SELF, DA_XYPERIODIC, DA_STENCIL_STAR, user.nx, user.ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &user.da); CHKERRQ(ierr);
	} else {
	    ierr = DACreate3d(PETSC_COMM_SELF, DA_XYZPERIODIC, DA_STENCIL_STAR, user.nx, user.ny, user.nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &user.da); CHKERRQ(ierr);
	}
    }
    else {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Using non-periodic boundary conditions\n"); CHKERRQ(ierr);
        ierr = DACreate(PETSC_COMM_SELF, user.ndim, DA_NONPERIODIC, DA_STENCIL_STAR, user.nx, user.ny, user.nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &user.da); CHKERRQ(ierr);
    }
    
    ierr = DACreateGlobalVector(user.da, &phi); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &u); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &G); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &phi_err); CHKERRQ(ierr);
    
    /* Create the eigensolver context */
    ierr = EPSCreate(PETSC_COMM_SELF, &user.eps); CHKERRQ(ierr);
    
    ierr = DAGetMatrix(user.da, MATSEQAIJ, &user.K); CHKERRQ(ierr);
    ierr = EPSSetOperators(user.eps, user.K, PETSC_NULL); CHKERRQ(ierr);
    ierr = EPSSetProblemType(user.eps, EPS_HEP); CHKERRQ(ierr);
    ierr = EPSGetST(user.eps, &st); CHKERRQ(ierr);
    ierr = EPSSetDimensions(user.eps, user.epsnum, 5*user.epsnum); CHKERRQ(ierr);
    
    ierr = STSetType(st, st_type); CHKERRQ(ierr);
    ierr = STSetShift(st, st_shift); CHKERRQ(ierr);
    
    ierr = STGetKSP(st, &eps_ksp); CHKERRQ(ierr);
    ierr = KSPGetPC(eps_ksp, &eps_pc); CHKERRQ(ierr);
    
    ierr = PCSetType(eps_pc, PCICC); CHKERRQ(ierr);
    ierr = KSPSetType(eps_ksp, KSPCG); CHKERRQ(ierr);
    
    ierr = STSetFromOptions(st); CHKERRQ(ierr);
    ierr = EPSSetFromOptions(user.eps); CHKERRQ(ierr);
    
    // Initializes PHI
    ierr = InitPhiRandom(user, phi); CHKERRQ(ierr);
    ierr = VecScale(phi, (PetscScalar) 1.0 / (PetscScalar) numprocs); CHKERRQ(ierr);
//    ierr = SimplexInteriorProjection(user, phi); CHKERRQ(ierr);
    ierr = SimplexProjection2(user, phi); CHKERRQ(ierr);
    
    F = 0.0;
    Fold = 0.0;
    it = 0;
    
    ierr = PetscLogStagePush(stages[0]); CHKERRQ(ierr);
    ierr = ComputeLambdaU(user, phi, &lambda, u); CHKERRQ(ierr);
    ierr = PetscLogStagePop(); CHKERRQ(ierr);
    
    MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    
    sprintf(filename, "Partition-level%d.log", level);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "%d   %e   ", it, F); CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%e   ", lambda); CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "%e \n", tol, it); CHKERRQ(ierr);
    
    for (level=0; level<user.numlevels; level++){
        if (SaveEnsight==PETSC_TRUE) {
            ierr = PetscLogStagePush(stages[1]); CHKERRQ(ierr);
            ierr = InitEnsight(user, u_prfx, phi_prfx, res_sfx, level); CHKERRQ(ierr);
                ierr = PetscLogStagePop(); CHKERRQ(ierr);

        }    
        error = tol + 1.0;    
        it = 0;
        do { 
            if ( (error < tol) && (it > 20) && (TwoPass == PETSC_TRUE) && (level == user.numlevels-1) ){
                ierr = PetscPrintf(PETSC_COMM_WORLD, "Switching to EXACT projection\n");; CHKERRQ(ierr);
                FirstPass = PETSC_FALSE;
                FinalPass = PETSC_TRUE;
    
                ierr = PetscLogStagePush(stages[1]); CHKERRQ(ierr);
                if (SaveTXT == PETSC_TRUE){
                    sprintf(filename, "%s%.3d-level%d-step1%s", u_prfx, myrank, level, txt_sfx);
                    ierr = VecView_TXT(u, filename); CHKERRQ(ierr);
                    sprintf(filename, "%s%.3d-level%d-step1%s", phi_prfx, myrank, level, txt_sfx);
                    ierr = VecView_TXT(phi, filename); CHKERRQ(ierr);
                }
                if (SaveVTK == PETSC_TRUE){
                    sprintf(filename, "%s%.3d-level%d-step1%s", u_prfx, myrank, level, vtk_sfx);
                    ierr = VecView_VTKASCII(u, filename); CHKERRQ(ierr);
                    sprintf(filename, "%s%.3d-level%d-step1%s", phi_prfx, myrank, level, vtk_sfx);
                    ierr = VecView_VTKASCII(phi, filename); CHKERRQ(ierr);
                }
                if (SaveEnsight == PETSC_TRUE){
                    sprintf(filename, "%s%.3d-level%d-step1%s", u_prfx, myrank, level, res_sfx);
                    ierr = VecView_EnsightASCII(u, filename); CHKERRQ(ierr);
                    sprintf(filename, "%s%.3d-level%d-step1%s", phi_prfx, myrank, level, res_sfx);
                    ierr = VecView_EnsightASCII(phi, filename); CHKERRQ(ierr);
                }
                if (SaveComposite == PETSC_TRUE){
                    sprintf(filename, "Partition_Phi_all-level%d-step1.txt", level);
                    ierr = SaveComposite_Phi(phi, filename); CHKERRQ(ierr);
        
                    sprintf(filename, "Partition_U_all-level%d-step1.txt", level);
                    ierr = SaveComposite_U(u, filename); CHKERRQ(ierr);
                }
                ierr = PetscLogStagePop(); CHKERRQ(ierr);
            } 


            it++;
            
            Fold = F;
            F = 0.0;
            ierr = PetscPrintf(PETSC_COMM_WORLD, "Level %d (%dx%dx%d) iteration %d:\n", level, user.nx, user.ny, user.nz, it); CHKERRQ(ierr);
            
            // Save the previous iteration
            ierr = VecCopy(phi, phi_err); CHKERRQ(ierr);
            
            // Compute the gradient of the objective function w.r.t. u
            ierr = ComputeG(user, G, u); CHKERRQ(ierr);
            
            // Update phi
            ierr = VecAXPY(phi, step, G); CHKERRQ(ierr);
            
            // Project phi onto the simplex \sum_k \phi^k_i=1 for i = 0 : nx*ny*nz-1
            if (FirstPass == PETSC_TRUE){
                ierr = SimplexProjection2(user, phi); CHKERRQ(ierr);
                ierr = PetscPrintf(PETSC_COMM_WORLD, "*** Using FAST projection\n"); CHKERRQ(ierr);
            } else {
                ierr = SimplexProjection(user, phi); CHKERRQ(ierr);
                ierr = PetscPrintf(PETSC_COMM_WORLD, "*** Using EXACT projection\n"); CHKERRQ(ierr);
            } 
            // Compute the distance the simplex:
            ierr = DistanceFromSimplex(&dist, phi); CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD, "   Distance from simplex: %f\n", dist); CHKERRQ(ierr);
            
            // Compute the L^\infty error on Phi
            ierr = VecAXPY(phi_err, -1.0, phi); CHKERRQ(ierr);
            ierr = VecNorm(phi_err, NORM_INFINITY, &myerror); CHKERRQ(ierr);
            ierr = PetscGlobalMax(&myerror, &error, PETSC_COMM_WORLD); CHKERRQ(ierr);
    
            //Compute the eigenvalues u associated to the new phi
            ierr = PetscLogStagePush(stages[0]); CHKERRQ(ierr);
            ierr = ComputeLambdaU(user, phi, &lambda, u); CHKERRQ(ierr);
            ierr = PetscLogStagePop(); CHKERRQ(ierr);
    
            //compute F= \sum_k lambda^k_epsnum
            MPI_Allreduce(&lambda, &F, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
            
            // Update the step
            if (F<=Fold) {
                step = step * 1.2;
                step = PetscMin(step, user.stepmax);
            } else {
                step = step / 2.0;
                step = PetscMax(step, user.stepmin);
            }
    
            //Display some stuff
            ierr = PetscPrintf(PETSC_COMM_WORLD, "F = %e, step = %e, error = %e, mu = %e\n\n", F, step, error, user.mu); CHKERRQ(ierr);
    
            //Save the same stuff in "Partition.log"
            ierr = PetscViewerASCIIPrintf(viewer, "%d   %e   ", it, F); CHKERRQ(ierr);
            ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%e   ", lambda); CHKERRQ(ierr);
            ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
            ierr = PetscViewerASCIIPrintf(viewer, "%e \n", error, it); CHKERRQ(ierr);
          
          
            if (it%modsave == 0){
                if (showphi == PETSC_TRUE) {
                    ierr = ShowComposite_Phi(phi, filename); CHKERRQ(ierr);
                }                
                ierr = PetscLogStagePush(stages[1]); CHKERRQ(ierr);
                if (SaveTXT == PETSC_TRUE){
                    sprintf(filename, "%s%.3d-level%d%s", u_prfx, myrank, level, txt_sfx);
                    ierr = VecView_TXT(u, filename); CHKERRQ(ierr);
                    sprintf(filename, "%s%.3d-level%d%s", phi_prfx, myrank, level, txt_sfx);
                    ierr = VecView_TXT(phi, filename); CHKERRQ(ierr);
                }
                if (SaveVTK == PETSC_TRUE){
                    sprintf(filename, "%s%.3d-level%d%s", u_prfx, myrank, level, vtk_sfx);
                    ierr = VecView_VTKASCII(u, filename); CHKERRQ(ierr);
                    sprintf(filename, "%s%.3d-level%d%s", phi_prfx, myrank, level, vtk_sfx);
                    ierr = VecView_VTKASCII(phi, filename); CHKERRQ(ierr);
                }
                if (SaveEnsight == PETSC_TRUE){
                    sprintf(filename, "%s%.3d-level%d%s", u_prfx, myrank, level, res_sfx);
                    ierr = VecView_EnsightASCII(u, filename); CHKERRQ(ierr);
                    sprintf(filename, "%s%.3d-level%d%s", phi_prfx, myrank, level, res_sfx);
                    ierr = VecView_EnsightASCII(phi, filename); CHKERRQ(ierr);
                }
                if (SaveComposite == PETSC_TRUE){
                    sprintf(filename, "Partition_Phi_all-level%d.txt", level);
                    ierr = SaveComposite_Phi(phi, filename); CHKERRQ(ierr);
        
                    sprintf(filename, "Partition_U_all-level%d.txt", level);
                    ierr = SaveComposite_U(u, filename); CHKERRQ(ierr);
                }
                ierr = PetscLogStagePop(); CHKERRQ(ierr);
            }
    
        // This must be the most convoluted stopping crterion. there HAS to be something better!
        } while ( (it < 20 ) || ( ( it < maxit ) && ((error > tol) || (FinalPass == PETSC_FALSE)) ) );
    
        ierr = PetscLogStagePush(stages[1]); CHKERRQ(ierr);
        if (SaveTXT == PETSC_TRUE){
            sprintf(filename, "%s%.3d-level%d%s", u_prfx, myrank, level, txt_sfx);
            ierr = VecView_TXT(u, filename); CHKERRQ(ierr);
            sprintf(filename, "%s%.3d-level%d%s", phi_prfx, myrank, level, txt_sfx);
            ierr = VecView_TXT(phi, filename); CHKERRQ(ierr);
        }
        if (SaveVTK == PETSC_TRUE){
            sprintf(filename, "%s%.3d-level%d%s", u_prfx, myrank, level, vtk_sfx);
            ierr = VecView_VTKASCII(u, filename); CHKERRQ(ierr);
            sprintf(filename, "%s%.3d-level%d%s", phi_prfx, myrank, level, vtk_sfx);
            ierr = VecView_VTKASCII(phi, filename); CHKERRQ(ierr);
        }
        if (SaveEnsight == PETSC_TRUE){
            sprintf(filename, "%s%.3d-level%d%s", u_prfx, myrank, level, res_sfx);
            ierr = VecView_EnsightASCII(u, filename); CHKERRQ(ierr);
            sprintf(filename, "%s%.3d-level%d%s", phi_prfx, myrank, level, res_sfx);
            ierr = VecView_EnsightASCII(phi, filename); CHKERRQ(ierr);
        }
        if (SaveComposite == PETSC_TRUE){
            sprintf(filename, "Partition_Phi_all-level%d.txt", level);
            ierr = SaveComposite_Phi(phi, filename); CHKERRQ(ierr);
    
            sprintf(filename, "Partition_U_all-level%d.txt", level);
            ierr = SaveComposite_U(u, filename); CHKERRQ(ierr);
        }
        ierr = PetscLogStagePop(); CHKERRQ(ierr);
        
        /* 
        Refine the grid and interpolate
        */
        if (level < user.numlevels-1){
            step = user.step;
        
            // Close the log file and open a new one
            ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
            sprintf(filename, "Partition-level%d.log", level+1);
            ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer); CHKERRQ(ierr);

            // Save phi
            ierr = VecCopy(phi, phi_err); CHKERRQ(ierr);
            
            // Destroy the fields, the matrix and the eps 
            ierr = VecDestroy(u); CHKERRQ(ierr);
            ierr = VecDestroy(phi); CHKERRQ(ierr);
            ierr = VecDestroy(G); CHKERRQ(ierr);
            ierr = DADestroy(user.da); CHKERRQ(ierr);
            ierr = MatDestroy(user.K); CHKERRQ(ierr);
            ierr = EPSDestroy(user.eps); CHKERRQ(ierr);
            
            // Create a temporary DA for the interpolation
            if (user.per) {
        	if (user.ndim == 2) {
                    ierr = DACreate2d(PETSC_COMM_SELF, DA_XYPERIODIC, DA_STENCIL_STAR, user.nx, user.ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &dac); CHKERRQ(ierr);
		} else {
		    ierr = DACreate3d(PETSC_COMM_SELF, DA_XYZPERIODIC, DA_STENCIL_STAR, user.nx, user.ny, user.nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dac); CHKERRQ(ierr);
		}
            } else {
                ierr = DACreate(PETSC_COMM_SELF, user.ndim, DA_NONPERIODIC, DA_STENCIL_STAR, user.nx, user.ny, user.nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dac); CHKERRQ(ierr);
            }
            
	    // Refine the DA and get the new sizes
	    ierr = DARefine(dac, PETSC_COMM_SELF, &user.da); CHKERRQ(ierr);
	    ierr = DAGetInfo(user.da, 0, &user.nx, &user.ny, &user.nz, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
   
            // Create the interpolation matrix K
            ierr = DAGetInterpolation(dac, user.da, &user.K, 0); CHKERRQ(ierr);
            
            // Create the new fields
            ierr = DACreateGlobalVector(user.da, &phi); CHKERRQ(ierr);
            ierr = VecDuplicate(phi, &u); CHKERRQ(ierr);
            ierr = VecDuplicate(phi, &G); CHKERRQ(ierr);
            
            // Project phi
            ierr = MatMult(user.K, phi_err, phi); CHKERRQ(ierr);
            
            // Destroy and re-create phi_err
            ierr = VecDestroy(phi_err); CHKERRQ(ierr);
            ierr = VecDuplicate(phi, &phi_err); CHKERRQ(ierr);
            
            // Destroy user.K
            ierr = MatDestroy(user.K); CHKERRQ(ierr);

            // recreate the eps and related stuff
            ierr = EPSCreate(PETSC_COMM_SELF, &user.eps); CHKERRQ(ierr);
            ierr = DAGetMatrix(user.da, MATSEQAIJ, &user.K); CHKERRQ(ierr);
            ierr = EPSSetOperators(user.eps, user.K, PETSC_NULL); CHKERRQ(ierr);
            ierr = EPSSetProblemType(user.eps, EPS_HEP); CHKERRQ(ierr);
            ierr = EPSGetST(user.eps, &st); CHKERRQ(ierr);
            ierr = EPSSetDimensions(user.eps, user.epsnum, 5*user.epsnum); CHKERRQ(ierr);
            
            ierr = STSetType(st, st_type); CHKERRQ(ierr);
            ierr = STSetShift(st, st_shift); CHKERRQ(ierr);
            
            ierr = STGetKSP(st, &eps_ksp); CHKERRQ(ierr);
            ierr = KSPGetPC(eps_ksp, &eps_pc); CHKERRQ(ierr);
            
            ierr = PCSetType(eps_pc, PCICC); CHKERRQ(ierr);
            ierr = KSPSetType(eps_ksp, KSPCG); CHKERRQ(ierr);
            
            ierr = STSetFromOptions(st); CHKERRQ(ierr);
            ierr = EPSSetFromOptions(user.eps); CHKERRQ(ierr);
        }

    }   
    // Be nice and deallocate
    VecDestroy(phi);
    VecDestroy(phi_err);
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
#define __FUNCT__ "SaveComposite_Phi"
PetscErrorCode SaveComposite_Phi(Vec phi, const char filename[]){
    PetscErrorCode ierr;
    Vec            psi, phi2;
    PetscMPIInt    myrank;
    PetscScalar    *phi_array, *psi_array;
    int            N;
    
    PetscFunctionBegin;
    
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    ierr = VecGetSize(phi, &N); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &phi2); CHKERRQ(ierr);

    ierr = VecCopy(phi, phi2); CHKERRQ(ierr);
    ierr = VecScale(phi2, (PetscScalar) myrank+1.0); CHKERRQ(ierr);
    ierr = VecGetArray(phi2, &phi_array); CHKERRQ(ierr);
    ierr = VecGetArray(psi,  &psi_array); CHKERRQ(ierr);            
    MPI_Allreduce(phi_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    ierr = VecRestoreArray(phi2, &phi_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(psi,  &psi_array); CHKERRQ(ierr);
    ierr = VecView_TXT(psi, filename); CHKERRQ(ierr);
    ierr = VecDestroy(psi); CHKERRQ(ierr);
    ierr = VecDestroy(phi2); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ShowComposite_Phi"
PetscErrorCode ShowComposite_Phi(Vec phi, const char filename[]){
    PetscErrorCode ierr;
    Vec            psi, phi2;
    PetscMPIInt    myrank;
    PetscScalar    *phi_array, *psi_array;
    int            N;
    
    PetscFunctionBegin;
    
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    ierr = VecGetSize(phi, &N); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);
    ierr = VecDuplicate(phi, &phi2); CHKERRQ(ierr);

    ierr = VecCopy(phi, phi2); CHKERRQ(ierr);
    ierr = VecScale(phi2, (PetscScalar) myrank+1.0); CHKERRQ(ierr);
    ierr = VecGetArray(phi2, &phi_array); CHKERRQ(ierr);
    ierr = VecGetArray(psi,  &psi_array); CHKERRQ(ierr);            
    MPI_Allreduce(phi_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    ierr = VecRestoreArray(phi2, &phi_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(psi,  &psi_array); CHKERRQ(ierr);
    if (myrank == 0){    
        ierr = VecView(psi, PETSC_VIEWER_DRAW_SELF); CHKERRQ(ierr);
    }
    ierr = VecDestroy(psi); CHKERRQ(ierr);
    ierr = VecDestroy(phi2); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SaveComposite_U"
PetscErrorCode SaveComposite_U(Vec u, const char filename[]){
    PetscErrorCode ierr;
    Vec            psi;
    PetscScalar    *u_array, *psi_array;
    int            N;
    
    PetscFunctionBegin;
    ierr = VecGetSize(u, &N); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &psi); CHKERRQ(ierr);

    ierr = VecGetArray(u, &u_array); CHKERRQ(ierr);
    ierr = VecGetArray(psi,  &psi_array); CHKERRQ(ierr);            
    MPI_Allreduce(u_array, psi_array, N, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    ierr = VecRestoreArray(u, &u_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(psi,  &psi_array); CHKERRQ(ierr);
    ierr = VecView_TXT(psi, filename); CHKERRQ(ierr);

    ierr = VecDestroy(psi); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeK2d"
PetscErrorCode ComputeK2d(AppCtx user, Vec phi)
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
#define __FUNCT__ "ComputeK3d"
PetscErrorCode ComputeK3d(AppCtx user, Vec phi)
{
    Mat				 K	 = user.K;
    DA				 da = user.da;
    PetscErrorCode   ierr;
    PetscInt		 i, j, k, xm, ym, zm, xs, ys, zs;
    PetscScalar	     v[7],Hx,Hy,Hz, HyHzdHx, HxHzdHy, HxHydHz ;
    MatStencil		 row,col[7];
    PetscScalar	     ***local_phi;
    
    PetscFunctionBegin;
    ierr = DAVecGetArray(user.da, phi, &local_phi); CHKERRQ(ierr);
   
    Hx = 1.0 / (PetscReal)(user.nx-1); 
    Hy = 1.0 / (PetscReal)(user.ny-1);
    Hz = 1.0 / (PetscReal)(user.nz-1);
    HyHzdHx = Hy*Hz/Hx;
    HxHzdHy = Hx*Hz/Hy;
    HxHydHz = Hx*Hy/Hz;
    
    ierr = DAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);

    for (k=zs; k<zs+zm; k++){
        for (j=ys; j<ys+ym; j++){
            for(i=xs; i<xs+xm; i++){
                row.i = i; row.j = j; row.k=k;
                if ( (!user.per) && (i==0 || j==0 || k==0 || i==user.nx-1 || j==user.ny-1 || k==user.nz-1)){
                    v[0]  = 2.0*(HyHzdHx + HxHzdHy + HxHydHz);
                    v[0] += user.mu * (1.0 - local_phi[k][j][i]);
                    ierr = MatSetValuesStencil(K, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
                } else {
                    v[0]  = -HxHydHz;                          col[0].i = i;   col[0].j = j;   col[0].k = k-1;
                    v[1]  = -HxHzdHy;                          col[1].i = i;   col[1].j = j-1; col[1].k = k;
                    v[2]  = -HyHzdHx;                          col[2].i = i-1; col[2].j = j;   col[2].k = k;
                    v[3]  = 2.0*(HyHzdHx + HxHzdHy + HxHydHz); col[3].i = i;   col[3].j = j;   col[3].k = k;
                    v[3] += user.mu * (1.0 - local_phi[k][j][i])*Hx*Hy;
                    v[4]  = -HyHzdHx;                          col[4].i = i+1; col[4].j = j;   col[4].k = k;
                    v[5]  = -HxHzdHy;                          col[5].i = i;   col[5].j = j+1; col[5].k = k;
                    v[6]  = -HxHydHz;                          col[6].i = i;   col[6].j = j;   col[6].k = k+1;
                    ierr = MatSetValuesStencil(K, 1 ,&row, 7, col, v, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
    }
    ierr = DAVecRestoreArray(user.da, phi, &local_phi); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
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
    
    if (user.ndim == 2){
        ierr = ComputeK2d(user, phi); CHKERRQ(ierr);
    } else {
        ierr = ComputeK3d(user, phi); CHKERRQ(ierr);
    }
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
    if (user.ndim==2){
        *lambda = *lambda * (PetscReal)(user.nx-1) * (PetscReal)(user.ny-1) / 2.0; 
    } else {
        *lambda = *lambda * (PetscReal)(user.nx-1) * (PetscReal)(user.ny-1) * (PetscReal)( user.nz - 1) / 2.0; 
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "        lambda[%d] = %e    EPSSolve converged in %f s for %d iterations\n", myrank, *lambda, eps_t, its); CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitPhiRandom"
extern PetscErrorCode InitPhiRandom(AppCtx user, Vec phi){
    PetscRandom	   rndm;
    PetscLogDouble tim;
    PetscMPIInt    rank, size;
    MPI_Comm       comm;
    PetscErrorCode ierr;
    unsigned long  seed;
    
    PetscFunctionBegin;
    ierr = PetscObjectGetComm((PetscObject) phi, &comm); CHKERRQ(ierr);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    ierr = PetscRandomCreate(comm, &rndm); CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rndm); CHKERRQ(ierr);
    ierr = PetscGetTime(&tim); CHKERRQ(ierr);
    seed = (unsigned long) (rank +1) * (unsigned long) tim;
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%i] Seed is %d\n", rank, seed); CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(rndm, seed); CHKERRQ(ierr);
    ierr = PetscRandomSeed(rndm); CHKERRQ(ierr);
    
    ierr = VecSetRandom(phi, rndm); CHKERRQ(ierr);
//    ierr = VecScale(phi, (PetscScalar) 1.0 / (PetscScalar) size); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(rndm); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeG"
extern PetscErrorCode ComputeG(AppCtx user, Vec G, Vec u){
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;
    ierr = VecPointwiseMult(G, u, u); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitEnsight"
PetscErrorCode InitEnsight(AppCtx user, const char u_prfx[], const char phi_prfx[], const char res_sfx[], int level){
    PetscErrorCode   ierr;
    MPI_Comm         comm;
    PetscMPIInt	     rank,numprocs;
    int              i;
    PetscViewer      viewer;
    char			 filename [ FILENAME_MAX ];

        
    PetscFunctionBegin;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &numprocs); CHKERRQ(ierr);
    if (!rank){
        sprintf(filename, "Partition-level%i.geo", level);
        ierr = DAView_GEOASCII(user.da, filename); CHKERRQ(ierr);
        sprintf(filename, "Partition-level%i.case", level);
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "FORMAT\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "type:  ensight gold\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "GEOMETRY\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "model: Partition-level%i.geo\n", level); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "VARIABLE\n"); CHKERRQ(ierr);
        for (i=0; i<numprocs; i++){
            ierr = PetscViewerASCIIPrintf(viewer, "scalar per node: U%i %s%.3d-level%i%s\n", i, u_prfx, i, level, res_sfx); CHKERRQ(ierr);
        }
        for (i=0; i<numprocs; i++){
            ierr = PetscViewerASCIIPrintf(viewer, "scalar per node: PHI%i %s%.3d-level%i%s\n", i, phi_prfx, i, level, res_sfx); CHKERRQ(ierr);
        }
        ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    }
    
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
    PetscInt          l, i, N;
    PetscMPIInt		  myrank, numprocs;
    PetscErrorCode    ierr;
    
    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    MPI_Comm_size(PETSC_COMM_WORLD, &numprocs);
    ierr = VecGetSize(phi, &N); CHKERRQ(ierr);
    
    ierr = VecDuplicate(phi, &psi); CHKERRQ(ierr);
    ierr = VecGetArray(psi, &psi_array); CHKERRQ(ierr);
    ierr = VecGetArray(phi, &phi_array); CHKERRQ(ierr);
    
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
      
      
      
      
      
      
      
