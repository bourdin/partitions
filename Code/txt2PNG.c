static char help[] = "Converts a matrix of values into a PNG file\n\n";

#include <stdio.h>
#include <stdlib.h>

#include "petscvec.h"
#include "petscda.h"

#include "png.h"


extern PetscErrorCode VecViewPNGJet(Vec x, const char filename[]);
extern PetscErrorCode VecReadTXT(Vec x, const char filename[]);
extern int JetR(PetscScalar x);
extern int JetG(PetscScalar x);
extern int JetB(PetscScalar x);


#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char ** argv) {
    char            filename [PETSC_MAX_PATH_LEN];
    char            prefix [PETSC_MAX_PATH_LEN];
    Vec             x;
    DA              da;
    int             nx, ny;
    PetscTruth		flag;
    PetscErrorCode  ierr;

    PetscFunctionBegin;
    PetscInitialize(&argc,&argv,(char *)0,help);

    ierr = PetscOptionsGetInt(PETSC_NULL, "-nx", &nx, &flag); CHKERRQ(ierr);
    if (flag == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG,"Mising -nx flag");
    ierr = PetscOptionsGetInt(PETSC_NULL, "-ny", &ny, &flag); CHKERRQ(ierr);
    if (flag == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG,"Missing -ny flag");
    ierr = PetscOptionsGetString(PETSC_NULL, "-f", prefix, PETSC_MAX_PATH_LEN-1, &flag);CHKERRQ(ierr);
    if (flag == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG,"Missing -f flag");

    ierr = DACreate2d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &da); CHKERRQ(ierr);
    
    ierr = DACreateGlobalVector(da, &x);CHKERRQ(ierr);
    
    
    sprintf(filename, "%s.txt", prefix);
    ierr = VecReadTXT(x, filename);CHKERRQ(ierr);
    ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    
    sprintf(filename, "%s.png", prefix);
    ierr = VecViewPNGJet(x, filename);CHKERRQ(ierr);
    
    ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    PetscFinalize();
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecReadTXT"
PetscErrorCode VecReadTXT(Vec x, const char filename[]){
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
#define __FUNCT__ "VecViewPNGJet"
PetscErrorCode VecViewPNGJet(Vec x, const char filename[]){
    Vec				 natural, io;
    VecScatter		 tozero;
    PetscMPIInt	     rank, size;
    int				 N;
    PetscScalar	     *io_array;
    DA				 da;
    int				 i, j, mx, my, mz;
    MPI_Comm         comm;
    png_structp      png_ptr;
    png_infop        info_ptr;
    FILE             *fp;
    png_byte**       image;
    png_byte**       row_pointers;
    const int        bytes_per_pixel = 3;
    PetscErrorCode   ierr;
    PetscScalar      xmin, xmax, rescaledpix;
    
    
    
    PetscFunctionBegin;    
    
    
    ierr = VecMin(x, PETSC_NULL, &xmin);CHKERRQ(ierr);
    ierr = VecMax(x, PETSC_NULL, &xmax);CHKERRQ(ierr);
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
    
    if (!rank) {
        ierr = PetscFOpen(PETSC_COMM_SELF, filename, "wb", &fp); CHKERRQ(ierr);
        
        // Begin init struct
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
				    NULL, // ptr to user error struct
				    NULL, // ptr to user error function
				    NULL);// ptr to user warning function
				    
			  if (!png_ptr) {
          SETERRQ(1,"Failed to init png_ptr");
        }
        
        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
          png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
          SETERRQ(1,"Failed to init info_ptr");
        }
        // end of init struct

        // Begin of writing header
        // prepare for errors.  Can't call setjmp() before png_ptr has been
        // allocated because the data inside png_ptr is access by png_jmpbuf!
        if (setjmp(png_jmpbuf(png_ptr))) {  // png_jmpbuf is a macro in pngconf.h
          png_destroy_read_struct(&png_ptr, &info_ptr, NULL /*end_info*/);
          SETERRQ(1,"initReadStructs: setjmp returned non-zero (i.e. an error occured.)\n");
        }

        png_init_io(png_ptr, fp);
        
        // prepare for errors.
        if (setjmp(png_jmpbuf(png_ptr))) {  // png_jmpbuf is a macro in pngconf.h
          png_destroy_read_struct(&png_ptr, &info_ptr, NULL /*end_info*/);
          SETERRQ(1,"writeHeader: setjmp returned non-zero (i.e. an error occured.)\n");
        }

        png_set_IHDR(png_ptr, info_ptr, mx, my,
               8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(png_ptr, info_ptr);
        // End of writing header
        
        image = (png_byte**)malloc(my * sizeof(png_byte*));
        row_pointers = (png_byte**)malloc(my * sizeof(png_byte*));
  

        // Not doing 3d DA yet
        for(i=0; i<my; i++) {
          image[i] = (png_byte*)malloc(mx * bytes_per_pixel * sizeof(png_byte));
          row_pointers[my-i-1] = image[i];
          for (j=0; j<mx; j++) {
            rescaledpix = 255* PetscMax(0,PetscMin(1, (io_array[i*mx + j]-xmin)/(xmax-xmin)));
            image[i][j*bytes_per_pixel+0] = JetR(rescaledpix);
            image[i][j*bytes_per_pixel+1] = JetG(rescaledpix);
            image[i][j*bytes_per_pixel+2] = JetB(rescaledpix);
          }
        }
        // Not doing 3d DA yet
        
        png_write_image(png_ptr, row_pointers);
        
        for(i=0;i<my;i++) free(image[i]);
        free(image);
        free(row_pointers);
        
        // Begin write end
        // prepare for errors.
        if (setjmp(png_jmpbuf(png_ptr))) {  // png_jmpbuf is a macro in pngconf.h
          png_destroy_read_struct(&png_ptr, &info_ptr, NULL /*end_info*/);
          SETERRQ(1,"writeEnd: setjmp returned non-zero (i.e. an error occured.)");
        }

        png_write_end(png_ptr, NULL);  // NULL because we don't need to
        // write any comments, etc.
        // End write end
        ierr = PetscFClose(PETSC_COMM_SELF, fp);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(io, &io_array); CHKERRQ(ierr);		
    ierr = VecDestroy(io); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "JetR"
int JetR(PetscScalar x){
    int R;
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;    
    
    if (x<159){
        R = PetscMin( 255, PetscMin(4*(x+32), 4*(159-x)));
    } else {
        R = 0;
    }
    
    PetscFunctionReturn(R);
}

#undef __FUNCT__
#define __FUNCT__ "JetG"
int JetG(PetscScalar x){
    int              G;
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;    
    
    if ( (x>32) && (x<224) ){
        G = PetscMin( 255, PetscMin(4*(x-31), 4*(223-x)));
    } else {
        G = 0;
    }
    
    PetscFunctionReturn(G);
}

#undef __FUNCT__
#define __FUNCT__ "JetB"
int JetB(PetscScalar x){
    int              B;
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;    
    
    if (x>95){
        B = PetscMin( 255, PetscMin(4*(x-95), 4*(287-x)));
    } else {
        B = 0;
    }
    
    PetscFunctionReturn(B);
}
