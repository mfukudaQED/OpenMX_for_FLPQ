/*
  cohp.c is a program to post-process a .cohp file to calculate COHP.
  Usage:  ./cohp Cgra.cohp
*/

 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Inputtools.h"

#define eV2Hartree 27.2113845
#define PI 3.1415926535897932384626
#define fp_bsize         1048576     /* buffer size for setvbuf */
#define YOUSO10  500
#define show_flag            0

typedef struct DCOMPLEX{double r,i;} dcomplex;  

 
char *Delete_path_extension(char *s0, char *r0);

int main(int argc, char **argv)
{
  char *file_cohp;
  int SpinP_switch,NumES;
  int COHP_num_pairs,pair_index;
  int i_vec[20];
  double *****ko;
  double ********COHP;
  int spin,tnoA,tnoB,i,j,k,m,p,method;
  int GA_AN,GB_AN,l1,l2,l3,k1,k2,k3,Numk1,Numk2,Numk3;
  int *GA_COHP,*GB_COHP,*tnoA_COHP,*tnoB_COHP;
  int *l1_COHP,*l2_COHP,*l3_COHP;
  double gaussian,tmp,sum0,sum; 
  char buf[2000],buf2[2000];
  double *tweight,*pweight,*weight;
  double nfac,emin,emax,de,e0,e,d,coe;
  int Solver;
  char basename[YOUSO10];
  char file1[YOUSO10];
  FILE *fp_cohp,*fp;
  int Dos_N;
  double ChemP;

  /**************************************************
                read data from file_cohp
  ***************************************************/

  file_cohp = argv[1];

  //printf("file_cohp=%s\n",file_cohp);

  fp_cohp = fopen(file_cohp, "r");  

  if (fp_cohp==NULL){
    printf("%s cannot be found.\n",file_cohp);
  }

  else{

    fread(&Solver,sizeof(int),1,fp_cohp);
    if (show_flag) printf("Solver=%2d\n",Solver);

    fread(&SpinP_switch,sizeof(int),1,fp_cohp);
    if (show_flag) printf("SpinP_switch=%2d\n",SpinP_switch);
    if (SpinP_switch==3){
      printf("NC-DFT is not supported.\n"); 
      exit(0);
    } 

    fread(&Numk1,sizeof(int),1,fp_cohp);
    fread(&Numk2,sizeof(int),1,fp_cohp);
    fread(&Numk3,sizeof(int),1,fp_cohp);

    if (show_flag){
      printf("Numk1=%2d Numk2=%2d Numk3=%2d\n",Numk1,Numk2,Numk3);
    } 

    fread(&NumES,sizeof(int),1,fp_cohp);
    if (show_flag) printf("NumES=%2d\n",NumES);

    fread(&COHP_num_pairs,sizeof(int),1,fp_cohp);
    if (show_flag) printf("COHP_num_pairs=%2d\n",COHP_num_pairs);

    fread(&ChemP,sizeof(double),1,fp_cohp);
    if (show_flag) printf("ChemP=%15.12f\n",ChemP);

    GA_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    GB_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    tnoA_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    tnoB_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    l1_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    l2_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    l3_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);

    ko = (double*****)malloc(sizeof(double****)*Numk1);
    for (k1=0; k1<Numk1; k1++){
      ko[k1] = (double****)malloc(sizeof(double***)*Numk2);
      for (k2=0; k2<Numk2; k2++){
	ko[k1][k2] = (double***)malloc(sizeof(double**)*Numk3);
	for (k3=0; k3<Numk3; k3++){
	  ko[k1][k2][k3] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
	  for (spin=0; spin<(SpinP_switch+1); spin++){
	    ko[k1][k2][k3][spin] = (double*)malloc(sizeof(double)*NumES);
	  }
	}
      }
    }

    COHP = (double********)malloc(sizeof(double*******)*Numk1);
    for (k1=0; k1<Numk1; k1++){
      COHP[k1] = (double*******)malloc(sizeof(double******)*Numk2);
      for (k2=0; k2<Numk2; k2++){
	COHP[k1][k2] = (double******)malloc(sizeof(double*****)*Numk3);
	for (k3=0; k3<Numk3; k3++){
	  COHP[k1][k2][k3] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));

	  for (spin=0; spin<=SpinP_switch; spin++){

	    COHP[k1][k2][k3][spin] = (double****)malloc(sizeof(double***)*NumES);

	    for (p=0; p<NumES; p++){

	      COHP[k1][k2][k3][spin][p] = (double***)malloc(sizeof(double**)*COHP_num_pairs);

	      fread(&i,sizeof(int),1,fp_cohp);
	      if (i!=spin){
		printf("The data seems not to follow the regular format #1.\n"); 
		exit(0);
	      }

	      fread(i_vec,sizeof(int),3,fp_cohp);
              if (i_vec[0]!=k1 || i_vec[1]!=k2 || i_vec[2]!=k3){
		printf("The data seems not to follow the regular format #2.\n"); 
		exit(0);
	      }

	      fread(&ko[k1][k2][k3][spin][p],sizeof(double),1,fp_cohp);
	      if (show_flag) printf("k1=%2d k2=%2d k3=%2d p=%2d ko=%15.12f\n",k1,k2,k3,p,ko[k1][k2][k3][spin][p]);

	      for (i=0; i<COHP_num_pairs; i++){

		fread(&j,sizeof(int),1,fp_cohp);

		if (i!=j){
		  printf("The data seems not to follow the regular format #3.\n"); 
		  exit(0);
		}

		fread(&GA_AN,sizeof(int),1,fp_cohp);
		fread(&GB_AN,sizeof(int),1,fp_cohp);
		fread(&tnoA,sizeof(int),1,fp_cohp);
		fread(&tnoB,sizeof(int),1,fp_cohp);
		fread(&l1,sizeof(int),1,fp_cohp);
		fread(&l2,sizeof(int),1,fp_cohp);
		fread(&l3,sizeof(int),1,fp_cohp);

		GA_COHP[i] = GA_AN; 
		GB_COHP[i] = GB_AN; 
		tnoA_COHP[i] = tnoA;
		tnoB_COHP[i] = tnoB;
		l1_COHP[i] = l1;
		l2_COHP[i] = l2;
		l3_COHP[i] = l3;

		COHP[k1][k2][k3][spin][p][i] = (double**)malloc(sizeof(double*)*tnoA);

		for (j=0; j<tnoA; j++){

		  COHP[k1][k2][k3][spin][p][i][j] = (double*)malloc(sizeof(double)*tnoB);
		  fread(&COHP[k1][k2][k3][spin][p][i][j][0],sizeof(double),tnoB,fp_cohp);

		  for (k=0; k<tnoB; k++){
		    if (show_flag) printf("spin=%2d p=%2d i=%2d j=%2d k=%2d COHPC=%15.12f\n",
					  spin,p,i,j,k,COHP[k1][k2][k3][spin][p][i][j][k]); 
		  }

		} // j
	      } // i
	    } // p
	  } // spin
	}  // k3
      } // k2
    } // k1

  } // else 

  /* get basename */

  Delete_path_extension(file_cohp,basename);

  /************************************
      start the calculation of cohp
  ************************************/

  if (Numk1==1 && Numk2==1 && Numk3==1){ // the case with only the gamma point
    printf("Kgrid= 1 1 1 => Gaussian Broadening is employed\n");
    method = 2;  
  }
  else{
    printf("Which method do you use?, Tetrahedron(1), Gaussian Broadening(2)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%d",&method);
  }

  /* tetrahedron method */

  if (method==1){ 




  }

  /* gaussian broadening method */

  else if (method==2){ 

    double spindeg;

    if      (SpinP_switch==0) spindeg = 2.0;
    else if (SpinP_switch==1) spindeg = 1.0;

    printf("Please input a value of gaussian (double) (eV)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%lf",&gaussian); gaussian = gaussian/eV2Hartree; 
    nfac = 1.0/(gaussian*sqrt(PI)*(double)(Numk1*Numk2*Numk3));

    printf("Please input the number of energy meshes\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%i",&Dos_N); 

    tweight = (double*)malloc(sizeof(double)*Dos_N);
    pweight = (double*)malloc(sizeof(double)*Dos_N);
    weight  = (double*)malloc(sizeof(double)*Dos_N);

    printf("Which pair for COHP : (1 to %d)\n",COHP_num_pairs);
    fgets(buf,1000,stdin); sscanf(buf,"%d",&pair_index);
    pair_index--; 
    if (show_flag) printf("pair_index=%2d\n",pair_index); 

    GA_AN = GA_COHP[pair_index];
    GB_AN = GB_COHP[pair_index];
    tnoA = tnoA_COHP[pair_index];
    tnoB = tnoB_COHP[pair_index];
    l1 = l1_COHP[pair_index];
    l2 = l2_COHP[pair_index];
    l3 = l3_COHP[pair_index];

    printf("generating...\n");

    /* find emin and emax */

    emin =  10000.0;
    emax = -10000.0;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (k1=0; k1<Numk1; k1++){
	for (k2=0; k2<Numk2; k2++){
	  for (k3=0; k3<Numk3; k3++){
            if (ko[k1][k2][k3][spin][0]<emin)       emin = ko[k1][k2][k3][spin][0];
            if (emax<ko[k1][k2][k3][spin][NumES-1]) emax = ko[k1][k2][k3][spin][NumES-1];
	  }
	}
      }
    }
 
    emin -= 10.0*gaussian;
    emax += 10.0*gaussian;
    de = (emax - emin)/(double)(Dos_N-1);

    /* calculate cohp */

    for (spin=0; spin<=SpinP_switch; spin++){

      for (m=0; m<Dos_N; m++) tweight[m] = 0.0;

      for (i=0; i<tnoA; i++){

        for (m=0; m<Dos_N; m++) pweight[m] = 0.0;

	for (j=0; j<tnoB; j++){

          for (m=0; m<Dos_N; m++) weight[m] = 0.0;

	  for (k1=0; k1<Numk1; k1++){
	    for (k2=0; k2<Numk2; k2++){
	      for (k3=0; k3<Numk3; k3++){
		for (p=0; p<NumES; p++){
          
		  e0 = ko[k1][k2][k3][spin][p];
                  coe = COHP[k1][k2][k3][spin][p][pair_index][i][j];

		  for (k=0; k<Dos_N; k++){
		    e = emin + de*(double)k;
		    d = (e-e0)/gaussian; 
		    tmp = coe*exp(-d*d)*nfac;
		    weight[k]  += tmp;
		    pweight[k] += tmp;
		    tweight[k] += tmp;
		  }
		}
	      }
	    }
	  }

          /* no sum */

	  sprintf(file1,"%s_i%i_o%i_o%i_s%i.cohp",basename,pair_index+1,i,j,spin);

  	  if ((fp = fopen(file1,"w")) != NULL){

            sum0 = 0.0;
            for (k=0; k<Dos_N; k++){
              e = emin + de*(double)k;
              if (e<=ChemP) sum0 += weight[k]*de;
	    }
            fprintf(fp,"# ChemP=%18.15f (Hartree), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
                        ChemP,sum0*spindeg,sum0*eV2Hartree*spindeg);
            fprintf(fp,"#\n");

            sum = 0.0;
            for (k=0; k<Dos_N; k++){
              e = emin + de*(double)k;
              sum += weight[k]*de;
              fprintf(fp,"%18.15f %18.15f %18.15f\n",(e-ChemP)*eV2Hartree,weight[k]*spindeg,sum*eV2Hartree*spindeg);
	    }

            printf(" %s\n",file1);
	    fclose(fp);
	  }
          else{
	    printf("Failure of saving %s\n",file1);
	  }

	} // j

        /* sum over j */

	sprintf(file1,"%s_i%i_o%i_s%i.cohp",basename,pair_index+1,i,spin);

	if ((fp = fopen(file1,"w")) != NULL){

	  sum0 = 0.0;
	  for (k=0; k<Dos_N; k++){
	    e = emin + de*(double)k;
	    if (e<=ChemP) sum0 += pweight[k]*de;
	  }
	  fprintf(fp,"# ChemP=%18.15f (Hartree), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
                      ChemP,sum0*spindeg,sum0*eV2Hartree*spindeg);
	  fprintf(fp,"#\n");

	  sum = 0.0;
	  for (k=0; k<Dos_N; k++){
	    e = emin + de*(double)k;
	    sum += pweight[k]*de;
	    fprintf(fp,"%18.15f %18.15f %18.15f\n",(e-ChemP)*eV2Hartree,pweight[k]*spindeg,sum*eV2Hartree*spindeg);
	  }

	  fclose(fp);
          printf(" %s\n",file1);

	}
	else{
	  printf("Failure of saving %s\n",file1);
	}

      } // i     

      /* sum over i and j */

      sprintf(file1,"%s_i%i_s%i.cohp",basename,pair_index+1,spin);

      if ((fp = fopen(file1,"w")) != NULL){

	sum0 = 0.0;
	for (k=0; k<Dos_N; k++){
	  e = emin + de*(double)k;
	  if (e<=ChemP) sum0 += tweight[k]*de;
	}
        fprintf(fp,"# ChemP=%18.15f (Hartree), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
                    ChemP,sum0*spindeg,sum0*eV2Hartree*spindeg);
	fprintf(fp,"#\n");

        sum = 0.0;
	for (k=0; k<Dos_N; k++){
	  e = emin + de*(double)k;
          sum += tweight[k]*de;
	  fprintf(fp,"%18.15f %18.15f %18.15f\n",(e-ChemP)*eV2Hartree,tweight[k]*spindeg,sum*eV2Hartree*spindeg);
	}

	fclose(fp);
        printf(" %s\n",file1);
      }
      else{
	printf("Failure of saving %s\n",file1);
      }

    } // spin
  }  

  /************************************
           freeing of arrays
  ************************************/

  if (method==2){ // gaussian broadening method

    for (k1=0; k1<Numk1; k1++){
      for (k2=0; k2<Numk2; k2++){
	for (k3=0; k3<Numk3; k3++){
	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (p=0; p<NumES; p++){
	      for (i=0; i<COHP_num_pairs; i++){
		for (j=0; j<tnoA_COHP[i]; j++){
		  free(COHP[k1][k2][k3][spin][p][i][j]);
		} // j
		free(COHP[k1][k2][k3][spin][p][i]);
	      } // i
	      free(COHP[k1][k2][k3][spin][p]);
	    } // p
	    free(COHP[k1][k2][k3][spin]);
	  } // spin
	  free(COHP[k1][k2][k3]);
	}  // k3
	free(COHP[k1][k2]);
      } // k2
      free(COHP[k1]);
    } // k1
    free(COHP);

    for (k1=0; k1<Numk1; k1++){
      for (k2=0; k2<Numk2; k2++){
	for (k3=0; k3<Numk3; k3++){
	  for (spin=0; spin<(SpinP_switch+1); spin++){
	    free(ko[k1][k2][k3][spin]);
	  }
	  free(ko[k1][k2][k3]);
	}
	free(ko[k1][k2]);
      }
      free(ko[k1]);
    }
    free(ko);

    free(tweight);
    free(pweight);
    free(weight);
  }

  free(GA_COHP);
  free(GB_COHP);
  free(tnoA_COHP);
  free(tnoB_COHP);
  free(l1_COHP);
  free(l2_COHP);
  free(l3_COHP);

  exit(0);
}





/* 
   input s0
   output r0
   if input='../test.Dos.val' output='test'
*/

char *Delete_path_extension(char *s0, char *r0)
{
  char *c;
  /* char s[YOUSO10]; */

  /* find '/' */
  c=rindex(s0,'/');
  if (c)  {
    strcpy(r0,c+1);
  }
  else {
    strcpy(r0,s0);
  }
  if (show_flag) printf("<%s>\n",r0);

  if (strlen(r0)==0 ) { return  NULL; }
  
  c =index(r0,'.');
  if (c) {
    *c='\0';
  }
  if (show_flag) printf("<%s>\n",r0);
  return r0;

}




