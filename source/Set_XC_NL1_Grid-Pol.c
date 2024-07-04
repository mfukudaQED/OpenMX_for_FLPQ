#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

static int Num_TPF,Num_Grid1,Num_Grid2;
static double NLX_coes[3][41][41];
static double min_rho16,max_rho16,min_R13,max_R13,al_R13,alpha_para;

static double mask_func_q(int m_flag, double q, double alpha_para);
static void gx_kernel3(int mu, double rho, double drho, double gx[3]);
static void gx_kernel5(int mu, double rho, double drho, double gx[3]);
static void set_gx_kernel_coes();


void Set_XC_NL1_Grid(int SCF_iter, int XC_P_switch, int XC_switch, 
		     double *Den0, double *Den1, 
		     double *Den2, double *Den3,
		     double *Vxc0, double *Vxc1,
		     double *Vxc2, double *Vxc3,
		     double ***dEXC_dGD, 
		     double ***dDen_Grid)
{
  /***********************************************************************************************
   XC_P_switch:
      0  \epsilon_XC (XC energy density)  
      1  \mu_XC      (XC potential)  
  ***********************************************************************************************/

  static int firsttime=0;
  int MN,MN1,MN2,i,j,k,ri,ri1,ri2;
  int i1,i2,j1,j2,k1,k2,k3,n,nmax;
  int Ng1,Ng2,Ng3;
  double den_min=1.0e-14; 

  double tmp0,tmp1,gx[3];
  double cot,sit,sip,cop,phi,theta;
  int numprocs,myid;

  double My_Ec[2],Ec[2],XC[2],C[2],X[2],rho0,rho1;
  double G[3],Ex[2],TotEx,re,im;
  double ***Re_a_r,***Im_a_r;
  double **Re_g_r, **Im_g_r, **Re_g_q, **Im_g_q;
  double **Re_rho_r, **Im_rho_r, **Re_rho_q, **Im_rho_q;
  double **dg_drho_r, dg_ddrho_r, ***Re_drho_r, ***Im_drho_r;
  double *ReTmpq0,*ImTmpq0,*ReTmpq1,*ImTmpq1,*ReTmpq2,*ImTmpq2;
  double *ReTmpr0,*ImTmpr0,*ReTmpr1,*ImTmpr1,*ReTmpr2,*ImTmpr2;

  double Re_dx,Re_dy,Re_dz,Im_dx,Im_dy,Im_dz;
  double sk1,sk2,sk3,time,tmpr,tmpi;
  double rho,drho;
  double G2,fmu;
  int spin,spinsize,N2D,GN,GNs,BN_CB,BN_AB,mu;

  /* for OpenMP */
  int OMPID,Nthrds;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
             initialize Vxc0 and Vxc1 
  ****************************************************/
 
  if (SpinP_switch==0){
    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      Vxc0[BN_AB] = 0.0;
    }
  }

  else if (SpinP_switch==1){
    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      Vxc0[BN_AB] = 0.0;
      Vxc1[BN_AB] = 0.0;
    }
  }

  else if (SpinP_switch==3){
    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      Vxc0[BN_AB] = 0.0;
      Vxc1[BN_AB] = 0.0;
      Vxc2[BN_AB] = 0.0;
      Vxc3[BN_AB] = 0.0;
    }
  }

  /****************************************************
                 allocation of arrays
  ****************************************************/

  if (firsttime==0){
    set_gx_kernel_coes();
    firsttime = 1;
  }

  if      (SpinP_switch==0) spinsize = 1;
  else if (SpinP_switch==1) spinsize = 2;
  else if (SpinP_switch==3) spinsize = 2;

  /* Re_a_r, Im_a_r */

  Re_a_r = (double***)malloc(sizeof(double**)*spinsize); 
  for (k=0; k<spinsize; k++){
    Re_a_r[k] = (double**)malloc(sizeof(double*)*3);
    for (i=0; i<3; i++){
      Re_a_r[k][i] = (double*)malloc(sizeof(double)*My_Max_NumGridB);
      for (j=0; j<My_Max_NumGridB; j++) Re_a_r[k][i][j] = 0.0;
    }
  }

  Im_a_r = (double***)malloc(sizeof(double**)*spinsize); 
  for (k=0; k<spinsize; k++){
    Im_a_r[k] = (double**)malloc(sizeof(double*)*3);
    for (i=0; i<3; i++){
      Im_a_r[k][i] = (double*)malloc(sizeof(double)*My_Max_NumGridB);
      for (j=0; j<My_Max_NumGridB; j++) Im_a_r[k][i][j] = 0.0;
    }
  }

  /* Re_g_r, Im_g_r, Re_g_q, Im_g_q */

  Re_g_r = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Re_g_r[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Re_g_r[k][j] = 0.0;
  }

  Im_g_r = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Im_g_r[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Im_g_r[k][j] = 0.0;
  }

  Re_g_q = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Re_g_q[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Re_g_q[k][j] = 0.0;
  }

  Im_g_q = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Im_g_q[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Im_g_q[k][j] = 0.0;
  }

  /* Re_rho_r, Im_rho_r, Re_rho_q, Im_rho_q */

  Re_rho_r = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Re_rho_r[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Re_rho_r[k][j] = 0.0;
  }

  Im_rho_r = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Im_rho_r[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Im_rho_r[k][j] = 0.0;
  }

  Re_rho_q = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Re_rho_q[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Re_rho_q[k][j] = 0.0;
  }

  Im_rho_q = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    Im_rho_q[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) Im_rho_q[k][j] = 0.0;
  }

  /* dg_drho_r */

  dg_drho_r = (double**)malloc(sizeof(double*)*spinsize); 
  for (k=0; k<spinsize; k++){
    dg_drho_r[k] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (j=0; j<My_Max_NumGridB; j++) dg_drho_r[k][j] = 0.0;
  }

  /* Re_drho_r */

  Re_drho_r = (double***)malloc(sizeof(double**)*spinsize); 
  for (k=0; k<spinsize; k++){
    Re_drho_r[k] = (double**)malloc(sizeof(double*)*3);
    for (i=0; i<3; i++){
      Re_drho_r[k][i] = (double*)malloc(sizeof(double)*My_Max_NumGridB);
      for (j=0; j<My_Max_NumGridB; j++) Re_drho_r[k][i][j] = 0.0;
    }
  }

  /* Im_drho_r */

  Im_drho_r = (double***)malloc(sizeof(double**)*spinsize); 
  for (k=0; k<spinsize; k++){
    Im_drho_r[k] = (double**)malloc(sizeof(double*)*3);
    for (i=0; i<3; i++){
      Im_drho_r[k][i] = (double*)malloc(sizeof(double)*My_Max_NumGridB);
      for (j=0; j<My_Max_NumGridB; j++) Im_drho_r[k][i][j] = 0.0;
    }
  }

  /* ReTmpq0, ImTmpq0, ReTmpq1, ImTmpq1, ReTmpq2, and ImTmpq2 */

  ReTmpq0 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpq0 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ReTmpq1 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpq1 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ReTmpq2 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpq2 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 

  ReTmpr0 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpr0 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ReTmpr1 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpr1 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ReTmpr2 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpr2 = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  

  /*
  {
    int i,k;
    double chi[3],t1,t2;
    double rho=1.0e-6;
    double drho=1.0e-1;
    double d;

    rho  = 0.001;
    drho = 0.001;

    d = rho/10000.0;

    k = 0;
    i = 2;

    gx_kernel(k, rho, drho, gx);
    
    printf("A d_gx=%18.15f\n",gx[i]);

    if (i==1){
      gx_kernel(k, rho+d, drho, gx);
      t1 = gx[0];
      gx_kernel(k, rho-d, drho, gx);
      t2 = gx[0];
    }
    else if (i==2){
      gx_kernel(k, rho, drho+d, gx);
      t1 = gx[0];
      gx_kernel(k, rho, drho-d, gx);
      t2 = gx[0];
    }

    printf("N d_gx=%18.15f\n",(t1-t2)/(2.0*d));
    MPI_Finalize();
    exit(0);
  }
  */

  /*
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    if (0.05<fabs(Density_Grid_B[0][BN_AB]+PCCDensity_Grid_B[0][BN_AB])){
      printf("VVV1 Den %2d %18.15f\n",BN_AB,Density_Grid_B[0][BN_AB]+PCCDensity_Grid_B[0][BN_AB]);
    }
  }
  */

  /*
  Density_Grid_B[0][124831] = Density_Grid_B[0][124831] - 0.00000;
  */
  
  /****************************************************
                    calculate drho_r
  ****************************************************/

  tmp0 = 1.0/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3;

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (spin=0; spin<spinsize; spin++){

    time = FFT_Density(9+spin,Re_rho_q[spin],Im_rho_q[spin]);

    for (i=0; i<3; i++){
      for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

	GN = BN_CB + GNs;     
	k3 = GN/(Ngrid2*Ngrid1);    
	k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
	k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

	if (k1<=Ngrid1/2) sk1 = (double)k1;
	else              sk1 = (double)(k1 - Ngrid1);

	if (k2<=Ngrid2/2) sk2 = (double)k2;
	else              sk2 = (double)(k2 - Ngrid2);

	if (k3<=Ngrid3/2) sk3 = (double)k3;
	else              sk3 = (double)(k3 - Ngrid3);

	G[0] = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
	G[1] = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
	G[2] = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];

	ReTmpq1[BN_CB] = -tmp0*G[i]*Im_rho_q[spin][BN_CB];
	ImTmpq1[BN_CB] =  tmp0*G[i]*Re_rho_q[spin][BN_CB];
      }

      Inverse_FFT_Poisson( Re_drho_r[spin][i], Im_drho_r[spin][i], ReTmpq1, ImTmpq1 );

    }
  }

  /************************************************************
     The first part in the functional derivatives.
     \sum_{\mu} \int dr2 f_{\mu} (r12) g_{\mu} (r2)  

     The second part in the functional derivatives.
     \sum_{\mu} dg_drho(r1) \int dr2 f_{\mu} (r12) rho (r2)  

     The third part in the functional derivatives.

  ************************************************************/

  tmp0 = 1.0/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3;

  for (spin=0; spin<spinsize; spin++){

    Ex[spin] = 0.0; 

    for (mu=0; mu<Num_TPF; mu++){

      /* calculate gx and the derivatives of gx with respect to rho and drho */

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

	rho = Density_Grid_B[spin][BN_AB] + PCCDensity_Grid_B[spin][BN_AB];

	drho = sqrt( Re_drho_r[spin][0][BN_AB]*Re_drho_r[spin][0][BN_AB] + Im_drho_r[spin][0][BN_AB]*Im_drho_r[spin][0][BN_AB]
                   + Re_drho_r[spin][1][BN_AB]*Re_drho_r[spin][1][BN_AB] + Im_drho_r[spin][1][BN_AB]*Im_drho_r[spin][1][BN_AB]
                   + Re_drho_r[spin][2][BN_AB]*Re_drho_r[spin][2][BN_AB] + Im_drho_r[spin][2][BN_AB]*Im_drho_r[spin][2][BN_AB] );

        /* gx and its derivatives */

        gx_kernel5(mu, 2.0*rho, 2.0*drho, gx);

        //printf("ZZZ1 BN_AN=%2d  %15.12f %15.12f %15.12f\n",BN_AB,gx[0],gx[1],gx[2]);

        Re_g_r[spin][BN_AB] = gx[0];
        Im_g_r[spin][BN_AB] = 0.0;

        /* In the following, the factor of 2 appears from 
           dg(2rho_{up,down}/d(2rho_{up,down}) * d(2rho_{up,down})/drho_{up,down}.
           The same is true for dg_ddrho_r.
        */

        dg_drho_r[spin][BN_AB] = 2.0*gx[1];  
        dg_ddrho_r = 2.0*gx[2];              

        /* 'a' vector */

	if (drho<1.0e-14){
	  Re_dx = 1.0;
	  Re_dy = 1.0;
	  Re_dz = 1.0;
	  Im_dx = 1.0;
	  Im_dy = 1.0;
	  Im_dz = 1.0;
	}
	else{
	  Re_dx = Re_drho_r[spin][0][BN_AB]/drho;
	  Re_dy = Re_drho_r[spin][1][BN_AB]/drho;
	  Re_dz = Re_drho_r[spin][2][BN_AB]/drho;
	  Im_dx = Im_drho_r[spin][0][BN_AB]/drho;
	  Im_dy = Im_drho_r[spin][1][BN_AB]/drho;
	  Im_dz = Im_drho_r[spin][2][BN_AB]/drho;
	}

	Re_a_r[spin][0][BN_AB] = dg_ddrho_r*Re_dx;
	Im_a_r[spin][0][BN_AB] = dg_ddrho_r*Im_dx;

	Re_a_r[spin][1][BN_AB] = dg_ddrho_r*Re_dy;
	Im_a_r[spin][1][BN_AB] = dg_ddrho_r*Im_dy;

	Re_a_r[spin][2][BN_AB] = dg_ddrho_r*Re_dz;
	Im_a_r[spin][2][BN_AB] = dg_ddrho_r*Im_dz;
      }

      /* FT of g_r */

      FFT_Poisson(Re_g_r[spin], Im_g_r[spin], Re_g_q[spin], Im_g_q[spin]);

      /* Multiplying fmu(G) with g_q */

      for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

	GN = BN_CB + GNs;     
	k3 = GN/(Ngrid2*Ngrid1);    
	k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
	k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

	if (k1<=Ngrid1/2) sk1 = (double)k1;
	else              sk1 = (double)(k1 - Ngrid1);

	if (k2<=Ngrid2/2) sk2 = (double)k2;
	else              sk2 = (double)(k2 - Ngrid2);

	if (k3<=Ngrid3/2) sk3 = (double)k3;
	else              sk3 = (double)(k3 - Ngrid3);

	G[0] = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
	G[1] = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
	G[2] = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];

        G2 = G[0]*G[0] + G[1]*G[1] + G[2]*G[2];        
        fmu = mask_func_q(mu, G2, alpha_para); 

        /* first contribution */

        ReTmpq1[BN_CB] = fmu*tmp0*Re_g_q[spin][BN_CB];
        ImTmpq1[BN_CB] = fmu*tmp0*Im_g_q[spin][BN_CB];

        /* second contribution */

        ReTmpq2[BN_CB] = fmu*tmp0*Re_rho_q[spin][BN_CB];
        ImTmpq2[BN_CB] = fmu*tmp0*Im_rho_q[spin][BN_CB];
      }        

      /* Inverse FT of Tmpq1: first contribution */

      Inverse_FFT_Poisson(ReTmpr1, ImTmpr1, ReTmpq1, ImTmpq1);

      /* store the first contribution to Vxc */

      if (spin==0){
	for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	  Vxc0[BN_AB] += ReTmpr1[BN_AB];
	}
      }

      else if (spin==1){
	for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	  Vxc1[BN_AB] += ReTmpr1[BN_AB];
	}
      }

      /* Inverse FT of g_q: second contribution */

      Inverse_FFT_Poisson(ReTmpr2, ImTmpr2, ReTmpq2, ImTmpq2);

      /* store the second contribution to Vxc */

      if (XC_P_switch==1){

	if (spin==0){
	  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	    Vxc0[BN_AB] += ReTmpr2[BN_AB]*dg_drho_r[spin][BN_AB];
	  }
	}

	else if (spin==1){
	  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	    Vxc1[BN_AB] += ReTmpr2[BN_AB]*dg_drho_r[spin][BN_AB];
	  }
	}
      }

      /* calculation of the third contribution */

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

        re = ReTmpr2[BN_AB]; /* s */
        im = ImTmpr2[BN_AB]; /* s */

        /* h*s */

	ReTmpr0[BN_AB] = re*Re_a_r[spin][0][BN_AB] - im*Im_a_r[spin][0][BN_AB];
	ImTmpr0[BN_AB] = re*Im_a_r[spin][0][BN_AB] + im*Re_a_r[spin][0][BN_AB];

	ReTmpr1[BN_AB] = re*Re_a_r[spin][1][BN_AB] - im*Im_a_r[spin][1][BN_AB];
	ImTmpr1[BN_AB] = re*Im_a_r[spin][1][BN_AB] + im*Re_a_r[spin][1][BN_AB];

	ReTmpr2[BN_AB] = re*Re_a_r[spin][2][BN_AB] - im*Im_a_r[spin][2][BN_AB];
	ImTmpr2[BN_AB] = re*Im_a_r[spin][2][BN_AB] + im*Re_a_r[spin][2][BN_AB];
      }

      /* h*s -> t */ 

      FFT_Poisson( ReTmpr0, ImTmpr0, ReTmpq0, ImTmpq0 );
      FFT_Poisson( ReTmpr1, ImTmpr1, ReTmpq1, ImTmpq1 );
      FFT_Poisson( ReTmpr2, ImTmpr2, ReTmpq2, ImTmpq2 );

      /* G*t */

      for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

	GN = BN_CB + GNs;     
	k3 = GN/(Ngrid2*Ngrid1);    
	k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
	k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

	if (k1<=Ngrid1/2) sk1 = (double)k1;
	else              sk1 = (double)(k1 - Ngrid1);

	if (k2<=Ngrid2/2) sk2 = (double)k2;
	else              sk2 = (double)(k2 - Ngrid2);

	if (k3<=Ngrid3/2) sk3 = (double)k3;
	else              sk3 = (double)(k3 - Ngrid3);

	G[0] = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
	G[1] = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
	G[2] = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];

        re = G[0]*ReTmpq0[BN_CB] + G[1]*ReTmpq1[BN_CB] + G[2]*ReTmpq2[BN_CB];
        im = G[0]*ImTmpq0[BN_CB] + G[1]*ImTmpq1[BN_CB] + G[2]*ImTmpq2[BN_CB];

        ReTmpq0[BN_CB] = re;
        ImTmpq0[BN_CB] = im;

      } /* BN_CB */

      Inverse_FFT_Poisson(ReTmpr0, ImTmpr0, ReTmpq0, ImTmpq0);

      /* store the third contribution to Vxc */

      if (XC_P_switch==1){

	if (spin==0){
	  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	    Vxc0[BN_AB] += ImTmpr0[BN_AB]*tmp0;
	  }
	}

	else if (spin==1){
	  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	    Vxc1[BN_AB] += ImTmpr0[BN_AB]*tmp0;
	  }
	}
      }

      /****************************************************
                       calculation of Ex
      ****************************************************/

      for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

	GN = BN_CB + GNs;     
	k3 = GN/(Ngrid2*Ngrid1);    
	k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
	k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

	if (k1<=Ngrid1/2) sk1 = (double)k1;
	else              sk1 = (double)(k1 - Ngrid1);

	if (k2<=Ngrid2/2) sk2 = (double)k2;
	else              sk2 = (double)(k2 - Ngrid2);

	if (k3<=Ngrid3/2) sk3 = (double)k3;
	else              sk3 = (double)(k3 - Ngrid3);

	G[0] = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
	G[1] = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
	G[2] = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];

        G2 = G[0]*G[0] + G[1]*G[1] + G[2]*G[2];        
        fmu = mask_func_q(mu, G2, alpha_para); 

        Ex[spin] += ( Re_rho_q[spin][BN_CB]*Re_g_q[spin][BN_CB]
                    + Im_rho_q[spin][BN_CB]*Im_g_q[spin][BN_CB])*fmu;

      } /* BN_CB */

    } /* mu */

    Ex[spin] *= (GridVol/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3);

  } /* spin */

  if      (SpinP_switch==0){
    Ex[1] = Ex[0];
    TotEx = 2.0*Ex[0];
  }
  else if (SpinP_switch==1){
    TotEx = Ex[0] + Ex[1];
  }

  /*
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    if (0.01<fabs(Vxc0[BN_AB])) printf("ABC %2d %18.15f\n",BN_AB,Vxc0[BN_AB]);
  }
  */

  /*
  BN_AB = 124831;
  printf("ABC %2d %18.15f\n",BN_AB,Vxc0[BN_AB]);
  */

  printf("Ex %18.15f %18.15f\n",Ex[0],Ex[1]);

  /*
  MPI_Finalize();
  exit(0);
  */

  /****************************************************
       correlation potential and energy by LDA
  ****************************************************/

  My_Ec[0] = 0.0;
  My_Ec[1] = 0.0;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    rho0 = Density_Grid_B[0][BN_AB] + PCCDensity_Grid_B[0][BN_AB];
    rho1 = Density_Grid_B[1][BN_AB] + PCCDensity_Grid_B[1][BN_AB];

    /* correlation energy */
   
    XC_CA_LSDA(1, rho0, rho1, XC, X, C, 0);

    My_Ec[0] += rho0*C[0];
    My_Ec[1] += rho1*C[1];

    /* XC_P_switch=0: correlation energy density
       XC_P_switch=1: correlation potentials */
   
    XC_CA_LSDA(1, rho0, rho1, XC, X, C, XC_P_switch);

    Vxc0[BN_AB] += C[0];
    Vxc1[BN_AB] += C[1];
  }   

  My_Ec[0] *= GridVol; 
  My_Ec[1] *= GridVol; 

  MPI_Allreduce(&My_Ec[0], &Ec[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Ec[1], &Ec[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (myid==Host_ID) printf("Ec %18.15f %18.15f\n",Ec[0],Ec[1]);

  /****************************************************
                 freeing of arrays
  ****************************************************/

  /* Re_a_r, Im_a_r */

  for (k=0; k<spinsize; k++){
    for (i=0; i<3; i++){
      free(Re_a_r[k][i]);
    }
    free(Re_a_r[k]);
  }
  free(Re_a_r);

  for (k=0; k<spinsize; k++){
    for (i=0; i<3; i++){
      free(Im_a_r[k][i]);
    }
    free(Im_a_r[k]);
  }
  free(Im_a_r);

  /* Re_g_r, Im_g_r, Re_g_q, Im_g_q */

  for (k=0; k<spinsize; k++){
    free(Re_g_r[k]);
  }
  free(Re_g_r);

  for (k=0; k<spinsize; k++){
    free(Im_g_r[k]);
  }
  free(Im_g_r);

  for (k=0; k<spinsize; k++){
    free(Re_g_q[k]);
  }
  free(Re_g_q);

  for (k=0; k<spinsize; k++){
    free(Im_g_q[k]);
  }
  free(Im_g_q);

  /* Re_rho_r, Im_rho_r, Re_rho_q, Im_rho_q */

  for (k=0; k<spinsize; k++){
    free(Re_rho_r[k]);
  }
  free(Re_rho_r);

  for (k=0; k<spinsize; k++){
    free(Im_rho_r[k]);
  }
  free(Im_rho_r);

  for (k=0; k<spinsize; k++){
    free(Re_rho_q[k]);
  }
  free(Re_rho_q);

  for (k=0; k<spinsize; k++){
    free(Im_rho_q[k]);
  }
  free(Im_rho_q);

  /* dg_drho_r */

  for (k=0; k<spinsize; k++){
    free(dg_drho_r[k]);
  }
  free(dg_drho_r);

  /* Re_drho_r */

  for (k=0; k<spinsize; k++){
    for (i=0; i<3; i++){
      free(Re_drho_r[k][i]);
    }
    free(Re_drho_r[k]);
  }
  free(Re_drho_r);

  /* Im_drho_r */

  for (k=0; k<spinsize; k++){
    for (i=0; i<3; i++){
      free(Im_drho_r[k][i]);
    }
    free(Im_drho_r[k]);
  }
  free(Im_drho_r);

  /* ReTmpq0, ImTmpq0, ReTmpq1, ImTmpq1, ReTmpq2, and ImTmpq2 */

  free(ReTmpq0);
  free(ImTmpq0);
  free(ReTmpq1);
  free(ImTmpq1);
  free(ReTmpq2);
  free(ImTmpq2);

  free(ReTmpr0);
  free(ImTmpr0);
  free(ReTmpr1);
  free(ImTmpr1);
  free(ReTmpr2);
  free(ImTmpr2);
}


void gx_kernel3(int mu, double rho, double drho, double gx[3])
{
  static int first_flag=0;
  int i,j,ii,jj,i1,j1,N1,N2;
  double tmp0,tmp1,s,s2,s3,s4,s5,t,t2,t3,t4,t5,al;
  double min_R,max_R,rho16,drho16_0,drho16_1;
  double dR0,dR1,R13,R,ln,x,y,d1,d2;
  double f[6][6],df0[6][6],df1[6][6];
  double threshold=1.0e-12;

  /* avoid the numerical instability if rho and drho are small. */

  if ( rho<threshold ){
    gx[0] = 0.0;
    gx[1] = 0.0;
    gx[2] = 0.0;
    return;
  }
  else if ( fabs(drho)<(0.1*threshold) ){
    drho = 0.1*threshold;
  }

  /* setting d1 and d2 */

  al = al_R13;

  min_R = (min_R13)*(min_R13)*(min_R13) + al*(min_R13);
  max_R = (max_R13)*(max_R13)*(max_R13) + al*(max_R13);

  N1 = Num_Grid1;
  N2 = Num_Grid2;
  d1 = (max_rho16-(min_rho16))/(double)N1;
  d2 = (max_R13-(min_R13))/(double)N2;

  /* find rho16 */

  x = rho;
  y = fabs(drho);

  rho16 = pow(x,1.0/6.0);
  drho16_0 = 1.0/6.0*pow(x,-5.0/6.0);
  drho16_1 = 0.0;

  if (rho16<(min_rho16)){
    x = min_rho16*min_rho16*min_rho16*min_rho16*min_rho16*min_rho16;
    rho16 = min_rho16;
    drho16_0 = 1.0/6.0*pow(x,-5.0/6.0);
    drho16_1 = 0.0;
  }
  else if ((max_rho16)<rho16){
    x = max_rho16*max_rho16*max_rho16*max_rho16*max_rho16*max_rho16;
    rho16 = max_rho16;
    drho16_0 = 1.0/6.0*pow(x,-5.0/6.0);
    drho16_1 = 0.0;
  }

  /* find R13 */

  ln = -log(8.0*PI) - 3.0*log(pow(x,4.0/3.0)/y);
  R = ln;
  R13 = -((pow(2.0/3.0,1.0/3.0)*al)/pow(9.0*R + sqrt(3.0)*sqrt(4.0*pow(al,3.0) + 27.0*pow(R,2.0)),1.0/3.0))
    + pow(9.0*R + sqrt(3.0)*sqrt(4.0*pow(al,3.0) + 27.0*pow(R,2.0)),0.3333333333333333)/(pow(2.0,1.0/3.0)*pow(3.0,2.0/3.0));

  dR0 = -4.0/x;
  dR1 = 3.0/y;

  if (R<(min_R)){
    R = min_R;
    R13 = min_R13;
    dR0 = 0.0;
    dR1 = 0.0;
  }
  else if ((max_R)<R){
    R = max_R;
    R13 = max_R13;
    dR0 = 0.0;
    dR1 = 0.0;
  }

  /* find indexes and convert variables to s and t */

  i1 = (int)((rho16 - (min_rho16))/d1);
  if (i1<1)           i1 = 1;
  else if ((N1-2)<i1) i1 = N1-2;

  j1 = (int)((R13 - (min_R13))/d2);
  if (j1<1)           j1 = 1;
  else if ((N2-2)<j1) j1 = N2-2;

  s = (rho16 - (min_rho16+(double)i1*d1))/d1;
  s2 = s*s;
  s3 = s*s2;

  t = (R13 - (min_R13+(double)j1*d2))/d2;
  t2 = t*t;
  t3 = t*t2;

  /* calculate f, df0, and df1 */

  /* f */

  f[0][0] = (s*t)/4. - (s2*t)/2. + (s3*t)/4. - (s*t2)/2. + s2*t2 - (s3*t2)/2. + (s*t3)/4. - (s2*t3)/2. + (s3*t3)/4.;

  f[0][1] = -s/2. + s2 - s3/2. + (5*s*t2)/4. - (5*s2*t2)/2. + (5*s3*t2)/4. - (3*s*t3)/4. + (3*s2*t3)/2. - (3*s3*t3)/4.;

  f[0][2] = -(s*t)/4. + (s2*t)/2. - (s3*t)/4. - s*t2 + 2*s2*t2 - s3*t2 + (3*s*t3)/4. - (3*s2*t3)/2. + (3*s3*t3)/4.;

  f[0][3] = (s*t2)/4. - (s2*t2)/2. + (s3*t2)/4. - (s*t3)/4. + (s2*t3)/2. - (s3*t3)/4.;

  f[1][0] = -t/2. + (5*s2*t)/4. - (3*s3*t)/4. + t2 - (5*s2*t2)/2. + (3*s3*t2)/2. - t3/2. + (5*s2*t3)/4. - (3*s3*t3)/4.;

  f[1][1] = 1 - (5*s2)/2. + (3*s3)/2. - (5*t2)/2. + (25*s2*t2)/4. - (15*s3*t2)/4. + (3*t3)/2. - (15*s2*t3)/4. + (9*s3*t3)/4.;

  f[1][2] = t/2. - (5*s2*t)/4. + (3*s3*t)/4. + 2*t2 - 5*s2*t2 + 3*s3*t2 - (3*t3)/2. + (15*s2*t3)/4. - (9*s3*t3)/4.;

  f[1][3] = -t2/2. + (5*s2*t2)/4. - (3*s3*t2)/4. + t3/2. - (5*s2*t3)/4. + (3*s3*t3)/4.;

  f[2][0] = -(s*t)/4. - s2*t + (3*s3*t)/4. + (s*t2)/2. + 2*s2*t2 - (3*s3*t2)/2. - (s*t3)/4. - s2*t3 + (3*s3*t3)/4.;

  f[2][1] = s/2. + 2*s2 - (3*s3)/2. - (5*s*t2)/4. - 5*s2*t2 + (15*s3*t2)/4. + (3*s*t3)/4. + 3*s2*t3 - (9*s3*t3)/4.;

  f[2][2] = (s*t)/4. + s2*t - (3*s3*t)/4. + s*t2 + 4*s2*t2 - 3*s3*t2 - (3*s*t3)/4. - 3*s2*t3 + (9*s3*t3)/4.;

  f[2][3] = -(s*t2)/4. - s2*t2 + (3*s3*t2)/4. + (s*t3)/4. + s2*t3 - (3*s3*t3)/4.;

  f[3][0] = (s2*t)/4. - (s3*t)/4. - (s2*t2)/2. + (s3*t2)/2. + (s2*t3)/4. - (s3*t3)/4.;

  f[3][1] = -s2/2. + s3/2. + (5*s2*t2)/4. - (5*s3*t2)/4. - (3*s2*t3)/4. + (3*s3*t3)/4.;

  f[3][2] = -(s2*t)/4. + (s3*t)/4. - s2*t2 + s3*t2 + (3*s2*t3)/4. - (3*s3*t3)/4.;

  f[3][3] = (s2*t2)/4. - (s3*t2)/4. - (s2*t3)/4. + (s3*t3)/4.;


  /* derivatives of coefficients of g w.r.t s  */ 

  df0[0][0] = t/4. - s*t + (3*s2*t)/4. - t2/2. + 2*s*t2 - (3*s2*t2)/2. + t3/4. - s*t3 + (3*s2*t3)/4.;

  df0[0][1] = -0.5 + 2*s - (3*s2)/2. + (5*t2)/4. - 5*s*t2 + (15*s2*t2)/4. - (3*t3)/4. + 3*s*t3 - (9*s2*t3)/4.;

  df0[0][2] = -t/4. + s*t - (3*s2*t)/4. - t2 + 4*s*t2 - 3*s2*t2 + (3*t3)/4. - 3*s*t3 + (9*s2*t3)/4.;

  df0[0][3] = t2/4. - s*t2 + (3*s2*t2)/4. - t3/4. + s*t3 - (3*s2*t3)/4.;

  df0[1][0] = (5*s*t)/2. - (9*s2*t)/4. - 5*s*t2 + (9*s2*t2)/2. + (5*s*t3)/2. - (9*s2*t3)/4.;

  df0[1][1] = -5*s + (9*s2)/2. + (25*s*t2)/2. - (45*s2*t2)/4. - (15*s*t3)/2. + (27*s2*t3)/4.;

  df0[1][2] = (-5*s*t)/2. + (9*s2*t)/4. - 10*s*t2 + 9*s2*t2 + (15*s*t3)/2. - (27*s2*t3)/4.;

  df0[1][3] = (5*s*t2)/2. - (9*s2*t2)/4. - (5*s*t3)/2. + (9*s2*t3)/4.;

  df0[2][0] = -t/4. - 2*s*t + (9*s2*t)/4. + t2/2. + 4*s*t2 - (9*s2*t2)/2. - t3/4. - 2*s*t3 + (9*s2*t3)/4.;

  df0[2][1] = 0.5 + 4*s - (9*s2)/2. - (5*t2)/4. - 10*s*t2 + (45*s2*t2)/4. + (3*t3)/4. + 6*s*t3 - (27*s2*t3)/4.;

  df0[2][2] = t/4. + 2*s*t - (9*s2*t)/4. + t2 + 8*s*t2 - 9*s2*t2 - (3*t3)/4. - 6*s*t3 + (27*s2*t3)/4.;

  df0[2][3] = -t2/4. - 2*s*t2 + (9*s2*t2)/4. + t3/4. + 2*s*t3 - (9*s2*t3)/4.;

  df0[3][0] = (s*t)/2. - (3*s2*t)/4. - s*t2 + (3*s2*t2)/2. + (s*t3)/2. - (3*s2*t3)/4.;

  df0[3][1] = -s + (3*s2)/2. + (5*s*t2)/2. - (15*s2*t2)/4. - (3*s*t3)/2. + (9*s2*t3)/4.;

  df0[3][2] = -(s*t)/2. + (3*s2*t)/4. - 2*s*t2 + 3*s2*t2 + (3*s*t3)/2. - (9*s2*t3)/4.;

  df0[3][3] = (s*t2)/2. - (3*s2*t2)/4. - (s*t3)/2. + (3*s2*t3)/4.;

  /* derivatives of coefficients of g w.r.t t  */ 

  df1[0][0] = s/4. - s2/2. + s3/4. - s*t + 2*s2*t - s3*t + (3*s*t2)/4. - (3*s2*t2)/2. + (3*s3*t2)/4.;

  df1[0][1] = (5*s*t)/2. - 5*s2*t + (5*s3*t)/2. - (9*s*t2)/4. + (9*s2*t2)/2. - (9*s3*t2)/4.;

  df1[0][2] = -s/4. + s2/2. - s3/4. - 2*s*t + 4*s2*t - 2*s3*t + (9*s*t2)/4. - (9*s2*t2)/2. + (9*s3*t2)/4.;

  df1[0][3] = (s*t)/2. - s2*t + (s3*t)/2. - (3*s*t2)/4. + (3*s2*t2)/2. - (3*s3*t2)/4.;

  df1[1][0] = -0.5 + (5*s2)/4. - (3*s3)/4. + 2*t - 5*s2*t + 3*s3*t - (3*t2)/2. + (15*s2*t2)/4. - (9*s3*t2)/4.;

  df1[1][1] = -5*t + (25*s2*t)/2. - (15*s3*t)/2. + (9*t2)/2. - (45*s2*t2)/4. + (27*s3*t2)/4.;

  df1[1][2] = 0.5 - (5*s2)/4. + (3*s3)/4. + 4*t - 10*s2*t + 6*s3*t - (9*t2)/2. + (45*s2*t2)/4. - (27*s3*t2)/4.;

  df1[1][3] = -t + (5*s2*t)/2. - (3*s3*t)/2. + (3*t2)/2. - (15*s2*t2)/4. + (9*s3*t2)/4.;

  df1[2][0] = -s/4. - s2 + (3*s3)/4. + s*t + 4*s2*t - 3*s3*t - (3*s*t2)/4. - 3*s2*t2 + (9*s3*t2)/4.;

  df1[2][1] = (-5*s*t)/2. - 10*s2*t + (15*s3*t)/2. + (9*s*t2)/4. + 9*s2*t2 - (27*s3*t2)/4.;

  df1[2][2] = s/4. + s2 - (3*s3)/4. + 2*s*t + 8*s2*t - 6*s3*t - (9*s*t2)/4. - 9*s2*t2 + (27*s3*t2)/4.;

  df1[2][3] = -(s*t)/2. - 2*s2*t + (3*s3*t)/2. + (3*s*t2)/4. + 3*s2*t2 - (9*s3*t2)/4.;

  df1[3][0] = s2/4. - s3/4. - s2*t + s3*t + (3*s2*t2)/4. - (3*s3*t2)/4.;

  df1[3][1] = (5*s2*t)/2. - (5*s3*t)/2. - (9*s2*t2)/4. + (9*s3*t2)/4.;

  df1[3][2] = -s2/4. + s3/4. - 2*s2*t + 2*s3*t + (9*s2*t2)/4. - (9*s3*t2)/4.;

  df1[3][3] = (s2*t)/2. - (s3*t)/2. - (3*s2*t2)/4. + (3*s3*t2)/4.;

  /* convert df0 and df1 to the derivatives w.r.t. rho and |nabla rho| */

  for (i=0; i<=3; i++){
    for (j=0; j<=3; j++){

      tmp0 = df0[i][j]/d1; // w.r.t. rho16
      tmp1 = df1[i][j]/d2; // w.r.t. R13 

      df0[i][j] = tmp0*drho16_0 + tmp1*dR0/(3.0*R13*R13+al);
      df1[i][j] = tmp0*drho16_1 + tmp1*dR1/(3.0*R13*R13+al);
    }
  }

  /* calculate gx[0,1,2] */

  gx[0] = 0.0;  gx[1] = 0.0;  gx[2] = 0.0;

  for (i=0; i<=3; i++){
    for (j=0; j<=3; j++){

      ii = i1+i-1;
      jj = j1+j-1;

      gx[0] += NLX_coes[mu][ii][jj]*f[i][j]; 
      gx[1] += NLX_coes[mu][ii][jj]*df0[i][j]; 
      gx[2] += NLX_coes[mu][ii][jj]*df1[i][j]; 
    }
  }

}



void gx_kernel5(int mu, double rho, double drho, double gx[3])
{
  static int first_flag=0;
  int i,j,ii,jj,i1,j1,N1,N2;
  double tmp0,tmp1,s,s2,s3,s4,s5,t,t2,t3,t4,t5,al;
  double min_R,max_R,rho16,drho16_0,drho16_1;
  double dR0,dR1,R13,R,ln,x,y,d1,d2;
  double f[6][6],df0[6][6],df1[6][6];
  double threshold=1.0e-12;

  /* avoid the numerical instability if rho and drho are small. */

  if ( rho<threshold ){
    gx[0] = 0.0;
    gx[1] = 0.0;
    gx[2] = 0.0;
    return;
  }
  else if ( fabs(drho)<(0.1*threshold) ){
    drho = 0.1*threshold;
  }

  /* setting d1 and d2 */

  al = al_R13;

  min_R = (min_R13)*(min_R13)*(min_R13) + al*(min_R13);
  max_R = (max_R13)*(max_R13)*(max_R13) + al*(max_R13);

  N1 = Num_Grid1;
  N2 = Num_Grid2;
  d1 = (max_rho16-(min_rho16))/(double)N1;
  d2 = (max_R13-(min_R13))/(double)N2;

  /* find rho16 */

  x = rho;
  y = fabs(drho);

  if (x<1.0e-20) x = 1.0e-20;
  if (y<1.0e-20) y = 1.0e-20;

  rho16 = pow(x,1.0/6.0);
  drho16_0 = 1.0/6.0*pow(x,-5.0/6.0);
  drho16_1 = 0.0;

  if (rho16<(min_rho16)){
    x = min_rho16*min_rho16*min_rho16*min_rho16*min_rho16*min_rho16;
    rho16 = min_rho16;
    drho16_0 = 1.0/6.0*pow(x,-5.0/6.0);
    drho16_1 = 0.0;
  }
  else if ((max_rho16)<rho16){
    x = max_rho16*max_rho16*max_rho16*max_rho16*max_rho16*max_rho16;
    rho16 = max_rho16;
    drho16_0 = 1.0/6.0*pow(x,-5.0/6.0);
    drho16_1 = 0.0;
  }

  /* find R13 */

  ln = -log(8.0*PI) - 3.0*log(pow(x,4.0/3.0)/y);
  R = ln;
  R13 = -((pow(2.0/3.0,1.0/3.0)*al)/pow(9.0*R + sqrt(3.0)*sqrt(4.0*pow(al,3.0) + 27.0*pow(R,2.0)),1.0/3.0))
    + pow(9.0*R + sqrt(3.0)*sqrt(4.0*pow(al,3.0) + 27.0*pow(R,2.0)),0.3333333333333333)/(pow(2.0,1.0/3.0)*pow(3.0,2.0/3.0));

  dR0 = -4.0/x;
  dR1 = 3.0/y;

  if (R<(min_R)){
    R = min_R;
    R13 = min_R13;
    dR0 = 0.0;
    dR1 = 0.0;
  }
  else if ((max_R)<R){
    R = max_R;
    R13 = max_R13;
    dR0 = 0.0;
    dR1 = 0.0;
  }

  /* find indexes and convert variables to s and t */

  i1 = (int)((rho16 - (min_rho16))/d1);
  if (i1<2)           i1 = 2;
  else if ((N1-3)<i1) i1 = N1-3;

  j1 = (int)((R13 - (min_R13))/d2);
  if (j1<2)           j1 = 2;
  else if ((N2-3)<j1) j1 = N2-3;

  s = (rho16 - (min_rho16+(double)i1*d1))/d1;
  s2 = s*s;
  s3 = s*s2;
  s4 = s2*s2;
  s5 = s2*s3;

  t = (R13 - (min_R13+(double)j1*d2))/d2;
  t2 = t*t;
  t3 = t*t2;
  t4 = t2*t2;
  t5 = t2*t3;

  /* calculate f, df0, and df1 */

  f[0][0] = (s2*t2)/64. - (3*s3*t2)/64. + (3*s4*t2)/64. - (s5*t2)/64. - (3*s2*t3)/64. + (9*s3*t3)/64. - (9*s4*t3)/64. + (3*s5*t3)/64. + (3*s2*t4)/64. - (9*s3*t4)/64. + (9*s4*t4)/64. - (3*s5*t4)/64. - (s2*t5)/64. + (3*s3*t5)/64. - (3*s4*t5)/64. + (s5*t5)/64.;

  f[0][1] = -(s2*t)/16. + (3*s3*t)/16. - (3*s4*t)/16. + (s5*t)/16. + (25*s2*t3)/64. - (75*s3*t3)/64. + (75*s4*t3)/64. - (25*s5*t3)/64. - (17*s2*t4)/32. + (51*s3*t4)/32. - (51*s4*t4)/32. + (17*s5*t4)/32. + (13*s2*t5)/64. - (39*s3*t5)/64. + (39*s4*t5)/64. - (13*s5*t5)/64.;

  f[0][2] = s2/8. - (3*s3)/8. + (3*s4)/8. - s5/8. - (s2*t2)/32. + (3*s3*t2)/32. - (3*s4*t2)/32. + (s5*t2)/32. - (29*s2*t3)/32. + (87*s3*t3)/32. - (87*s4*t3)/32. + (29*s5*t3)/32. + (43*s2*t4)/32. - (129*s3*t4)/32. + (129*s4*t4)/32. - (43*s5*t4)/32. - (17*s2*t5)/32. + (51*s3*t5)/32. - (51*s4*t5)/32. + (17*s5*t5)/32.;

  f[0][3] = (s2*t)/16. - (3*s3*t)/16. + (3*s4*t)/16. - (s5*t)/16. + (27*s2*t3)/32. - (81*s3*t3)/32. + (81*s4*t3)/32. - (27*s5*t3)/32. - (21*s2*t4)/16. + (63*s3*t4)/16. - (63*s4*t4)/16. + (21*s5*t4)/16. + (17*s2*t5)/32. - (51*s3*t5)/32. + (51*s4*t5)/32. - (17*s5*t5)/32.;

  f[0][4] = (s2*t2)/64. - (3*s3*t2)/64. + (3*s4*t2)/64. - (s5*t2)/64. - (19*s2*t3)/64. + (57*s3*t3)/64. - (57*s4*t3)/64. + (19*s5*t3)/64. + (31*s2*t4)/64. - (93*s3*t4)/64. + (93*s4*t4)/64. - (31*s5*t4)/64. - (13*s2*t5)/64. + (39*s3*t5)/64. - (39*s4*t5)/64. + (13*s5*t5)/64.;

  f[0][5] = (s2*t3)/64. - (3*s3*t3)/64. + (3*s4*t3)/64. - (s5*t3)/64. - (s2*t4)/32. + (3*s3*t4)/32. - (3*s4*t4)/32. + (s5*t4)/32. + (s2*t5)/64. - (3*s3*t5)/64. + (3*s4*t5)/64. - (s5*t5)/64.;

  f[1][0] = -(s*t2)/16. + (25*s3*t2)/64. - (17*s4*t2)/32. + (13*s5*t2)/64. + (3*s*t3)/16. - (75*s3*t3)/64. + (51*s4*t3)/32. - (39*s5*t3)/64. - (3*s*t4)/16. + (75*s3*t4)/64. - (51*s4*t4)/32. + (39*s5*t4)/64. + (s*t5)/16. - (25*s3*t5)/64. + (17*s4*t5)/32. - (13*s5*t5)/64.;

  f[1][1] = (s*t)/4. - (25*s3*t)/16. + (17*s4*t)/8. - (13*s5*t)/16. - (25*s*t3)/16. + (625*s3*t3)/64. - (425*s4*t3)/32. + (325*s5*t3)/64. + (17*s*t4)/8. - (425*s3*t4)/32. + (289*s4*t4)/16. - (221*s5*t4)/32. - (13*s*t5)/16. + (325*s3*t5)/64. - (221*s4*t5)/32. + (169*s5*t5)/64.;

  f[1][2] = -s/2. + (25*s3)/8. - (17*s4)/4. + (13*s5)/8. + (s*t2)/8. - (25*s3*t2)/32. + (17*s4*t2)/16. - (13*s5*t2)/32. + (29*s*t3)/8. - (725*s3*t3)/32. + (493*s4*t3)/16. - (377*s5*t3)/32. - (43*s*t4)/8. + (1075*s3*t4)/32. - (731*s4*t4)/16. + (559*s5*t4)/32. + (17*s*t5)/8. - (425*s3*t5)/32. + (289*s4*t5)/16. - (221*s5*t5)/32.;

  f[1][3] = -(s*t)/4. + (25*s3*t)/16. - (17*s4*t)/8. + (13*s5*t)/16. - (27*s*t3)/8. + (675*s3*t3)/32. - (459*s4*t3)/16. + (351*s5*t3)/32. + (21*s*t4)/4. - (525*s3*t4)/16. + (357*s4*t4)/8. - (273*s5*t4)/16. - (17*s*t5)/8. + (425*s3*t5)/32. - (289*s4*t5)/16. + (221*s5*t5)/32.;

  f[1][4] = -(s*t2)/16. + (25*s3*t2)/64. - (17*s4*t2)/32. + (13*s5*t2)/64. + (19*s*t3)/16. - (475*s3*t3)/64. + (323*s4*t3)/32. - (247*s5*t3)/64. - (31*s*t4)/16. + (775*s3*t4)/64. - (527*s4*t4)/32. + (403*s5*t4)/64. + (13*s*t5)/16. - (325*s3*t5)/64. + (221*s4*t5)/32. - (169*s5*t5)/64.;

  f[1][5] = -(s*t3)/16. + (25*s3*t3)/64. - (17*s4*t3)/32. + (13*s5*t3)/64. + (s*t4)/8. - (25*s3*t4)/32. + (17*s4*t4)/16. - (13*s5*t4)/32. - (s*t5)/16. + (25*s3*t5)/64. - (17*s4*t5)/32. + (13*s5*t5)/64.;

  f[2][0] = t2/8. - (s2*t2)/32. - (29*s3*t2)/32. + (43*s4*t2)/32. - (17*s5*t2)/32. - (3*t3)/8. + (3*s2*t3)/32. + (87*s3*t3)/32. - (129*s4*t3)/32. + (51*s5*t3)/32. + (3*t4)/8. - (3*s2*t4)/32. - (87*s3*t4)/32. + (129*s4*t4)/32. - (51*s5*t4)/32. - t5/8. + (s2*t5)/32. + (29*s3*t5)/32. - (43*s4*t5)/32. + (17*s5*t5)/32.;

  f[2][1] = -t/2. + (s2*t)/8. + (29*s3*t)/8. - (43*s4*t)/8. + (17*s5*t)/8. + (25*t3)/8. - (25*s2*t3)/32. - (725*s3*t3)/32. + (1075*s4*t3)/32. - (425*s5*t3)/32. - (17*t4)/4. + (17*s2*t4)/16. + (493*s3*t4)/16. - (731*s4*t4)/16. + (289*s5*t4)/16. + (13*t5)/8. - (13*s2*t5)/32. - (377*s3*t5)/32. + (559*s4*t5)/32. - (221*s5*t5)/32.;

  f[2][2] = 1 - s2/4. - (29*s3)/4. + (43*s4)/4. - (17*s5)/4. - t2/4. + (s2*t2)/16. + (29*s3*t2)/16. - (43*s4*t2)/16. + (17*s5*t2)/16. - (29*t3)/4. + (29*s2*t3)/16. + (841*s3*t3)/16. - (1247*s4*t3)/16. + (493*s5*t3)/16. + (43*t4)/4. - (43*s2*t4)/16. - (1247*s3*t4)/16. + (1849*s4*t4)/16. - (731*s5*t4)/16. - (17*t5)/4. + (17*s2*t5)/16. + (493*s3*t5)/16. - (731*s4*t5)/16. + (289*s5*t5)/16.;

  f[2][3] = t/2. - (s2*t)/8. - (29*s3*t)/8. + (43*s4*t)/8. - (17*s5*t)/8. + (27*t3)/4. - (27*s2*t3)/16. - (783*s3*t3)/16. + (1161*s4*t3)/16. - (459*s5*t3)/16. - (21*t4)/2. + (21*s2*t4)/8. + (609*s3*t4)/8. - (903*s4*t4)/8. + (357*s5*t4)/8. + (17*t5)/4. - (17*s2*t5)/16. - (493*s3*t5)/16. + (731*s4*t5)/16. - (289*s5*t5)/16.;

  f[2][4] = t2/8. - (s2*t2)/32. - (29*s3*t2)/32. + (43*s4*t2)/32. - (17*s5*t2)/32. - (19*t3)/8. + (19*s2*t3)/32. + (551*s3*t3)/32. - (817*s4*t3)/32. + (323*s5*t3)/32. + (31*t4)/8. - (31*s2*t4)/32. - (899*s3*t4)/32. + (1333*s4*t4)/32. - (527*s5*t4)/32. - (13*t5)/8. + (13*s2*t5)/32. + (377*s3*t5)/32. - (559*s4*t5)/32. + (221*s5*t5)/32.;

  f[2][5] = t3/8. - (s2*t3)/32. - (29*s3*t3)/32. + (43*s4*t3)/32. - (17*s5*t3)/32. - t4/4. + (s2*t4)/16. + (29*s3*t4)/16. - (43*s4*t4)/16. + (17*s5*t4)/16. + t5/8. - (s2*t5)/32. - (29*s3*t5)/32. + (43*s4*t5)/32. - (17*s5*t5)/32.;

  f[3][0] = (s*t2)/16. + (27*s3*t2)/32. - (21*s4*t2)/16. + (17*s5*t2)/32. - (3*s*t3)/16. - (81*s3*t3)/32. + (63*s4*t3)/16. - (51*s5*t3)/32. + (3*s*t4)/16. + (81*s3*t4)/32. - (63*s4*t4)/16. + (51*s5*t4)/32. - (s*t5)/16. - (27*s3*t5)/32. + (21*s4*t5)/16. - (17*s5*t5)/32.;

  f[3][1] = -(s*t)/4. - (27*s3*t)/8. + (21*s4*t)/4. - (17*s5*t)/8. + (25*s*t3)/16. + (675*s3*t3)/32. - (525*s4*t3)/16. + (425*s5*t3)/32. - (17*s*t4)/8. - (459*s3*t4)/16. + (357*s4*t4)/8. - (289*s5*t4)/16. + (13*s*t5)/16. + (351*s3*t5)/32. - (273*s4*t5)/16. + (221*s5*t5)/32.;

  f[3][2] = s/2. + (27*s3)/4. - (21*s4)/2. + (17*s5)/4. - (s*t2)/8. - (27*s3*t2)/16. + (21*s4*t2)/8. - (17*s5*t2)/16. - (29*s*t3)/8. - (783*s3*t3)/16. + (609*s4*t3)/8. - (493*s5*t3)/16. + (43*s*t4)/8. + (1161*s3*t4)/16. - (903*s4*t4)/8. + (731*s5*t4)/16. - (17*s*t5)/8. - (459*s3*t5)/16. + (357*s4*t5)/8. - (289*s5*t5)/16.;

  f[3][3] = (s*t)/4. + (27*s3*t)/8. - (21*s4*t)/4. + (17*s5*t)/8. + (27*s*t3)/8. + (729*s3*t3)/16. - (567*s4*t3)/8. + (459*s5*t3)/16. - (21*s*t4)/4. - (567*s3*t4)/8. + (441*s4*t4)/4. - (357*s5*t4)/8. + (17*s*t5)/8. + (459*s3*t5)/16. - (357*s4*t5)/8. + (289*s5*t5)/16.;

  f[3][4] = (s*t2)/16. + (27*s3*t2)/32. - (21*s4*t2)/16. + (17*s5*t2)/32. - (19*s*t3)/16. - (513*s3*t3)/32. + (399*s4*t3)/16. - (323*s5*t3)/32. + (31*s*t4)/16. + (837*s3*t4)/32. - (651*s4*t4)/16. + (527*s5*t4)/32. - (13*s*t5)/16. - (351*s3*t5)/32. + (273*s4*t5)/16. - (221*s5*t5)/32.;

  f[3][5] = (s*t3)/16. + (27*s3*t3)/32. - (21*s4*t3)/16. + (17*s5*t3)/32. - (s*t4)/8. - (27*s3*t4)/16. + (21*s4*t4)/8. - (17*s5*t4)/16. + (s*t5)/16. + (27*s3*t5)/32. - (21*s4*t5)/16. + (17*s5*t5)/32.;

  f[4][0] = (s2*t2)/64. - (19*s3*t2)/64. + (31*s4*t2)/64. - (13*s5*t2)/64. - (3*s2*t3)/64. + (57*s3*t3)/64. - (93*s4*t3)/64. + (39*s5*t3)/64. + (3*s2*t4)/64. - (57*s3*t4)/64. + (93*s4*t4)/64. - (39*s5*t4)/64. - (s2*t5)/64. + (19*s3*t5)/64. - (31*s4*t5)/64. + (13*s5*t5)/64.;

  f[4][1] = -(s2*t)/16. + (19*s3*t)/16. - (31*s4*t)/16. + (13*s5*t)/16. + (25*s2*t3)/64. - (475*s3*t3)/64. + (775*s4*t3)/64. - (325*s5*t3)/64. - (17*s2*t4)/32. + (323*s3*t4)/32. - (527*s4*t4)/32. + (221*s5*t4)/32. + (13*s2*t5)/64. - (247*s3*t5)/64. + (403*s4*t5)/64. - (169*s5*t5)/64.;

  f[4][2] = s2/8. - (19*s3)/8. + (31*s4)/8. - (13*s5)/8. - (s2*t2)/32. + (19*s3*t2)/32. - (31*s4*t2)/32. + (13*s5*t2)/32. - (29*s2*t3)/32. + (551*s3*t3)/32. - (899*s4*t3)/32. + (377*s5*t3)/32. + (43*s2*t4)/32. - (817*s3*t4)/32. + (1333*s4*t4)/32. - (559*s5*t4)/32. - (17*s2*t5)/32. + (323*s3*t5)/32. - (527*s4*t5)/32. + (221*s5*t5)/32.;

  f[4][3] = (s2*t)/16. - (19*s3*t)/16. + (31*s4*t)/16. - (13*s5*t)/16. + (27*s2*t3)/32. - (513*s3*t3)/32. + (837*s4*t3)/32. - (351*s5*t3)/32. - (21*s2*t4)/16. + (399*s3*t4)/16. - (651*s4*t4)/16. + (273*s5*t4)/16. + (17*s2*t5)/32. - (323*s3*t5)/32. + (527*s4*t5)/32. - (221*s5*t5)/32.;

  f[4][4] = (s2*t2)/64. - (19*s3*t2)/64. + (31*s4*t2)/64. - (13*s5*t2)/64. - (19*s2*t3)/64. + (361*s3*t3)/64. - (589*s4*t3)/64. + (247*s5*t3)/64. + (31*s2*t4)/64. - (589*s3*t4)/64. + (961*s4*t4)/64. - (403*s5*t4)/64. - (13*s2*t5)/64. + (247*s3*t5)/64. - (403*s4*t5)/64. + (169*s5*t5)/64.;

  f[4][5] = (s2*t3)/64. - (19*s3*t3)/64. + (31*s4*t3)/64. - (13*s5*t3)/64. - (s2*t4)/32. + (19*s3*t4)/32. - (31*s4*t4)/32. + (13*s5*t4)/32. + (s2*t5)/64. - (19*s3*t5)/64. + (31*s4*t5)/64. - (13*s5*t5)/64.;

  f[5][0] = (s3*t2)/64. - (s4*t2)/32. + (s5*t2)/64. - (3*s3*t3)/64. + (3*s4*t3)/32. - (3*s5*t3)/64. + (3*s3*t4)/64. - (3*s4*t4)/32. + (3*s5*t4)/64. - (s3*t5)/64. + (s4*t5)/32. - (s5*t5)/64.;

  f[5][1] = -(s3*t)/16. + (s4*t)/8. - (s5*t)/16. + (25*s3*t3)/64. - (25*s4*t3)/32. + (25*s5*t3)/64. - (17*s3*t4)/32. + (17*s4*t4)/16. - (17*s5*t4)/32. + (13*s3*t5)/64. - (13*s4*t5)/32. + (13*s5*t5)/64.;

  f[5][2] = s3/8. - s4/4. + s5/8. - (s3*t2)/32. + (s4*t2)/16. - (s5*t2)/32. - (29*s3*t3)/32. + (29*s4*t3)/16. - (29*s5*t3)/32. + (43*s3*t4)/32. - (43*s4*t4)/16. + (43*s5*t4)/32. - (17*s3*t5)/32. + (17*s4*t5)/16. - (17*s5*t5)/32.;

  f[5][3] = (s3*t)/16. - (s4*t)/8. + (s5*t)/16. + (27*s3*t3)/32. - (27*s4*t3)/16. + (27*s5*t3)/32. - (21*s3*t4)/16. + (21*s4*t4)/8. - (21*s5*t4)/16. + (17*s3*t5)/32. - (17*s4*t5)/16. + (17*s5*t5)/32.;

  f[5][4] = (s3*t2)/64. - (s4*t2)/32. + (s5*t2)/64. - (19*s3*t3)/64. + (19*s4*t3)/32. - (19*s5*t3)/64. + (31*s3*t4)/64. - (31*s4*t4)/32. + (31*s5*t4)/64. - (13*s3*t5)/64. + (13*s4*t5)/32. - (13*s5*t5)/64.;

  f[5][5] = (s3*t3)/64. - (s4*t3)/32. + (s5*t3)/64. - (s3*t4)/32. + (s4*t4)/16. - (s5*t4)/32. + (s3*t5)/64. - (s4*t5)/32. + (s5*t5)/64.;

  /* derivatives of coefficients of g w.r.t s  */ 

  df0[0][0] = (s*t2)/32. - (9*s2*t2)/64. + (3*s3*t2)/16. - (5*s4*t2)/64. - (3*s*t3)/32. + (27*s2*t3)/64. - (9*s3*t3)/16. + (15*s4*t3)/64. + (3*s*t4)/32. - (27*s2*t4)/64. + (9*s3*t4)/16. - (15*s4*t4)/64. - (s*t5)/32. + (9*s2*t5)/64. - (3*s3*t5)/16. + (5*s4*t5)/64.;

  df0[0][1] = -(s*t)/8. + (9*s2*t)/16. - (3*s3*t)/4. + (5*s4*t)/16. + (25*s*t3)/32. - (225*s2*t3)/64. + (75*s3*t3)/16. - (125*s4*t3)/64. - (17*s*t4)/16. + (153*s2*t4)/32. - (51*s3*t4)/8. + (85*s4*t4)/32. + (13*s*t5)/32. - (117*s2*t5)/64. + (39*s3*t5)/16. - (65*s4*t5)/64.;

  df0[0][2] = s/4. - (9*s2)/8. + (3*s3)/2. - (5*s4)/8. - (s*t2)/16. + (9*s2*t2)/32. - (3*s3*t2)/8. + (5*s4*t2)/32. - (29*s*t3)/16. + (261*s2*t3)/32. - (87*s3*t3)/8. + (145*s4*t3)/32. + (43*s*t4)/16. - (387*s2*t4)/32. + (129*s3*t4)/8. - (215*s4*t4)/32. - (17*s*t5)/16. + (153*s2*t5)/32. - (51*s3*t5)/8. + (85*s4*t5)/32.;

  df0[0][3] = (s*t)/8. - (9*s2*t)/16. + (3*s3*t)/4. - (5*s4*t)/16. + (27*s*t3)/16. - (243*s2*t3)/32. + (81*s3*t3)/8. - (135*s4*t3)/32. - (21*s*t4)/8. + (189*s2*t4)/16. - (63*s3*t4)/4. + (105*s4*t4)/16. + (17*s*t5)/16. - (153*s2*t5)/32. + (51*s3*t5)/8. - (85*s4*t5)/32.;

  df0[0][4] = (s*t2)/32. - (9*s2*t2)/64. + (3*s3*t2)/16. - (5*s4*t2)/64. - (19*s*t3)/32. + (171*s2*t3)/64. - (57*s3*t3)/16. + (95*s4*t3)/64. + (31*s*t4)/32. - (279*s2*t4)/64. + (93*s3*t4)/16. - (155*s4*t4)/64. - (13*s*t5)/32. + (117*s2*t5)/64. - (39*s3*t5)/16. + (65*s4*t5)/64.;

  df0[0][5] = (s*t3)/32. - (9*s2*t3)/64. + (3*s3*t3)/16. - (5*s4*t3)/64. - (s*t4)/16. + (9*s2*t4)/32. - (3*s3*t4)/8. + (5*s4*t4)/32. + (s*t5)/32. - (9*s2*t5)/64. + (3*s3*t5)/16. - (5*s4*t5)/64.;

  df0[1][0] = -t2/16. + (75*s2*t2)/64. - (17*s3*t2)/8. + (65*s4*t2)/64. + (3*t3)/16. - (225*s2*t3)/64. + (51*s3*t3)/8. - (195*s4*t3)/64. - (3*t4)/16. + (225*s2*t4)/64. - (51*s3*t4)/8. + (195*s4*t4)/64. + t5/16. - (75*s2*t5)/64. + (17*s3*t5)/8. - (65*s4*t5)/64.;

  df0[1][1] = t/4. - (75*s2*t)/16. + (17*s3*t)/2. - (65*s4*t)/16. - (25*t3)/16. + (1875*s2*t3)/64. - (425*s3*t3)/8. + (1625*s4*t3)/64. + (17*t4)/8. - (1275*s2*t4)/32. + (289*s3*t4)/4. - (1105*s4*t4)/32. - (13*t5)/16. + (975*s2*t5)/64. - (221*s3*t5)/8. + (845*s4*t5)/64.;

  df0[1][2] = -0.5 + (75*s2)/8. - 17*s3 + (65*s4)/8. + t2/8. - (75*s2*t2)/32. + (17*s3*t2)/4. - (65*s4*t2)/32. + (29*t3)/8. - (2175*s2*t3)/32. + (493*s3*t3)/4. - (1885*s4*t3)/32. - (43*t4)/8. + (3225*s2*t4)/32. - (731*s3*t4)/4. + (2795*s4*t4)/32. + (17*t5)/8. - (1275*s2*t5)/32. + (289*s3*t5)/4. - (1105*s4*t5)/32.;

  df0[1][3] = -t/4. + (75*s2*t)/16. - (17*s3*t)/2. + (65*s4*t)/16. - (27*t3)/8. + (2025*s2*t3)/32. - (459*s3*t3)/4. + (1755*s4*t3)/32. + (21*t4)/4. - (1575*s2*t4)/16. + (357*s3*t4)/2. - (1365*s4*t4)/16. - (17*t5)/8. + (1275*s2*t5)/32. - (289*s3*t5)/4. + (1105*s4*t5)/32.;

  df0[1][4] = -t2/16. + (75*s2*t2)/64. - (17*s3*t2)/8. + (65*s4*t2)/64. + (19*t3)/16. - (1425*s2*t3)/64. + (323*s3*t3)/8. - (1235*s4*t3)/64. - (31*t4)/16. + (2325*s2*t4)/64. - (527*s3*t4)/8. + (2015*s4*t4)/64. + (13*t5)/16. - (975*s2*t5)/64. + (221*s3*t5)/8. - (845*s4*t5)/64.;

  df0[1][5] = -t3/16. + (75*s2*t3)/64. - (17*s3*t3)/8. + (65*s4*t3)/64. + t4/8. - (75*s2*t4)/32. + (17*s3*t4)/4. - (65*s4*t4)/32. - t5/16. + (75*s2*t5)/64. - (17*s3*t5)/8. + (65*s4*t5)/64.;

  df0[2][0] = -(s*t2)/16. - (87*s2*t2)/32. + (43*s3*t2)/8. - (85*s4*t2)/32. + (3*s*t3)/16. + (261*s2*t3)/32. - (129*s3*t3)/8. + (255*s4*t3)/32. - (3*s*t4)/16. - (261*s2*t4)/32. + (129*s3*t4)/8. - (255*s4*t4)/32. + (s*t5)/16. + (87*s2*t5)/32. - (43*s3*t5)/8. + (85*s4*t5)/32.;

  df0[2][1] = (s*t)/4. + (87*s2*t)/8. - (43*s3*t)/2. + (85*s4*t)/8. - (25*s*t3)/16. - (2175*s2*t3)/32. + (1075*s3*t3)/8. - (2125*s4*t3)/32. + (17*s*t4)/8. + (1479*s2*t4)/16. - (731*s3*t4)/4. + (1445*s4*t4)/16. - (13*s*t5)/16. - (1131*s2*t5)/32. + (559*s3*t5)/8. - (1105*s4*t5)/32.;

  df0[2][2] = -s/2. - (87*s2)/4. + 43*s3 - (85*s4)/4. + (s*t2)/8. + (87*s2*t2)/16. - (43*s3*t2)/4. + (85*s4*t2)/16. + (29*s*t3)/8. + (2523*s2*t3)/16. - (1247*s3*t3)/4. + (2465*s4*t3)/16. - (43*s*t4)/8. - (3741*s2*t4)/16. + (1849*s3*t4)/4. - (3655*s4*t4)/16. + (17*s*t5)/8. + (1479*s2*t5)/16. - (731*s3*t5)/4. + (1445*s4*t5)/16.;

  df0[2][3] = -(s*t)/4. - (87*s2*t)/8. + (43*s3*t)/2. - (85*s4*t)/8. - (27*s*t3)/8. - (2349*s2*t3)/16. + (1161*s3*t3)/4. - (2295*s4*t3)/16. + (21*s*t4)/4. + (1827*s2*t4)/8. - (903*s3*t4)/2. + (1785*s4*t4)/8. - (17*s*t5)/8. - (1479*s2*t5)/16. + (731*s3*t5)/4. - (1445*s4*t5)/16.;

  df0[2][4] = -(s*t2)/16. - (87*s2*t2)/32. + (43*s3*t2)/8. - (85*s4*t2)/32. + (19*s*t3)/16. + (1653*s2*t3)/32. - (817*s3*t3)/8. + (1615*s4*t3)/32. - (31*s*t4)/16. - (2697*s2*t4)/32. + (1333*s3*t4)/8. - (2635*s4*t4)/32. + (13*s*t5)/16. + (1131*s2*t5)/32. - (559*s3*t5)/8. + (1105*s4*t5)/32.;

  df0[2][5] = -(s*t3)/16. - (87*s2*t3)/32. + (43*s3*t3)/8. - (85*s4*t3)/32. + (s*t4)/8. + (87*s2*t4)/16. - (43*s3*t4)/4. + (85*s4*t4)/16. - (s*t5)/16. - (87*s2*t5)/32. + (43*s3*t5)/8. - (85*s4*t5)/32.;

  df0[3][0] = t2/16. + (81*s2*t2)/32. - (21*s3*t2)/4. + (85*s4*t2)/32. - (3*t3)/16. - (243*s2*t3)/32. + (63*s3*t3)/4. - (255*s4*t3)/32. + (3*t4)/16. + (243*s2*t4)/32. - (63*s3*t4)/4. + (255*s4*t4)/32. - t5/16. - (81*s2*t5)/32. + (21*s3*t5)/4. - (85*s4*t5)/32.;

  df0[3][1] = -t/4. - (81*s2*t)/8. + 21*s3*t - (85*s4*t)/8. + (25*t3)/16. + (2025*s2*t3)/32. - (525*s3*t3)/4. + (2125*s4*t3)/32. - (17*t4)/8. - (1377*s2*t4)/16. + (357*s3*t4)/2. - (1445*s4*t4)/16. + (13*t5)/16. + (1053*s2*t5)/32. - (273*s3*t5)/4. + (1105*s4*t5)/32.;

  df0[3][2] = 0.5 + (81*s2)/4. - 42*s3 + (85*s4)/4. - t2/8. - (81*s2*t2)/16. + (21*s3*t2)/2. - (85*s4*t2)/16. - (29*t3)/8. - (2349*s2*t3)/16. + (609*s3*t3)/2. - (2465*s4*t3)/16. + (43*t4)/8. + (3483*s2*t4)/16. - (903*s3*t4)/2. + (3655*s4*t4)/16. - (17*t5)/8. - (1377*s2*t5)/16. + (357*s3*t5)/2. - (1445*s4*t5)/16.;

  df0[3][3] = t/4. + (81*s2*t)/8. - 21*s3*t + (85*s4*t)/8. + (27*t3)/8. + (2187*s2*t3)/16. - (567*s3*t3)/2. + (2295*s4*t3)/16. - (21*t4)/4. - (1701*s2*t4)/8. + 441*s3*t4 - (1785*s4*t4)/8. + (17*t5)/8. + (1377*s2*t5)/16. - (357*s3*t5)/2. + (1445*s4*t5)/16.;

  df0[3][4] = t2/16. + (81*s2*t2)/32. - (21*s3*t2)/4. + (85*s4*t2)/32. - (19*t3)/16. - (1539*s2*t3)/32. + (399*s3*t3)/4. - (1615*s4*t3)/32. + (31*t4)/16. + (2511*s2*t4)/32. - (651*s3*t4)/4. + (2635*s4*t4)/32. - (13*t5)/16. - (1053*s2*t5)/32. + (273*s3*t5)/4. - (1105*s4*t5)/32.;

  df0[3][5] = t3/16. + (81*s2*t3)/32. - (21*s3*t3)/4. + (85*s4*t3)/32. - t4/8. - (81*s2*t4)/16. + (21*s3*t4)/2. - (85*s4*t4)/16. + t5/16. + (81*s2*t5)/32. - (21*s3*t5)/4. + (85*s4*t5)/32.;

  df0[4][0] = (s*t2)/32. - (57*s2*t2)/64. + (31*s3*t2)/16. - (65*s4*t2)/64. - (3*s*t3)/32. + (171*s2*t3)/64. - (93*s3*t3)/16. + (195*s4*t3)/64. + (3*s*t4)/32. - (171*s2*t4)/64. + (93*s3*t4)/16. - (195*s4*t4)/64. - (s*t5)/32. + (57*s2*t5)/64. - (31*s3*t5)/16. + (65*s4*t5)/64.;

  df0[4][1] = -(s*t)/8. + (57*s2*t)/16. - (31*s3*t)/4. + (65*s4*t)/16. + (25*s*t3)/32. - (1425*s2*t3)/64. + (775*s3*t3)/16. - (1625*s4*t3)/64. - (17*s*t4)/16. + (969*s2*t4)/32. - (527*s3*t4)/8. + (1105*s4*t4)/32. + (13*s*t5)/32. - (741*s2*t5)/64. + (403*s3*t5)/16. - (845*s4*t5)/64.;

  df0[4][2] = s/4. - (57*s2)/8. + (31*s3)/2. - (65*s4)/8. - (s*t2)/16. + (57*s2*t2)/32. - (31*s3*t2)/8. + (65*s4*t2)/32. - (29*s*t3)/16. + (1653*s2*t3)/32. - (899*s3*t3)/8. + (1885*s4*t3)/32. + (43*s*t4)/16. - (2451*s2*t4)/32. + (1333*s3*t4)/8. - (2795*s4*t4)/32. - (17*s*t5)/16. + (969*s2*t5)/32. - (527*s3*t5)/8. + (1105*s4*t5)/32.;

  df0[4][3] = (s*t)/8. - (57*s2*t)/16. + (31*s3*t)/4. - (65*s4*t)/16. + (27*s*t3)/16. - (1539*s2*t3)/32. + (837*s3*t3)/8. - (1755*s4*t3)/32. - (21*s*t4)/8. + (1197*s2*t4)/16. - (651*s3*t4)/4. + (1365*s4*t4)/16. + (17*s*t5)/16. - (969*s2*t5)/32. + (527*s3*t5)/8. - (1105*s4*t5)/32.;

  df0[4][4] = (s*t2)/32. - (57*s2*t2)/64. + (31*s3*t2)/16. - (65*s4*t2)/64. - (19*s*t3)/32. + (1083*s2*t3)/64. - (589*s3*t3)/16. + (1235*s4*t3)/64. + (31*s*t4)/32. - (1767*s2*t4)/64. + (961*s3*t4)/16. - (2015*s4*t4)/64. - (13*s*t5)/32. + (741*s2*t5)/64. - (403*s3*t5)/16. + (845*s4*t5)/64.;

  df0[4][5] = (s*t3)/32. - (57*s2*t3)/64. + (31*s3*t3)/16. - (65*s4*t3)/64. - (s*t4)/16. + (57*s2*t4)/32. - (31*s3*t4)/8. + (65*s4*t4)/32. + (s*t5)/32. - (57*s2*t5)/64. + (31*s3*t5)/16. - (65*s4*t5)/64.;

  df0[5][0] = (3*s2*t2)/64. - (s3*t2)/8. + (5*s4*t2)/64. - (9*s2*t3)/64. + (3*s3*t3)/8. - (15*s4*t3)/64. + (9*s2*t4)/64. - (3*s3*t4)/8. + (15*s4*t4)/64. - (3*s2*t5)/64. + (s3*t5)/8. - (5*s4*t5)/64.;

  df0[5][1] = (-3*s2*t)/16. + (s3*t)/2. - (5*s4*t)/16. + (75*s2*t3)/64. - (25*s3*t3)/8. + (125*s4*t3)/64. - (51*s2*t4)/32. + (17*s3*t4)/4. - (85*s4*t4)/32. + (39*s2*t5)/64. - (13*s3*t5)/8. + (65*s4*t5)/64.;

  df0[5][2] = (3*s2)/8. - s3 + (5*s4)/8. - (3*s2*t2)/32. + (s3*t2)/4. - (5*s4*t2)/32. - (87*s2*t3)/32. + (29*s3*t3)/4. - (145*s4*t3)/32. + (129*s2*t4)/32. - (43*s3*t4)/4. + (215*s4*t4)/32. - (51*s2*t5)/32. + (17*s3*t5)/4. - (85*s4*t5)/32.;

  df0[5][3] = (3*s2*t)/16. - (s3*t)/2. + (5*s4*t)/16. + (81*s2*t3)/32. - (27*s3*t3)/4. + (135*s4*t3)/32. - (63*s2*t4)/16. + (21*s3*t4)/2. - (105*s4*t4)/16. + (51*s2*t5)/32. - (17*s3*t5)/4. + (85*s4*t5)/32.;

  df0[5][4] = (3*s2*t2)/64. - (s3*t2)/8. + (5*s4*t2)/64. - (57*s2*t3)/64. + (19*s3*t3)/8. - (95*s4*t3)/64. + (93*s2*t4)/64. - (31*s3*t4)/8. + (155*s4*t4)/64. - (39*s2*t5)/64. + (13*s3*t5)/8. - (65*s4*t5)/64.;

  df0[5][5] = (3*s2*t3)/64. - (s3*t3)/8. + (5*s4*t3)/64. - (3*s2*t4)/32. + (s3*t4)/4. - (5*s4*t4)/32. + (3*s2*t5)/64. - (s3*t5)/8. + (5*s4*t5)/64.;

  /* derivatives of coefficients of g w.r.t t  */ 

  df1[0][0] = (s2*t)/32. - (3*s3*t)/32. + (3*s4*t)/32. - (s5*t)/32. - (9*s2*t2)/64. + (27*s3*t2)/64. - (27*s4*t2)/64. + (9*s5*t2)/64. + (3*s2*t3)/16. - (9*s3*t3)/16. + (9*s4*t3)/16. - (3*s5*t3)/16. - (5*s2*t4)/64. + (15*s3*t4)/64. - (15*s4*t4)/64. + (5*s5*t4)/64.;

  df1[0][1] = -s2/16. + (3*s3)/16. - (3*s4)/16. + s5/16. + (75*s2*t2)/64. - (225*s3*t2)/64. + (225*s4*t2)/64. - (75*s5*t2)/64. - (17*s2*t3)/8. + (51*s3*t3)/8. - (51*s4*t3)/8. + (17*s5*t3)/8. + (65*s2*t4)/64. - (195*s3*t4)/64. + (195*s4*t4)/64. - (65*s5*t4)/64.;

  df1[0][2] = -(s2*t)/16. + (3*s3*t)/16. - (3*s4*t)/16. + (s5*t)/16. - (87*s2*t2)/32. + (261*s3*t2)/32. - (261*s4*t2)/32. + (87*s5*t2)/32. + (43*s2*t3)/8. - (129*s3*t3)/8. + (129*s4*t3)/8. - (43*s5*t3)/8. - (85*s2*t4)/32. + (255*s3*t4)/32. - (255*s4*t4)/32. + (85*s5*t4)/32.;

  df1[0][3] = s2/16. - (3*s3)/16. + (3*s4)/16. - s5/16. + (81*s2*t2)/32. - (243*s3*t2)/32. + (243*s4*t2)/32. - (81*s5*t2)/32. - (21*s2*t3)/4. + (63*s3*t3)/4. - (63*s4*t3)/4. + (21*s5*t3)/4. + (85*s2*t4)/32. - (255*s3*t4)/32. + (255*s4*t4)/32. - (85*s5*t4)/32.;

  df1[0][4] = (s2*t)/32. - (3*s3*t)/32. + (3*s4*t)/32. - (s5*t)/32. - (57*s2*t2)/64. + (171*s3*t2)/64. - (171*s4*t2)/64. + (57*s5*t2)/64. + (31*s2*t3)/16. - (93*s3*t3)/16. + (93*s4*t3)/16. - (31*s5*t3)/16. - (65*s2*t4)/64. + (195*s3*t4)/64. - (195*s4*t4)/64. + (65*s5*t4)/64.;

  df1[0][5] = (3*s2*t2)/64. - (9*s3*t2)/64. + (9*s4*t2)/64. - (3*s5*t2)/64. - (s2*t3)/8. + (3*s3*t3)/8. - (3*s4*t3)/8. + (s5*t3)/8. + (5*s2*t4)/64. - (15*s3*t4)/64. + (15*s4*t4)/64. - (5*s5*t4)/64.;

  df1[1][0] = -(s*t)/8. + (25*s3*t)/32. - (17*s4*t)/16. + (13*s5*t)/32. + (9*s*t2)/16. - (225*s3*t2)/64. + (153*s4*t2)/32. - (117*s5*t2)/64. - (3*s*t3)/4. + (75*s3*t3)/16. - (51*s4*t3)/8. + (39*s5*t3)/16. + (5*s*t4)/16. - (125*s3*t4)/64. + (85*s4*t4)/32. - (65*s5*t4)/64.;

  df1[1][1] = s/4. - (25*s3)/16. + (17*s4)/8. - (13*s5)/16. - (75*s*t2)/16. + (1875*s3*t2)/64. - (1275*s4*t2)/32. + (975*s5*t2)/64. + (17*s*t3)/2. - (425*s3*t3)/8. + (289*s4*t3)/4. - (221*s5*t3)/8. - (65*s*t4)/16. + (1625*s3*t4)/64. - (1105*s4*t4)/32. + (845*s5*t4)/64.;

  df1[1][2] = (s*t)/4. - (25*s3*t)/16. + (17*s4*t)/8. - (13*s5*t)/16. + (87*s*t2)/8. - (2175*s3*t2)/32. + (1479*s4*t2)/16. - (1131*s5*t2)/32. - (43*s*t3)/2. + (1075*s3*t3)/8. - (731*s4*t3)/4. + (559*s5*t3)/8. + (85*s*t4)/8. - (2125*s3*t4)/32. + (1445*s4*t4)/16. - (1105*s5*t4)/32.;

  df1[1][3] = -s/4. + (25*s3)/16. - (17*s4)/8. + (13*s5)/16. - (81*s*t2)/8. + (2025*s3*t2)/32. - (1377*s4*t2)/16. + (1053*s5*t2)/32. + 21*s*t3 - (525*s3*t3)/4. + (357*s4*t3)/2. - (273*s5*t3)/4. - (85*s*t4)/8. + (2125*s3*t4)/32. - (1445*s4*t4)/16. + (1105*s5*t4)/32.;

  df1[1][4] = -(s*t)/8. + (25*s3*t)/32. - (17*s4*t)/16. + (13*s5*t)/32. + (57*s*t2)/16. - (1425*s3*t2)/64. + (969*s4*t2)/32. - (741*s5*t2)/64. - (31*s*t3)/4. + (775*s3*t3)/16. - (527*s4*t3)/8. + (403*s5*t3)/16. + (65*s*t4)/16. - (1625*s3*t4)/64. + (1105*s4*t4)/32. - (845*s5*t4)/64.;

  df1[1][5] = (-3*s*t2)/16. + (75*s3*t2)/64. - (51*s4*t2)/32. + (39*s5*t2)/64. + (s*t3)/2. - (25*s3*t3)/8. + (17*s4*t3)/4. - (13*s5*t3)/8. - (5*s*t4)/16. + (125*s3*t4)/64. - (85*s4*t4)/32. + (65*s5*t4)/64.;

  df1[2][0] = t/4. - (s2*t)/16. - (29*s3*t)/16. + (43*s4*t)/16. - (17*s5*t)/16. - (9*t2)/8. + (9*s2*t2)/32. + (261*s3*t2)/32. - (387*s4*t2)/32. + (153*s5*t2)/32. + (3*t3)/2. - (3*s2*t3)/8. - (87*s3*t3)/8. + (129*s4*t3)/8. - (51*s5*t3)/8. - (5*t4)/8. + (5*s2*t4)/32. + (145*s3*t4)/32. - (215*s4*t4)/32. + (85*s5*t4)/32.;

  df1[2][1] = -0.5 + s2/8. + (29*s3)/8. - (43*s4)/8. + (17*s5)/8. + (75*t2)/8. - (75*s2*t2)/32. - (2175*s3*t2)/32. + (3225*s4*t2)/32. - (1275*s5*t2)/32. - 17*t3 + (17*s2*t3)/4. + (493*s3*t3)/4. - (731*s4*t3)/4. + (289*s5*t3)/4. + (65*t4)/8. - (65*s2*t4)/32. - (1885*s3*t4)/32. + (2795*s4*t4)/32. - (1105*s5*t4)/32.;

  df1[2][2] = -t/2. + (s2*t)/8. + (29*s3*t)/8. - (43*s4*t)/8. + (17*s5*t)/8. - (87*t2)/4. + (87*s2*t2)/16. + (2523*s3*t2)/16. - (3741*s4*t2)/16. + (1479*s5*t2)/16. + 43*t3 - (43*s2*t3)/4. - (1247*s3*t3)/4. + (1849*s4*t3)/4. - (731*s5*t3)/4. - (85*t4)/4. + (85*s2*t4)/16. + (2465*s3*t4)/16. - (3655*s4*t4)/16. + (1445*s5*t4)/16.;

  df1[2][3] = 0.5 - s2/8. - (29*s3)/8. + (43*s4)/8. - (17*s5)/8. + (81*t2)/4. - (81*s2*t2)/16. - (2349*s3*t2)/16. + (3483*s4*t2)/16. - (1377*s5*t2)/16. - 42*t3 + (21*s2*t3)/2. + (609*s3*t3)/2. - (903*s4*t3)/2. + (357*s5*t3)/2. + (85*t4)/4. - (85*s2*t4)/16. - (2465*s3*t4)/16. + (3655*s4*t4)/16. - (1445*s5*t4)/16.;

  df1[2][4] = t/4. - (s2*t)/16. - (29*s3*t)/16. + (43*s4*t)/16. - (17*s5*t)/16. - (57*t2)/8. + (57*s2*t2)/32. + (1653*s3*t2)/32. - (2451*s4*t2)/32. + (969*s5*t2)/32. + (31*t3)/2. - (31*s2*t3)/8. - (899*s3*t3)/8. + (1333*s4*t3)/8. - (527*s5*t3)/8. - (65*t4)/8. + (65*s2*t4)/32. + (1885*s3*t4)/32. - (2795*s4*t4)/32. + (1105*s5*t4)/32.;

  df1[2][5] = (3*t2)/8. - (3*s2*t2)/32. - (87*s3*t2)/32. + (129*s4*t2)/32. - (51*s5*t2)/32. - t3 + (s2*t3)/4. + (29*s3*t3)/4. - (43*s4*t3)/4. + (17*s5*t3)/4. + (5*t4)/8. - (5*s2*t4)/32. - (145*s3*t4)/32. + (215*s4*t4)/32. - (85*s5*t4)/32.;

  df1[3][0] = (s*t)/8. + (27*s3*t)/16. - (21*s4*t)/8. + (17*s5*t)/16. - (9*s*t2)/16. - (243*s3*t2)/32. + (189*s4*t2)/16. - (153*s5*t2)/32. + (3*s*t3)/4. + (81*s3*t3)/8. - (63*s4*t3)/4. + (51*s5*t3)/8. - (5*s*t4)/16. - (135*s3*t4)/32. + (105*s4*t4)/16. - (85*s5*t4)/32.;

  df1[3][1] = -s/4. - (27*s3)/8. + (21*s4)/4. - (17*s5)/8. + (75*s*t2)/16. + (2025*s3*t2)/32. - (1575*s4*t2)/16. + (1275*s5*t2)/32. - (17*s*t3)/2. - (459*s3*t3)/4. + (357*s4*t3)/2. - (289*s5*t3)/4. + (65*s*t4)/16. + (1755*s3*t4)/32. - (1365*s4*t4)/16. + (1105*s5*t4)/32.;

  df1[3][2] = -(s*t)/4. - (27*s3*t)/8. + (21*s4*t)/4. - (17*s5*t)/8. - (87*s*t2)/8. - (2349*s3*t2)/16. + (1827*s4*t2)/8. - (1479*s5*t2)/16. + (43*s*t3)/2. + (1161*s3*t3)/4. - (903*s4*t3)/2. + (731*s5*t3)/4. - (85*s*t4)/8. - (2295*s3*t4)/16. + (1785*s4*t4)/8. - (1445*s5*t4)/16.;

  df1[3][3] = s/4. + (27*s3)/8. - (21*s4)/4. + (17*s5)/8. + (81*s*t2)/8. + (2187*s3*t2)/16. - (1701*s4*t2)/8. + (1377*s5*t2)/16. - 21*s*t3 - (567*s3*t3)/2. + 441*s4*t3 - (357*s5*t3)/2. + (85*s*t4)/8. + (2295*s3*t4)/16. - (1785*s4*t4)/8. + (1445*s5*t4)/16.;

  df1[3][4] = (s*t)/8. + (27*s3*t)/16. - (21*s4*t)/8. + (17*s5*t)/16. - (57*s*t2)/16. - (1539*s3*t2)/32. + (1197*s4*t2)/16. - (969*s5*t2)/32. + (31*s*t3)/4. + (837*s3*t3)/8. - (651*s4*t3)/4. + (527*s5*t3)/8. - (65*s*t4)/16. - (1755*s3*t4)/32. + (1365*s4*t4)/16. - (1105*s5*t4)/32.;

  df1[3][5] = (3*s*t2)/16. + (81*s3*t2)/32. - (63*s4*t2)/16. + (51*s5*t2)/32. - (s*t3)/2. - (27*s3*t3)/4. + (21*s4*t3)/2. - (17*s5*t3)/4. + (5*s*t4)/16. + (135*s3*t4)/32. - (105*s4*t4)/16. + (85*s5*t4)/32.;

  df1[4][0] = (s2*t)/32. - (19*s3*t)/32. + (31*s4*t)/32. - (13*s5*t)/32. - (9*s2*t2)/64. + (171*s3*t2)/64. - (279*s4*t2)/64. + (117*s5*t2)/64. + (3*s2*t3)/16. - (57*s3*t3)/16. + (93*s4*t3)/16. - (39*s5*t3)/16. - (5*s2*t4)/64. + (95*s3*t4)/64. - (155*s4*t4)/64. + (65*s5*t4)/64.;

  df1[4][1] = -s2/16. + (19*s3)/16. - (31*s4)/16. + (13*s5)/16. + (75*s2*t2)/64. - (1425*s3*t2)/64. + (2325*s4*t2)/64. - (975*s5*t2)/64. - (17*s2*t3)/8. + (323*s3*t3)/8. - (527*s4*t3)/8. + (221*s5*t3)/8. + (65*s2*t4)/64. - (1235*s3*t4)/64. + (2015*s4*t4)/64. - (845*s5*t4)/64.;

  df1[4][2] = -(s2*t)/16. + (19*s3*t)/16. - (31*s4*t)/16. + (13*s5*t)/16. - (87*s2*t2)/32. + (1653*s3*t2)/32. - (2697*s4*t2)/32. + (1131*s5*t2)/32. + (43*s2*t3)/8. - (817*s3*t3)/8. + (1333*s4*t3)/8. - (559*s5*t3)/8. - (85*s2*t4)/32. + (1615*s3*t4)/32. - (2635*s4*t4)/32. + (1105*s5*t4)/32.;

  df1[4][3] = s2/16. - (19*s3)/16. + (31*s4)/16. - (13*s5)/16. + (81*s2*t2)/32. - (1539*s3*t2)/32. + (2511*s4*t2)/32. - (1053*s5*t2)/32. - (21*s2*t3)/4. + (399*s3*t3)/4. - (651*s4*t3)/4. + (273*s5*t3)/4. + (85*s2*t4)/32. - (1615*s3*t4)/32. + (2635*s4*t4)/32. - (1105*s5*t4)/32.;

  df1[4][4] = (s2*t)/32. - (19*s3*t)/32. + (31*s4*t)/32. - (13*s5*t)/32. - (57*s2*t2)/64. + (1083*s3*t2)/64. - (1767*s4*t2)/64. + (741*s5*t2)/64. + (31*s2*t3)/16. - (589*s3*t3)/16. + (961*s4*t3)/16. - (403*s5*t3)/16. - (65*s2*t4)/64. + (1235*s3*t4)/64. - (2015*s4*t4)/64. + (845*s5*t4)/64.;

  df1[4][5] = (3*s2*t2)/64. - (57*s3*t2)/64. + (93*s4*t2)/64. - (39*s5*t2)/64. - (s2*t3)/8. + (19*s3*t3)/8. - (31*s4*t3)/8. + (13*s5*t3)/8. + (5*s2*t4)/64. - (95*s3*t4)/64. + (155*s4*t4)/64. - (65*s5*t4)/64.;

  df1[5][0] = (s3*t)/32. - (s4*t)/16. + (s5*t)/32. - (9*s3*t2)/64. + (9*s4*t2)/32. - (9*s5*t2)/64. + (3*s3*t3)/16. - (3*s4*t3)/8. + (3*s5*t3)/16. - (5*s3*t4)/64. + (5*s4*t4)/32. - (5*s5*t4)/64.;

  df1[5][1] = -s3/16. + s4/8. - s5/16. + (75*s3*t2)/64. - (75*s4*t2)/32. + (75*s5*t2)/64. - (17*s3*t3)/8. + (17*s4*t3)/4. - (17*s5*t3)/8. + (65*s3*t4)/64. - (65*s4*t4)/32. + (65*s5*t4)/64.;

  df1[5][2] = -(s3*t)/16. + (s4*t)/8. - (s5*t)/16. - (87*s3*t2)/32. + (87*s4*t2)/16. - (87*s5*t2)/32. + (43*s3*t3)/8. - (43*s4*t3)/4. + (43*s5*t3)/8. - (85*s3*t4)/32. + (85*s4*t4)/16. - (85*s5*t4)/32.;

  df1[5][3] = s3/16. - s4/8. + s5/16. + (81*s3*t2)/32. - (81*s4*t2)/16. + (81*s5*t2)/32. - (21*s3*t3)/4. + (21*s4*t3)/2. - (21*s5*t3)/4. + (85*s3*t4)/32. - (85*s4*t4)/16. + (85*s5*t4)/32.;

  df1[5][4] = (s3*t)/32. - (s4*t)/16. + (s5*t)/32. - (57*s3*t2)/64. + (57*s4*t2)/32. - (57*s5*t2)/64. + (31*s3*t3)/16. - (31*s4*t3)/8. + (31*s5*t3)/16. - (65*s3*t4)/64. + (65*s4*t4)/32. - (65*s5*t4)/64.;

  df1[5][5] = (3*s3*t2)/64. - (3*s4*t2)/32. + (3*s5*t2)/64. - (s3*t3)/8. + (s4*t3)/4. - (s5*t3)/8. + (5*s3*t4)/64. - (5*s4*t4)/32. + (5*s5*t4)/64.;

  /* convert df0 and df1 to the derivatives w.r.t. rho and |nabla rho| */

  for (i=0; i<=5; i++){
    for (j=0; j<=5; j++){

      tmp0 = df0[i][j]/d1; // w.r.t. rho16
      tmp1 = df1[i][j]/d2; // w.r.t. R13 

      df0[i][j] = tmp0*drho16_0 + tmp1*dR0/(3.0*R13*R13+al);
      df1[i][j] = tmp0*drho16_1 + tmp1*dR1/(3.0*R13*R13+al);
    }
  }

  /* calculate gx[0,1,2] */

  gx[0] = 0.0;  gx[1] = 0.0;  gx[2] = 0.0;

  for (i=0; i<=5; i++){
    for (j=0; j<=5; j++){

      ii = i1+i-2;
      jj = j1+j-2;

      gx[0] += NLX_coes[mu][ii][jj]*f[i][j]; 
      gx[1] += NLX_coes[mu][ii][jj]*df0[i][j]; 
      gx[2] += NLX_coes[mu][ii][jj]*df1[i][j]; 
    }
  }

}




double mask_func_q(int m_flag, double q2, double alpha_para)
{
  double al1,al2,al3,al4,al6;
  double mfq,deno1,deno2,deno3,deno4;

  al1 = alpha_para;
  al2 = al1*al1;
  al4 = al2*al2;
  al6 = al2*al4;

  deno1 = q2 + al1*al1;
  deno2 = deno1*deno1;
  deno3 = deno1*deno2;
  deno4 = deno2*deno2;

  switch (m_flag){
  
  case 0:

    mfq = al2/deno1;

    break;

  case 1:

    mfq = al2/deno1 - al4/deno2;

    break;

  case 2:

    mfq = al2/deno1 - 4.0*al4/deno2 + al4*(3.0*al2-q2)/deno3;

    break;

  case 3:

    mfq = al2/deno1 - 7.0*al4/deno2 + 4.0*al4*(3.0*al2-q2)/deno3 - 6.0*al6*(al2-q2)/deno4;

    break;

  default:
    printf("m_flag is invalid.\n"); exit(0);

  }

  return mfq;
}




void set_gx_kernel_coes()
{
  Num_TPF = 3;
  Num_Grid1 = 40;
  Num_Grid2 = 40;
  min_rho16 =  0.000000000000000;
  max_rho16 =  8.000000000000000;
  min_R13 = -5.000000000000000;
  max_R13 =  4.000000000000000;
  al_R13 =  2.000000000000000;
  alpha_para =  6.000000000000000;

  NLX_coes[0][  0][  0] =+6.273940716738721e-10;
  NLX_coes[0][  0][  1] =+1.282933682030845e-01;
  NLX_coes[0][  0][  2] =+2.390128181078433e-01;
  NLX_coes[0][  0][  3] =+3.275387433369281e-01;
  NLX_coes[0][  0][  4] =+3.883949417298855e-01;
  NLX_coes[0][  0][  5] =+4.174270828960163e-01;
  NLX_coes[0][  0][  6] =+4.115060409206118e-01;
  NLX_coes[0][  0][  7] =+3.690440291142205e-01;
  NLX_coes[0][  0][  8] =+2.909981190344654e-01;
  NLX_coes[0][  0][  9] =+1.819554198531759e-01;
  NLX_coes[0][  0][ 10] =+4.974181647380519e-02;
  NLX_coes[0][  0][ 11] =-9.674274670790148e-02;
  NLX_coes[0][  0][ 12] =-2.488377548958450e-01;
  NLX_coes[0][  0][ 13] =-3.983170548542610e-01;
  NLX_coes[0][  0][ 14] =-5.364574706362186e-01;
  NLX_coes[0][  0][ 15] =-6.526192987362706e-01;
  NLX_coes[0][  0][ 16] =-7.343649451050817e-01;
  NLX_coes[0][  0][ 17] =-7.704197502350637e-01;
  NLX_coes[0][  0][ 18] =-7.539453197340459e-01;
  NLX_coes[0][  0][ 19] =-6.859969399921365e-01;
  NLX_coes[0][  0][ 20] =-5.820684124424350e-01;
  NLX_coes[0][  0][ 21] =-4.765667585519340e-01;
  NLX_coes[0][  0][ 22] =-4.112029560661311e-01;
  NLX_coes[0][  0][ 23] =-4.124259241517927e-01;
  NLX_coes[0][  0][ 24] =-4.759282282331279e-01;
  NLX_coes[0][  0][ 25] =-5.761829697861403e-01;
  NLX_coes[0][  0][ 26] =-6.950614873578826e-01;
  NLX_coes[0][  0][ 27] =-7.926683941502738e-01;
  NLX_coes[0][  0][ 28] =-8.011439012260357e-01;
  NLX_coes[0][  0][ 29] =-7.347915992048881e-01;
  NLX_coes[0][  0][ 30] =-6.297684081023929e-01;
  NLX_coes[0][  0][ 31] =-3.787366192042060e-01;
  NLX_coes[0][  0][ 32] =+7.423342763135357e-03;
  NLX_coes[0][  0][ 33] =+4.565751336205221e-01;
  NLX_coes[0][  0][ 34] =+8.617029730740566e-01;
  NLX_coes[0][  0][ 35] =+1.132196448153300e+00;
  NLX_coes[0][  0][ 36] =+1.303283439758838e+00;
  NLX_coes[0][  0][ 37] =+1.386311326111667e+00;
  NLX_coes[0][  0][ 38] =+1.394258869735335e+00;
  NLX_coes[0][  0][ 39] =+1.339985220112877e+00;
  NLX_coes[0][  0][ 40] =+1.240981227093959e+00;
  NLX_coes[0][  1][  0] =-2.954235103836267e-02;
  NLX_coes[0][  1][  1] =+6.187318992082803e-02;
  NLX_coes[0][  1][  2] =+1.526024428929203e-01;
  NLX_coes[0][  1][  3] =+2.289147526905033e-01;
  NLX_coes[0][  1][  4] =+2.842210482605796e-01;
  NLX_coes[0][  1][  5] =+3.130634405525524e-01;
  NLX_coes[0][  1][  6] =+3.109069227981626e-01;
  NLX_coes[0][  1][  7] =+2.745412099991651e-01;
  NLX_coes[0][  1][  8] =+2.031882419891572e-01;
  NLX_coes[0][  1][  9] =+1.002975184007372e-01;
  NLX_coes[0][  1][ 10] =-2.557601531157641e-02;
  NLX_coes[0][  1][ 11] =-1.647557446188209e-01;
  NLX_coes[0][  1][ 12] =-3.101042455913172e-01;
  NLX_coes[0][  1][ 13] =-4.545846064102385e-01;
  NLX_coes[0][  1][ 14] =-5.923881989178795e-01;
  NLX_coes[0][  1][ 15] =-7.126636750553454e-01;
  NLX_coes[0][  1][ 16] =-8.022350026102552e-01;
  NLX_coes[0][  1][ 17] =-8.475204863900602e-01;
  NLX_coes[0][  1][ 18] =-8.389686002249815e-01;
  NLX_coes[0][  1][ 19] =-7.726691759226714e-01;
  NLX_coes[0][  1][ 20] =-6.566711582095712e-01;
  NLX_coes[0][  1][ 21] =-5.234011504638999e-01;
  NLX_coes[0][  1][ 22] =-4.248325771768651e-01;
  NLX_coes[0][  1][ 23] =-3.950412260829402e-01;
  NLX_coes[0][  1][ 24] =-4.149467915450897e-01;
  NLX_coes[0][  1][ 25] =-4.363379799615563e-01;
  NLX_coes[0][  1][ 26] =-4.382115620912595e-01;
  NLX_coes[0][  1][ 27] =-4.468200764427819e-01;
  NLX_coes[0][  1][ 28] =-3.091485134426797e-01;
  NLX_coes[0][  1][ 29] =-9.693740890198570e-02;
  NLX_coes[0][  1][ 30] =+1.694957588623872e-01;
  NLX_coes[0][  1][ 31] =+3.912424992523851e-01;
  NLX_coes[0][  1][ 32] =+5.596090541303688e-01;
  NLX_coes[0][  1][ 33] =+7.612571533595391e-01;
  NLX_coes[0][  1][ 34] =+9.500692731753074e-01;
  NLX_coes[0][  1][ 35] =+1.082443724593749e+00;
  NLX_coes[0][  1][ 36] =+1.137109993792141e+00;
  NLX_coes[0][  1][ 37] =+1.128102590785791e+00;
  NLX_coes[0][  1][ 38] =+1.066617248299861e+00;
  NLX_coes[0][  1][ 39] =+9.644149977168528e-01;
  NLX_coes[0][  1][ 40] =+8.324830500516693e-01;
  NLX_coes[0][  2][  0] =-1.181694027367114e-01;
  NLX_coes[0][  2][  1] =-3.652329165819170e-02;
  NLX_coes[0][  2][  2] =+4.089663967552955e-02;
  NLX_coes[0][  2][  3] =+1.096129415222350e-01;
  NLX_coes[0][  2][  4] =+1.631301523156576e-01;
  NLX_coes[0][  2][  5] =+1.953830864760924e-01;
  NLX_coes[0][  2][  6] =+2.008738501165102e-01;
  NLX_coes[0][  2][  7] =+1.746829184491775e-01;
  NLX_coes[0][  2][  8] =+1.132534344163842e-01;
  NLX_coes[0][  2][  9] =+1.714429577196655e-02;
  NLX_coes[0][  2][ 10] =-1.028373403226784e-01;
  NLX_coes[0][  2][ 11] =-2.318337342485715e-01;
  NLX_coes[0][  2][ 12] =-3.650251845877320e-01;
  NLX_coes[0][  2][ 13] =-5.000952508747490e-01;
  NLX_coes[0][  2][ 14] =-6.361314538008853e-01;
  NLX_coes[0][  2][ 15] =-7.603830959642681e-01;
  NLX_coes[0][  2][ 16] =-8.603080267335879e-01;
  NLX_coes[0][  2][ 17] =-9.203471753264963e-01;
  NLX_coes[0][  2][ 18] =-9.281880709087569e-01;
  NLX_coes[0][  2][ 19] =-8.790703646787548e-01;
  NLX_coes[0][  2][ 20] =-7.658282221149062e-01;
  NLX_coes[0][  2][ 21] =-6.012861751346329e-01;
  NLX_coes[0][  2][ 22] =-4.718239886331345e-01;
  NLX_coes[0][  2][ 23] =-4.400077230392646e-01;
  NLX_coes[0][  2][ 24] =-4.471187234403020e-01;
  NLX_coes[0][  2][ 25] =-4.105381094622509e-01;
  NLX_coes[0][  2][ 26] =-2.574383230900756e-01;
  NLX_coes[0][  2][ 27] =-2.974186728255620e-02;
  NLX_coes[0][  2][ 28] =+2.730832997741678e-01;
  NLX_coes[0][  2][ 29] =+6.217062097847613e-01;
  NLX_coes[0][  2][ 30] =+8.622412073663779e-01;
  NLX_coes[0][  2][ 31] =+8.940800978661191e-01;
  NLX_coes[0][  2][ 32] =+8.740692733905655e-01;
  NLX_coes[0][  2][ 33] =+9.176917140535276e-01;
  NLX_coes[0][  2][ 34] =+9.697100285226026e-01;
  NLX_coes[0][  2][ 35] =+9.544381078388292e-01;
  NLX_coes[0][  2][ 36] =+8.977378890082161e-01;
  NLX_coes[0][  2][ 37] =+8.028521886386508e-01;
  NLX_coes[0][  2][ 38] =+6.754125655742024e-01;
  NLX_coes[0][  2][ 39] =+5.237372495989985e-01;
  NLX_coes[0][  2][ 40] =+3.578385803270835e-01;
  NLX_coes[0][  3][  0] =-2.658811559360342e-01;
  NLX_coes[0][  3][  1] =-1.864969053523237e-01;
  NLX_coes[0][  3][  2] =-1.116183369853079e-01;
  NLX_coes[0][  3][  3] =-4.353330530005333e-02;
  NLX_coes[0][  3][  4] =+1.315018101372011e-02;
  NLX_coes[0][  3][  5] =+5.325496337657107e-02;
  NLX_coes[0][  3][  6] =+7.139664757520645e-02;
  NLX_coes[0][  3][  7] =+6.169120363603976e-02;
  NLX_coes[0][  3][  8] =+1.724838898648975e-02;
  NLX_coes[0][  3][  9] =-6.809829065840795e-02;
  NLX_coes[0][  3][ 10] =-1.854147815206285e-01;
  NLX_coes[0][  3][ 11] =-3.016809938623912e-01;
  NLX_coes[0][  3][ 12] =-4.058581790093029e-01;
  NLX_coes[0][  3][ 13] =-5.297187353264806e-01;
  NLX_coes[0][  3][ 14] =-6.602680743740723e-01;
  NLX_coes[0][  3][ 15] =-7.834242939647194e-01;
  NLX_coes[0][  3][ 16] =-8.917921599819270e-01;
  NLX_coes[0][  3][ 17] =-9.704975361306006e-01;
  NLX_coes[0][  3][ 18] =-9.947758552695741e-01;
  NLX_coes[0][  3][ 19] =-9.858182219951046e-01;
  NLX_coes[0][  3][ 20] =-9.209531104076530e-01;
  NLX_coes[0][  3][ 21] =-7.469534452042487e-01;
  NLX_coes[0][  3][ 22] =-6.241307930059602e-01;
  NLX_coes[0][  3][ 23] =-6.292050406452884e-01;
  NLX_coes[0][  3][ 24] =-6.112237876039729e-01;
  NLX_coes[0][  3][ 25] =-5.249689661692810e-01;
  NLX_coes[0][  3][ 26] =-3.637619395596108e-01;
  NLX_coes[0][  3][ 27] =-1.243284023986997e-01;
  NLX_coes[0][  3][ 28] =+1.798283572190721e-01;
  NLX_coes[0][  3][ 29] =+5.410776970835702e-01;
  NLX_coes[0][  3][ 30] =+8.864446685502388e-01;
  NLX_coes[0][  3][ 31] =+9.874163919584820e-01;
  NLX_coes[0][  3][ 32] =+9.637857398359363e-01;
  NLX_coes[0][  3][ 33] =+8.926686414231216e-01;
  NLX_coes[0][  3][ 34] =+8.004483249125781e-01;
  NLX_coes[0][  3][ 35] =+6.869016552237586e-01;
  NLX_coes[0][  3][ 36] =+5.415879395254742e-01;
  NLX_coes[0][  3][ 37] =+3.712568365532308e-01;
  NLX_coes[0][  3][ 38] =+1.832348954224937e-01;
  NLX_coes[0][  3][ 39] =-1.578553832067887e-02;
  NLX_coes[0][  3][ 40] =-2.193476445526304e-01;
  NLX_coes[0][  4][  0] =-4.726776104972109e-01;
  NLX_coes[0][  4][  1] =-3.914244112805098e-01;
  NLX_coes[0][  4][  2] =-3.135905574103442e-01;
  NLX_coes[0][  4][  3] =-2.411623311279955e-01;
  NLX_coes[0][  4][  4] =-1.773191763787128e-01;
  NLX_coes[0][  4][  5] =-1.259437739907138e-01;
  NLX_coes[0][  4][  6] =-9.126883131907482e-02;
  NLX_coes[0][  4][  7] =-7.808709863134873e-02;
  NLX_coes[0][  4][  8] =-9.335726448168774e-02;
  NLX_coes[0][  4][  9] =-1.511261086826196e-01;
  NLX_coes[0][  4][ 10] =-2.769436074961124e-01;
  NLX_coes[0][  4][ 11] =-3.870244344454746e-01;
  NLX_coes[0][  4][ 12] =-4.129953501090752e-01;
  NLX_coes[0][  4][ 13] =-5.464064424541358e-01;
  NLX_coes[0][  4][ 14] =-6.615359959635017e-01;
  NLX_coes[0][  4][ 15] =-7.671658315188038e-01;
  NLX_coes[0][  4][ 16] =-8.710043464662292e-01;
  NLX_coes[0][  4][ 17] =-9.660602030370375e-01;
  NLX_coes[0][  4][ 18] =-9.913635268759973e-01;
  NLX_coes[0][  4][ 19] =-1.012351758828612e+00;
  NLX_coes[0][  4][ 20] =-1.026268488353244e+00;
  NLX_coes[0][  4][ 21] =-1.040142634735299e+00;
  NLX_coes[0][  4][ 22] =-1.066011408362632e+00;
  NLX_coes[0][  4][ 23] =-1.105651074246872e+00;
  NLX_coes[0][  4][ 24] =-1.140550876535046e+00;
  NLX_coes[0][  4][ 25] =-1.174715283067622e+00;
  NLX_coes[0][  4][ 26] =-1.204970699244379e+00;
  NLX_coes[0][  4][ 27] =-1.220249940512021e+00;
  NLX_coes[0][  4][ 28] =-1.107497629237132e+00;
  NLX_coes[0][  4][ 29] =-2.331133466894047e-01;
  NLX_coes[0][  4][ 30] =+4.220232359310719e-01;
  NLX_coes[0][  4][ 31] =+7.050292883033619e-01;
  NLX_coes[0][  4][ 32] =+7.339285970100898e-01;
  NLX_coes[0][  4][ 33] =+6.277221282134261e-01;
  NLX_coes[0][  4][ 34] =+4.580275840643683e-01;
  NLX_coes[0][  4][ 35] =+2.576771888283257e-01;
  NLX_coes[0][  4][ 36] =+3.580504855630789e-02;
  NLX_coes[0][  4][ 37] =-1.995102428196345e-01;
  NLX_coes[0][  4][ 38] =-4.414215594137895e-01;
  NLX_coes[0][  4][ 39] =-6.846204073587721e-01;
  NLX_coes[0][  4][ 40] =-9.256783237691176e-01;
  NLX_coes[0][  5][  0] =-7.385587663783327e-01;
  NLX_coes[0][  5][  1] =-6.533985647688548e-01;
  NLX_coes[0][  5][  2] =-5.703489632633215e-01;
  NLX_coes[0][  5][  3] =-4.912986221847392e-01;
  NLX_coes[0][  5][  4] =-4.185933689469233e-01;
  NLX_coes[0][  5][  5] =-3.550141253543996e-01;
  NLX_coes[0][  5][  6] =-3.035701547902955e-01;
  NLX_coes[0][  5][  7] =-2.673404449891008e-01;
  NLX_coes[0][  5][  8] =-2.492547609504346e-01;
  NLX_coes[0][  5][  9] =-2.539044194274331e-01;
  NLX_coes[0][  5][ 10] =-3.144410807105673e-01;
  NLX_coes[0][  5][ 11] =-3.803743053423627e-01;
  NLX_coes[0][  5][ 12] =-4.588988599763673e-01;
  NLX_coes[0][  5][ 13] =-5.301752188418373e-01;
  NLX_coes[0][  5][ 14] =-6.012242054107159e-01;
  NLX_coes[0][  5][ 15] =-6.735190027567751e-01;
  NLX_coes[0][  5][ 16] =-7.425705631502009e-01;
  NLX_coes[0][  5][ 17] =-7.985381252827686e-01;
  NLX_coes[0][  5][ 18] =-8.579799554406400e-01;
  NLX_coes[0][  5][ 19] =-9.493172384091703e-01;
  NLX_coes[0][  5][ 20] =-1.051016469370032e+00;
  NLX_coes[0][  5][ 21] =-1.139961824511676e+00;
  NLX_coes[0][  5][ 22] =-1.214475669318518e+00;
  NLX_coes[0][  5][ 23] =-1.282412706800113e+00;
  NLX_coes[0][  5][ 24] =-1.355296648420847e+00;
  NLX_coes[0][  5][ 25] =-1.444431257185224e+00;
  NLX_coes[0][  5][ 26] =-1.500554196327404e+00;
  NLX_coes[0][  5][ 27] =-1.558893182039945e+00;
  NLX_coes[0][  5][ 28] =-1.318653538746881e+00;
  NLX_coes[0][  5][ 29] =-5.326525178585110e-01;
  NLX_coes[0][  5][ 30] =+4.195867095917079e-02;
  NLX_coes[0][  5][ 31] =+2.968219442157055e-01;
  NLX_coes[0][  5][ 32] =+2.991607437282964e-01;
  NLX_coes[0][  5][ 33] =+1.469843207628371e-01;
  NLX_coes[0][  5][ 34] =-8.590485276378690e-02;
  NLX_coes[0][  5][ 35] =-3.563091737816749e-01;
  NLX_coes[0][  5][ 36] =-6.428015557659245e-01;
  NLX_coes[0][  5][ 37] =-9.340297747475880e-01;
  NLX_coes[0][  5][ 38] =-1.223079802208192e+00;
  NLX_coes[0][  5][ 39] =-1.506093729078381e+00;
  NLX_coes[0][  5][ 40] =-1.782153737567771e+00;
  NLX_coes[0][  6][  0] =-1.063524623574674e+00;
  NLX_coes[0][  6][  1] =-9.741413736575538e-01;
  NLX_coes[0][  6][  2] =-8.857203477638340e-01;
  NLX_coes[0][  6][  3] =-8.000920810133004e-01;
  NLX_coes[0][  6][  4] =-7.191867802128190e-01;
  NLX_coes[0][  6][  5] =-6.451345205502523e-01;
  NLX_coes[0][  6][  6] =-5.802137820586539e-01;
  NLX_coes[0][  6][  7] =-5.269306265054363e-01;
  NLX_coes[0][  6][  8] =-4.887555925853307e-01;
  NLX_coes[0][  6][  9] =-4.715882431077306e-01;
  NLX_coes[0][  6][ 10] =-4.788352345548180e-01;
  NLX_coes[0][  6][ 11] =-5.201077687907069e-01;
  NLX_coes[0][  6][ 12] =-6.082173131870149e-01;
  NLX_coes[0][  6][ 13] =-6.646825512017679e-01;
  NLX_coes[0][  6][ 14] =-7.301362105022505e-01;
  NLX_coes[0][  6][ 15] =-7.977924846860383e-01;
  NLX_coes[0][  6][ 16] =-8.496740940799149e-01;
  NLX_coes[0][  6][ 17] =-8.485344325048869e-01;
  NLX_coes[0][  6][ 18] =-7.432612090038561e-01;
  NLX_coes[0][  6][ 19] =-7.584453525971573e-01;
  NLX_coes[0][  6][ 20] =-8.184033514718633e-01;
  NLX_coes[0][  6][ 21] =-9.589762112332842e-01;
  NLX_coes[0][  6][ 22] =-1.124671364701054e+00;
  NLX_coes[0][  6][ 23] =-1.278094223915899e+00;
  NLX_coes[0][  6][ 24] =-1.378093938742839e+00;
  NLX_coes[0][  6][ 25] =-1.512820293234814e+00;
  NLX_coes[0][  6][ 26] =-1.166562721234844e+00;
  NLX_coes[0][  6][ 27] =-8.315441185993907e-01;
  NLX_coes[0][  6][ 28] =-5.383186914635248e-01;
  NLX_coes[0][  6][ 29] =-3.009443701715058e-01;
  NLX_coes[0][  6][ 30] =-1.479353684656629e-01;
  NLX_coes[0][  6][ 31] =-1.429521443037592e-01;
  NLX_coes[0][  6][ 32] =-2.838501806734466e-01;
  NLX_coes[0][  6][ 33] =-5.297173828300524e-01;
  NLX_coes[0][  6][ 34] =-8.361622421149814e-01;
  NLX_coes[0][  6][ 35] =-1.170371148541349e+00;
  NLX_coes[0][  6][ 36] =-1.512234725908906e+00;
  NLX_coes[0][  6][ 37] =-1.850614820833980e+00;
  NLX_coes[0][  6][ 38] =-2.179613781528233e+00;
  NLX_coes[0][  6][ 39] =-2.497007192237198e+00;
  NLX_coes[0][  6][ 40] =-2.803956677897758e+00;
  NLX_coes[0][  7][  0] =-1.447575182085253e+00;
  NLX_coes[0][  7][  1] =-1.354987580965625e+00;
  NLX_coes[0][  7][  2] =-1.262518888242299e+00;
  NLX_coes[0][  7][  3] =-1.172019138729097e+00;
  NLX_coes[0][  7][  4] =-1.085320301431816e+00;
  NLX_coes[0][  7][  5] =-1.004413272919645e+00;
  NLX_coes[0][  7][  6] =-9.315856290064849e-01;
  NLX_coes[0][  7][  7] =-8.696801487673984e-01;
  NLX_coes[0][  7][  8] =-8.226300006463338e-01;
  NLX_coes[0][  7][  9] =-7.962184694898017e-01;
  NLX_coes[0][  7][ 10] =-7.960101465140710e-01;
  NLX_coes[0][  7][ 11] =-8.317585982559101e-01;
  NLX_coes[0][  7][ 12] =-9.004198301179439e-01;
  NLX_coes[0][  7][ 13] =-9.766995095693569e-01;
  NLX_coes[0][  7][ 14] =-1.064284362247320e+00;
  NLX_coes[0][  7][ 15] =-1.158853855977757e+00;
  NLX_coes[0][  7][ 16] =-1.238361632537656e+00;
  NLX_coes[0][  7][ 17] =-1.258298649332464e+00;
  NLX_coes[0][  7][ 18] =-1.169067731204866e+00;
  NLX_coes[0][  7][ 19] =-1.031679491301615e+00;
  NLX_coes[0][  7][ 20] =-8.732737542366577e-01;
  NLX_coes[0][  7][ 21] =-9.227422671163530e-01;
  NLX_coes[0][  7][ 22] =-1.064633634734714e+00;
  NLX_coes[0][  7][ 23] =-1.287948515962996e+00;
  NLX_coes[0][  7][ 24] =-1.360260977028571e+00;
  NLX_coes[0][  7][ 25] =-1.325668912841921e+00;
  NLX_coes[0][  7][ 26] =-6.723658465274287e-01;
  NLX_coes[0][  7][ 27] =+1.339099947814951e-01;
  NLX_coes[0][  7][ 28] =+2.043678172656981e-01;
  NLX_coes[0][  7][ 29] =-1.054949230517039e-01;
  NLX_coes[0][  7][ 30] =-4.118122976044201e-01;
  NLX_coes[0][  7][ 31] =-7.170541655298287e-01;
  NLX_coes[0][  7][ 32] =-1.050453063477908e+00;
  NLX_coes[0][  7][ 33] =-1.416320135037120e+00;
  NLX_coes[0][  7][ 34] =-1.803164233922507e+00;
  NLX_coes[0][  7][ 35] =-2.196523255724065e+00;
  NLX_coes[0][  7][ 36] =-2.584999367441934e+00;
  NLX_coes[0][  7][ 37] =-2.961497421628859e+00;
  NLX_coes[0][  7][ 38] =-3.322582940709061e+00;
  NLX_coes[0][  7][ 39] =-3.667959950893464e+00;
  NLX_coes[0][  7][ 40] =-4.000523707566035e+00;
  NLX_coes[0][  8][  0] =-1.890710441909454e+00;
  NLX_coes[0][  8][  1] =-1.796876792260145e+00;
  NLX_coes[0][  8][  2] =-1.702648941515548e+00;
  NLX_coes[0][  8][  3] =-1.609973501841513e+00;
  NLX_coes[0][  8][  4] =-1.520794376198154e+00;
  NLX_coes[0][  8][  5] =-1.437273002570321e+00;
  NLX_coes[0][  8][  6] =-1.362016256230371e+00;
  NLX_coes[0][  8][  7] =-1.298382846351843e+00;
  NLX_coes[0][  8][  8] =-1.250856002269556e+00;
  NLX_coes[0][  8][  9] =-1.225117201181172e+00;
  NLX_coes[0][  8][ 10] =-1.227554544950981e+00;
  NLX_coes[0][  8][ 11] =-1.263858539361595e+00;
  NLX_coes[0][  8][ 12] =-1.331868851621333e+00;
  NLX_coes[0][  8][ 13] =-1.426078231141567e+00;
  NLX_coes[0][  8][ 14] =-1.543081105027529e+00;
  NLX_coes[0][  8][ 15] =-1.679712127404823e+00;
  NLX_coes[0][  8][ 16] =-1.821975530182614e+00;
  NLX_coes[0][  8][ 17] =-1.933830245177675e+00;
  NLX_coes[0][  8][ 18] =-1.934242613499348e+00;
  NLX_coes[0][  8][ 19] =-1.670066477683217e+00;
  NLX_coes[0][  8][ 20] =-1.410623080091100e+00;
  NLX_coes[0][  8][ 21] =-1.281441497434938e+00;
  NLX_coes[0][  8][ 22] =-1.270534492038195e+00;
  NLX_coes[0][  8][ 23] =-1.345954289638166e+00;
  NLX_coes[0][  8][ 24] =-1.432366824632547e+00;
  NLX_coes[0][  8][ 25] =-1.396166899214812e+00;
  NLX_coes[0][  8][ 26] =-6.758752156404960e-01;
  NLX_coes[0][  8][ 27] =+9.158278745708491e-02;
  NLX_coes[0][  8][ 28] =+3.064995075827934e-02;
  NLX_coes[0][  8][ 29] =-5.040648493087492e-01;
  NLX_coes[0][  8][ 30] =-1.070285788409989e+00;
  NLX_coes[0][  8][ 31] =-1.592726391550910e+00;
  NLX_coes[0][  8][ 32] =-2.080960040126083e+00;
  NLX_coes[0][  8][ 33] =-2.549745018848179e+00;
  NLX_coes[0][  8][ 34] =-3.004751742330534e+00;
  NLX_coes[0][  8][ 35] =-3.445308287703235e+00;
  NLX_coes[0][  8][ 36] =-3.868738782697207e+00;
  NLX_coes[0][  8][ 37] =-4.272912918125302e+00;
  NLX_coes[0][  8][ 38] =-4.657303707386653e+00;
  NLX_coes[0][  8][ 39] =-5.023479922062161e+00;
  NLX_coes[0][  8][ 40] =-5.375621957253851e+00;
  NLX_coes[0][  9][  0] =-2.392930403046609e+00;
  NLX_coes[0][  9][  1] =-2.300387392543601e+00;
  NLX_coes[0][  9][  2] =-2.207211372072796e+00;
  NLX_coes[0][  9][  3] =-2.115479164338292e+00;
  NLX_coes[0][  9][  4] =-2.027326011450438e+00;
  NLX_coes[0][  9][  5] =-1.945166922899498e+00;
  NLX_coes[0][  9][  6] =-1.871930158726033e+00;
  NLX_coes[0][  9][  7] =-1.811302975905174e+00;
  NLX_coes[0][  9][  8] =-1.767909983670943e+00;
  NLX_coes[0][  9][  9] =-1.747179956079689e+00;
  NLX_coes[0][  9][ 10] =-1.754822239874975e+00;
  NLX_coes[0][  9][ 11] =-1.795055515978474e+00;
  NLX_coes[0][  9][ 12] =-1.868781947944537e+00;
  NLX_coes[0][  9][ 13] =-1.975474890878165e+00;
  NLX_coes[0][  9][ 14] =-2.113094286391262e+00;
  NLX_coes[0][  9][ 15] =-2.278404025939980e+00;
  NLX_coes[0][  9][ 16] =-2.463374471912030e+00;
  NLX_coes[0][  9][ 17] =-2.653718461656962e+00;
  NLX_coes[0][  9][ 18] =-2.797423674233391e+00;
  NLX_coes[0][  9][ 19] =-2.485564519935617e+00;
  NLX_coes[0][  9][ 20] =-2.191003487026853e+00;
  NLX_coes[0][  9][ 21] =-1.993985104199627e+00;
  NLX_coes[0][  9][ 22] =-1.880927648271762e+00;
  NLX_coes[0][  9][ 23] =-1.839513831187482e+00;
  NLX_coes[0][  9][ 24] =-1.801600328023963e+00;
  NLX_coes[0][  9][ 25] =-1.596574864009813e+00;
  NLX_coes[0][  9][ 26] =-1.223897674894336e+00;
  NLX_coes[0][  9][ 27] =-9.899682820657354e-01;
  NLX_coes[0][  9][ 28] =-1.174555119683799e+00;
  NLX_coes[0][  9][ 29] =-1.670900124036070e+00;
  NLX_coes[0][  9][ 30] =-2.277852336859583e+00;
  NLX_coes[0][  9][ 31] =-2.876805693289371e+00;
  NLX_coes[0][  9][ 32] =-3.437496706809801e+00;
  NLX_coes[0][  9][ 33] =-3.961426984894362e+00;
  NLX_coes[0][  9][ 34] =-4.454924849621722e+00;
  NLX_coes[0][  9][ 35] =-4.922123431636924e+00;
  NLX_coes[0][  9][ 36] =-5.364982295684486e+00;
  NLX_coes[0][  9][ 37] =-5.784678219695658e+00;
  NLX_coes[0][  9][ 38] =-6.182781545410378e+00;
  NLX_coes[0][  9][ 39] =-6.562105803781304e+00;
  NLX_coes[0][  9][ 40] =-6.927457370307100e+00;
  NLX_coes[0][ 10][  0] =-2.954235065495995e+00;
  NLX_coes[0][ 10][  1] =-2.865808349962760e+00;
  NLX_coes[0][ 10][  2] =-2.776689429592923e+00;
  NLX_coes[0][ 10][  3] =-2.689069225919674e+00;
  NLX_coes[0][ 10][  4] =-2.605249547711754e+00;
  NLX_coes[0][ 10][  5] =-2.527830826092306e+00;
  NLX_coes[0][ 10][  6] =-2.459899820157177e+00;
  NLX_coes[0][ 10][  7] =-2.405184144439514e+00;
  NLX_coes[0][ 10][  8] =-2.368090945301529e+00;
  NLX_coes[0][ 10][  9] =-2.353504188406798e+00;
  NLX_coes[0][ 10][ 10] =-2.366227921073936e+00;
  NLX_coes[0][ 10][ 11] =-2.409941568060195e+00;
  NLX_coes[0][ 10][ 12] =-2.486613658528201e+00;
  NLX_coes[0][ 10][ 13] =-2.596502811896659e+00;
  NLX_coes[0][ 10][ 14] =-2.737549295673893e+00;
  NLX_coes[0][ 10][ 15] =-2.903778312201632e+00;
  NLX_coes[0][ 10][ 16] =-3.079613333490689e+00;
  NLX_coes[0][ 10][ 17] =-3.224256372765240e+00;
  NLX_coes[0][ 10][ 18] =-3.240744655003815e+00;
  NLX_coes[0][ 10][ 19] =-3.005071020083158e+00;
  NLX_coes[0][ 10][ 20] =-2.876715213823822e+00;
  NLX_coes[0][ 10][ 21] =-2.798232573116712e+00;
  NLX_coes[0][ 10][ 22] =-2.726218810313521e+00;
  NLX_coes[0][ 10][ 23] =-2.654422472327812e+00;
  NLX_coes[0][ 10][ 24] =-2.578892657451209e+00;
  NLX_coes[0][ 10][ 25] =-2.519554358639780e+00;
  NLX_coes[0][ 10][ 26] =-2.533132269197410e+00;
  NLX_coes[0][ 10][ 27] =-2.662473774721537e+00;
  NLX_coes[0][ 10][ 28] =-2.977802458491540e+00;
  NLX_coes[0][ 10][ 29] =-3.446824944848166e+00;
  NLX_coes[0][ 10][ 30] =-4.002972964392913e+00;
  NLX_coes[0][ 10][ 31] =-4.576055828482862e+00;
  NLX_coes[0][ 10][ 32] =-5.130727305371652e+00;
  NLX_coes[0][ 10][ 33] =-5.656349096766708e+00;
  NLX_coes[0][ 10][ 34] =-6.152713093682061e+00;
  NLX_coes[0][ 10][ 35] =-6.622196917167827e+00;
  NLX_coes[0][ 10][ 36] =-7.067125086331231e+00;
  NLX_coes[0][ 10][ 37] =-7.489524709589459e+00;
  NLX_coes[0][ 10][ 38] =-7.891632847769033e+00;
  NLX_coes[0][ 10][ 39] =-8.276546693458455e+00;
  NLX_coes[0][ 10][ 40] =-8.648902997528266e+00;
  NLX_coes[0][ 11][  0] =-3.574624429256941e+00;
  NLX_coes[0][ 11][  1] =-3.493231357228402e+00;
  NLX_coes[0][ 11][  2] =-3.411172866872597e+00;
  NLX_coes[0][ 11][  3] =-3.330704276598492e+00;
  NLX_coes[0][ 11][  4] =-3.254211557966415e+00;
  NLX_coes[0][ 11][  5] =-3.184348531534476e+00;
  NLX_coes[0][ 11][  6] =-3.124161486378654e+00;
  NLX_coes[0][ 11][  7] =-3.077160440620784e+00;
  NLX_coes[0][ 11][  8] =-3.047274810463933e+00;
  NLX_coes[0][ 11][  9] =-3.038625587361029e+00;
  NLX_coes[0][ 11][ 10] =-3.055057258529362e+00;
  NLX_coes[0][ 11][ 11] =-3.099505532264633e+00;
  NLX_coes[0][ 11][ 12] =-3.173469369979400e+00;
  NLX_coes[0][ 11][ 13] =-3.276328418583828e+00;
  NLX_coes[0][ 11][ 14] =-3.404032424271513e+00;
  NLX_coes[0][ 11][ 15] =-3.546100981224121e+00;
  NLX_coes[0][ 11][ 16] =-3.679779450612623e+00;
  NLX_coes[0][ 11][ 17] =-3.762076023205938e+00;
  NLX_coes[0][ 11][ 18] =-3.731977269071706e+00;
  NLX_coes[0][ 11][ 19] =-3.589772646143095e+00;
  NLX_coes[0][ 11][ 20] =-3.441193463679973e+00;
  NLX_coes[0][ 11][ 21] =-3.330998794654921e+00;
  NLX_coes[0][ 11][ 22] =-3.290578162073044e+00;
  NLX_coes[0][ 11][ 23] =-3.369362935847145e+00;
  NLX_coes[0][ 11][ 24] =-3.575981332920285e+00;
  NLX_coes[0][ 11][ 25] =-3.931839432905338e+00;
  NLX_coes[0][ 11][ 26] =-4.513848176020126e+00;
  NLX_coes[0][ 11][ 27] =-4.869847669606336e+00;
  NLX_coes[0][ 11][ 28] =-5.232571460123181e+00;
  NLX_coes[0][ 11][ 29] =-5.651111010241460e+00;
  NLX_coes[0][ 11][ 30] =-6.123644524573809e+00;
  NLX_coes[0][ 11][ 31] =-6.622228326558394e+00;
  NLX_coes[0][ 11][ 32] =-7.121923959782479e+00;
  NLX_coes[0][ 11][ 33] =-7.609047435121552e+00;
  NLX_coes[0][ 11][ 34] =-8.078223762275883e+00;
  NLX_coes[0][ 11][ 35] =-8.528217097726555e+00;
  NLX_coes[0][ 11][ 36] =-8.959424736373800e+00;
  NLX_coes[0][ 11][ 37] =-9.372913701949340e+00;
  NLX_coes[0][ 11][ 38] =-9.770313153110093e+00;
  NLX_coes[0][ 11][ 39] =-1.015408257401898e+01;
  NLX_coes[0][ 11][ 40] =-1.052794160496038e+01;
  NLX_coes[0][ 12][  0] =-4.254098494328934e+00;
  NLX_coes[0][ 12][  1] =-4.182638269741771e+00;
  NLX_coes[0][ 12][  2] =-4.110557650375473e+00;
  NLX_coes[0][ 12][  3] =-4.040115226912210e+00;
  NLX_coes[0][ 12][  4] =-3.973684572169069e+00;
  NLX_coes[0][ 12][  5] =-3.913839298334643e+00;
  NLX_coes[0][ 12][  6] =-3.863417077721762e+00;
  NLX_coes[0][ 12][  7] =-3.825526101771975e+00;
  NLX_coes[0][ 12][  8] =-3.803451772221265e+00;
  NLX_coes[0][ 12][  9] =-3.800426101734308e+00;
  NLX_coes[0][ 12][ 10] =-3.819244560010542e+00;
  NLX_coes[0][ 12][ 11] =-3.861780693461253e+00;
  NLX_coes[0][ 12][ 12] =-3.928417169629476e+00;
  NLX_coes[0][ 12][ 13] =-4.017228352213539e+00;
  NLX_coes[0][ 12][ 14] =-4.122636841384303e+00;
  NLX_coes[0][ 12][ 15] =-4.233129105655749e+00;
  NLX_coes[0][ 12][ 16] =-4.328189962947190e+00;
  NLX_coes[0][ 12][ 17] =-4.376793807487215e+00;
  NLX_coes[0][ 12][ 18] =-4.345984867904026e+00;
  NLX_coes[0][ 12][ 19] =-4.231493850471092e+00;
  NLX_coes[0][ 12][ 20] =-4.053793413581471e+00;
  NLX_coes[0][ 12][ 21] =-3.915574108228376e+00;
  NLX_coes[0][ 12][ 22] =-3.938573390757186e+00;
  NLX_coes[0][ 12][ 23] =-4.258980988203769e+00;
  NLX_coes[0][ 12][ 24] =-4.801014111095417e+00;
  NLX_coes[0][ 12][ 25] =-5.630630939644236e+00;
  NLX_coes[0][ 12][ 26] =-6.873941048598124e+00;
  NLX_coes[0][ 12][ 27] =-7.390624651177879e+00;
  NLX_coes[0][ 12][ 28] =-7.763924472899679e+00;
  NLX_coes[0][ 12][ 29] =-8.123300653239825e+00;
  NLX_coes[0][ 12][ 30] =-8.508544968141672e+00;
  NLX_coes[0][ 12][ 31] =-8.919536454577235e+00;
  NLX_coes[0][ 12][ 32] =-9.344390188413632e+00;
  NLX_coes[0][ 12][ 33] =-9.772182571453680e+00;
  NLX_coes[0][ 12][ 34] =-1.019588818320350e+01;
  NLX_coes[0][ 12][ 35] =-1.061171138401059e+01;
  NLX_coes[0][ 12][ 36] =-1.101787574376455e+01;
  NLX_coes[0][ 12][ 37] =-1.141382560642666e+01;
  NLX_coes[0][ 12][ 38] =-1.179990200857754e+01;
  NLX_coes[0][ 12][ 39] =-1.217733202005676e+01;
  NLX_coes[0][ 12][ 40] =-1.254839119062703e+01;
  NLX_coes[0][ 13][  0] =-4.992657260711666e+00;
  NLX_coes[0][ 13][  1] =-4.933966214063433e+00;
  NLX_coes[0][ 13][  2] =-4.874684715007176e+00;
  NLX_coes[0][ 13][  3] =-4.817020565468085e+00;
  NLX_coes[0][ 13][  4] =-4.763254033161331e+00;
  NLX_coes[0][ 13][  5] =-4.715778184697490e+00;
  NLX_coes[0][ 13][  6] =-4.677113266815789e+00;
  NLX_coes[0][ 13][  7] =-4.649866592647754e+00;
  NLX_coes[0][ 13][  8] =-4.636610411113169e+00;
  NLX_coes[0][ 13][  9] =-4.639658127284781e+00;
  NLX_coes[0][ 13][ 10] =-4.660738306184879e+00;
  NLX_coes[0][ 13][ 11] =-4.700587334167276e+00;
  NLX_coes[0][ 13][ 12] =-4.758458668040594e+00;
  NLX_coes[0][ 13][ 13] =-4.831517143717219e+00;
  NLX_coes[0][ 13][ 14] =-4.914099305089186e+00;
  NLX_coes[0][ 13][ 15] =-4.996974644582090e+00;
  NLX_coes[0][ 13][ 16] =-5.067185192233442e+00;
  NLX_coes[0][ 13][ 17] =-5.109510075193329e+00;
  NLX_coes[0][ 13][ 18] =-5.109391155866046e+00;
  NLX_coes[0][ 13][ 19] =-5.046662301620228e+00;
  NLX_coes[0][ 13][ 20] =-4.879713197397805e+00;
  NLX_coes[0][ 13][ 21] =-4.696290656776880e+00;
  NLX_coes[0][ 13][ 22] =-4.921723250858646e+00;
  NLX_coes[0][ 13][ 23] =-5.492150247271743e+00;
  NLX_coes[0][ 13][ 24] =-6.295003865878650e+00;
  NLX_coes[0][ 13][ 25] =-7.823070765848646e+00;
  NLX_coes[0][ 13][ 26] =-9.369113640335025e+00;
  NLX_coes[0][ 13][ 27] =-1.000138269287614e+01;
  NLX_coes[0][ 13][ 28] =-1.039465811042366e+01;
  NLX_coes[0][ 13][ 29] =-1.071896174126694e+01;
  NLX_coes[0][ 13][ 30] =-1.103769431103314e+01;
  NLX_coes[0][ 13][ 31] =-1.137204578793639e+01;
  NLX_coes[0][ 13][ 32] =-1.172401781761359e+01;
  NLX_coes[0][ 13][ 33] =-1.208896349175479e+01;
  NLX_coes[0][ 13][ 34] =-1.246141983148734e+01;
  NLX_coes[0][ 13][ 35] =-1.283700665829811e+01;
  NLX_coes[0][ 13][ 36] =-1.321267260018924e+01;
  NLX_coes[0][ 13][ 37] =-1.358649176346403e+01;
  NLX_coes[0][ 13][ 38] =-1.395746142123609e+01;
  NLX_coes[0][ 13][ 39] =-1.432539855791923e+01;
  NLX_coes[0][ 13][ 40] =-1.469091280900309e+01;
  NLX_coes[0][ 14][  0] =-5.790300728405012e+00;
  NLX_coes[0][ 14][  1] =-5.747144119600711e+00;
  NLX_coes[0][ 14][  2] =-5.703409804064990e+00;
  NLX_coes[0][ 14][  3] =-5.661219350867505e+00;
  NLX_coes[0][ 14][  4] =-5.622710332984674e+00;
  NLX_coes[0][ 14][  5] =-5.590043211497586e+00;
  NLX_coes[0][ 14][  6] =-5.565379800503202e+00;
  NLX_coes[0][ 14][  7] =-5.550812963594068e+00;
  NLX_coes[0][ 14][  8] =-5.548230517155404e+00;
  NLX_coes[0][ 14][  9] =-5.559104810960579e+00;
  NLX_coes[0][ 14][ 10] =-5.584214237672399e+00;
  NLX_coes[0][ 14][ 11] =-5.623317882955321e+00;
  NLX_coes[0][ 14][ 12] =-5.674817791492231e+00;
  NLX_coes[0][ 14][ 13] =-5.735482487132355e+00;
  NLX_coes[0][ 14][ 14] =-5.800401152373698e+00;
  NLX_coes[0][ 14][ 15] =-5.863542772626974e+00;
  NLX_coes[0][ 14][ 16] =-5.919579585672779e+00;
  NLX_coes[0][ 14][ 17] =-5.967666059861587e+00;
  NLX_coes[0][ 14][ 18] =-6.015822623167505e+00;
  NLX_coes[0][ 14][ 19] =-6.073304549860596e+00;
  NLX_coes[0][ 14][ 20] =-6.083295801084322e+00;
  NLX_coes[0][ 14][ 21] =-5.794822369153653e+00;
  NLX_coes[0][ 14][ 22] =-6.169644804322099e+00;
  NLX_coes[0][ 14][ 23] =-7.122874493265253e+00;
  NLX_coes[0][ 14][ 24] =-8.548853533127414e+00;
  NLX_coes[0][ 14][ 25] =-1.043658801534681e+01;
  NLX_coes[0][ 14][ 26] =-1.175349327476991e+01;
  NLX_coes[0][ 14][ 27] =-1.254598133077735e+01;
  NLX_coes[0][ 14][ 28] =-1.300938558079618e+01;
  NLX_coes[0][ 14][ 29] =-1.333450693299791e+01;
  NLX_coes[0][ 14][ 30] =-1.361686586437305e+01;
  NLX_coes[0][ 14][ 31] =-1.389781600202145e+01;
  NLX_coes[0][ 14][ 32] =-1.419229709026633e+01;
  NLX_coes[0][ 14][ 33] =-1.450323991008036e+01;
  NLX_coes[0][ 14][ 34] =-1.482882966395263e+01;
  NLX_coes[0][ 14][ 35] =-1.516591058614189e+01;
  NLX_coes[0][ 14][ 36] =-1.551138163408402e+01;
  NLX_coes[0][ 14][ 37] =-1.586265965113181e+01;
  NLX_coes[0][ 14][ 38] =-1.621776700897301e+01;
  NLX_coes[0][ 14][ 39] =-1.657528154000983e+01;
  NLX_coes[0][ 14][ 40] =-1.693421910404802e+01;
  NLX_coes[0][ 15][  0] =-6.647028897408996e+00;
  NLX_coes[0][ 15][  1] =-6.622103231752217e+00;
  NLX_coes[0][ 15][  2] =-6.596615135648793e+00;
  NLX_coes[0][ 15][  3] =-6.572587875040070e+00;
  NLX_coes[0][ 15][  4] =-6.552003323386466e+00;
  NLX_coes[0][ 15][  5] =-6.536787460622878e+00;
  NLX_coes[0][ 15][  6] =-6.528766986130202e+00;
  NLX_coes[0][ 15][  7] =-6.529583950251355e+00;
  NLX_coes[0][ 15][  8] =-6.540559104151407e+00;
  NLX_coes[0][ 15][  9] =-6.562502724019875e+00;
  NLX_coes[0][ 15][ 10] =-6.595484529843373e+00;
  NLX_coes[0][ 15][ 11] =-6.638592078983321e+00;
  NLX_coes[0][ 15][ 12] =-6.689737184078732e+00;
  NLX_coes[0][ 15][ 13] =-6.745630180353159e+00;
  NLX_coes[0][ 15][ 14] =-6.802156039502067e+00;
  NLX_coes[0][ 15][ 15] =-6.855565568909039e+00;
  NLX_coes[0][ 15][ 16] =-6.905042567958829e+00;
  NLX_coes[0][ 15][ 17] =-6.956942324757038e+00;
  NLX_coes[0][ 15][ 18] =-7.029576755302965e+00;
  NLX_coes[0][ 15][ 19] =-7.154780413744102e+00;
  NLX_coes[0][ 15][ 20] =-7.359510219431935e+00;
  NLX_coes[0][ 15][ 21] =-7.277016127399527e+00;
  NLX_coes[0][ 15][ 22] =-7.878799250164420e+00;
  NLX_coes[0][ 15][ 23] =-9.227870592245839e+00;
  NLX_coes[0][ 15][ 24] =-1.123241873897144e+01;
  NLX_coes[0][ 15][ 25] =-1.312647942541119e+01;
  NLX_coes[0][ 15][ 26] =-1.428437375052525e+01;
  NLX_coes[0][ 15][ 27] =-1.506564954378138e+01;
  NLX_coes[0][ 15][ 28] =-1.556524581635907e+01;
  NLX_coes[0][ 15][ 29] =-1.590815378162112e+01;
  NLX_coes[0][ 15][ 30] =-1.618151475025919e+01;
  NLX_coes[0][ 15][ 31] =-1.643506094353614e+01;
  NLX_coes[0][ 15][ 32] =-1.669320401817780e+01;
  NLX_coes[0][ 15][ 33] =-1.696588868853313e+01;
  NLX_coes[0][ 15][ 34] =-1.725567234652467e+01;
  NLX_coes[0][ 15][ 35] =-1.756173721768407e+01;
  NLX_coes[0][ 15][ 36] =-1.788196088800268e+01;
  NLX_coes[0][ 15][ 37] =-1.821388777933800e+01;
  NLX_coes[0][ 15][ 38] =-1.855511504410973e+01;
  NLX_coes[0][ 15][ 39] =-1.890336350662046e+01;
  NLX_coes[0][ 15][ 40] =-1.925634785446126e+01;
  NLX_coes[0][ 16][  0] =-7.562841767723708e+00;
  NLX_coes[0][ 16][  1] =-7.558768845686020e+00;
  NLX_coes[0][ 16][  2] =-7.554181747861639e+00;
  NLX_coes[0][ 16][  3] =-7.551016030420392e+00;
  NLX_coes[0][ 16][  4] =-7.551118277492102e+00;
  NLX_coes[0][ 16][  5] =-7.556221450751897e+00;
  NLX_coes[0][ 16][  6] =-7.567892905410172e+00;
  NLX_coes[0][ 16][  7] =-7.587448031712385e+00;
  NLX_coes[0][ 16][  8] =-7.615826025370820e+00;
  NLX_coes[0][ 16][  9] =-7.653431391964260e+00;
  NLX_coes[0][ 16][ 10] =-7.699956000926599e+00;
  NLX_coes[0][ 16][ 11] =-7.754213431747059e+00;
  NLX_coes[0][ 16][ 12] =-7.814045006243168e+00;
  NLX_coes[0][ 16][ 13] =-7.876403472835482e+00;
  NLX_coes[0][ 16][ 14] =-7.937797163567391e+00;
  NLX_coes[0][ 16][ 15] =-7.995393746074440e+00;
  NLX_coes[0][ 16][ 16] =-8.049223670328981e+00;
  NLX_coes[0][ 16][ 17] =-8.105939161625120e+00;
  NLX_coes[0][ 16][ 18] =-8.183769072748767e+00;
  NLX_coes[0][ 16][ 19] =-8.315243043341946e+00;
  NLX_coes[0][ 16][ 20] =-8.548787340450739e+00;
  NLX_coes[0][ 16][ 21] =-9.031671918694030e+00;
  NLX_coes[0][ 16][ 22] =-1.007689390877418e+01;
  NLX_coes[0][ 16][ 23] =-1.198528664330513e+01;
  NLX_coes[0][ 16][ 24] =-1.416812335978660e+01;
  NLX_coes[0][ 16][ 25] =-1.568258550637504e+01;
  NLX_coes[0][ 16][ 26] =-1.677950493102665e+01;
  NLX_coes[0][ 16][ 27] =-1.753521229729139e+01;
  NLX_coes[0][ 16][ 28] =-1.804886286754809e+01;
  NLX_coes[0][ 16][ 29] =-1.841121348521460e+01;
  NLX_coes[0][ 16][ 30] =-1.869312219292609e+01;
  NLX_coes[0][ 16][ 31] =-1.894180744747585e+01;
  NLX_coes[0][ 16][ 32] =-1.918516092302705e+01;
  NLX_coes[0][ 16][ 33] =-1.943778857274132e+01;
  NLX_coes[0][ 16][ 34] =-1.970614372307888e+01;
  NLX_coes[0][ 16][ 35] =-1.999209321597035e+01;
  NLX_coes[0][ 16][ 36] =-2.029510854123165e+01;
  NLX_coes[0][ 16][ 37] =-2.061348523663423e+01;
  NLX_coes[0][ 16][ 38] =-2.094493614410798e+01;
  NLX_coes[0][ 16][ 39] =-2.128678239899594e+01;
  NLX_coes[0][ 16][ 40] =-2.163585528977003e+01;
  NLX_coes[0][ 17][  0] =-8.537739339349264e+00;
  NLX_coes[0][ 17][  1] =-8.557041837888642e+00;
  NLX_coes[0][ 17][  2] =-8.575942444057358e+00;
  NLX_coes[0][ 17][  3] =-8.596317684227198e+00;
  NLX_coes[0][ 17][  4] =-8.619924601743920e+00;
  NLX_coes[0][ 17][  5] =-8.648375497262561e+00;
  NLX_coes[0][ 17][  6] =-8.683088971838218e+00;
  NLX_coes[0][ 17][  7] =-8.725215148377234e+00;
  NLX_coes[0][ 17][  8] =-8.775536081012767e+00;
  NLX_coes[0][ 17][  9] =-8.834348350431224e+00;
  NLX_coes[0][ 17][ 10] =-8.901343928942158e+00;
  NLX_coes[0][ 17][ 11] =-8.975518262544487e+00;
  NLX_coes[0][ 17][ 12] =-9.055152046623666e+00;
  NLX_coes[0][ 17][ 13] =-9.137934389709157e+00;
  NLX_coes[0][ 17][ 14] =-9.221314174152148e+00;
  NLX_coes[0][ 17][ 15] =-9.303180509033959e+00;
  NLX_coes[0][ 17][ 16] =-9.383050841727588e+00;
  NLX_coes[0][ 17][ 17] =-9.464439585312943e+00;
  NLX_coes[0][ 17][ 18] =-9.561082385784385e+00;
  NLX_coes[0][ 17][ 19] =-9.716361642055668e+00;
  NLX_coes[0][ 17][ 20] =-1.006427916545451e+01;
  NLX_coes[0][ 17][ 21] =-1.091677882269999e+01;
  NLX_coes[0][ 17][ 22] =-1.257107247003239e+01;
  NLX_coes[0][ 17][ 23] =-1.500972597789286e+01;
  NLX_coes[0][ 17][ 24] =-1.684971578979076e+01;
  NLX_coes[0][ 17][ 25] =-1.817900935327739e+01;
  NLX_coes[0][ 17][ 26] =-1.918033899819353e+01;
  NLX_coes[0][ 17][ 27] =-1.990980560867693e+01;
  NLX_coes[0][ 17][ 28] =-2.043681290937992e+01;
  NLX_coes[0][ 17][ 29] =-2.082509694845536e+01;
  NLX_coes[0][ 17][ 30] =-2.112977105089996e+01;
  NLX_coes[0][ 17][ 31] =-2.139216420074551e+01;
  NLX_coes[0][ 17][ 32] =-2.164003399644904e+01;
  NLX_coes[0][ 17][ 33] =-2.189031968579060e+01;
  NLX_coes[0][ 17][ 34] =-2.215232417532252e+01;
  NLX_coes[0][ 17][ 35] =-2.243038158788681e+01;
  NLX_coes[0][ 17][ 36] =-2.272576483929829e+01;
  NLX_coes[0][ 17][ 37] =-2.303790017871437e+01;
  NLX_coes[0][ 17][ 38] =-2.336504230576430e+01;
  NLX_coes[0][ 17][ 39] =-2.370454549930865e+01;
  NLX_coes[0][ 17][ 40] =-2.405281257301509e+01;
  NLX_coes[0][ 18][  0] =-9.571721612285778e+00;
  NLX_coes[0][ 18][  1] =-9.616777819194191e+00;
  NLX_coes[0][ 18][  2] =-9.661632475188627e+00;
  NLX_coes[0][ 18][  3] =-9.708142995682064e+00;
  NLX_coes[0][ 18][  4] =-9.758037179345781e+00;
  NLX_coes[0][ 18][  5] =-9.812894657615306e+00;
  NLX_coes[0][ 18][  6] =-9.874110179104113e+00;
  NLX_coes[0][ 18][  7] =-9.942840532399266e+00;
  NLX_coes[0][ 18][  8] =-1.001993982852230e+01;
  NLX_coes[0][ 18][  9] =-1.010589305753775e+01;
  NLX_coes[0][ 18][ 10] =-1.020076521645744e+01;
  NLX_coes[0][ 18][ 11] =-1.030419265683099e+01;
  NLX_coes[0][ 18][ 12] =-1.041545389560568e+01;
  NLX_coes[0][ 18][ 13] =-1.053366720046976e+01;
  NLX_coes[0][ 18][ 14] =-1.065816598874624e+01;
  NLX_coes[0][ 18][ 15] =-1.078907388742773e+01;
  NLX_coes[0][ 18][ 16] =-1.092797530639031e+01;
  NLX_coes[0][ 18][ 17] =-1.107859505686117e+01;
  NLX_coes[0][ 18][ 18] =-1.125015858888984e+01;
  NLX_coes[0][ 18][ 19] =-1.147876409146498e+01;
  NLX_coes[0][ 18][ 20] =-1.188721176869535e+01;
  NLX_coes[0][ 18][ 21] =-1.290047451273462e+01;
  NLX_coes[0][ 18][ 22] =-1.544670061012464e+01;
  NLX_coes[0][ 18][ 23] =-1.772094672156515e+01;
  NLX_coes[0][ 18][ 24] =-1.925325398573852e+01;
  NLX_coes[0][ 18][ 25] =-2.047918280856887e+01;
  NLX_coes[0][ 18][ 26] =-2.143311062378441e+01;
  NLX_coes[0][ 18][ 27] =-2.216204934018497e+01;
  NLX_coes[0][ 18][ 28] =-2.271486488325531e+01;
  NLX_coes[0][ 18][ 29] =-2.313984897169469e+01;
  NLX_coes[0][ 18][ 30] =-2.348048221807797e+01;
  NLX_coes[0][ 18][ 31] =-2.377209620641747e+01;
  NLX_coes[0][ 18][ 32] =-2.404081739810376e+01;
  NLX_coes[0][ 18][ 33] =-2.430443816678795e+01;
  NLX_coes[0][ 18][ 34] =-2.457413663964959e+01;
  NLX_coes[0][ 18][ 35] =-2.485626414314901e+01;
  NLX_coes[0][ 18][ 36] =-2.515382290037468e+01;
  NLX_coes[0][ 18][ 37] =-2.546752367917226e+01;
  NLX_coes[0][ 18][ 38] =-2.579643991136647e+01;
  NLX_coes[0][ 18][ 39] =-2.613831098420069e+01;
  NLX_coes[0][ 18][ 40] =-2.648953697878352e+01;
  NLX_coes[0][ 19][  0] =-1.066478858653333e+01;
  NLX_coes[0][ 19][  1] =-1.073776988174405e+01;
  NLX_coes[0][ 19][  2] =-1.081085049405389e+01;
  NLX_coes[0][ 19][  3] =-1.088591226063662e+01;
  NLX_coes[0][ 19][  4] =-1.096471720747530e+01;
  NLX_coes[0][ 19][  5] =-1.104890047911270e+01;
  NLX_coes[0][ 19][  6] =-1.113995195571626e+01;
  NLX_coes[0][ 19][  7] =-1.123919137615466e+01;
  NLX_coes[0][ 19][  8] =-1.134774479847747e+01;
  NLX_coes[0][ 19][  9] =-1.146653519672703e+01;
  NLX_coes[0][ 19][ 10] =-1.159630689302887e+01;
  NLX_coes[0][ 19][ 11] =-1.173771226409903e+01;
  NLX_coes[0][ 19][ 12] =-1.189149928753590e+01;
  NLX_coes[0][ 19][ 13] =-1.205884919772009e+01;
  NLX_coes[0][ 19][ 14] =-1.224193015828544e+01;
  NLX_coes[0][ 19][ 15] =-1.244480282977692e+01;
  NLX_coes[0][ 19][ 16] =-1.267506666219486e+01;
  NLX_coes[0][ 19][ 17] =-1.294671793174693e+01;
  NLX_coes[0][ 19][ 18] =-1.328042840772724e+01;
  NLX_coes[0][ 19][ 19] =-1.367903648265766e+01;
  NLX_coes[0][ 19][ 20] =-1.407945539893263e+01;
  NLX_coes[0][ 19][ 21] =-1.531274553750861e+01;
  NLX_coes[0][ 19][ 22] =-1.809290033316757e+01;
  NLX_coes[0][ 19][ 23] =-1.984905569440027e+01;
  NLX_coes[0][ 19][ 24] =-2.135085386502713e+01;
  NLX_coes[0][ 19][ 25] =-2.256347934559265e+01;
  NLX_coes[0][ 19][ 26] =-2.352612494474753e+01;
  NLX_coes[0][ 19][ 27] =-2.428478335446021e+01;
  NLX_coes[0][ 19][ 28] =-2.488119393012206e+01;
  NLX_coes[0][ 19][ 29] =-2.535532482143039e+01;
  NLX_coes[0][ 19][ 30] =-2.574356669697013e+01;
  NLX_coes[0][ 19][ 31] =-2.607680618789210e+01;
  NLX_coes[0][ 19][ 32] =-2.637937895355208e+01;
  NLX_coes[0][ 19][ 33] =-2.666918847957416e+01;
  NLX_coes[0][ 19][ 34] =-2.695856801896904e+01;
  NLX_coes[0][ 19][ 35] =-2.725540649301626e+01;
  NLX_coes[0][ 19][ 36] =-2.756421576555486e+01;
  NLX_coes[0][ 19][ 37] =-2.788697877232653e+01;
  NLX_coes[0][ 19][ 38] =-2.822372451534233e+01;
  NLX_coes[0][ 19][ 39] =-2.857282702589432e+01;
  NLX_coes[0][ 19][ 40] =-2.893103855688134e+01;
  NLX_coes[0][ 20][  0] =-1.181694026209192e+01;
  NLX_coes[0][ 20][  1] =-1.191973863990268e+01;
  NLX_coes[0][ 20][  2] =-1.202303714225329e+01;
  NLX_coes[0][ 20][  3] =-1.212878206756914e+01;
  NLX_coes[0][ 20][  4] =-1.223882617895110e+01;
  NLX_coes[0][ 20][  5] =-1.235493522740192e+01;
  NLX_coes[0][ 20][  6] =-1.247879040601221e+01;
  NLX_coes[0][ 20][  7] =-1.261199358576138e+01;
  NLX_coes[0][ 20][  8] =-1.275608545424155e+01;
  NLX_coes[0][ 20][  9] =-1.291259172326084e+01;
  NLX_coes[0][ 20][ 10] =-1.308311960605036e+01;
  NLX_coes[0][ 20][ 11] =-1.326953631868735e+01;
  NLX_coes[0][ 20][ 12] =-1.347427504464331e+01;
  NLX_coes[0][ 20][ 13] =-1.370083611680217e+01;
  NLX_coes[0][ 20][ 14] =-1.395458825907691e+01;
  NLX_coes[0][ 20][ 15] =-1.424401075013465e+01;
  NLX_coes[0][ 20][ 16] =-1.458250310704232e+01;
  NLX_coes[0][ 20][ 17] =-1.499142027104194e+01;
  NLX_coes[0][ 20][ 18] =-1.551117307829538e+01;
  NLX_coes[0][ 20][ 19] =-1.625198749985537e+01;
  NLX_coes[0][ 20][ 20] =-1.750668977522379e+01;
  NLX_coes[0][ 20][ 21] =-1.741254084717542e+01;
  NLX_coes[0][ 20][ 22] =-1.979616001641605e+01;
  NLX_coes[0][ 20][ 23] =-2.171254380140613e+01;
  NLX_coes[0][ 20][ 24] =-2.324491249647187e+01;
  NLX_coes[0][ 20][ 25] =-2.448333488320992e+01;
  NLX_coes[0][ 20][ 26] =-2.548649896759346e+01;
  NLX_coes[0][ 20][ 27] =-2.629558194300182e+01;
  NLX_coes[0][ 20][ 28] =-2.694857722630793e+01;
  NLX_coes[0][ 20][ 29] =-2.748103902902244e+01;
  NLX_coes[0][ 20][ 30] =-2.792515281282480e+01;
  NLX_coes[0][ 20][ 31] =-2.830871665673473e+01;
  NLX_coes[0][ 20][ 32] =-2.865448708409241e+01;
  NLX_coes[0][ 20][ 33] =-2.898011096933441e+01;
  NLX_coes[0][ 20][ 34] =-2.929855461432264e+01;
  NLX_coes[0][ 20][ 35] =-2.961879715095006e+01;
  NLX_coes[0][ 20][ 36] =-2.994657180672399e+01;
  NLX_coes[0][ 20][ 37] =-3.028501149877936e+01;
  NLX_coes[0][ 20][ 38] =-3.063512375677404e+01;
  NLX_coes[0][ 20][ 39] =-3.099606454930356e+01;
  NLX_coes[0][ 20][ 40] =-3.136520204873385e+01;
  NLX_coes[0][ 21][  0] =-1.302817663896151e+01;
  NLX_coes[0][ 21][  1] =-1.316233102724767e+01;
  NLX_coes[0][ 21][  2] =-1.329747371680371e+01;
  NLX_coes[0][ 21][  3] =-1.343564641000890e+01;
  NLX_coes[0][ 21][  4] =-1.357883459966124e+01;
  NLX_coes[0][ 21][  5] =-1.372898731635492e+01;
  NLX_coes[0][ 21][  6] =-1.388803935019670e+01;
  NLX_coes[0][ 21][  7] =-1.405794368634529e+01;
  NLX_coes[0][ 21][  8] =-1.424072527030823e+01;
  NLX_coes[0][ 21][  9] =-1.443857207245351e+01;
  NLX_coes[0][ 21][ 10] =-1.465398608153636e+01;
  NLX_coes[0][ 21][ 11] =-1.489002596495659e+01;
  NLX_coes[0][ 21][ 12] =-1.515068578690093e+01;
  NLX_coes[0][ 21][ 13] =-1.544147204047486e+01;
  NLX_coes[0][ 21][ 14] =-1.577027061892570e+01;
  NLX_coes[0][ 21][ 15] =-1.614867476608094e+01;
  NLX_coes[0][ 21][ 16] =-1.659419184959714e+01;
  NLX_coes[0][ 21][ 17] =-1.713394323668521e+01;
  NLX_coes[0][ 21][ 18] =-1.780703265623535e+01;
  NLX_coes[0][ 21][ 19] =-1.864298983194630e+01;
  NLX_coes[0][ 21][ 20] =-1.956656936248838e+01;
  NLX_coes[0][ 21][ 21] =-2.038463311689682e+01;
  NLX_coes[0][ 21][ 22] =-2.187290508810974e+01;
  NLX_coes[0][ 21][ 23] =-2.356197150427785e+01;
  NLX_coes[0][ 21][ 24] =-2.506063096003153e+01;
  NLX_coes[0][ 21][ 25] =-2.632153185573772e+01;
  NLX_coes[0][ 21][ 26] =-2.736627624852140e+01;
  NLX_coes[0][ 21][ 27] =-2.822816885234747e+01;
  NLX_coes[0][ 21][ 28] =-2.894028094064257e+01;
  NLX_coes[0][ 21][ 29] =-2.953375968602804e+01;
  NLX_coes[0][ 21][ 30] =-3.003711687426077e+01;
  NLX_coes[0][ 21][ 31] =-3.047545822407560e+01;
  NLX_coes[0][ 21][ 32] =-3.086995089071306e+01;
  NLX_coes[0][ 21][ 33] =-3.123766297314247e+01;
  NLX_coes[0][ 21][ 34] =-3.159174771489821e+01;
  NLX_coes[0][ 21][ 35] =-3.194185146742065e+01;
  NLX_coes[0][ 21][ 36] =-3.229460797067370e+01;
  NLX_coes[0][ 21][ 37] =-3.265410853341444e+01;
  NLX_coes[0][ 21][ 38] =-3.302227630095888e+01;
  NLX_coes[0][ 21][ 39] =-3.339910550965551e+01;
  NLX_coes[0][ 21][ 40] =-3.378274843139030e+01;
  NLX_coes[0][ 22][  0] =-1.429849771714205e+01;
  NLX_coes[0][ 22][  1] =-1.446512741616464e+01;
  NLX_coes[0][ 22][  2] =-1.463329929684205e+01;
  NLX_coes[0][ 22][  3] =-1.480516879469944e+01;
  NLX_coes[0][ 22][  4] =-1.498287737399781e+01;
  NLX_coes[0][ 22][  5] =-1.516858314319241e+01;
  NLX_coes[0][ 22][  6] =-1.536449881154441e+01;
  NLX_coes[0][ 22][  7] =-1.557294438598912e+01;
  NLX_coes[0][ 22][  8] =-1.579642508430375e+01;
  NLX_coes[0][ 22][  9] =-1.603774887932815e+01;
  NLX_coes[0][ 22][ 10] =-1.630020290926126e+01;
  NLX_coes[0][ 22][ 11] =-1.658781359392800e+01;
  NLX_coes[0][ 22][ 12] =-1.690572136734000e+01;
  NLX_coes[0][ 22][ 13] =-1.726070660097377e+01;
  NLX_coes[0][ 22][ 14] =-1.766190297040949e+01;
  NLX_coes[0][ 22][ 15] =-1.812168830128799e+01;
  NLX_coes[0][ 22][ 16] =-1.865646547066075e+01;
  NLX_coes[0][ 22][ 17] =-1.928605145213715e+01;
  NLX_coes[0][ 22][ 18] =-2.002831411617600e+01;
  NLX_coes[0][ 22][ 19] =-2.088337338233069e+01;
  NLX_coes[0][ 22][ 20] =-2.179284881624038e+01;
  NLX_coes[0][ 22][ 21] =-2.284795611720073e+01;
  NLX_coes[0][ 22][ 22] =-2.410232835786852e+01;
  NLX_coes[0][ 22][ 23] =-2.553901165360472e+01;
  NLX_coes[0][ 22][ 24] =-2.692168500321395e+01;
  NLX_coes[0][ 22][ 25] =-2.815455279665673e+01;
  NLX_coes[0][ 22][ 26] =-2.921665523285528e+01;
  NLX_coes[0][ 22][ 27] =-3.011886115851904e+01;
  NLX_coes[0][ 22][ 28] =-3.088294251531284e+01;
  NLX_coes[0][ 22][ 29] =-3.153342347996958e+01;
  NLX_coes[0][ 22][ 30] =-3.209434118981297e+01;
  NLX_coes[0][ 22][ 31] =-3.258772957514645e+01;
  NLX_coes[0][ 22][ 32] =-3.303282383471814e+01;
  NLX_coes[0][ 22][ 33] =-3.344570691286783e+01;
  NLX_coes[0][ 22][ 34] =-3.383927863410059e+01;
  NLX_coes[0][ 22][ 35] =-3.422344065401619e+01;
  NLX_coes[0][ 22][ 36] =-3.460539277999314e+01;
  NLX_coes[0][ 22][ 37] =-3.498995276196384e+01;
  NLX_coes[0][ 22][ 38] =-3.537983618435462e+01;
  NLX_coes[0][ 22][ 39] =-3.577585730800774e+01;
  NLX_coes[0][ 22][ 40] =-3.617703161666476e+01;
  NLX_coes[0][ 23][  0] =-1.562790349663341e+01;
  NLX_coes[0][ 23][  1] =-1.582765526262386e+01;
  NLX_coes[0][ 23][  2] =-1.602954186775170e+01;
  NLX_coes[0][ 23][  3] =-1.623583691021777e+01;
  NLX_coes[0][ 23][  4] =-1.644884146967846e+01;
  NLX_coes[0][ 23][  5] =-1.667092200836000e+01;
  NLX_coes[0][ 23][  6] =-1.690455818943746e+01;
  NLX_coes[0][ 23][  7] =-1.715240656901057e+01;
  NLX_coes[0][ 23][  8] =-1.741738846217855e+01;
  NLX_coes[0][ 23][  9] =-1.770281262960228e+01;
  NLX_coes[0][ 23][ 10] =-1.801254529167161e+01;
  NLX_coes[0][ 23][ 11] =-1.835124022147099e+01;
  NLX_coes[0][ 23][ 12] =-1.872463753380986e+01;
  NLX_coes[0][ 23][ 13] =-1.913992388298497e+01;
  NLX_coes[0][ 23][ 14] =-1.960609919334814e+01;
  NLX_coes[0][ 23][ 15] =-2.013416972199394e+01;
  NLX_coes[0][ 23][ 16] =-2.073670984845201e+01;
  NLX_coes[0][ 23][ 17] =-2.142587404412382e+01;
  NLX_coes[0][ 23][ 18] =-2.220864914109335e+01;
  NLX_coes[0][ 23][ 19] =-2.308040718203345e+01;
  NLX_coes[0][ 23][ 20] =-2.403191832074251e+01;
  NLX_coes[0][ 23][ 21] =-2.510543839535048e+01;
  NLX_coes[0][ 23][ 22] =-2.628908554924769e+01;
  NLX_coes[0][ 23][ 23] =-2.756694569734481e+01;
  NLX_coes[0][ 23][ 24] =-2.883661578398776e+01;
  NLX_coes[0][ 23][ 25] =-3.001684782697309e+01;
  NLX_coes[0][ 23][ 26] =-3.107211476625275e+01;
  NLX_coes[0][ 23][ 27] =-3.199666093430467e+01;
  NLX_coes[0][ 23][ 28] =-3.280015897176140e+01;
  NLX_coes[0][ 23][ 29] =-3.349905708998755e+01;
  NLX_coes[0][ 23][ 30] =-3.411196833965962e+01;
  NLX_coes[0][ 23][ 31] =-3.465725735814556e+01;
  NLX_coes[0][ 23][ 32] =-3.515176071830160e+01;
  NLX_coes[0][ 23][ 33] =-3.561013786956069e+01;
  NLX_coes[0][ 23][ 34] =-3.604461001993545e+01;
  NLX_coes[0][ 23][ 35] =-3.646494401294662e+01;
  NLX_coes[0][ 23][ 36] =-3.687857771858795e+01;
  NLX_coes[0][ 23][ 37] =-3.729080775857769e+01;
  NLX_coes[0][ 23][ 38] =-3.770498218005746e+01;
  NLX_coes[0][ 23][ 39] =-3.812266111484116e+01;
  NLX_coes[0][ 23][ 40] =-3.854372677266298e+01;
  NLX_coes[0][ 24][  0] =-1.701639397743549e+01;
  NLX_coes[0][ 24][  1] =-1.724940672386385e+01;
  NLX_coes[0][ 24][  2] =-1.748515751027349e+01;
  NLX_coes[0][ 24][  3] =-1.772602931693608e+01;
  NLX_coes[0][ 24][  4] =-1.797446888874089e+01;
  NLX_coes[0][ 24][  5] =-1.823302798984307e+01;
  NLX_coes[0][ 24][  6] =-1.850441471661830e+01;
  NLX_coes[0][ 24][  7] =-1.879155865111938e+01;
  NLX_coes[0][ 24][  8] =-1.909769504013471e+01;
  NLX_coes[0][ 24][  9] =-1.942647374270240e+01;
  NLX_coes[0][ 24][ 10] =-1.978209747958343e+01;
  NLX_coes[0][ 24][ 11] =-2.016948909619979e+01;
  NLX_coes[0][ 24][ 12] =-2.059447539804367e+01;
  NLX_coes[0][ 24][ 13] =-2.106394847022201e+01;
  NLX_coes[0][ 24][ 14] =-2.158591248674662e+01;
  NLX_coes[0][ 24][ 15] =-2.216923320285850e+01;
  NLX_coes[0][ 24][ 16] =-2.282279477151586e+01;
  NLX_coes[0][ 24][ 17] =-2.355376670915479e+01;
  NLX_coes[0][ 24][ 18] =-2.436521576612215e+01;
  NLX_coes[0][ 24][ 19] =-2.525523111520734e+01;
  NLX_coes[0][ 24][ 20] =-2.622579415819320e+01;
  NLX_coes[0][ 24][ 21] =-2.728349511783187e+01;
  NLX_coes[0][ 24][ 22] =-2.841796091875472e+01;
  NLX_coes[0][ 24][ 23] =-2.960205782120025e+01;
  NLX_coes[0][ 24][ 24] =-3.078450831672144e+01;
  NLX_coes[0][ 24][ 25] =-3.191057857909292e+01;
  NLX_coes[0][ 24][ 26] =-3.294641156932052e+01;
  NLX_coes[0][ 24][ 27] =-3.387885652312011e+01;
  NLX_coes[0][ 24][ 28] =-3.470900938072424e+01;
  NLX_coes[0][ 24][ 29] =-3.544612431453049e+01;
  NLX_coes[0][ 24][ 30] =-3.610337256933585e+01;
  NLX_coes[0][ 24][ 31] =-3.669518781339916e+01;
  NLX_coes[0][ 24][ 32] =-3.723568154359467e+01;
  NLX_coes[0][ 24][ 33] =-3.773773841071475e+01;
  NLX_coes[0][ 24][ 34] =-3.821254492702442e+01;
  NLX_coes[0][ 24][ 35] =-3.866939540556270e+01;
  NLX_coes[0][ 24][ 36] =-3.911566887414620e+01;
  NLX_coes[0][ 24][ 37] =-3.955690095423761e+01;
  NLX_coes[0][ 24][ 38] =-3.999689707434735e+01;
  NLX_coes[0][ 24][ 39] =-4.043785241043123e+01;
  NLX_coes[0][ 24][ 40] =-4.088046109987527e+01;
  NLX_coes[0][ 25][  0] =-1.846396915954817e+01;
  NLX_coes[0][ 25][  1] =-1.872985752239198e+01;
  NLX_coes[0][ 25][  2] =-1.899907154252124e+01;
  NLX_coes[0][ 25][  3] =-1.927408371675723e+01;
  NLX_coes[0][ 25][  4] =-1.955745883146791e+01;
  NLX_coes[0][ 25][  5] =-1.985189509583015e+01;
  NLX_coes[0][ 25][  6] =-2.016027329484951e+01;
  NLX_coes[0][ 25][  7] =-2.048571535125388e+01;
  NLX_coes[0][ 25][  8] =-2.083165427046411e+01;
  NLX_coes[0][ 25][  9] =-2.120191660011999e+01;
  NLX_coes[0][ 25][ 10] =-2.160081526191581e+01;
  NLX_coes[0][ 25][ 11] =-2.203324328706740e+01;
  NLX_coes[0][ 25][ 12] =-2.250474514589684e+01;
  NLX_coes[0][ 25][ 13] =-2.302151905168303e+01;
  NLX_coes[0][ 25][ 14] =-2.359026998459932e+01;
  NLX_coes[0][ 25][ 15] =-2.421779901698405e+01;
  NLX_coes[0][ 25][ 16] =-2.491021995639167e+01;
  NLX_coes[0][ 25][ 17] =-2.567184068101274e+01;
  NLX_coes[0][ 25][ 18] =-2.650420931787901e+01;
  NLX_coes[0][ 25][ 19] =-2.740663586904713e+01;
  NLX_coes[0][ 25][ 20] =-2.837883933290915e+01;
  NLX_coes[0][ 25][ 21] =-2.941673861921941e+01;
  NLX_coes[0][ 25][ 22] =-3.050879420725368e+01;
  NLX_coes[0][ 25][ 23] =-3.163077892744553e+01;
  NLX_coes[0][ 25][ 24] =-3.274955283743634e+01;
  NLX_coes[0][ 25][ 25] =-3.382897260667779e+01;
  NLX_coes[0][ 25][ 26] =-3.484172039563507e+01;
  NLX_coes[0][ 25][ 27] =-3.577321308591542e+01;
  NLX_coes[0][ 25][ 28] =-3.661984826769633e+01;
  NLX_coes[0][ 25][ 29] =-3.738564671865405e+01;
  NLX_coes[0][ 25][ 30] =-3.807915287047302e+01;
  NLX_coes[0][ 25][ 31] =-3.871110604558704e+01;
  NLX_coes[0][ 25][ 32] =-3.929285195922910e+01;
  NLX_coes[0][ 25][ 33] =-3.983532572057289e+01;
  NLX_coes[0][ 25][ 34] =-4.034844284559791e+01;
  NLX_coes[0][ 25][ 35] =-4.084077089623388e+01;
  NLX_coes[0][ 25][ 36] =-4.131938739776039e+01;
  NLX_coes[0][ 25][ 37] =-4.178985506672766e+01;
  NLX_coes[0][ 25][ 38] =-4.225626549042723e+01;
  NLX_coes[0][ 25][ 39] =-4.272131957643780e+01;
  NLX_coes[0][ 25][ 40] =-4.318642872647892e+01;
  NLX_coes[0][ 26][  0] =-1.997062904297139e+01;
  NLX_coes[0][ 26][  1] =-2.026848461905075e+01;
  NLX_coes[0][ 26][  2] =-2.057021632921535e+01;
  NLX_coes[0][ 26][  3] =-2.087835812663853e+01;
  NLX_coes[0][ 26][  4] =-2.119555631946804e+01;
  NLX_coes[0][ 26][  5] =-2.152460814937587e+01;
  NLX_coes[0][ 26][  6] =-2.186850461978079e+01;
  NLX_coes[0][ 26][  7] =-2.223047685020280e+01;
  NLX_coes[0][ 26][  8] =-2.261404540717868e+01;
  NLX_coes[0][ 26][  9] =-2.302307056829951e+01;
  NLX_coes[0][ 26][ 10] =-2.346179756670049e+01;
  NLX_coes[0][ 26][ 11] =-2.393488371205441e+01;
  NLX_coes[0][ 26][ 12] =-2.444738325766314e+01;
  NLX_coes[0][ 26][ 13] =-2.500465157527411e+01;
  NLX_coes[0][ 26][ 14] =-2.561211707737327e+01;
  NLX_coes[0][ 26][ 15] =-2.627487074135527e+01;
  NLX_coes[0][ 26][ 16] =-2.699706715851526e+01;
  NLX_coes[0][ 26][ 17] =-2.778125666533955e+01;
  NLX_coes[0][ 26][ 18] =-2.862797433160879e+01;
  NLX_coes[0][ 26][ 19] =-2.953597045507707e+01;
  NLX_coes[0][ 26][ 20] =-3.050217874411019e+01;
  NLX_coes[0][ 26][ 21] =-3.151968547089268e+01;
  NLX_coes[0][ 26][ 22] =-3.257622627777321e+01;
  NLX_coes[0][ 26][ 23] =-3.365237650736638e+01;
  NLX_coes[0][ 26][ 24] =-3.472389915469556e+01;
  NLX_coes[0][ 26][ 25] =-3.576545587637663e+01;
  NLX_coes[0][ 26][ 26] =-3.675597927840442e+01;
  NLX_coes[0][ 26][ 27] =-3.768204489726542e+01;
  NLX_coes[0][ 26][ 28] =-3.853805159372319e+01;
  NLX_coes[0][ 26][ 29] =-3.932470088701167e+01;
  NLX_coes[0][ 26][ 30] =-4.004703520781197e+01;
  NLX_coes[0][ 26][ 31] =-4.071266497347258e+01;
  NLX_coes[0][ 26][ 32] =-4.133038800899342e+01;
  NLX_coes[0][ 26][ 33] =-4.190920427027349e+01;
  NLX_coes[0][ 26][ 34] =-4.245765942100400e+01;
  NLX_coes[0][ 26][ 35] =-4.298343925736046e+01;
  NLX_coes[0][ 26][ 36] =-4.349314565472388e+01;
  NLX_coes[0][ 26][ 37] =-4.399219857188453e+01;
  NLX_coes[0][ 26][ 38] =-4.448482296614275e+01;
  NLX_coes[0][ 26][ 39] =-4.497409321936631e+01;
  NLX_coes[0][ 26][ 40] =-4.546202089168346e+01;
  NLX_coes[0][ 27][  0] =-2.153637362770505e+01;
  NLX_coes[0][ 27][  1] =-2.186478083840107e+01;
  NLX_coes[0][ 27][  2] =-2.219756189566349e+01;
  NLX_coes[0][ 27][  3] =-2.253727895749909e+01;
  NLX_coes[0][ 27][  4] =-2.288661914368851e+01;
  NLX_coes[0][ 27][  5] =-2.324842946129920e+01;
  NLX_coes[0][ 27][  6] =-2.362575086141279e+01;
  NLX_coes[0][ 27][  7] =-2.402184925139851e+01;
  NLX_coes[0][ 27][  8] =-2.444024159426931e+01;
  NLX_coes[0][ 27][  9] =-2.488471380954171e+01;
  NLX_coes[0][ 27][ 10] =-2.535932368978011e+01;
  NLX_coes[0][ 27][ 11] =-2.586837646954951e+01;
  NLX_coes[0][ 27][ 12] =-2.641635370236932e+01;
  NLX_coes[0][ 27][ 13] =-2.700776982832745e+01;
  NLX_coes[0][ 27][ 14] =-2.764693016198364e+01;
  NLX_coes[0][ 27][ 15] =-2.833757795825048e+01;
  NLX_coes[0][ 27][ 16] =-2.908245795889657e+01;
  NLX_coes[0][ 27][ 17] =-2.988289039507043e+01;
  NLX_coes[0][ 27][ 18] =-3.073849530233811e+01;
  NLX_coes[0][ 27][ 19] =-3.164706584557742e+01;
  NLX_coes[0][ 27][ 20] =-3.260409241692012e+01;
  NLX_coes[0][ 27][ 21] =-3.360190336801067e+01;
  NLX_coes[0][ 27][ 22] =-3.462886763787250e+01;
  NLX_coes[0][ 27][ 23] =-3.566923211838050e+01;
  NLX_coes[0][ 27][ 24] =-3.670443239035872e+01;
  NLX_coes[0][ 27][ 25] =-3.771558735959093e+01;
  NLX_coes[0][ 27][ 26] =-3.868635981687378e+01;
  NLX_coes[0][ 27][ 27] =-3.960518341595166e+01;
  NLX_coes[0][ 27][ 28] =-4.046594845101503e+01;
  NLX_coes[0][ 27][ 29] =-4.126745975732010e+01;
  NLX_coes[0][ 27][ 30] =-4.201231110281426e+01;
  NLX_coes[0][ 27][ 31] =-4.270565342410191e+01;
  NLX_coes[0][ 27][ 32] =-4.335410411967403e+01;
  NLX_coes[0][ 27][ 33] =-4.396488512594672e+01;
  NLX_coes[0][ 27][ 34] =-4.454519349033956e+01;
  NLX_coes[0][ 27][ 35] =-4.510177306963503e+01;
  NLX_coes[0][ 27][ 36] =-4.564064655478349e+01;
  NLX_coes[0][ 27][ 37] =-4.616696940423997e+01;
  NLX_coes[0][ 27][ 38] =-4.668497461586239e+01;
  NLX_coes[0][ 27][ 39] =-4.719798657478356e+01;
  NLX_coes[0][ 27][ 40] =-4.770849226726890e+01;
  NLX_coes[0][ 28][  0] =-2.316120291374916e+01;
  NLX_coes[0][ 28][  1] =-2.351826527431950e+01;
  NLX_coes[0][ 28][  2] =-2.388013706987501e+01;
  NLX_coes[0][ 28][  3] =-2.424937280855629e+01;
  NLX_coes[0][ 28][  4] =-2.462865952697965e+01;
  NLX_coes[0][ 28][  5] =-2.502084831867684e+01;
  NLX_coes[0][ 28][  6] =-2.542897860559226e+01;
  NLX_coes[0][ 28][  7] =-2.585629245773524e+01;
  NLX_coes[0][ 28][  8] =-2.630623725983274e+01;
  NLX_coes[0][ 28][  9] =-2.678245411689878e+01;
  NLX_coes[0][ 28][ 10] =-2.728874662518900e+01;
  NLX_coes[0][ 28][ 11] =-2.782902070715571e+01;
  NLX_coes[0][ 28][ 12] =-2.840718243984957e+01;
  NLX_coes[0][ 28][ 13] =-2.902697937454047e+01;
  NLX_coes[0][ 28][ 14] =-2.969177508939248e+01;
  NLX_coes[0][ 28][ 15] =-3.040426055884496e+01;
  NLX_coes[0][ 28][ 16] =-3.116613052215465e+01;
  NLX_coes[0][ 28][ 17] =-3.197777805259355e+01;
  NLX_coes[0][ 28][ 18] =-3.283805089375713e+01;
  NLX_coes[0][ 28][ 19] =-3.374401495868204e+01;
  NLX_coes[0][ 28][ 20] =-3.469058771250678e+01;
  NLX_coes[0][ 28][ 21] =-3.567013864297657e+01;
  NLX_coes[0][ 28][ 22] =-3.667220875542529e+01;
  NLX_coes[0][ 28][ 23] =-3.768376896776385e+01;
  NLX_coes[0][ 28][ 24] =-3.869015894567091e+01;
  NLX_coes[0][ 28][ 25] =-3.967669541149593e+01;
  NLX_coes[0][ 28][ 26] =-4.063042994474221e+01;
  NLX_coes[0][ 28][ 27] =-4.154158545872989e+01;
  NLX_coes[0][ 28][ 28] =-4.240423349838352e+01;
  NLX_coes[0][ 28][ 29] =-4.321619546473836e+01;
  NLX_coes[0][ 28][ 30] =-4.397844244796850e+01;
  NLX_coes[0][ 28][ 31] =-4.469428848384217e+01;
  NLX_coes[0][ 28][ 32] =-4.536858175666026e+01;
  NLX_coes[0][ 28][ 33] =-4.600700170342019e+01;
  NLX_coes[0][ 28][ 34] =-4.661550190667761e+01;
  NLX_coes[0][ 28][ 35] =-4.719990004858749e+01;
  NLX_coes[0][ 28][ 36] =-4.776559837496885e+01;
  NLX_coes[0][ 28][ 37] =-4.831741274882931e+01;
  NLX_coes[0][ 28][ 38] =-4.885948981923764e+01;
  NLX_coes[0][ 28][ 39] =-4.939529680690030e+01;
  NLX_coes[0][ 28][ 40] =-4.992767511807320e+01;
  NLX_coes[0][ 29][  0] =-2.484511690110369e+01;
  NLX_coes[0][ 29][  1] =-2.522848892876635e+01;
  NLX_coes[0][ 29][  2] =-2.561704025631151e+01;
  NLX_coes[0][ 29][  3] =-2.601328109286479e+01;
  NLX_coes[0][ 29][  4] =-2.641986035986872e+01;
  NLX_coes[0][ 29][  5] =-2.683959523266925e+01;
  NLX_coes[0][ 29][  6] =-2.727548533486309e+01;
  NLX_coes[0][ 29][  7] =-2.773070962575044e+01;
  NLX_coes[0][ 29][  8] =-2.820860620916305e+01;
  NLX_coes[0][ 29][  9] =-2.871263488030258e+01;
  NLX_coes[0][ 29][ 10] =-2.924631979097225e+01;
  NLX_coes[0][ 29][ 11] =-2.981316658946047e+01;
  NLX_coes[0][ 29][ 12] =-3.041654637068537e+01;
  NLX_coes[0][ 29][ 13] =-3.105953939006553e+01;
  NLX_coes[0][ 29][ 14] =-3.174473636797490e+01;
  NLX_coes[0][ 29][ 15] =-3.247400510981526e+01;
  NLX_coes[0][ 29][ 16] =-3.324824275262168e+01;
  NLX_coes[0][ 29][ 17] =-3.406714042558095e+01;
  NLX_coes[0][ 29][ 18] =-3.492897248758788e+01;
  NLX_coes[0][ 29][ 19] =-3.583038636795067e+01;
  NLX_coes[0][ 29][ 20] =-3.676618154900892e+01;
  NLX_coes[0][ 29][ 21] =-3.772913997252672e+01;
  NLX_coes[0][ 29][ 22] =-3.871002080905750e+01;
  NLX_coes[0][ 29][ 23] =-3.969789609744679e+01;
  NLX_coes[0][ 29][ 24] =-4.068088017872522e+01;
  NLX_coes[0][ 29][ 25] =-4.164719855612688e+01;
  NLX_coes[0][ 29][ 26] =-4.258634523354679e+01;
  NLX_coes[0][ 29][ 27] =-4.349004759063861e+01;
  NLX_coes[0][ 29][ 28] =-4.435281816348286e+01;
  NLX_coes[0][ 29][ 29] =-4.517202972687635e+01;
  NLX_coes[0][ 29][ 30] =-4.594761678574770e+01;
  NLX_coes[0][ 29][ 31] =-4.668156681088289e+01;
  NLX_coes[0][ 29][ 32] =-4.737734475719093e+01;
  NLX_coes[0][ 29][ 33] =-4.803934673898551e+01;
  NLX_coes[0][ 29][ 34] =-4.867243377581011e+01;
  NLX_coes[0][ 29][ 35] =-4.928156461148543e+01;
  NLX_coes[0][ 29][ 36] =-4.987152782109729e+01;
  NLX_coes[0][ 29][ 37] =-5.044676451704788e+01;
  NLX_coes[0][ 29][ 38] =-5.101127057362977e+01;
  NLX_coes[0][ 29][ 39] =-5.156856884619089e+01;
  NLX_coes[0][ 29][ 40] =-5.212174561269589e+01;
  NLX_coes[0][ 30][  0] =-2.658811558976871e+01;
  NLX_coes[0][ 30][  1] =-2.699503548471384e+01;
  NLX_coes[0][ 30][  2] =-2.740743986608982e+01;
  NLX_coes[0][ 30][  3] =-2.782775801179544e+01;
  NLX_coes[0][ 30][  4] =-2.825856775931344e+01;
  NLX_coes[0][ 30][  5] =-2.870262521364313e+01;
  NLX_coes[0][ 30][  6] =-2.916286814675019e+01;
  NLX_coes[0][ 30][  7] =-2.964239381710950e+01;
  NLX_coes[0][ 30][  8] =-3.014441576077162e+01;
  NLX_coes[0][ 30][  9] =-3.067220344019571e+01;
  NLX_coes[0][ 30][ 10] =-3.122900538922885e+01;
  NLX_coes[0][ 30][ 11] =-3.181795328615000e+01;
  NLX_coes[0][ 30][ 12] =-3.244194287563744e+01;
  NLX_coes[0][ 30][ 13] =-3.310348873235234e+01;
  NLX_coes[0][ 30][ 14] =-3.380455384647198e+01;
  NLX_coes[0][ 30][ 15] =-3.454636128895132e+01;
  NLX_coes[0][ 30][ 16] =-3.532920128084925e+01;
  NLX_coes[0][ 30][ 17] =-3.615224814550666e+01;
  NLX_coes[0][ 30][ 18] =-3.701339432861344e+01;
  NLX_coes[0][ 30][ 19] =-3.790910185504708e+01;
  NLX_coes[0][ 30][ 20] =-3.883428770996235e+01;
  NLX_coes[0][ 30][ 21] =-3.978228604923437e+01;
  NLX_coes[0][ 30][ 22] =-4.074495935460274e+01;
  NLX_coes[0][ 30][ 23] =-4.171302807643992e+01;
  NLX_coes[0][ 30][ 24] =-4.267663895083785e+01;
  NLX_coes[0][ 30][ 25] =-4.362611333899142e+01;
  NLX_coes[0][ 30][ 26] =-4.455274203492270e+01;
  NLX_coes[0][ 30][ 27] =-4.544946131435381e+01;
  NLX_coes[0][ 30][ 28] =-4.631128274291900e+01;
  NLX_coes[0][ 30][ 29] =-4.713542388842088e+01;
  NLX_coes[0][ 30][ 30] =-4.792117447930939e+01;
  NLX_coes[0][ 30][ 31] =-4.866958381059582e+01;
  NLX_coes[0][ 30][ 32] =-4.938306199308339e+01;
  NLX_coes[0][ 30][ 33] =-5.006496892435350e+01;
  NLX_coes[0][ 30][ 34] =-5.071923913758668e+01;
  NLX_coes[0][ 30][ 35] =-5.135006815885600e+01;
  NLX_coes[0][ 30][ 36] =-5.196167011515774e+01;
  NLX_coes[0][ 30][ 37] =-5.255810697248050e+01;
  NLX_coes[0][ 30][ 38] =-5.314318554912735e+01;
  NLX_coes[0][ 30][ 39] =-5.372041777818823e+01;
  NLX_coes[0][ 30][ 40] =-5.429304121189473e+01;
  NLX_coes[0][ 31][  0] =-2.839019897974433e+01;
  NLX_coes[0][ 31][  1] =-2.881751731610649e+01;
  NLX_coes[0][ 31][  2] =-2.925056474405717e+01;
  NLX_coes[0][ 31][  3] =-2.969165269041524e+01;
  NLX_coes[0][ 31][  4] =-3.014326185662282e+01;
  NLX_coes[0][ 31][  5] =-3.060807421063937e+01;
  NLX_coes[0][ 31][  6] =-3.108896254154958e+01;
  NLX_coes[0][ 31][  7] =-3.158894475609772e+01;
  NLX_coes[0][ 31][  8] =-3.211111563153521e+01;
  NLX_coes[0][ 31][  9] =-3.265856542453814e+01;
  NLX_coes[0][ 31][ 10] =-3.323428873415006e+01;
  NLX_coes[0][ 31][ 11] =-3.384108262830724e+01;
  NLX_coes[0][ 31][ 12] =-3.448143144572175e+01;
  NLX_coes[0][ 31][ 13] =-3.515737674102314e+01;
  NLX_coes[0][ 31][ 14] =-3.587037394373833e+01;
  NLX_coes[0][ 31][ 15] =-3.662114138328128e+01;
  NLX_coes[0][ 31][ 16] =-3.740951066519886e+01;
  NLX_coes[0][ 31][ 17] =-3.823428809955424e+01;
  NLX_coes[0][ 31][ 18] =-3.909313528062708e+01;
  NLX_coes[0][ 31][ 19] =-3.998247844636389e+01;
  NLX_coes[0][ 31][ 20] =-4.089746540149055e+01;
  NLX_coes[0][ 31][ 21] =-4.183200041243164e+01;
  NLX_coes[0][ 31][ 22] =-4.277889660222271e+01;
  NLX_coes[0][ 31][ 23] =-4.373017447084854e+01;
  NLX_coes[0][ 31][ 24] =-4.467750683855696e+01;
  NLX_coes[0][ 31][ 25] =-4.561276543155953e+01;
  NLX_coes[0][ 31][ 26] =-4.652858804402508e+01;
  NLX_coes[0][ 31][ 27] =-4.741886871501792e+01;
  NLX_coes[0][ 31][ 28] =-4.827909179252242e+01;
  NLX_coes[0][ 31][ 29] =-4.910647134745874e+01;
  NLX_coes[0][ 31][ 30] =-4.989990499317033e+01;
  NLX_coes[0][ 31][ 31] =-5.065978639596566e+01;
  NLX_coes[0][ 31][ 32] =-5.138773384325515e+01;
  NLX_coes[0][ 31][ 33] =-5.208628776146355e+01;
  NLX_coes[0][ 31][ 34] =-5.275861713524749e+01;
  NLX_coes[0][ 31][ 35] =-5.340826049786650e+01;
  NLX_coes[0][ 31][ 36] =-5.403891530955224e+01;
  NLX_coes[0][ 31][ 37] =-5.465428135783029e+01;
  NLX_coes[0][ 31][ 38] =-5.525795916328830e+01;
  NLX_coes[0][ 31][ 39] =-5.585340254763365e+01;
  NLX_coes[0][ 31][ 40] =-5.644392463907369e+01;
  NLX_coes[0][ 32][  0] =-3.025136707103068e+01;
  NLX_coes[0][ 32][  1] =-3.069556675212517e+01;
  NLX_coes[0][ 32][  2] =-3.114568452926764e+01;
  NLX_coes[0][ 32][  3] =-3.160387530551917e+01;
  NLX_coes[0][ 32][  4] =-3.207250609000258e+01;
  NLX_coes[0][ 32][  5] =-3.255419067860634e+01;
  NLX_coes[0][ 32][  6] =-3.305175622675132e+01;
  NLX_coes[0][ 32][  7] =-3.356816428665967e+01;
  NLX_coes[0][ 32][  8] =-3.410641336288337e+01;
  NLX_coes[0][ 32][  9] =-3.466943804530655e+01;
  NLX_coes[0][ 32][ 10] =-3.526000864924886e+01;
  NLX_coes[0][ 32][ 11] =-3.588062949100912e+01;
  NLX_coes[0][ 32][ 12] =-3.653343259644544e+01;
  NLX_coes[0][ 32][ 13] =-3.722006506826379e+01;
  NLX_coes[0][ 32][ 14] =-3.794157115755301e+01;
  NLX_coes[0][ 32][ 15] =-3.869827320266510e+01;
  NLX_coes[0][ 32][ 16] =-3.948965803029757e+01;
  NLX_coes[0][ 32][ 17] =-4.031427662662519e+01;
  NLX_coes[0][ 32][ 18] =-4.116966568205957e+01;
  NLX_coes[0][ 32][ 19] =-4.205230209697040e+01;
  NLX_coes[0][ 32][ 20] =-4.295760607452817e+01;
  NLX_coes[0][ 32][ 21] =-4.388001296323002e+01;
  NLX_coes[0][ 32][ 22] =-4.481313420400600e+01;
  NLX_coes[0][ 32][ 23] =-4.575001821173707e+01;
  NLX_coes[0][ 32][ 24] =-4.668350326817329e+01;
  NLX_coes[0][ 32][ 25] =-4.760662989149428e+01;
  NLX_coes[0][ 32][ 26] =-4.851306023865644e+01;
  NLX_coes[0][ 32][ 27] =-4.939744445022700e+01;
  NLX_coes[0][ 32][ 28] =-5.025568332191834e+01;
  NLX_coes[0][ 32][ 29] =-5.108505965061281e+01;
  NLX_coes[0][ 32][ 30] =-5.188423815048908e+01;
  NLX_coes[0][ 32][ 31] =-5.265315681334186e+01;
  NLX_coes[0][ 32][ 32] =-5.339284477924748e+01;
  NLX_coes[0][ 32][ 33] =-5.410520319761961e+01;
  NLX_coes[0][ 32][ 34] =-5.479278000235720e+01;
  NLX_coes[0][ 32][ 35] =-5.545856119173828e+01;
  NLX_coes[0][ 32][ 36] =-5.610579304334663e+01;
  NLX_coes[0][ 32][ 37] =-5.673784322865631e+01;
  NLX_coes[0][ 32][ 38] =-5.735810450932671e+01;
  NLX_coes[0][ 32][ 39] =-5.796994247076925e+01;
  NLX_coes[0][ 32][ 40] =-5.857668821995492e+01;
  NLX_coes[0][ 33][  0] =-3.217161986362800e+01;
  NLX_coes[0][ 33][  1] =-3.262882217367488e+01;
  NLX_coes[0][ 33][  2] =-3.309207853917587e+01;
  NLX_coes[0][ 33][  3] =-3.356334432249764e+01;
  NLX_coes[0][ 33][  4] =-3.404487153868976e+01;
  NLX_coes[0][ 33][  5] =-3.453924034654743e+01;
  NLX_coes[0][ 33][  6] =-3.504927911423717e+01;
  NLX_coes[0][ 33][  7] =-3.557793503407205e+01;
  NLX_coes[0][ 33][  8] =-3.612814300140504e+01;
  NLX_coes[0][ 33][  9] =-3.670270902003504e+01;
  NLX_coes[0][ 33][ 10] =-3.730420742610890e+01;
  NLX_coes[0][ 33][ 11] =-3.793488582644930e+01;
  NLX_coes[0][ 33][ 12] =-3.859657216005455e+01;
  NLX_coes[0][ 33][ 13] =-3.929058101504038e+01;
  NLX_coes[0][ 33][ 14] =-4.001761951622223e+01;
  NLX_coes[0][ 33][ 15] =-4.077769595423408e+01;
  NLX_coes[0][ 33][ 16] =-4.157003647863878e+01;
  NLX_coes[0][ 33][ 17] =-4.239301663114235e+01;
  NLX_coes[0][ 33][ 18] =-4.324411578200853e+01;
  NLX_coes[0][ 33][ 19] =-4.411990426901309e+01;
  NLX_coes[0][ 33][ 20] =-4.501607495022446e+01;
  NLX_coes[0][ 33][ 21] =-4.592753173298379e+01;
  NLX_coes[0][ 33][ 22] =-4.684854506494896e+01;
  NLX_coes[0][ 33][ 23] =-4.777297680377654e+01;
  NLX_coes[0][ 33][ 24] =-4.869456455588954e+01;
  NLX_coes[0][ 33][ 25] =-4.960724150883696e+01;
  NLX_coes[0][ 33][ 26] =-5.050545663699450e+01;
  NLX_coes[0][ 33][ 27] =-5.138445646536961e+01;
  NLX_coes[0][ 33][ 28] =-5.224049513468048e+01;
  NLX_coes[0][ 33][ 29] =-5.307095293231340e+01;
  NLX_coes[0][ 33][ 30] =-5.387436015337000e+01;
  NLX_coes[0][ 33][ 31] =-5.465033815743934e+01;
  NLX_coes[0][ 33][ 32] =-5.539947912137435e+01;
  NLX_coes[0][ 33][ 33] =-5.612318927341417e+01;
  NLX_coes[0][ 33][ 34] =-5.682351866997535e+01;
  NLX_coes[0][ 33][ 35] =-5.750299605167053e+01;
  NLX_coes[0][ 33][ 36] =-5.816448196079347e+01;
  NLX_coes[0][ 33][ 37] =-5.881104844180967e+01;
  NLX_coes[0][ 33][ 38] =-5.944588999723520e+01;
  NLX_coes[0][ 33][ 39] =-6.007226824876025e+01;
  NLX_coes[0][ 33][ 40] =-6.069349200501507e+01;
  NLX_coes[0][ 34][  0] =-3.415095735753677e+01;
  NLX_coes[0][ 34][  1] =-3.461690750889693e+01;
  NLX_coes[0][ 34][  2] =-3.508898885010472e+01;
  NLX_coes[0][ 34][  3] =-3.556890599107646e+01;
  NLX_coes[0][ 34][  4] =-3.605882601132874e+01;
  NLX_coes[0][ 34][  5] =-3.656137686906660e+01;
  NLX_coes[0][ 34][  6] =-3.707946711885040e+01;
  NLX_coes[0][ 34][  7] =-3.761608423615349e+01;
  NLX_coes[0][ 34][  8] =-3.817413135622180e+01;
  NLX_coes[0][ 34][  9] =-3.875630530181634e+01;
  NLX_coes[0][ 34][ 10] =-3.936500197588049e+01;
  NLX_coes[0][ 34][ 11] =-4.000223562641800e+01;
  NLX_coes[0][ 34][ 12] =-4.066956342624532e+01;
  NLX_coes[0][ 34][ 13] =-4.136801168231682e+01;
  NLX_coes[0][ 34][ 14] =-4.209800377333880e+01;
  NLX_coes[0][ 34][ 15] =-4.285929264402644e+01;
  NLX_coes[0][ 34][ 16] =-4.365090257110778e+01;
  NLX_coes[0][ 34][ 17] =-4.447108621782592e+01;
  NLX_coes[0][ 34][ 18] =-4.531730401249646e+01;
  NLX_coes[0][ 34][ 19] =-4.618623374937115e+01;
  NLX_coes[0][ 34][ 20] =-4.707381871590705e+01;
  NLX_coes[0][ 34][ 21] =-4.797536182762286e+01;
  NLX_coes[0][ 34][ 22] =-4.888567016115409e+01;
  NLX_coes[0][ 34][ 23] =-4.979924844914124e+01;
  NLX_coes[0][ 34][ 24] =-5.071053202038458e+01;
  NLX_coes[0][ 34][ 25] =-5.161414122635389e+01;
  NLX_coes[0][ 34][ 26] =-5.250513307437646e+01;
  NLX_coes[0][ 34][ 27] =-5.337922395802045e+01;
  NLX_coes[0][ 34][ 28] =-5.423296095156913e+01;
  NLX_coes[0][ 34][ 29] =-5.506382734743588e+01;
  NLX_coes[0][ 34][ 30] =-5.587027861512291e+01;
  NLX_coes[0][ 34][ 31] =-5.665171492894476e+01;
  NLX_coes[0][ 34][ 32] =-5.740840362577136e+01;
  NLX_coes[0][ 34][ 33] =-5.814136843949581e+01;
  NLX_coes[0][ 34][ 34] =-5.885226243193691e+01;
  NLX_coes[0][ 34][ 35] =-5.954323925350441e+01;
  NLX_coes[0][ 34][ 36] =-6.021683390415301e+01;
  NLX_coes[0][ 34][ 37] =-6.087586050573726e+01;
  NLX_coes[0][ 34][ 38] =-6.152333146441119e+01;
  NLX_coes[0][ 34][ 39] =-6.216240037349715e+01;
  NLX_coes[0][ 34][ 40] =-6.279632969345081e+01;
  NLX_coes[0][ 35][  0] =-3.618937955275778e+01;
  NLX_coes[0][ 35][  1] =-3.665940163017123e+01;
  NLX_coes[0][ 35][  2] =-3.713554646752122e+01;
  NLX_coes[0][ 35][  3] =-3.761920371902864e+01;
  NLX_coes[0][ 35][  4] =-3.811256602734754e+01;
  NLX_coes[0][ 35][  5] =-3.861846499808522e+01;
  NLX_coes[0][ 35][  6] =-3.913999580564044e+01;
  NLX_coes[0][ 35][  7] =-3.968023520714498e+01;
  NLX_coes[0][ 35][  8] =-4.024206697614706e+01;
  NLX_coes[0][ 35][  9] =-4.082807641115707e+01;
  NLX_coes[0][ 35][ 10] =-4.144047863277719e+01;
  NLX_coes[0][ 35][ 11] =-4.208106001602529e+01;
  NLX_coes[0][ 35][ 12] =-4.275112325702501e+01;
  NLX_coes[0][ 35][ 13] =-4.345143321680499e+01;
  NLX_coes[0][ 35][ 14] =-4.418216446718867e+01;
  NLX_coes[0][ 35][ 15] =-4.494285369671849e+01;
  NLX_coes[0][ 35][ 16] =-4.573236151743120e+01;
  NLX_coes[0][ 35][ 17] =-4.654884909179930e+01;
  NLX_coes[0][ 35][ 18] =-4.738977553641749e+01;
  NLX_coes[0][ 35][ 19] =-4.825192223494542e+01;
  NLX_coes[0][ 35][ 20] =-4.913144977034419e+01;
  NLX_coes[0][ 35][ 21] =-5.002399173667835e+01;
  NLX_coes[0][ 35][ 22] =-5.092478680336510e+01;
  NLX_coes[0][ 35][ 23] =-5.182884601090961e+01;
  NLX_coes[0][ 35][ 24] =-5.273114688595690e+01;
  NLX_coes[0][ 35][ 25] =-5.362684071513829e+01;
  NLX_coes[0][ 35][ 26] =-5.451145564017622e+01;
  NLX_coes[0][ 35][ 27] =-5.538107736789611e+01;
  NLX_coes[0][ 35][ 28] =-5.623249172655296e+01;
  NLX_coes[0][ 35][ 29] =-5.706327858201414e+01;
  NLX_coes[0][ 35][ 30] =-5.787185347539630e+01;
  NLX_coes[0][ 35][ 31] =-5.865746013126379e+01;
  NLX_coes[0][ 35][ 32] =-5.942012232461396e+01;
  NLX_coes[0][ 35][ 33] =-6.016056671237961e+01;
  NLX_coes[0][ 35][ 34] =-6.088012905937953e+01;
  NLX_coes[0][ 35][ 35] =-6.158065524665247e+01;
  NLX_coes[0][ 35][ 36] =-6.226440616439707e+01;
  NLX_coes[0][ 35][ 37] =-6.293397267518420e+01;
  NLX_coes[0][ 35][ 38] =-6.359220382713482e+01;
  NLX_coes[0][ 35][ 39] =-6.424214968372004e+01;
  NLX_coes[0][ 35][ 40] =-6.488701768392120e+01;
  NLX_coes[0][ 36][  0] =-3.828688644929272e+01;
  NLX_coes[0][ 36][  1] =-3.875579188599207e+01;
  NLX_coes[0][ 36][  2] =-3.923063788821082e+01;
  NLX_coes[0][ 36][  3] =-3.971244438778766e+01;
  NLX_coes[0][ 36][  4] =-4.020375287459436e+01;
  NLX_coes[0][ 36][  5] =-4.070784004271891e+01;
  NLX_coes[0][ 36][  6] =-4.122808425935618e+01;
  NLX_coes[0][ 36][  7] =-4.176765461602175e+01;
  NLX_coes[0][ 36][  8] =-4.232938146865858e+01;
  NLX_coes[0][ 36][  9] =-4.291570009788548e+01;
  NLX_coes[0][ 36][ 10] =-4.352861618152996e+01;
  NLX_coes[0][ 36][ 11] =-4.416967381551729e+01;
  NLX_coes[0][ 36][ 12] =-4.483992089234276e+01;
  NLX_coes[0][ 36][ 13] =-4.553987222999723e+01;
  NLX_coes[0][ 36][ 14] =-4.626947318515167e+01;
  NLX_coes[0][ 36][ 15] =-4.702806765263585e+01;
  NLX_coes[0][ 36][ 16] =-4.781437501882421e+01;
  NLX_coes[0][ 36][ 17] =-4.862648099258216e+01;
  NLX_coes[0][ 36][ 18] =-4.946184730823282e+01;
  NLX_coes[0][ 36][ 19] =-5.031734499897808e+01;
  NLX_coes[0][ 36][ 20] =-5.118931510280399e+01;
  NLX_coes[0][ 36][ 21] =-5.207365906764672e+01;
  NLX_coes[0][ 36][ 22] =-5.296595863110998e+01;
  NLX_coes[0][ 36][ 23] =-5.386162164884186e+01;
  NLX_coes[0][ 36][ 24] =-5.475604665314081e+01;
  NLX_coes[0][ 36][ 25] =-5.564479555536817e+01;
  NLX_coes[0][ 36][ 26] =-5.652376170163196e+01;
  NLX_coes[0][ 36][ 27] =-5.738932012921747e+01;
  NLX_coes[0][ 36][ 28] =-5.823844861511039e+01;
  NLX_coes[0][ 36][ 29] =-5.906881167584175e+01;
  NLX_coes[0][ 36][ 30] =-5.987880432721708e+01;
  NLX_coes[0][ 36][ 31] =-6.066755717573709e+01;
  NLX_coes[0][ 36][ 32] =-6.143490840585683e+01;
  NLX_coes[0][ 36][ 33] =-6.218135087249936e+01;
  NLX_coes[0][ 36][ 34] =-6.290796361154033e+01;
  NLX_coes[0][ 36][ 35] =-6.361633674626469e+01;
  NLX_coes[0][ 36][ 36] =-6.430849723029269e+01;
  NLX_coes[0][ 36][ 37] =-6.498684035257946e+01;
  NLX_coes[0][ 36][ 38] =-6.565406874338161e+01;
  NLX_coes[0][ 36][ 39] =-6.631313730593642e+01;
  NLX_coes[0][ 36][ 40] =-6.696720447875867e+01;
  NLX_coes[0][ 37][  0] =-4.044347804714784e+01;
  NLX_coes[0][ 37][  1] =-4.090541421977448e+01;
  NLX_coes[0][ 37][  2] =-4.137259435350880e+01;
  NLX_coes[0][ 37][  3] =-4.184594964641551e+01;
  NLX_coes[0][ 37][  4] =-4.232909976216624e+01;
  NLX_coes[0][ 37][  5] =-4.282599973357227e+01;
  NLX_coes[0][ 37][  6] =-4.334028711140419e+01;
  NLX_coes[0][ 37][  7] =-4.387511698348550e+01;
  NLX_coes[0][ 37][  8] =-4.443316042886001e+01;
  NLX_coes[0][ 37][  9] =-4.501662191135267e+01;
  NLX_coes[0][ 37][ 10] =-4.562724370328461e+01;
  NLX_coes[0][ 37][ 11] =-4.626629638897404e+01;
  NLX_coes[0][ 37][ 12] =-4.693455966353783e+01;
  NLX_coes[0][ 37][ 13] =-4.763229800633456e+01;
  NLX_coes[0][ 37][ 14] =-4.835923575592142e+01;
  NLX_coes[0][ 37][ 15] =-4.911453611402007e+01;
  NLX_coes[0][ 37][ 16] =-4.989678860374624e+01;
  NLX_coes[0][ 37][ 17] =-5.070400941551494e+01;
  NLX_coes[0][ 37][ 18] =-5.153365879717064e+01;
  NLX_coes[0][ 37][ 19] =-5.238267906909130e+01;
  NLX_coes[0][ 37][ 20] =-5.324755584308780e+01;
  NLX_coes[0][ 37][ 21] =-5.412440348987808e+01;
  NLX_coes[0][ 37][ 22] =-5.500907381359566e+01;
  NLX_coes[0][ 37][ 23] =-5.589728438530707e+01;
  NLX_coes[0][ 37][ 24] =-5.678476038048001e+01;
  NLX_coes[0][ 37][ 25] =-5.766738152570240e+01;
  NLX_coes[0][ 37][ 26] =-5.854132439565140e+01;
  NLX_coes[0][ 37][ 27] =-5.940319020301540e+01;
  NLX_coes[0][ 37][ 28] =-6.025010952976704e+01;
  NLX_coes[0][ 37][ 29] =-6.107981797726327e+01;
  NLX_coes[0][ 37][ 30] =-6.189070000944250e+01;
  NLX_coes[0][ 37][ 31] =-6.268180172760746e+01;
  NLX_coes[0][ 37][ 32] =-6.345281637044546e+01;
  NLX_coes[0][ 37][ 33] =-6.420404856413634e+01;
  NLX_coes[0][ 37][ 34] =-6.493636455493451e+01;
  NLX_coes[0][ 37][ 35] =-6.565113580415763e+01;
  NLX_coes[0][ 37][ 36] =-6.635018240620903e+01;
  NLX_coes[0][ 37][ 37] =-6.703572074850983e+01;
  NLX_coes[0][ 37][ 38] =-6.771031661808230e+01;
  NLX_coes[0][ 37][ 39] =-6.837683484799729e+01;
  NLX_coes[0][ 37][ 40] =-6.903840026699423e+01;
  NLX_coes[0][ 38][  0] =-4.265915434637478e+01;
  NLX_coes[0][ 38][  1] =-4.310738274439702e+01;
  NLX_coes[0][ 38][  2] =-4.355829674064363e+01;
  NLX_coes[0][ 38][  3] =-4.401529968348158e+01;
  NLX_coes[0][ 38][  4] =-4.448379100977888e+01;
  NLX_coes[0][ 38][  5] =-4.496828058323644e+01;
  NLX_coes[0][ 38][  6] =-4.547233185956939e+01;
  NLX_coes[0][ 38][  7] =-4.599882678018196e+01;
  NLX_coes[0][ 38][  8] =-4.655010818214461e+01;
  NLX_coes[0][ 38][  9] =-4.712804209871701e+01;
  NLX_coes[0][ 38][ 10] =-4.773404010958320e+01;
  NLX_coes[0][ 38][ 11] =-4.836905970260503e+01;
  NLX_coes[0][ 38][ 12] =-4.903359218646799e+01;
  NLX_coes[0][ 38][ 13] =-4.972764474789308e+01;
  NLX_coes[0][ 38][ 14] =-5.045072194502603e+01;
  NLX_coes[0][ 38][ 15] =-5.120181125733059e+01;
  NLX_coes[0][ 38][ 16] =-5.197937690759922e+01;
  NLX_coes[0][ 38][ 17] =-5.278136581562566e+01;
  NLX_coes[0][ 38][ 18] =-5.360522907882149e+01;
  NLX_coes[0][ 38][ 19] =-5.444796168054500e+01;
  NLX_coes[0][ 38][ 20] =-5.530616210917830e+01;
  NLX_coes[0][ 38][ 21] =-5.617611218259906e+01;
  NLX_coes[0][ 38][ 22] =-5.705387564237894e+01;
  NLX_coes[0][ 38][ 23] =-5.793541214306990e+01;
  NLX_coes[0][ 38][ 24] =-5.881670135704055e+01;
  NLX_coes[0][ 38][ 25] =-5.969387036284436e+01;
  NLX_coes[0][ 38][ 26] =-6.056331660771215e+01;
  NLX_coes[0][ 38][ 27] =-6.142181876903624e+01;
  NLX_coes[0][ 38][ 28] =-6.226662885899887e+01;
  NLX_coes[0][ 38][ 29] =-6.309554079694357e+01;
  NLX_coes[0][ 38][ 30] =-6.390693311938561e+01;
  NLX_coes[0][ 38][ 31] =-6.469978611517661e+01;
  NLX_coes[0][ 38][ 32] =-6.547367607446400e+01;
  NLX_coes[0][ 38][ 33] =-6.622875122988668e+01;
  NLX_coes[0][ 38][ 34] =-6.696569519938554e+01;
  NLX_coes[0][ 38][ 35] =-6.768568430935137e+01;
  NLX_coes[0][ 38][ 36] =-6.839034509525374e+01;
  NLX_coes[0][ 38][ 37] =-6.908171703538703e+01;
  NLX_coes[0][ 38][ 38] =-6.976222117143860e+01;
  NLX_coes[0][ 38][ 39] =-7.043463196153168e+01;
  NLX_coes[0][ 38][ 40] =-7.110203021902907e+01;
  NLX_coes[0][ 39][  0] =-4.493391534710742e+01;
  NLX_coes[0][ 39][  1] =-4.535943757462063e+01;
  NLX_coes[0][ 39][  2] =-4.578075952775160e+01;
  NLX_coes[0][ 39][  3] =-4.621288826702438e+01;
  NLX_coes[0][ 39][  4] =-4.666095273580623e+01;
  NLX_coes[0][ 39][  5] =-4.712871750139040e+01;
  NLX_coes[0][ 39][  6] =-4.761910507361533e+01;
  NLX_coes[0][ 39][  7] =-4.813444465813998e+01;
  NLX_coes[0][ 39][  8] =-4.867658745005440e+01;
  NLX_coes[0][ 39][  9] =-4.924695964653039e+01;
  NLX_coes[0][ 39][ 10] =-4.984657961715018e+01;
  NLX_coes[0][ 39][ 11] =-5.047605492244710e+01;
  NLX_coes[0][ 39][ 12] =-5.113556886043842e+01;
  NLX_coes[0][ 39][ 13] =-5.182486300937651e+01;
  NLX_coes[0][ 39][ 14] =-5.254322064090508e+01;
  NLX_coes[0][ 39][ 15] =-5.328945507148345e+01;
  NLX_coes[0][ 39][ 16] =-5.406190653773486e+01;
  NLX_coes[0][ 39][ 17] =-5.485845076799738e+01;
  NLX_coes[0][ 39][ 18] =-5.567652192198263e+01;
  NLX_coes[0][ 39][ 19] =-5.651315187989929e+01;
  NLX_coes[0][ 39][ 20] =-5.736502691915891e+01;
  NLX_coes[0][ 39][ 21] =-5.822856160597794e+01;
  NLX_coes[0][ 39][ 22] =-5.909998829829105e+01;
  NLX_coes[0][ 39][ 23] =-5.997545912584846e+01;
  NLX_coes[0][ 39][ 24] =-6.085115587120091e+01;
  NLX_coes[0][ 39][ 25] =-6.172340204906175e+01;
  NLX_coes[0][ 39][ 26] =-6.258877088984726e+01;
  NLX_coes[0][ 39][ 27] =-6.344418302898198e+01;
  NLX_coes[0][ 39][ 28] =-6.428698852750456e+01;
  NLX_coes[0][ 39][ 29] =-6.511502931069118e+01;
  NLX_coes[0][ 39][ 30] =-6.592668000338557e+01;
  NLX_coes[0][ 39][ 31] =-6.672086718939481e+01;
  NLX_coes[0][ 39][ 32] =-6.749706905026328e+01;
  NLX_coes[0][ 39][ 33] =-6.825529892747404e+01;
  NLX_coes[0][ 39][ 34] =-6.899607748784601e+01;
  NLX_coes[0][ 39][ 35] =-6.972039887785786e+01;
  NLX_coes[0][ 39][ 36] =-7.042969675902957e+01;
  NLX_coes[0][ 39][ 37] =-7.112581718257026e+01;
  NLX_coes[0][ 39][ 38] =-7.181100852892246e+01;
  NLX_coes[0][ 39][ 39] =-7.248796347358007e+01;
  NLX_coes[0][ 39][ 40] =-7.315950377133593e+01;
  NLX_coes[0][ 40][  0] =-4.726776104764826e+01;
  NLX_coes[0][ 40][  1] =-4.763960967450107e+01;
  NLX_coes[0][ 40][  2] =-4.802526365453279e+01;
  NLX_coes[0][ 40][  3] =-4.842791132541273e+01;
  NLX_coes[0][ 40][  4] =-4.885200606316729e+01;
  NLX_coes[0][ 40][  5] =-4.930029704722603e+01;
  NLX_coes[0][ 40][  6] =-4.977483646226501e+01;
  NLX_coes[0][ 40][  7] =-5.027723336343566e+01;
  NLX_coes[0][ 40][  8] =-5.080873970538874e+01;
  NLX_coes[0][ 40][  9] =-5.137027469237326e+01;
  NLX_coes[0][ 40][ 10] =-5.196242227321898e+01;
  NLX_coes[0][ 40][ 11] =-5.258541588259888e+01;
  NLX_coes[0][ 40][ 12] =-5.323911793133706e+01;
  NLX_coes[0][ 40][ 13] =-5.392299889000281e+01;
  NLX_coes[0][ 40][ 14] =-5.463611962674503e+01;
  NLX_coes[0][ 40][ 15] =-5.537712009088848e+01;
  NLX_coes[0][ 40][ 16] =-5.614421708471499e+01;
  NLX_coes[0][ 40][ 17] =-5.693521353222568e+01;
  NLX_coes[0][ 40][ 18] =-5.774752121762407e+01;
  NLX_coes[0][ 40][ 19] =-5.857819835668777e+01;
  NLX_coes[0][ 40][ 20] =-5.942400254606547e+01;
  NLX_coes[0][ 40][ 21] =-6.028145861339420e+01;
  NLX_coes[0][ 40][ 22] =-6.114693971729284e+01;
  NLX_coes[0][ 40][ 23] =-6.201675882488463e+01;
  NLX_coes[0][ 40][ 24] =-6.288726657415049e+01;
  NLX_coes[0][ 40][ 25] =-6.375495067998602e+01;
  NLX_coes[0][ 40][ 26] =-6.461653162466050e+01;
  NLX_coes[0][ 40][ 27] =-6.546904949240282e+01;
  NLX_coes[0][ 40][ 28] =-6.630993748853297e+01;
  NLX_coes[0][ 40][ 29] =-6.713707885539984e+01;
  NLX_coes[0][ 40][ 30] =-6.794884540620431e+01;
  NLX_coes[0][ 40][ 31] =-6.874411753840052e+01;
  NLX_coes[0][ 40][ 32] =-6.952228714897883e+01;
  NLX_coes[0][ 40][ 33] =-7.028324618005156e+01;
  NLX_coes[0][ 40][ 34] =-7.102736447169686e+01;
  NLX_coes[0][ 40][ 35] =-7.175546118317888e+01;
  NLX_coes[0][ 40][ 36] =-7.246877438318134e+01;
  NLX_coes[0][ 40][ 37] =-7.316893374438993e+01;
  NLX_coes[0][ 40][ 38] =-7.385794206147827e+01;
  NLX_coes[0][ 40][ 39] =-7.453815334752117e+01;
  NLX_coes[0][ 40][ 40] =-7.521266265460252e+01;
  NLX_coes[1][  0][  0] =+3.859695423399047e+01;
  NLX_coes[1][  0][  1] =+3.797207847414818e+01;
  NLX_coes[1][  0][  2] =+3.734282595190464e+01;
  NLX_coes[1][  0][  3] =+3.670801705659031e+01;
  NLX_coes[1][  0][  4] =+3.606741756395382e+01;
  NLX_coes[1][  0][  5] =+3.542128784332451e+01;
  NLX_coes[1][  0][  6] =+3.477040294125945e+01;
  NLX_coes[1][  0][  7] =+3.411616320630522e+01;
  NLX_coes[1][  0][  8] =+3.346081723328594e+01;
  NLX_coes[1][  0][  9] =+3.280771214342277e+01;
  NLX_coes[1][  0][ 10] =+3.216121463670752e+01;
  NLX_coes[1][  0][ 11] =+3.152623847086193e+01;
  NLX_coes[1][  0][ 12] =+3.090801279350151e+01;
  NLX_coes[1][  0][ 13] =+3.031226383138137e+01;
  NLX_coes[1][  0][ 14] =+2.974577367421826e+01;
  NLX_coes[1][  0][ 15] =+2.921709758324421e+01;
  NLX_coes[1][  0][ 16] =+2.873677263208726e+01;
  NLX_coes[1][  0][ 17] =+2.831669128279444e+01;
  NLX_coes[1][  0][ 18] =+2.796942723229029e+01;
  NLX_coes[1][  0][ 19] =+2.770751123495178e+01;
  NLX_coes[1][  0][ 20] =+2.754070314329465e+01;
  NLX_coes[1][  0][ 21] =+2.747090318310385e+01;
  NLX_coes[1][  0][ 22] =+2.748745456463463e+01;
  NLX_coes[1][  0][ 23] =+2.756634116170914e+01;
  NLX_coes[1][  0][ 24] =+2.766656590200581e+01;
  NLX_coes[1][  0][ 25] =+2.773368246084392e+01;
  NLX_coes[1][  0][ 26] =+2.771864636805773e+01;
  NLX_coes[1][  0][ 27] =+2.760521888165905e+01;
  NLX_coes[1][  0][ 28] =+2.741243409740100e+01;
  NLX_coes[1][  0][ 29] =+2.719789626773710e+01;
  NLX_coes[1][  0][ 30] =+2.699536597962134e+01;
  NLX_coes[1][  0][ 31] =+2.676421223916923e+01;
  NLX_coes[1][  0][ 32] =+2.650379301702800e+01;
  NLX_coes[1][  0][ 33] =+2.621651671946891e+01;
  NLX_coes[1][  0][ 34] =+2.588428358368893e+01;
  NLX_coes[1][  0][ 35] =+2.551489580931660e+01;
  NLX_coes[1][  0][ 36] =+2.509789869010296e+01;
  NLX_coes[1][  0][ 37] =+2.463372853658335e+01;
  NLX_coes[1][  0][ 38] =+2.412570150969221e+01;
  NLX_coes[1][  0][ 39] =+2.357935345895826e+01;
  NLX_coes[1][  0][ 40] =+2.300470729367903e+01;
  NLX_coes[1][  1][  0] =+3.253214186760952e+01;
  NLX_coes[1][  1][  1] =+3.191055759799314e+01;
  NLX_coes[1][  1][  2] =+3.128586964702736e+01;
  NLX_coes[1][  1][  3] =+3.065744018375716e+01;
  NLX_coes[1][  1][  4] =+3.002493359434403e+01;
  NLX_coes[1][  1][  5] =+2.938852139505850e+01;
  NLX_coes[1][  1][  6] =+2.874881688119394e+01;
  NLX_coes[1][  1][  7] =+2.810696015982019e+01;
  NLX_coes[1][  1][  8] =+2.746487000589865e+01;
  NLX_coes[1][  1][  9] =+2.682565777202961e+01;
  NLX_coes[1][  1][ 10] =+2.619383001707181e+01;
  NLX_coes[1][  1][ 11] =+2.557437262131374e+01;
  NLX_coes[1][  1][ 12] =+2.497190936587052e+01;
  NLX_coes[1][  1][ 13] =+2.439140297718761e+01;
  NLX_coes[1][  1][ 14] =+2.383838916300470e+01;
  NLX_coes[1][  1][ 15] =+2.332082055223169e+01;
  NLX_coes[1][  1][ 16] =+2.284886219565857e+01;
  NLX_coes[1][  1][ 17] =+2.243425513856220e+01;
  NLX_coes[1][  1][ 18] =+2.208939908803936e+01;
  NLX_coes[1][  1][ 19] =+2.182747815052794e+01;
  NLX_coes[1][  1][ 20] =+2.166208913498565e+01;
  NLX_coes[1][  1][ 21] =+2.159988402575951e+01;
  NLX_coes[1][  1][ 22] =+2.163329443198131e+01;
  NLX_coes[1][  1][ 23] =+2.174038429103567e+01;
  NLX_coes[1][  1][ 24] =+2.188380908954428e+01;
  NLX_coes[1][  1][ 25] =+2.200236602886206e+01;
  NLX_coes[1][  1][ 26] =+2.202444746785917e+01;
  NLX_coes[1][  1][ 27] =+2.192637306346084e+01;
  NLX_coes[1][  1][ 28] =+2.172723411549974e+01;
  NLX_coes[1][  1][ 29] =+2.148568876145780e+01;
  NLX_coes[1][  1][ 30] =+2.128369719412384e+01;
  NLX_coes[1][  1][ 31] =+2.103615387902867e+01;
  NLX_coes[1][  1][ 32] =+2.074311879139371e+01;
  NLX_coes[1][  1][ 33] =+2.038451792231752e+01;
  NLX_coes[1][  1][ 34] =+1.999077570364000e+01;
  NLX_coes[1][  1][ 35] =+1.955403752899738e+01;
  NLX_coes[1][  1][ 36] =+1.907790598973590e+01;
  NLX_coes[1][  1][ 37] =+1.856374558820279e+01;
  NLX_coes[1][  1][ 38] =+1.801601774136501e+01;
  NLX_coes[1][  1][ 39] =+1.744124593721914e+01;
  NLX_coes[1][  1][ 40] =+1.684624027110913e+01;
  NLX_coes[1][  2][  0] =+2.646058914132556e+01;
  NLX_coes[1][  2][  1] =+2.584265060346078e+01;
  NLX_coes[1][  2][  2] =+2.522305874284588e+01;
  NLX_coes[1][  2][  3] =+2.460123975459344e+01;
  NLX_coes[1][  2][  4] =+2.397705684653461e+01;
  NLX_coes[1][  2][  5] =+2.335072712703499e+01;
  NLX_coes[1][  2][  6] =+2.272277122614830e+01;
  NLX_coes[1][  2][  7] =+2.209402730901134e+01;
  NLX_coes[1][  2][  8] =+2.146584161756788e+01;
  NLX_coes[1][  2][  9] =+2.084071019529564e+01;
  NLX_coes[1][  2][ 10] =+2.022369267747033e+01;
  NLX_coes[1][  2][ 11] =+1.962098026577726e+01;
  NLX_coes[1][  2][ 12] =+1.903656879431021e+01;
  NLX_coes[1][  2][ 13] =+1.847371889009080e+01;
  NLX_coes[1][  2][ 14] =+1.793610698516264e+01;
  NLX_coes[1][  2][ 15] =+1.743135108554281e+01;
  NLX_coes[1][  2][ 16] =+1.696918945355791e+01;
  NLX_coes[1][  2][ 17] =+1.656141135234562e+01;
  NLX_coes[1][  2][ 18] =+1.621894326107216e+01;
  NLX_coes[1][  2][ 19] =+1.595205385589668e+01;
  NLX_coes[1][  2][ 20] =+1.577877968095148e+01;
  NLX_coes[1][  2][ 21] =+1.571483158186338e+01;
  NLX_coes[1][  2][ 22] =+1.575995234234940e+01;
  NLX_coes[1][  2][ 23] =+1.588460288379681e+01;
  NLX_coes[1][  2][ 24] =+1.607499036661062e+01;
  NLX_coes[1][  2][ 25] =+1.627415520957279e+01;
  NLX_coes[1][  2][ 26] =+1.635965959892387e+01;
  NLX_coes[1][  2][ 27] =+1.624929122970036e+01;
  NLX_coes[1][  2][ 28] =+1.608680111835358e+01;
  NLX_coes[1][  2][ 29] =+1.591670217739031e+01;
  NLX_coes[1][  2][ 30] =+1.567560776071611e+01;
  NLX_coes[1][  2][ 31] =+1.533835056573467e+01;
  NLX_coes[1][  2][ 32] =+1.495983537077894e+01;
  NLX_coes[1][  2][ 33] =+1.454268545191571e+01;
  NLX_coes[1][  2][ 34] =+1.407129595773934e+01;
  NLX_coes[1][  2][ 35] =+1.356604860974470e+01;
  NLX_coes[1][  2][ 36] =+1.302745136443052e+01;
  NLX_coes[1][  2][ 37] =+1.246074009700065e+01;
  NLX_coes[1][  2][ 38] =+1.187127606662413e+01;
  NLX_coes[1][  2][ 39] =+1.126464623511290e+01;
  NLX_coes[1][  2][ 40] =+1.064723925147879e+01;
  NLX_coes[1][  3][  0] =+2.037797108620724e+01;
  NLX_coes[1][  3][  1] =+1.976432069264757e+01;
  NLX_coes[1][  3][  2] =+1.915005989133650e+01;
  NLX_coes[1][  3][  3] =+1.853502746807746e+01;
  NLX_coes[1][  3][  4] =+1.791936221224629e+01;
  NLX_coes[1][  3][  5] =+1.730345947068533e+01;
  NLX_coes[1][  3][  6] =+1.668791057825267e+01;
  NLX_coes[1][  3][  7] =+1.607338843188621e+01;
  NLX_coes[1][  3][  8] =+1.546052955463586e+01;
  NLX_coes[1][  3][  9] =+1.485034805401931e+01;
  NLX_coes[1][  3][ 10] =+1.424755968942851e+01;
  NLX_coes[1][  3][ 11] =+1.366267495968945e+01;
  NLX_coes[1][  3][ 12] =+1.310140196031989e+01;
  NLX_coes[1][  3][ 13] =+1.255875121194411e+01;
  NLX_coes[1][  3][ 14] =+1.203962752341747e+01;
  NLX_coes[1][  3][ 15] =+1.155129162726512e+01;
  NLX_coes[1][  3][ 16] =+1.110238318563302e+01;
  NLX_coes[1][  3][ 17] =+1.070591031426430e+01;
  NLX_coes[1][  3][ 18] =+1.037243759843165e+01;
  NLX_coes[1][  3][ 19] =+1.010302666744987e+01;
  NLX_coes[1][  3][ 20] =+9.907026682621495e+00;
  NLX_coes[1][  3][ 21] =+9.823891737077954e+00;
  NLX_coes[1][  3][ 22] =+9.878474200203197e+00;
  NLX_coes[1][  3][ 23] =+9.994552631380243e+00;
  NLX_coes[1][  3][ 24] =+1.019847357105895e+01;
  NLX_coes[1][  3][ 25] =+1.047283206482656e+01;
  NLX_coes[1][  3][ 26] =+1.068504778422037e+01;
  NLX_coes[1][  3][ 27] =+1.067305773588804e+01;
  NLX_coes[1][  3][ 28] =+1.056649540242995e+01;
  NLX_coes[1][  3][ 29] =+1.039507461622766e+01;
  NLX_coes[1][  3][ 30] =+1.006406492330813e+01;
  NLX_coes[1][  3][ 31] =+9.661573122732770e+00;
  NLX_coes[1][  3][ 32] =+9.191848262400313e+00;
  NLX_coes[1][  3][ 33] =+8.669903386471272e+00;
  NLX_coes[1][  3][ 34] =+8.112307959723278e+00;
  NLX_coes[1][  3][ 35] =+7.526400284445382e+00;
  NLX_coes[1][  3][ 36] =+6.920478244198174e+00;
  NLX_coes[1][  3][ 37] =+6.299234422245092e+00;
  NLX_coes[1][  3][ 38] =+5.666786853255743e+00;
  NLX_coes[1][  3][ 39] =+5.027005442597626e+00;
  NLX_coes[1][  3][ 40] =+4.383598442958999e+00;
  NLX_coes[1][  4][  0] =+1.428031396996780e+01;
  NLX_coes[1][  4][  1] =+1.367117932755620e+01;
  NLX_coes[1][  4][  2] =+1.306233145003531e+01;
  NLX_coes[1][  4][  3] =+1.245406351913095e+01;
  NLX_coes[1][  4][  4] =+1.184685629407664e+01;
  NLX_coes[1][  4][  5] =+1.124139277720817e+01;
  NLX_coes[1][  4][  6] =+1.063853806419534e+01;
  NLX_coes[1][  4][  7] =+1.003924343930702e+01;
  NLX_coes[1][  4][  8] =+9.444170132283894e+00;
  NLX_coes[1][  4][  9] =+8.852578460777274e+00;
  NLX_coes[1][  4][ 10] =+8.261361060850314e+00;
  NLX_coes[1][  4][ 11] =+7.692830154819413e+00;
  NLX_coes[1][  4][ 12] =+7.166380799622500e+00;
  NLX_coes[1][  4][ 13] =+6.642322409366135e+00;
  NLX_coes[1][  4][ 14] =+6.145947088533454e+00;
  NLX_coes[1][  4][ 15] =+5.680998224936219e+00;
  NLX_coes[1][  4][ 16] =+5.252034380590458e+00;
  NLX_coes[1][  4][ 17] =+4.871333401433299e+00;
  NLX_coes[1][  4][ 18] =+4.565013309451770e+00;
  NLX_coes[1][  4][ 19] =+4.330221326913430e+00;
  NLX_coes[1][  4][ 20] =+4.091366310129726e+00;
  NLX_coes[1][  4][ 21] =+3.955121886602946e+00;
  NLX_coes[1][  4][ 22] =+3.986391208140598e+00;
  NLX_coes[1][  4][ 23] =+4.143631818846109e+00;
  NLX_coes[1][  4][ 24] =+4.273928343382027e+00;
  NLX_coes[1][  4][ 25] =+4.459218015431055e+00;
  NLX_coes[1][  4][ 26] =+4.636862988147875e+00;
  NLX_coes[1][  4][ 27] =+4.850504979740205e+00;
  NLX_coes[1][  4][ 28] =+4.976714504669078e+00;
  NLX_coes[1][  4][ 29] =+4.823493979783034e+00;
  NLX_coes[1][  4][ 30] =+4.438770184481709e+00;
  NLX_coes[1][  4][ 31] =+3.928292909027446e+00;
  NLX_coes[1][  4][ 32] =+3.344840922686783e+00;
  NLX_coes[1][  4][ 33] =+2.717448969869152e+00;
  NLX_coes[1][  4][ 34] =+2.065664027194263e+00;
  NLX_coes[1][  4][ 35] =+1.399803776593599e+00;
  NLX_coes[1][  4][ 36] =+7.273451809542616e-01;
  NLX_coes[1][  4][ 37] =+5.248206802255515e-02;
  NLX_coes[1][  4][ 38] =-6.221468408773887e-01;
  NLX_coes[1][  4][ 39] =-1.294857312509402e+00;
  NLX_coes[1][  4][ 40] =-1.964919880034165e+00;
  NLX_coes[1][  5][  0] =+8.163606314650540e+00;
  NLX_coes[1][  5][  1] =+7.558887014307927e+00;
  NLX_coes[1][  5][  2] =+6.955252504221261e+00;
  NLX_coes[1][  5][  3] =+6.353420946658123e+00;
  NLX_coes[1][  5][  4] =+5.754223706931986e+00;
  NLX_coes[1][  5][  5] =+5.158651496234638e+00;
  NLX_coes[1][  5][  6] =+4.567875657707297e+00;
  NLX_coes[1][  5][  7] =+3.983265125439129e+00;
  NLX_coes[1][  5][  8] =+3.406439206145861e+00;
  NLX_coes[1][  5][  9] =+2.838877486892506e+00;
  NLX_coes[1][  5][ 10] =+2.275788646317864e+00;
  NLX_coes[1][  5][ 11] =+1.731800529647359e+00;
  NLX_coes[1][  5][ 12] =+1.208939085314469e+00;
  NLX_coes[1][  5][ 13] =+7.218380048543085e-01;
  NLX_coes[1][  5][ 14] =+2.554353682985126e-01;
  NLX_coes[1][  5][ 15] =-1.774115292683141e-01;
  NLX_coes[1][  5][ 16] =-5.722410292774062e-01;
  NLX_coes[1][  5][ 17] =-9.266153079300277e-01;
  NLX_coes[1][  5][ 18] =-1.183664387611132e+00;
  NLX_coes[1][  5][ 19] =-1.341283140524751e+00;
  NLX_coes[1][  5][ 20] =-1.509315948725631e+00;
  NLX_coes[1][  5][ 21] =-1.697737897448824e+00;
  NLX_coes[1][  5][ 22] =-1.857690848376863e+00;
  NLX_coes[1][  5][ 23] =-1.917919275765736e+00;
  NLX_coes[1][  5][ 24] =-1.816621992554556e+00;
  NLX_coes[1][  5][ 25] =-1.569299378240279e+00;
  NLX_coes[1][  5][ 26] =-1.181000232616356e+00;
  NLX_coes[1][  5][ 27] =-7.044126736312339e-01;
  NLX_coes[1][  5][ 28] =-5.495184946689688e-01;
  NLX_coes[1][  5][ 29] =-8.119441515153270e-01;
  NLX_coes[1][  5][ 30] =-1.306882533239526e+00;
  NLX_coes[1][  5][ 31] =-1.937601396221067e+00;
  NLX_coes[1][  5][ 32] =-2.637553351668145e+00;
  NLX_coes[1][  5][ 33] =-3.368565107971391e+00;
  NLX_coes[1][  5][ 34] =-4.109535688267794e+00;
  NLX_coes[1][  5][ 35] =-4.849413074838931e+00;
  NLX_coes[1][  5][ 36] =-5.582422601103232e+00;
  NLX_coes[1][  5][ 37] =-6.305818494043346e+00;
  NLX_coes[1][  5][ 38] =-7.018751317630626e+00;
  NLX_coes[1][  5][ 39] =-7.721717144206576e+00;
  NLX_coes[1][  5][ 40] =-8.416656751789866e+00;
  NLX_coes[1][  6][  0] =+2.023882728312727e+00;
  NLX_coes[1][  6][  1] =+1.423182898161009e+00;
  NLX_coes[1][  6][  2] =+8.242616124634629e-01;
  NLX_coes[1][  6][  3] =+2.282053496292417e-01;
  NLX_coes[1][  6][  4] =-3.638496839907372e-01;
  NLX_coes[1][  6][  5] =-9.506587125532117e-01;
  NLX_coes[1][  6][  6] =-1.530830153211819e+00;
  NLX_coes[1][  6][  7] =-2.102831965426995e+00;
  NLX_coes[1][  6][  8] =-2.665141716220925e+00;
  NLX_coes[1][  6][  9] =-3.216545908150128e+00;
  NLX_coes[1][  6][ 10] =-3.754939989440578e+00;
  NLX_coes[1][  6][ 11] =-4.279146922155687e+00;
  NLX_coes[1][  6][ 12] =-4.787961977196730e+00;
  NLX_coes[1][  6][ 13] =-5.259614247253478e+00;
  NLX_coes[1][  6][ 14] =-5.703073163331437e+00;
  NLX_coes[1][  6][ 15] =-6.109690732295087e+00;
  NLX_coes[1][  6][ 16] =-6.464157217440068e+00;
  NLX_coes[1][  6][ 17] =-6.737462320616108e+00;
  NLX_coes[1][  6][ 18] =-6.885823926150665e+00;
  NLX_coes[1][  6][ 19] =-6.933170309706870e+00;
  NLX_coes[1][  6][ 20] =-6.957261689360363e+00;
  NLX_coes[1][  6][ 21] =-7.262700954032274e+00;
  NLX_coes[1][  6][ 22] =-7.372673311880793e+00;
  NLX_coes[1][  6][ 23] =-7.384361154564018e+00;
  NLX_coes[1][  6][ 24] =-7.309352187947329e+00;
  NLX_coes[1][  6][ 25] =-6.721491042818786e+00;
  NLX_coes[1][  6][ 26] =-6.422288488462145e+00;
  NLX_coes[1][  6][ 27] =-6.259330674471381e+00;
  NLX_coes[1][  6][ 28] =-6.237093880523780e+00;
  NLX_coes[1][  6][ 29] =-6.617265744601984e+00;
  NLX_coes[1][  6][ 30] =-7.253316568151649e+00;
  NLX_coes[1][  6][ 31] =-8.011026106748856e+00;
  NLX_coes[1][  6][ 32] =-8.818139468091271e+00;
  NLX_coes[1][  6][ 33] =-9.638366561509935e+00;
  NLX_coes[1][  6][ 34] =-1.045328283704383e+01;
  NLX_coes[1][  6][ 35] =-1.125378020361709e+01;
  NLX_coes[1][  6][ 36] =-1.203594475918023e+01;
  NLX_coes[1][  6][ 37] =-1.279874794272283e+01;
  NLX_coes[1][  6][ 38] =-1.354302242539384e+01;
  NLX_coes[1][  6][ 39] =-1.427109656875198e+01;
  NLX_coes[1][  6][ 40] =-1.498700610546811e+01;
  NLX_coes[1][  7][  0] =-4.142694551848274e+00;
  NLX_coes[1][  7][  1] =-4.740028825127355e+00;
  NLX_coes[1][  7][  2] =-5.335006068390241e+00;
  NLX_coes[1][  7][  3] =-5.926239380555170e+00;
  NLX_coes[1][  7][  4] =-6.512359485028830e+00;
  NLX_coes[1][  7][  5] =-7.091963035176864e+00;
  NLX_coes[1][  7][  6] =-7.663590733956225e+00;
  NLX_coes[1][  7][  7] =-8.225751451090842e+00;
  NLX_coes[1][  7][  8] =-8.777009551795924e+00;
  NLX_coes[1][  7][  9] =-9.316110009514038e+00;
  NLX_coes[1][  7][ 10] =-9.841433080874312e+00;
  NLX_coes[1][  7][ 11] =-1.035191641043882e+01;
  NLX_coes[1][  7][ 12] =-1.084312870380853e+01;
  NLX_coes[1][  7][ 13] =-1.130587982426313e+01;
  NLX_coes[1][  7][ 14] =-1.173629313138919e+01;
  NLX_coes[1][  7][ 15] =-1.212536614642628e+01;
  NLX_coes[1][  7][ 16] =-1.245319625608304e+01;
  NLX_coes[1][  7][ 17] =-1.268439320086717e+01;
  NLX_coes[1][  7][ 18] =-1.277575041675360e+01;
  NLX_coes[1][  7][ 19] =-1.272965536455930e+01;
  NLX_coes[1][  7][ 20] =-1.262908207042739e+01;
  NLX_coes[1][  7][ 21] =-1.269077569082683e+01;
  NLX_coes[1][  7][ 22] =-1.273548262954374e+01;
  NLX_coes[1][  7][ 23] =-1.288155520536499e+01;
  NLX_coes[1][  7][ 24] =-1.291244180430767e+01;
  NLX_coes[1][  7][ 25] =-1.265360819101473e+01;
  NLX_coes[1][  7][ 26] =-1.205955558698995e+01;
  NLX_coes[1][  7][ 27] =-1.177424475041959e+01;
  NLX_coes[1][  7][ 28] =-1.207907553891010e+01;
  NLX_coes[1][  7][ 29] =-1.272032597774551e+01;
  NLX_coes[1][  7][ 30] =-1.350870459516877e+01;
  NLX_coes[1][  7][ 31] =-1.436787557202315e+01;
  NLX_coes[1][  7][ 32] =-1.525168749778372e+01;
  NLX_coes[1][  7][ 33] =-1.613305279431438e+01;
  NLX_coes[1][  7][ 34] =-1.699766996680647e+01;
  NLX_coes[1][  7][ 35] =-1.783881597505235e+01;
  NLX_coes[1][  7][ 36] =-1.865414618802404e+01;
  NLX_coes[1][  7][ 37] =-1.944389269940318e+01;
  NLX_coes[1][  7][ 38] =-2.021003174496692e+01;
  NLX_coes[1][  7][ 39] =-2.095608276032642e+01;
  NLX_coes[1][  7][ 40] =-2.168741021312208e+01;
  NLX_coes[1][  8][  0] =-1.033975670612302e+01;
  NLX_coes[1][  8][  1] =-1.093459345257499e+01;
  NLX_coes[1][  8][  2] =-1.152661051937846e+01;
  NLX_coes[1][  8][  3] =-1.211418718489326e+01;
  NLX_coes[1][  8][  4] =-1.269578877826007e+01;
  NLX_coes[1][  8][  5] =-1.326992490776456e+01;
  NLX_coes[1][  8][  6] =-1.383513695885378e+01;
  NLX_coes[1][  8][  7] =-1.439002007673153e+01;
  NLX_coes[1][  8][  8] =-1.493326469303056e+01;
  NLX_coes[1][  8][  9] =-1.546362448145718e+01;
  NLX_coes[1][  8][ 10] =-1.597974533897527e+01;
  NLX_coes[1][  8][ 11] =-1.647978383984815e+01;
  NLX_coes[1][  8][ 12] =-1.695974234654574e+01;
  NLX_coes[1][  8][ 13] =-1.741439393463466e+01;
  NLX_coes[1][  8][ 14] =-1.783736183307906e+01;
  NLX_coes[1][  8][ 15] =-1.821957377255375e+01;
  NLX_coes[1][  8][ 16] =-1.854445585286789e+01;
  NLX_coes[1][  8][ 17] =-1.878330790818260e+01;
  NLX_coes[1][  8][ 18] =-1.888982786461280e+01;
  NLX_coes[1][  8][ 19] =-1.880784274898708e+01;
  NLX_coes[1][  8][ 20] =-1.866161794139612e+01;
  NLX_coes[1][  8][ 21] =-1.855318777602763e+01;
  NLX_coes[1][  8][ 22] =-1.853182857543388e+01;
  NLX_coes[1][  8][ 23] =-1.872899806155253e+01;
  NLX_coes[1][  8][ 24] =-1.885734952756407e+01;
  NLX_coes[1][  8][ 25] =-1.884674641736290e+01;
  NLX_coes[1][  8][ 26] =-1.841524244108985e+01;
  NLX_coes[1][  8][ 27] =-1.821086916270558e+01;
  NLX_coes[1][  8][ 28] =-1.858068521957949e+01;
  NLX_coes[1][  8][ 29] =-1.930449296079992e+01;
  NLX_coes[1][  8][ 30] =-2.015693985370590e+01;
  NLX_coes[1][  8][ 31] =-2.105899818640151e+01;
  NLX_coes[1][  8][ 32] =-2.197292263786881e+01;
  NLX_coes[1][  8][ 33] =-2.287766995663902e+01;
  NLX_coes[1][  8][ 34] =-2.376173339331997e+01;
  NLX_coes[1][  8][ 35] =-2.461956520287900e+01;
  NLX_coes[1][  8][ 36] =-2.544933564744731e+01;
  NLX_coes[1][  8][ 37] =-2.625159781212709e+01;
  NLX_coes[1][  8][ 38] =-2.702863543633862e+01;
  NLX_coes[1][  8][ 39] =-2.778433998674025e+01;
  NLX_coes[1][  8][ 40] =-2.852453063321816e+01;
  NLX_coes[1][  9][  0] =-1.657066504305729e+01;
  NLX_coes[1][  9][  1] =-1.716404160607031e+01;
  NLX_coes[1][  9][  2] =-1.775424021260956e+01;
  NLX_coes[1][  9][  3] =-1.833946228602608e+01;
  NLX_coes[1][  9][  4] =-1.891805527102955e+01;
  NLX_coes[1][  9][  5] =-1.948847608878048e+01;
  NLX_coes[1][  9][  6] =-2.004927734322287e+01;
  NLX_coes[1][  9][  7] =-2.059911058092534e+01;
  NLX_coes[1][  9][  8] =-2.113672105904907e+01;
  NLX_coes[1][  9][  9] =-2.166087127734084e+01;
  NLX_coes[1][  9][ 10] =-2.217016961021065e+01;
  NLX_coes[1][  9][ 11] =-2.266260092377047e+01;
  NLX_coes[1][  9][ 12] =-2.313500977579577e+01;
  NLX_coes[1][  9][ 13] =-2.358322329709589e+01;
  NLX_coes[1][  9][ 14] =-2.400142047730606e+01;
  NLX_coes[1][  9][ 15] =-2.438155910240263e+01;
  NLX_coes[1][  9][ 16] =-2.471230567512421e+01;
  NLX_coes[1][  9][ 17] =-2.498004737066802e+01;
  NLX_coes[1][  9][ 18] =-2.515910074305533e+01;
  NLX_coes[1][  9][ 19] =-2.508126185591641e+01;
  NLX_coes[1][  9][ 20] =-2.501015073119295e+01;
  NLX_coes[1][  9][ 21] =-2.499515728971552e+01;
  NLX_coes[1][  9][ 22] =-2.505486232696663e+01;
  NLX_coes[1][  9][ 23] =-2.515440354710724e+01;
  NLX_coes[1][  9][ 24] =-2.529637913777690e+01;
  NLX_coes[1][  9][ 25] =-2.532921449104333e+01;
  NLX_coes[1][  9][ 26] =-2.525070499036827e+01;
  NLX_coes[1][  9][ 27] =-2.534038167487731e+01;
  NLX_coes[1][  9][ 28] =-2.574597703281297e+01;
  NLX_coes[1][  9][ 29] =-2.641176060357509e+01;
  NLX_coes[1][  9][ 30] =-2.722231430639898e+01;
  NLX_coes[1][  9][ 31] =-2.809529887890560e+01;
  NLX_coes[1][  9][ 32] =-2.898738743801770e+01;
  NLX_coes[1][  9][ 33] =-2.987582300679468e+01;
  NLX_coes[1][  9][ 34] =-3.074832069728025e+01;
  NLX_coes[1][  9][ 35] =-3.159853669519902e+01;
  NLX_coes[1][  9][ 36] =-3.242381949180903e+01;
  NLX_coes[1][  9][ 37] =-3.322399426587290e+01;
  NLX_coes[1][  9][ 38] =-3.400076126096373e+01;
  NLX_coes[1][  9][ 39] =-3.475755218457469e+01;
  NLX_coes[1][  9][ 40] =-3.549977906694893e+01;
  NLX_coes[1][ 10][  0] =-2.283846691377952e+01;
  NLX_coes[1][ 10][  1] =-2.343155013937888e+01;
  NLX_coes[1][ 10][  2] =-2.402118426957923e+01;
  NLX_coes[1][ 10][  3] =-2.460543678778945e+01;
  NLX_coes[1][ 10][  4] =-2.518256673468134e+01;
  NLX_coes[1][ 10][  5] =-2.575098762093925e+01;
  NLX_coes[1][ 10][  6] =-2.630924514115888e+01;
  NLX_coes[1][ 10][  7] =-2.685599980353227e+01;
  NLX_coes[1][ 10][  8] =-2.738999135936800e+01;
  NLX_coes[1][ 10][  9] =-2.790995102093575e+01;
  NLX_coes[1][ 10][ 10] =-2.841442858170257e+01;
  NLX_coes[1][ 10][ 11] =-2.890149029271563e+01;
  NLX_coes[1][ 10][ 12] =-2.936846978186252e+01;
  NLX_coes[1][ 10][ 13] =-2.981173168443716e+01;
  NLX_coes[1][ 10][ 14] =-3.022621015505735e+01;
  NLX_coes[1][ 10][ 15] =-3.060469095629459e+01;
  NLX_coes[1][ 10][ 16] =-3.093614071705695e+01;
  NLX_coes[1][ 10][ 17] =-3.120108721670292e+01;
  NLX_coes[1][ 10][ 18] =-3.136265725773212e+01;
  NLX_coes[1][ 10][ 19] =-3.138503807879393e+01;
  NLX_coes[1][ 10][ 20] =-3.146919038366082e+01;
  NLX_coes[1][ 10][ 21] =-3.159483843622063e+01;
  NLX_coes[1][ 10][ 22] =-3.176346454869524e+01;
  NLX_coes[1][ 10][ 23] =-3.194865676535591e+01;
  NLX_coes[1][ 10][ 24] =-3.214797382070257e+01;
  NLX_coes[1][ 10][ 25] =-3.236552726743425e+01;
  NLX_coes[1][ 10][ 26] =-3.256427883100024e+01;
  NLX_coes[1][ 10][ 27] =-3.285452369665868e+01;
  NLX_coes[1][ 10][ 28] =-3.330686910996761e+01;
  NLX_coes[1][ 10][ 29] =-3.391569309413060e+01;
  NLX_coes[1][ 10][ 30] =-3.464285549951240e+01;
  NLX_coes[1][ 10][ 31] =-3.543963926248298e+01;
  NLX_coes[1][ 10][ 32] =-3.626955418076874e+01;
  NLX_coes[1][ 10][ 33] =-3.710906839969478e+01;
  NLX_coes[1][ 10][ 34] =-3.794385845603864e+01;
  NLX_coes[1][ 10][ 35] =-3.876558729387045e+01;
  NLX_coes[1][ 10][ 36] =-3.956985114492237e+01;
  NLX_coes[1][ 10][ 37] =-4.035497996325019e+01;
  NLX_coes[1][ 10][ 38] =-4.112140136908105e+01;
  NLX_coes[1][ 10][ 39] =-4.187141371748265e+01;
  NLX_coes[1][ 10][ 40] =-4.260929965926319e+01;
  NLX_coes[1][ 11][  0] =-2.914586722410859e+01;
  NLX_coes[1][ 11][  1] =-2.973992153528662e+01;
  NLX_coes[1][ 11][  2] =-3.033032742721297e+01;
  NLX_coes[1][ 11][  3] =-3.091505410273376e+01;
  NLX_coes[1][ 11][  4] =-3.149229241364673e+01;
  NLX_coes[1][ 11][  5] =-3.206041353526146e+01;
  NLX_coes[1][ 11][  6] =-3.261793634837385e+01;
  NLX_coes[1][ 11][  7] =-3.316349413783347e+01;
  NLX_coes[1][ 11][  8] =-3.369578438378112e+01;
  NLX_coes[1][ 11][  9] =-3.421348263838230e+01;
  NLX_coes[1][ 11][ 10] =-3.471510173909777e+01;
  NLX_coes[1][ 11][ 11] =-3.519880284388560e+01;
  NLX_coes[1][ 11][ 12] =-3.566219830790135e+01;
  NLX_coes[1][ 11][ 13] =-3.610206477247274e+01;
  NLX_coes[1][ 11][ 14] =-3.651385505049218e+01;
  NLX_coes[1][ 11][ 15] =-3.689074506176451e+01;
  NLX_coes[1][ 11][ 16] =-3.722198760911221e+01;
  NLX_coes[1][ 11][ 17] =-3.749116890732268e+01;
  NLX_coes[1][ 11][ 18] =-3.767980075145935e+01;
  NLX_coes[1][ 11][ 19] =-3.780484733947823e+01;
  NLX_coes[1][ 11][ 20] =-3.793607821769054e+01;
  NLX_coes[1][ 11][ 21] =-3.811196862769429e+01;
  NLX_coes[1][ 11][ 22] =-3.836084863044560e+01;
  NLX_coes[1][ 11][ 23] =-3.870526216487712e+01;
  NLX_coes[1][ 11][ 24] =-3.914079694842452e+01;
  NLX_coes[1][ 11][ 25] =-3.967661265431803e+01;
  NLX_coes[1][ 11][ 26] =-4.025011537805065e+01;
  NLX_coes[1][ 11][ 27] =-4.067144287870791e+01;
  NLX_coes[1][ 11][ 28] =-4.113312092984420e+01;
  NLX_coes[1][ 11][ 29] =-4.168068467950742e+01;
  NLX_coes[1][ 11][ 30] =-4.231699330748140e+01;
  NLX_coes[1][ 11][ 31] =-4.302150578768423e+01;
  NLX_coes[1][ 11][ 32] =-4.376966589504671e+01;
  NLX_coes[1][ 11][ 33] =-4.454111302304747e+01;
  NLX_coes[1][ 11][ 34] =-4.532111648852442e+01;
  NLX_coes[1][ 11][ 35] =-4.609978136919067e+01;
  NLX_coes[1][ 11][ 36] =-4.687092002345035e+01;
  NLX_coes[1][ 11][ 37] =-4.763115020871116e+01;
  NLX_coes[1][ 11][ 38] =-4.837931171062269e+01;
  NLX_coes[1][ 11][ 39] =-4.911617548160871e+01;
  NLX_coes[1][ 11][ 40] =-4.984440958069614e+01;
  NLX_coes[1][ 12][  0] =-3.549520864676349e+01;
  NLX_coes[1][ 12][  1] =-3.609157111803393e+01;
  NLX_coes[1][ 12][  2] =-3.668414910700278e+01;
  NLX_coes[1][ 12][  3] =-3.727084638653469e+01;
  NLX_coes[1][ 12][  4] =-3.784980540558607e+01;
  NLX_coes[1][ 12][  5] =-3.841936032885841e+01;
  NLX_coes[1][ 12][  6] =-3.897799536808433e+01;
  NLX_coes[1][ 12][  7] =-3.952430128647906e+01;
  NLX_coes[1][ 12][  8] =-4.005692019549635e+01;
  NLX_coes[1][ 12][  9] =-4.057446891532251e+01;
  NLX_coes[1][ 12][ 10] =-4.107543528250056e+01;
  NLX_coes[1][ 12][ 11] =-4.155805516260091e+01;
  NLX_coes[1][ 12][ 12] =-4.202016923314812e+01;
  NLX_coes[1][ 12][ 13] =-4.245902134916968e+01;
  NLX_coes[1][ 12][ 14] =-4.287093940767209e+01;
  NLX_coes[1][ 12][ 15] =-4.325081579202593e+01;
  NLX_coes[1][ 12][ 16] =-4.359148801663900e+01;
  NLX_coes[1][ 12][ 17] =-4.388389154524403e+01;
  NLX_coes[1][ 12][ 18] =-4.412121812664613e+01;
  NLX_coes[1][ 12][ 19] =-4.431193117400404e+01;
  NLX_coes[1][ 12][ 20] =-4.447747428009512e+01;
  NLX_coes[1][ 12][ 21] =-4.468903549551563e+01;
  NLX_coes[1][ 12][ 22] =-4.502542476836201e+01;
  NLX_coes[1][ 12][ 23] =-4.556622523839092e+01;
  NLX_coes[1][ 12][ 24] =-4.625007540783314e+01;
  NLX_coes[1][ 12][ 25] =-4.708738258227658e+01;
  NLX_coes[1][ 12][ 26] =-4.809030528321883e+01;
  NLX_coes[1][ 12][ 27] =-4.862651327363636e+01;
  NLX_coes[1][ 12][ 28] =-4.909278304935021e+01;
  NLX_coes[1][ 12][ 29] =-4.958829314630252e+01;
  NLX_coes[1][ 12][ 30] =-5.014485249155059e+01;
  NLX_coes[1][ 12][ 31] =-5.076205212873548e+01;
  NLX_coes[1][ 12][ 32] =-5.142724438180807e+01;
  NLX_coes[1][ 12][ 33] =-5.212560593405836e+01;
  NLX_coes[1][ 12][ 34] =-5.284412218092940e+01;
  NLX_coes[1][ 12][ 35] =-5.357268716298869e+01;
  NLX_coes[1][ 12][ 36] =-5.430405846920842e+01;
  NLX_coes[1][ 12][ 37] =-5.503347782847551e+01;
  NLX_coes[1][ 12][ 38] =-5.575828756268303e+01;
  NLX_coes[1][ 12][ 39] =-5.647765504543794e+01;
  NLX_coes[1][ 12][ 40] =-5.719242839699300e+01;
  NLX_coes[1][ 13][  0] =-4.188845422260966e+01;
  NLX_coes[1][ 13][  1] =-4.248851352301713e+01;
  NLX_coes[1][ 13][  2] =-4.308471490385055e+01;
  NLX_coes[1][ 13][  3] =-4.367493147618588e+01;
  NLX_coes[1][ 13][  4] =-4.425728268399168e+01;
  NLX_coes[1][ 13][  5] =-4.483008199808098e+01;
  NLX_coes[1][ 13][  6] =-4.539178869640117e+01;
  NLX_coes[1][ 13][  7] =-4.594095894433006e+01;
  NLX_coes[1][ 13][  8] =-4.647619096105482e+01;
  NLX_coes[1][ 13][  9] =-4.699606096837974e+01;
  NLX_coes[1][ 13][ 10] =-4.749905192116559e+01;
  NLX_coes[1][ 13][ 11] =-4.798348424049590e+01;
  NLX_coes[1][ 13][ 12] =-4.844745804304473e+01;
  NLX_coes[1][ 13][ 13] =-4.888882069199854e+01;
  NLX_coes[1][ 13][ 14] =-4.930519175440344e+01;
  NLX_coes[1][ 13][ 15] =-4.969413166160063e+01;
  NLX_coes[1][ 13][ 16] =-5.005363326741477e+01;
  NLX_coes[1][ 13][ 17] =-5.038303685459333e+01;
  NLX_coes[1][ 13][ 18] =-5.068332413613390e+01;
  NLX_coes[1][ 13][ 19] =-5.095064480938251e+01;
  NLX_coes[1][ 13][ 20] =-5.116560479265176e+01;
  NLX_coes[1][ 13][ 21] =-5.138725131053571e+01;
  NLX_coes[1][ 13][ 22] =-5.188356847682638e+01;
  NLX_coes[1][ 13][ 23] =-5.260680070980122e+01;
  NLX_coes[1][ 13][ 24] =-5.346940029604826e+01;
  NLX_coes[1][ 13][ 25] =-5.473765392665672e+01;
  NLX_coes[1][ 13][ 26] =-5.594825480167673e+01;
  NLX_coes[1][ 13][ 27] =-5.657851914319762e+01;
  NLX_coes[1][ 13][ 28] =-5.706742246702331e+01;
  NLX_coes[1][ 13][ 29] =-5.753855843533849e+01;
  NLX_coes[1][ 13][ 30] =-5.804053476598560e+01;
  NLX_coes[1][ 13][ 31] =-5.858894140872838e+01;
  NLX_coes[1][ 13][ 32] =-5.918281759450490e+01;
  NLX_coes[1][ 13][ 33] =-5.981427024462334e+01;
  NLX_coes[1][ 13][ 34] =-6.047370770036620e+01;
  NLX_coes[1][ 13][ 35] =-6.115225321704220e+01;
  NLX_coes[1][ 13][ 36] =-6.184264166096820e+01;
  NLX_coes[1][ 13][ 37] =-6.253940770233049e+01;
  NLX_coes[1][ 13][ 38] =-6.323879002985490e+01;
  NLX_coes[1][ 13][ 39] =-6.393854894361343e+01;
  NLX_coes[1][ 13][ 40] =-6.463777125490236e+01;
  NLX_coes[1][ 14][  0] =-4.832716896904061e+01;
  NLX_coes[1][ 14][  1] =-4.893234456905667e+01;
  NLX_coes[1][ 14][  2] =-4.953365704760871e+01;
  NLX_coes[1][ 14][  3] =-5.012898848123056e+01;
  NLX_coes[1][ 14][  4] =-5.071646920522200e+01;
  NLX_coes[1][ 14][  5] =-5.129442164995369e+01;
  NLX_coes[1][ 14][  6] =-5.186130851011071e+01;
  NLX_coes[1][ 14][  7] =-5.241568220257567e+01;
  NLX_coes[1][ 14][  8] =-5.295613350178173e+01;
  NLX_coes[1][ 14][  9] =-5.348124022235075e+01;
  NLX_coes[1][ 14][ 10] =-5.398952263938644e+01;
  NLX_coes[1][ 14][ 11] =-5.447942072701471e+01;
  NLX_coes[1][ 14][ 12] =-5.494932083050946e+01;
  NLX_coes[1][ 14][ 13] =-5.539768430781679e+01;
  NLX_coes[1][ 14][ 14] =-5.582337510070488e+01;
  NLX_coes[1][ 14][ 15] =-5.622635279504749e+01;
  NLX_coes[1][ 14][ 16] =-5.660895914498512e+01;
  NLX_coes[1][ 14][ 17] =-5.697791043955746e+01;
  NLX_coes[1][ 14][ 18] =-5.734591428594722e+01;
  NLX_coes[1][ 14][ 19] =-5.772547687815374e+01;
  NLX_coes[1][ 14][ 20] =-5.808501746612650e+01;
  NLX_coes[1][ 14][ 21] =-5.826630030013594e+01;
  NLX_coes[1][ 14][ 22] =-5.887540606006385e+01;
  NLX_coes[1][ 14][ 23] =-5.984915654003229e+01;
  NLX_coes[1][ 14][ 24] =-6.110178223359368e+01;
  NLX_coes[1][ 14][ 25] =-6.259444030373083e+01;
  NLX_coes[1][ 14][ 26] =-6.368931796926879e+01;
  NLX_coes[1][ 14][ 27] =-6.443999257393439e+01;
  NLX_coes[1][ 14][ 28] =-6.498736182517531e+01;
  NLX_coes[1][ 14][ 29] =-6.546454938546614e+01;
  NLX_coes[1][ 14][ 30] =-6.594010464352664e+01;
  NLX_coes[1][ 14][ 31] =-6.644379952875762e+01;
  NLX_coes[1][ 14][ 32] =-6.698498743600771e+01;
  NLX_coes[1][ 14][ 33] =-6.756285925022868e+01;
  NLX_coes[1][ 14][ 34] =-6.817217515963098e+01;
  NLX_coes[1][ 14][ 35] =-6.880637343223194e+01;
  NLX_coes[1][ 14][ 36] =-6.945912349525588e+01;
  NLX_coes[1][ 14][ 37] =-7.012499672000574e+01;
  NLX_coes[1][ 14][ 38] =-7.079966541991325e+01;
  NLX_coes[1][ 14][ 39] =-7.147985553263234e+01;
  NLX_coes[1][ 14][ 40] =-7.216315516159378e+01;
  NLX_coes[1][ 15][  0] =-5.481250041701297e+01;
  NLX_coes[1][ 15][  1] =-5.542421924335151e+01;
  NLX_coes[1][ 15][  2] =-5.603214667766936e+01;
  NLX_coes[1][ 15][  3] =-5.663421913682529e+01;
  NLX_coes[1][ 15][  4] =-5.722862041334152e+01;
  NLX_coes[1][ 15][  5] =-5.781372408300813e+01;
  NLX_coes[1][ 15][  6] =-5.838804130653769e+01;
  NLX_coes[1][ 15][  7] =-5.895017207635720e+01;
  NLX_coes[1][ 15][  8] =-5.949875960875998e+01;
  NLX_coes[1][ 15][  9] =-6.003245091477774e+01;
  NLX_coes[1][ 15][ 10] =-6.054987235947453e+01;
  NLX_coes[1][ 15][ 11] =-6.104963831263271e+01;
  NLX_coes[1][ 15][ 12] =-6.153042715137281e+01;
  NLX_coes[1][ 15][ 13] =-6.199118804583019e+01;
  NLX_coes[1][ 15][ 14] =-6.243159161427253e+01;
  NLX_coes[1][ 15][ 15] =-6.285290680137665e+01;
  NLX_coes[1][ 15][ 16] =-6.325952322315874e+01;
  NLX_coes[1][ 15][ 17] =-6.366115114044749e+01;
  NLX_coes[1][ 15][ 18] =-6.407496719562815e+01;
  NLX_coes[1][ 15][ 19] =-6.452553665273810e+01;
  NLX_coes[1][ 15][ 20] =-6.503201174393828e+01;
  NLX_coes[1][ 15][ 21] =-6.536051205744917e+01;
  NLX_coes[1][ 15][ 22] =-6.612225033421231e+01;
  NLX_coes[1][ 15][ 23] =-6.734779732898048e+01;
  NLX_coes[1][ 15][ 24] =-6.895855917679391e+01;
  NLX_coes[1][ 15][ 25] =-7.046189417312968e+01;
  NLX_coes[1][ 15][ 26] =-7.147941157585105e+01;
  NLX_coes[1][ 15][ 27] =-7.224457724827397e+01;
  NLX_coes[1][ 15][ 28] =-7.283143039360243e+01;
  NLX_coes[1][ 15][ 29] =-7.332977339244384e+01;
  NLX_coes[1][ 15][ 30] =-7.380230618761007e+01;
  NLX_coes[1][ 15][ 31] =-7.428471128110766e+01;
  NLX_coes[1][ 15][ 32] =-7.479369217584423e+01;
  NLX_coes[1][ 15][ 33] =-7.533454211512260e+01;
  NLX_coes[1][ 15][ 34] =-7.590641241980494e+01;
  NLX_coes[1][ 15][ 35] =-7.650557384910256e+01;
  NLX_coes[1][ 15][ 36] =-7.712728855745755e+01;
  NLX_coes[1][ 15][ 37] =-7.776679256717614e+01;
  NLX_coes[1][ 15][ 38] =-7.841972770024226e+01;
  NLX_coes[1][ 15][ 39] =-7.908222910552600e+01;
  NLX_coes[1][ 15][ 40] =-7.975077300577327e+01;
  NLX_coes[1][ 16][  0] =-6.134516050977822e+01;
  NLX_coes[1][ 16][  1] =-6.196482963218062e+01;
  NLX_coes[1][ 16][  2] =-6.258086430546294e+01;
  NLX_coes[1][ 16][  3] =-6.319130538597324e+01;
  NLX_coes[1][ 16][  4] =-6.379443937446037e+01;
  NLX_coes[1][ 16][  5] =-6.438874255570201e+01;
  NLX_coes[1][ 16][  6] =-6.497283198321648e+01;
  NLX_coes[1][ 16][  7] =-6.554542202094774e+01;
  NLX_coes[1][ 16][  8] =-6.610528703854192e+01;
  NLX_coes[1][ 16][  9] =-6.665123390671528e+01;
  NLX_coes[1][ 16][ 10] =-6.718209271853502e+01;
  NLX_coes[1][ 16][ 11] =-6.769674156354858e+01;
  NLX_coes[1][ 16][ 12] =-6.819419295438448e+01;
  NLX_coes[1][ 16][ 13] =-6.867378832620355e+01;
  NLX_coes[1][ 16][ 14] =-6.913557731057050e+01;
  NLX_coes[1][ 16][ 15] =-6.958100692491310e+01;
  NLX_coes[1][ 16][ 16] =-7.001411465220662e+01;
  NLX_coes[1][ 16][ 17] =-7.044344116610810e+01;
  NLX_coes[1][ 16][ 18] =-7.088442952932705e+01;
  NLX_coes[1][ 16][ 19] =-7.136019347713589e+01;
  NLX_coes[1][ 16][ 20] =-7.190146563697404e+01;
  NLX_coes[1][ 16][ 21] =-7.259858695619420e+01;
  NLX_coes[1][ 16][ 22] =-7.364458090434390e+01;
  NLX_coes[1][ 16][ 23] =-7.521863236041452e+01;
  NLX_coes[1][ 16][ 24] =-7.694030340232403e+01;
  NLX_coes[1][ 16][ 25] =-7.821959915882718e+01;
  NLX_coes[1][ 16][ 26] =-7.921601219509714e+01;
  NLX_coes[1][ 16][ 27] =-7.998459863968507e+01;
  NLX_coes[1][ 16][ 28] =-8.059726712269688e+01;
  NLX_coes[1][ 16][ 29] =-8.112004166482618e+01;
  NLX_coes[1][ 16][ 30] =-8.160444504438183e+01;
  NLX_coes[1][ 16][ 31] =-8.208468141590238e+01;
  NLX_coes[1][ 16][ 32] =-8.258043608100488e+01;
  NLX_coes[1][ 16][ 33] =-8.310108591691082e+01;
  NLX_coes[1][ 16][ 34] =-8.364946866527819e+01;
  NLX_coes[1][ 16][ 35] =-8.422466397005888e+01;
  NLX_coes[1][ 16][ 36] =-8.482382412427003e+01;
  NLX_coes[1][ 16][ 37] =-8.544326755068579e+01;
  NLX_coes[1][ 16][ 38] =-8.607904488862138e+01;
  NLX_coes[1][ 16][ 39] =-8.672712966641447e+01;
  NLX_coes[1][ 16][ 40] =-8.738331981681213e+01;
  NLX_coes[1][ 17][  0] =-6.792541259294715e+01;
  NLX_coes[1][ 17][  1] =-6.855438803281528e+01;
  NLX_coes[1][ 17][  2] =-6.917997599991409e+01;
  NLX_coes[1][ 17][  3] =-6.980037405801723e+01;
  NLX_coes[1][ 17][  4] =-7.041402379275468e+01;
  NLX_coes[1][ 17][  5] =-7.101956026305184e+01;
  NLX_coes[1][ 17][  6] =-7.161576988446278e+01;
  NLX_coes[1][ 17][  7] =-7.220155599187645e+01;
  NLX_coes[1][ 17][  8] =-7.277591317679493e+01;
  NLX_coes[1][ 17][  9] =-7.333791397309328e+01;
  NLX_coes[1][ 17][ 10] =-7.388671480547133e+01;
  NLX_coes[1][ 17][ 11] =-7.442159247636329e+01;
  NLX_coes[1][ 17][ 12] =-7.494202768166204e+01;
  NLX_coes[1][ 17][ 13] =-7.544785668359648e+01;
  NLX_coes[1][ 17][ 14] =-7.593951277837704e+01;
  NLX_coes[1][ 17][ 15] =-7.641837689019594e+01;
  NLX_coes[1][ 17][ 16] =-7.688730240818963e+01;
  NLX_coes[1][ 17][ 17] =-7.735170504024074e+01;
  NLX_coes[1][ 17][ 18] =-7.782289775069125e+01;
  NLX_coes[1][ 17][ 19] =-7.832956528286324e+01;
  NLX_coes[1][ 17][ 20] =-7.895521031611446e+01;
  NLX_coes[1][ 17][ 21] =-7.989279044833259e+01;
  NLX_coes[1][ 17][ 22] =-8.132361543859963e+01;
  NLX_coes[1][ 17][ 23] =-8.322909885463331e+01;
  NLX_coes[1][ 17][ 24] =-8.474249958475387e+01;
  NLX_coes[1][ 17][ 25] =-8.591758829815728e+01;
  NLX_coes[1][ 17][ 26] =-8.686956852130928e+01;
  NLX_coes[1][ 17][ 27] =-8.763888363312057e+01;
  NLX_coes[1][ 17][ 28] =-8.827597584168572e+01;
  NLX_coes[1][ 17][ 29] =-8.882791431943524e+01;
  NLX_coes[1][ 17][ 30] =-8.933525295264884e+01;
  NLX_coes[1][ 17][ 31] =-8.982823307378625e+01;
  NLX_coes[1][ 17][ 32] =-9.032680317401858e+01;
  NLX_coes[1][ 17][ 33] =-9.084252680901218e+01;
  NLX_coes[1][ 17][ 34] =-9.138091196471467e+01;
  NLX_coes[1][ 17][ 35] =-9.194346009434635e+01;
  NLX_coes[1][ 17][ 36] =-9.252920645201783e+01;
  NLX_coes[1][ 17][ 37] =-9.313575051736404e+01;
  NLX_coes[1][ 17][ 38] =-9.375985430686148e+01;
  NLX_coes[1][ 17][ 39] =-9.439769088996850e+01;
  NLX_coes[1][ 17][ 40] =-9.504479860779026e+01;
  NLX_coes[1][ 18][  0] =-7.455306734684845e+01;
  NLX_coes[1][ 18][  1] =-7.519262038426277e+01;
  NLX_coes[1][ 18][  2] =-7.582912220436479e+01;
  NLX_coes[1][ 18][  3] =-7.646097797255173e+01;
  NLX_coes[1][ 18][  4] =-7.708683537891805e+01;
  NLX_coes[1][ 18][  5] =-7.770554294192341e+01;
  NLX_coes[1][ 18][  6] =-7.831611828365062e+01;
  NLX_coes[1][ 18][  7] =-7.891772630023377e+01;
  NLX_coes[1][ 18][  8] =-7.950966891665709e+01;
  NLX_coes[1][ 18][  9] =-8.009139017938692e+01;
  NLX_coes[1][ 18][ 10] =-8.066250281449209e+01;
  NLX_coes[1][ 18][ 11] =-8.122284481958586e+01;
  NLX_coes[1][ 18][ 12] =-8.177257654342918e+01;
  NLX_coes[1][ 18][ 13] =-8.231232903977519e+01;
  NLX_coes[1][ 18][ 14] =-8.284341071788633e+01;
  NLX_coes[1][ 18][ 15] =-8.336805784189211e+01;
  NLX_coes[1][ 18][ 16] =-8.388964047344471e+01;
  NLX_coes[1][ 18][ 17] =-8.441276388522407e+01;
  NLX_coes[1][ 18][ 18] =-8.494495504065631e+01;
  NLX_coes[1][ 18][ 19] =-8.550955910421116e+01;
  NLX_coes[1][ 18][ 20] =-8.618271915406066e+01;
  NLX_coes[1][ 18][ 21] =-8.722750833402112e+01;
  NLX_coes[1][ 18][ 22] =-8.921805900532054e+01;
  NLX_coes[1][ 18][ 23] =-9.102528950301270e+01;
  NLX_coes[1][ 18][ 24] =-9.235435336861434e+01;
  NLX_coes[1][ 18][ 25] =-9.347537450972311e+01;
  NLX_coes[1][ 18][ 26] =-9.441133646937151e+01;
  NLX_coes[1][ 18][ 27] =-9.519543190060996e+01;
  NLX_coes[1][ 18][ 28] =-9.586341837772866e+01;
  NLX_coes[1][ 18][ 29] =-9.645106641742143e+01;
  NLX_coes[1][ 18][ 30] =-9.699059326743226e+01;
  NLX_coes[1][ 18][ 31] =-9.750803048987447e+01;
  NLX_coes[1][ 18][ 32] =-9.802235230654183e+01;
  NLX_coes[1][ 18][ 33] =-9.854605312341121e+01;
  NLX_coes[1][ 18][ 34] =-9.908640396129665e+01;
  NLX_coes[1][ 18][ 35] =-9.964681575381866e+01;
  NLX_coes[1][ 18][ 36] =-1.002280109648135e+02;
  NLX_coes[1][ 18][ 37] =-1.008288934344944e+02;
  NLX_coes[1][ 18][ 38] =-1.014471039153954e+02;
  NLX_coes[1][ 18][ 39] =-1.020792845847190e+02;
  NLX_coes[1][ 18][ 40] =-1.027210770323855e+02;
  NLX_coes[1][ 19][  0] =-8.122749071210536e+01;
  NLX_coes[1][ 19][  1] =-8.187877394914356e+01;
  NLX_coes[1][ 19][  2] =-8.252742423062104e+01;
  NLX_coes[1][ 19][  3] =-8.317209995938843e+01;
  NLX_coes[1][ 19][  4] =-8.381169992330597e+01;
  NLX_coes[1][ 19][  5] =-8.444533340848517e+01;
  NLX_coes[1][ 19][  6] =-8.507230156587812e+01;
  NLX_coes[1][ 19][  7] =-8.569209086889919e+01;
  NLX_coes[1][ 19][  8] =-8.630438125397545e+01;
  NLX_coes[1][ 19][  9] =-8.690907355573428e+01;
  NLX_coes[1][ 19][ 10] =-8.750634321800133e+01;
  NLX_coes[1][ 19][ 11] =-8.809672998806184e+01;
  NLX_coes[1][ 19][ 12] =-8.868127626264150e+01;
  NLX_coes[1][ 19][ 13] =-8.926172995158564e+01;
  NLX_coes[1][ 19][ 14] =-8.984083570362083e+01;
  NLX_coes[1][ 19][ 15] =-9.042278278700594e+01;
  NLX_coes[1][ 19][ 16] =-9.101404235062896e+01;
  NLX_coes[1][ 19][ 17] =-9.162489247712969e+01;
  NLX_coes[1][ 19][ 18] =-9.226929384520386e+01;
  NLX_coes[1][ 19][ 19] =-9.294932171780688e+01;
  NLX_coes[1][ 19][ 20] =-9.362496160113022e+01;
  NLX_coes[1][ 19][ 21] =-9.481233362483290e+01;
  NLX_coes[1][ 19][ 22] =-9.695342509479769e+01;
  NLX_coes[1][ 19][ 23] =-9.844371024259959e+01;
  NLX_coes[1][ 19][ 24] =-9.976120901329533e+01;
  NLX_coes[1][ 19][ 25] =-1.008838642797696e+02;
  NLX_coes[1][ 19][ 26] =-1.018374477665704e+02;
  NLX_coes[1][ 19][ 27] =-1.026535483007996e+02;
  NLX_coes[1][ 19][ 28] =-1.033621997361068e+02;
  NLX_coes[1][ 19][ 29] =-1.039927327478849e+02;
  NLX_coes[1][ 19][ 30] =-1.045720111747956e+02;
  NLX_coes[1][ 19][ 31] =-1.051227741935212e+02;
  NLX_coes[1][ 19][ 32] =-1.056627467220519e+02;
  NLX_coes[1][ 19][ 33] =-1.062046617856787e+02;
  NLX_coes[1][ 19][ 34] =-1.067568745754222e+02;
  NLX_coes[1][ 19][ 35] =-1.073242110912125e+02;
  NLX_coes[1][ 19][ 36] =-1.079088013034599e+02;
  NLX_coes[1][ 19][ 37] =-1.085107609967580e+02;
  NLX_coes[1][ 19][ 38] =-1.091286657331921e+02;
  NLX_coes[1][ 19][ 39] =-1.097598027381720e+02;
  NLX_coes[1][ 19][ 40] =-1.104002022508979e+02;
  NLX_coes[1][ 20][  0] =-8.794762546892942e+01;
  NLX_coes[1][ 20][  1] =-8.861164130052302e+01;
  NLX_coes[1][ 20][  2] =-8.927351089340129e+01;
  NLX_coes[1][ 20][  3] =-8.993218269474569e+01;
  NLX_coes[1][ 20][  4] =-9.058684145774912e+01;
  NLX_coes[1][ 20][  5] =-9.123689203702875e+01;
  NLX_coes[1][ 20][  6] =-9.188195518190716e+01;
  NLX_coes[1][ 20][  7] =-9.252187720427270e+01;
  NLX_coes[1][ 20][  8] =-9.315675715993615e+01;
  NLX_coes[1][ 20][  9] =-9.378699735265755e+01;
  NLX_coes[1][ 20][ 10] =-9.441338580342114e+01;
  NLX_coes[1][ 20][ 11] =-9.503722331665568e+01;
  NLX_coes[1][ 20][ 12] =-9.566051422392484e+01;
  NLX_coes[1][ 20][ 13] =-9.628625217472471e+01;
  NLX_coes[1][ 20][ 14] =-9.691885527770017e+01;
  NLX_coes[1][ 20][ 15] =-9.756482993084839e+01;
  NLX_coes[1][ 20][ 16] =-9.823374038726598e+01;
  NLX_coes[1][ 20][ 17] =-9.893990325224038e+01;
  NLX_coes[1][ 20][ 18] =-9.970909329975356e+01;
  NLX_coes[1][ 20][ 19] =-1.006099899650773e+02;
  NLX_coes[1][ 20][ 20] =-1.018243472489816e+02;
  NLX_coes[1][ 20][ 21] =-1.021894728563791e+02;
  NLX_coes[1][ 20][ 22] =-1.040888175664099e+02;
  NLX_coes[1][ 20][ 23] =-1.056851163719476e+02;
  NLX_coes[1][ 20][ 24] =-1.070292573287286e+02;
  NLX_coes[1][ 20][ 25] =-1.081774060638016e+02;
  NLX_coes[1][ 20][ 26] =-1.091673267272862e+02;
  NLX_coes[1][ 20][ 27] =-1.100269874438243e+02;
  NLX_coes[1][ 20][ 28] =-1.107831875395501e+02;
  NLX_coes[1][ 20][ 29] =-1.114615779802749e+02;
  NLX_coes[1][ 20][ 30] =-1.120856365324346e+02;
  NLX_coes[1][ 20][ 31] =-1.126756773443303e+02;
  NLX_coes[1][ 20][ 32] =-1.132482220587000e+02;
  NLX_coes[1][ 20][ 33] =-1.138158665651603e+02;
  NLX_coes[1][ 20][ 34] =-1.143875591925235e+02;
  NLX_coes[1][ 20][ 35] =-1.149691081749412e+02;
  NLX_coes[1][ 20][ 36] =-1.155637481871561e+02;
  NLX_coes[1][ 20][ 37] =-1.161726479413449e+02;
  NLX_coes[1][ 20][ 38] =-1.167952918448296e+02;
  NLX_coes[1][ 20][ 39] =-1.174297043743261e+02;
  NLX_coes[1][ 20][ 40] =-1.180725058448711e+02;
  NLX_coes[1][ 21][  0] =-9.471202649120355e+01;
  NLX_coes[1][ 21][  1] =-9.538960054196401e+01;
  NLX_coes[1][ 21][  2] =-9.606556495017455e+01;
  NLX_coes[1][ 21][  3] =-9.673918349219208e+01;
  NLX_coes[1][ 21][  4] =-9.740994874953732e+01;
  NLX_coes[1][ 21][  5] =-9.807758005548489e+01;
  NLX_coes[1][ 21][  6] =-9.874203334742015e+01;
  NLX_coes[1][ 21][  7] =-9.940352570881419e+01;
  NLX_coes[1][ 21][  8] =-1.000625790122688e+02;
  NLX_coes[1][ 21][  9] =-1.007200892085162e+02;
  NLX_coes[1][ 21][ 10] =-1.013774307666961e+02;
  NLX_coes[1][ 21][ 11] =-1.020366101467751e+02;
  NLX_coes[1][ 21][ 12] =-1.027004889711357e+02;
  NLX_coes[1][ 21][ 13] =-1.033731082861550e+02;
  NLX_coes[1][ 21][ 14] =-1.040601645628307e+02;
  NLX_coes[1][ 21][ 15] =-1.047697406435813e+02;
  NLX_coes[1][ 21][ 16] =-1.055135543437608e+02;
  NLX_coes[1][ 21][ 17] =-1.063091179781515e+02;
  NLX_coes[1][ 21][ 18] =-1.071810680356336e+02;
  NLX_coes[1][ 21][ 19] =-1.081476109966778e+02;
  NLX_coes[1][ 21][ 20] =-1.091611311141346e+02;
  NLX_coes[1][ 21][ 21] =-1.101001036262170e+02;
  NLX_coes[1][ 21][ 22] =-1.114468353403572e+02;
  NLX_coes[1][ 21][ 23] =-1.129076880924011e+02;
  NLX_coes[1][ 21][ 24] =-1.142382859340111e+02;
  NLX_coes[1][ 21][ 25] =-1.154093210237049e+02;
  NLX_coes[1][ 21][ 26] =-1.164353145797327e+02;
  NLX_coes[1][ 21][ 27] =-1.173388895598226e+02;
  NLX_coes[1][ 21][ 28] =-1.181431357511063e+02;
  NLX_coes[1][ 21][ 29] =-1.188702640627332e+02;
  NLX_coes[1][ 21][ 30] =-1.195408500415046e+02;
  NLX_coes[1][ 21][ 31] =-1.201730835610185e+02;
  NLX_coes[1][ 21][ 32] =-1.207822467604866e+02;
  NLX_coes[1][ 21][ 33] =-1.213805118401736e+02;
  NLX_coes[1][ 21][ 34] =-1.219770297497489e+02;
  NLX_coes[1][ 21][ 35] =-1.225782126529265e+02;
  NLX_coes[1][ 21][ 36] =-1.231881006141588e+02;
  NLX_coes[1][ 21][ 37] =-1.238087223784920e+02;
  NLX_coes[1][ 21][ 38] =-1.244403890620325e+02;
  NLX_coes[1][ 21][ 39] =-1.250818855352345e+02;
  NLX_coes[1][ 21][ 40] =-1.257305433122805e+02;
  NLX_coes[1][ 22][  0] =-1.015189081194511e+02;
  NLX_coes[1][ 22][  1] =-1.022106697003991e+02;
  NLX_coes[1][ 22][  2] =-1.029013864667761e+02;
  NLX_coes[1][ 22][  3] =-1.035906498471320e+02;
  NLX_coes[1][ 22][  4] =-1.042782678617953e+02;
  NLX_coes[1][ 22][  5] =-1.049642762428287e+02;
  NLX_coes[1][ 22][  6] =-1.056489603219387e+02;
  NLX_coes[1][ 22][  7] =-1.063328909906402e+02;
  NLX_coes[1][ 22][  8] =-1.070169792594604e+02;
  NLX_coes[1][ 22][  9] =-1.077025555510507e+02;
  NLX_coes[1][ 22][ 10] =-1.083914819592135e+02;
  NLX_coes[1][ 22][ 11] =-1.090863083990379e+02;
  NLX_coes[1][ 22][ 12] =-1.097904869212883e+02;
  NLX_coes[1][ 22][ 13] =-1.105086622780376e+02;
  NLX_coes[1][ 22][ 14] =-1.112470580459373e+02;
  NLX_coes[1][ 22][ 15] =-1.120139517348134e+02;
  NLX_coes[1][ 22][ 16] =-1.128200639465346e+02;
  NLX_coes[1][ 22][ 17] =-1.136780727478063e+02;
  NLX_coes[1][ 22][ 18] =-1.145991755636423e+02;
  NLX_coes[1][ 22][ 19] =-1.155831794749379e+02;
  NLX_coes[1][ 22][ 20] =-1.165931001482986e+02;
  NLX_coes[1][ 22][ 21] =-1.176849807343382e+02;
  NLX_coes[1][ 22][ 22] =-1.188914109420645e+02;
  NLX_coes[1][ 22][ 23] =-1.202011704030874e+02;
  NLX_coes[1][ 22][ 24] =-1.214667069420668e+02;
  NLX_coes[1][ 22][ 25] =-1.226286206353683e+02;
  NLX_coes[1][ 22][ 26] =-1.236747893419092e+02;
  NLX_coes[1][ 22][ 27] =-1.246135275516077e+02;
  NLX_coes[1][ 22][ 28] =-1.254604118313468e+02;
  NLX_coes[1][ 22][ 29] =-1.262330011559564e+02;
  NLX_coes[1][ 22][ 30] =-1.269485953155035e+02;
  NLX_coes[1][ 22][ 31] =-1.276230861045184e+02;
  NLX_coes[1][ 22][ 32] =-1.282703044627752e+02;
  NLX_coes[1][ 22][ 33] =-1.289017010928093e+02;
  NLX_coes[1][ 22][ 34] =-1.295262831650459e+02;
  NLX_coes[1][ 22][ 35] =-1.301507287892186e+02;
  NLX_coes[1][ 22][ 36] =-1.307795988690482e+02;
  NLX_coes[1][ 22][ 37] =-1.314155765201027e+02;
  NLX_coes[1][ 22][ 38] =-1.320596821003265e+02;
  NLX_coes[1][ 22][ 39] =-1.327114307461113e+02;
  NLX_coes[1][ 22][ 40] =-1.333689159129550e+02;
  NLX_coes[1][ 23][  0] =-1.083662008310128e+02;
  NLX_coes[1][ 23][  1] =-1.090725716843722e+02;
  NLX_coes[1][ 23][  2] =-1.097784683207910e+02;
  NLX_coes[1][ 23][  3] =-1.104838091259490e+02;
  NLX_coes[1][ 23][  4] =-1.111887113818844e+02;
  NLX_coes[1][ 23][  5] =-1.118935133729390e+02;
  NLX_coes[1][ 23][  6] =-1.125988050633499e+02;
  NLX_coes[1][ 23][  7] =-1.133054706607745e+02;
  NLX_coes[1][ 23][  8] =-1.140147469332581e+02;
  NLX_coes[1][ 23][  9] =-1.147283017666371e+02;
  NLX_coes[1][ 23][ 10] =-1.154483378930538e+02;
  NLX_coes[1][ 23][ 11] =-1.161777264084311e+02;
  NLX_coes[1][ 23][ 12] =-1.169201720873792e+02;
  NLX_coes[1][ 23][ 13] =-1.176804032143695e+02;
  NLX_coes[1][ 23][ 14] =-1.184643504950581e+02;
  NLX_coes[1][ 23][ 15] =-1.192792040384538e+02;
  NLX_coes[1][ 23][ 16] =-1.201330680427909e+02;
  NLX_coes[1][ 23][ 17] =-1.210336497723059e+02;
  NLX_coes[1][ 23][ 18] =-1.219852422057999e+02;
  NLX_coes[1][ 23][ 19] =-1.229846784455340e+02;
  NLX_coes[1][ 23][ 20] =-1.240257262119029e+02;
  NLX_coes[1][ 23][ 21] =-1.251341354360184e+02;
  NLX_coes[1][ 23][ 22] =-1.263019322919111e+02;
  NLX_coes[1][ 23][ 23] =-1.275188078945038e+02;
  NLX_coes[1][ 23][ 24] =-1.287208667330833e+02;
  NLX_coes[1][ 23][ 25] =-1.298578304073929e+02;
  NLX_coes[1][ 23][ 26] =-1.309083958276586e+02;
  NLX_coes[1][ 23][ 27] =-1.318702679685225e+02;
  NLX_coes[1][ 23][ 28] =-1.327511277553174e+02;
  NLX_coes[1][ 23][ 29] =-1.335630988887987e+02;
  NLX_coes[1][ 23][ 30] =-1.343197132348807e+02;
  NLX_coes[1][ 23][ 31] =-1.350342524416377e+02;
  NLX_coes[1][ 23][ 32] =-1.357188232025370e+02;
  NLX_coes[1][ 23][ 33] =-1.363838632742808e+02;
  NLX_coes[1][ 23][ 34] =-1.370379281965376e+02;
  NLX_coes[1][ 23][ 35] =-1.376876639954193e+02;
  NLX_coes[1][ 23][ 36] =-1.383378921388639e+02;
  NLX_coes[1][ 23][ 37] =-1.389917473170083e+02;
  NLX_coes[1][ 23][ 38] =-1.396508233582768e+02;
  NLX_coes[1][ 23][ 39] =-1.403152977188725e+02;
  NLX_coes[1][ 23][ 40] =-1.409840194803381e+02;
  NLX_coes[1][ 24][  0] =-1.152516135910933e+02;
  NLX_coes[1][ 24][  1] =-1.159728052854115e+02;
  NLX_coes[1][ 24][  2] =-1.166940780188623e+02;
  NLX_coes[1][ 24][  3] =-1.174156646318823e+02;
  NLX_coes[1][ 24][  4] =-1.181379736354567e+02;
  NLX_coes[1][ 24][  5] =-1.188616193395156e+02;
  NLX_coes[1][ 24][  6] =-1.195874575190179e+02;
  NLX_coes[1][ 24][  7] =-1.203166294893845e+02;
  NLX_coes[1][ 24][  8] =-1.210506172355438e+02;
  NLX_coes[1][ 24][  9] =-1.217913117131657e+02;
  NLX_coes[1][ 24][ 10] =-1.225410950807829e+02;
  NLX_coes[1][ 24][ 11] =-1.233029343540956e+02;
  NLX_coes[1][ 24][ 12] =-1.240804765843604e+02;
  NLX_coes[1][ 24][ 13] =-1.248781197992982e+02;
  NLX_coes[1][ 24][ 14] =-1.257010023675297e+02;
  NLX_coes[1][ 24][ 15] =-1.265547989153974e+02;
  NLX_coes[1][ 24][ 16] =-1.274451433960850e+02;
  NLX_coes[1][ 24][ 17] =-1.283765016440916e+02;
  NLX_coes[1][ 24][ 18] =-1.293506498574611e+02;
  NLX_coes[1][ 24][ 19] =-1.303661221118492e+02;
  NLX_coes[1][ 24][ 20] =-1.314237679448505e+02;
  NLX_coes[1][ 24][ 21] =-1.325272663319638e+02;
  NLX_coes[1][ 24][ 22] =-1.336697753793214e+02;
  NLX_coes[1][ 24][ 23] =-1.348341686022493e+02;
  NLX_coes[1][ 24][ 24] =-1.359885336846208e+02;
  NLX_coes[1][ 24][ 25] =-1.370991402488650e+02;
  NLX_coes[1][ 24][ 26] =-1.381456369139677e+02;
  NLX_coes[1][ 24][ 27] =-1.391209253233534e+02;
  NLX_coes[1][ 24][ 28] =-1.400270677879232e+02;
  NLX_coes[1][ 24][ 29] =-1.408713916464988e+02;
  NLX_coes[1][ 24][ 30] =-1.416637276299309e+02;
  NLX_coes[1][ 24][ 31] =-1.424146377436864e+02;
  NLX_coes[1][ 24][ 32] =-1.431343255040733e+02;
  NLX_coes[1][ 24][ 33] =-1.438319943254517e+02;
  NLX_coes[1][ 24][ 34] =-1.445155032389005e+02;
  NLX_coes[1][ 24][ 35] =-1.451912205028497e+02;
  NLX_coes[1][ 24][ 36] =-1.458640036720541e+02;
  NLX_coes[1][ 24][ 37] =-1.465372523400498e+02;
  NLX_coes[1][ 24][ 38] =-1.472129940664383e+02;
  NLX_coes[1][ 24][ 39] =-1.478919773132724e+02;
  NLX_coes[1][ 24][ 40] =-1.485737580724007e+02;
  NLX_coes[1][ 25][  0] =-1.221726980023583e+02;
  NLX_coes[1][ 25][  1] =-1.229087174745693e+02;
  NLX_coes[1][ 25][  2] =-1.236453398937692e+02;
  NLX_coes[1][ 25][  3] =-1.243830904153552e+02;
  NLX_coes[1][ 25][  4] =-1.251226418074649e+02;
  NLX_coes[1][ 25][  5] =-1.258648493580032e+02;
  NLX_coes[1][ 25][  6] =-1.266107878498064e+02;
  NLX_coes[1][ 25][  7] =-1.273617927709896e+02;
  NLX_coes[1][ 25][  8] =-1.281195070109711e+02;
  NLX_coes[1][ 25][  9] =-1.288859328365055e+02;
  NLX_coes[1][ 25][ 10] =-1.296634863497152e+02;
  NLX_coes[1][ 25][ 11] =-1.304550469212642e+02;
  NLX_coes[1][ 25][ 12] =-1.312639857600246e+02;
  NLX_coes[1][ 25][ 13] =-1.320941440055734e+02;
  NLX_coes[1][ 25][ 14] =-1.329497109775826e+02;
  NLX_coes[1][ 25][ 15] =-1.338349335324863e+02;
  NLX_coes[1][ 25][ 16] =-1.347535927170483e+02;
  NLX_coes[1][ 25][ 17] =-1.357082769261990e+02;
  NLX_coes[1][ 25][ 18] =-1.366997704829943e+02;
  NLX_coes[1][ 25][ 19] =-1.377273817622761e+02;
  NLX_coes[1][ 25][ 20] =-1.387906210513376e+02;
  NLX_coes[1][ 25][ 21] =-1.398866292560556e+02;
  NLX_coes[1][ 25][ 22] =-1.410079681620757e+02;
  NLX_coes[1][ 25][ 23] =-1.421394384566558e+02;
  NLX_coes[1][ 25][ 24] =-1.432605028859423e+02;
  NLX_coes[1][ 25][ 25] =-1.443490192594236e+02;
  NLX_coes[1][ 25][ 26] =-1.453886326870688e+02;
  NLX_coes[1][ 25][ 27] =-1.463711965141002e+02;
  NLX_coes[1][ 25][ 28] =-1.472956303492227e+02;
  NLX_coes[1][ 25][ 29] =-1.481657532351973e+02;
  NLX_coes[1][ 25][ 30] =-1.489882634336699e+02;
  NLX_coes[1][ 25][ 31] =-1.497711966015782e+02;
  NLX_coes[1][ 25][ 32] =-1.505228538808100e+02;
  NLX_coes[1][ 25][ 33] =-1.512511018839764e+02;
  NLX_coes[1][ 25][ 34] =-1.519629459336464e+02;
  NLX_coes[1][ 25][ 35] =-1.526642969438928e+02;
  NLX_coes[1][ 25][ 36] =-1.533598704987052e+02;
  NLX_coes[1][ 25][ 37] =-1.540531713058178e+02;
  NLX_coes[1][ 25][ 38] =-1.547465286750771e+02;
  NLX_coes[1][ 25][ 39] =-1.554411602088771e+02;
  NLX_coes[1][ 25][ 40] =-1.561372520939829e+02;
  NLX_coes[1][ 26][  0] =-1.291269105929507e+02;
  NLX_coes[1][ 26][  1] =-1.298775726377237e+02;
  NLX_coes[1][ 26][  2] =-1.306293124359529e+02;
  NLX_coes[1][ 26][  3] =-1.313829184032241e+02;
  NLX_coes[1][ 26][  4] =-1.321392949832416e+02;
  NLX_coes[1][ 26][  5] =-1.328994992390745e+02;
  NLX_coes[1][ 26][  6] =-1.336647760760760e+02;
  NLX_coes[1][ 26][  7] =-1.344365935217566e+02;
  NLX_coes[1][ 26][  8] =-1.352166781254399e+02;
  NLX_coes[1][ 26][  9] =-1.360070486005540e+02;
  NLX_coes[1][ 26][ 10] =-1.368100428771282e+02;
  NLX_coes[1][ 26][ 11] =-1.376283292245132e+02;
  NLX_coes[1][ 26][ 12] =-1.384648855990030e+02;
  NLX_coes[1][ 26][ 13] =-1.393229231795522e+02;
  NLX_coes[1][ 26][ 14] =-1.402057228955750e+02;
  NLX_coes[1][ 26][ 15] =-1.411163558479706e+02;
  NLX_coes[1][ 26][ 16] =-1.420572872432200e+02;
  NLX_coes[1][ 26][ 17] =-1.430299428035867e+02;
  NLX_coes[1][ 26][ 18] =-1.440344456163744e+02;
  NLX_coes[1][ 26][ 19] =-1.450697675269122e+02;
  NLX_coes[1][ 26][ 20] =-1.461337340216361e+02;
  NLX_coes[1][ 26][ 21] =-1.472217989858091e+02;
  NLX_coes[1][ 26][ 22] =-1.483261595782904e+02;
  NLX_coes[1][ 26][ 23] =-1.494346833130990e+02;
  NLX_coes[1][ 26][ 24] =-1.505324204793125e+02;
  NLX_coes[1][ 26][ 25] =-1.516039509207017e+02;
  NLX_coes[1][ 26][ 26] =-1.526367444321210e+02;
  NLX_coes[1][ 26][ 27] =-1.536232494486298e+02;
  NLX_coes[1][ 26][ 28] =-1.545609686533177e+02;
  NLX_coes[1][ 26][ 29] =-1.554514554834685e+02;
  NLX_coes[1][ 26][ 30] =-1.562990243273366e+02;
  NLX_coes[1][ 26][ 31] =-1.571095753457724e+02;
  NLX_coes[1][ 26][ 32] =-1.578896683299950e+02;
  NLX_coes[1][ 26][ 33] =-1.586458530367228e+02;
  NLX_coes[1][ 26][ 34] =-1.593842176946877e+02;
  NLX_coes[1][ 26][ 35] =-1.601101077983216e+02;
  NLX_coes[1][ 26][ 36] =-1.608279709044895e+02;
  NLX_coes[1][ 26][ 37] =-1.615412907556730e+02;
  NLX_coes[1][ 26][ 38] =-1.622525827362633e+02;
  NLX_coes[1][ 26][ 39] =-1.629634316721686e+02;
  NLX_coes[1][ 26][ 40] =-1.636745621570523e+02;
  NLX_coes[1][ 27][  0] =-1.361116701769147e+02;
  NLX_coes[1][ 27][  1] =-1.368766152334574e+02;
  NLX_coes[1][ 27][  2] =-1.376430567449142e+02;
  NLX_coes[1][ 27][  3] =-1.384120132864769e+02;
  NLX_coes[1][ 27][  4] =-1.391845860117512e+02;
  NLX_coes[1][ 27][  5] =-1.399619942666200e+02;
  NLX_coes[1][ 27][  6] =-1.407456068470134e+02;
  NLX_coes[1][ 27][  7] =-1.415369697186138e+02;
  NLX_coes[1][ 27][  8] =-1.423378294904427e+02;
  NLX_coes[1][ 27][  9] =-1.431501499528529e+02;
  NLX_coes[1][ 27][ 10] =-1.439761163514797e+02;
  NLX_coes[1][ 27][ 11] =-1.448181186721612e+02;
  NLX_coes[1][ 27][ 12] =-1.456787013260103e+02;
  NLX_coes[1][ 27][ 13] =-1.465604634308493e+02;
  NLX_coes[1][ 27][ 14] =-1.474658943402555e+02;
  NLX_coes[1][ 27][ 15] =-1.483971386564531e+02;
  NLX_coes[1][ 27][ 16] =-1.493557105484883e+02;
  NLX_coes[1][ 27][ 17] =-1.503422192297316e+02;
  NLX_coes[1][ 27][ 18] =-1.513561960786116e+02;
  NLX_coes[1][ 27][ 19] =-1.523960251859302e+02;
  NLX_coes[1][ 27][ 20] =-1.534586677745423e+02;
  NLX_coes[1][ 27][ 21] =-1.545391579910238e+02;
  NLX_coes[1][ 27][ 22] =-1.556301440150145e+02;
  NLX_coes[1][ 27][ 23] =-1.567218259906051e+02;
  NLX_coes[1][ 27][ 24] =-1.578028070580351e+02;
  NLX_coes[1][ 27][ 25] =-1.588616709241683e+02;
  NLX_coes[1][ 27][ 26] =-1.598887715900049e+02;
  NLX_coes[1][ 27][ 27] =-1.608776101525952e+02;
  NLX_coes[1][ 27][ 28] =-1.618252336573131e+02;
  NLX_coes[1][ 27][ 29] =-1.627318545399999e+02;
  NLX_coes[1][ 27][ 30] =-1.636000973221959e+02;
  NLX_coes[1][ 27][ 31] =-1.644341758621835e+02;
  NLX_coes[1][ 27][ 32] =-1.652391607045697e+02;
  NLX_coes[1][ 27][ 33] =-1.660203955271640e+02;
  NLX_coes[1][ 27][ 34] =-1.667830677467899e+02;
  NLX_coes[1][ 27][ 35] =-1.675319148738003e+02;
  NLX_coes[1][ 27][ 36] =-1.682710410189639e+02;
  NLX_coes[1][ 27][ 37] =-1.690038186109206e+02;
  NLX_coes[1][ 27][ 38] =-1.697328546687075e+02;
  NLX_coes[1][ 27][ 39] =-1.704600069700925e+02;
  NLX_coes[1][ 27][ 40] =-1.711864422927847e+02;
  NLX_coes[1][ 28][  0] =-1.431244080298554e+02;
  NLX_coes[1][ 28][  1] =-1.439031234301397e+02;
  NLX_coes[1][ 28][  2] =-1.446836935477290e+02;
  NLX_coes[1][ 28][  3] =-1.454673326595654e+02;
  NLX_coes[1][ 28][  4] =-1.462553040397953e+02;
  NLX_coes[1][ 28][  5] =-1.470489524859097e+02;
  NLX_coes[1][ 28][  6] =-1.478497302350181e+02;
  NLX_coes[1][ 28][  7] =-1.486592166953894e+02;
  NLX_coes[1][ 28][  8] =-1.494791309501286e+02;
  NLX_coes[1][ 28][  9] =-1.503113342551517e+02;
  NLX_coes[1][ 28][ 10] =-1.511578177471220e+02;
  NLX_coes[1][ 28][ 11] =-1.520206684598593e+02;
  NLX_coes[1][ 28][ 12] =-1.529020050434487e+02;
  NLX_coes[1][ 28][ 13] =-1.538038744425182e+02;
  NLX_coes[1][ 28][ 14] =-1.547281041844093e+02;
  NLX_coes[1][ 28][ 15] =-1.556761142216362e+02;
  NLX_coes[1][ 28][ 16] =-1.566487081052054e+02;
  NLX_coes[1][ 28][ 17] =-1.576458790834947e+02;
  NLX_coes[1][ 28][ 18] =-1.586666605361635e+02;
  NLX_coes[1][ 28][ 19] =-1.597089882744463e+02;
  NLX_coes[1][ 28][ 20] =-1.607694895128910e+02;
  NLX_coes[1][ 28][ 21] =-1.618432579540270e+02;
  NLX_coes[1][ 28][ 22] =-1.629237069842352e+02;
  NLX_coes[1][ 28][ 23] =-1.640027587787254e+02;
  NLX_coes[1][ 28][ 24] =-1.650714530735229e+02;
  NLX_coes[1][ 28][ 25] =-1.661209645859741e+02;
  NLX_coes[1][ 28][ 26] =-1.671437018688047e+02;
  NLX_coes[1][ 28][ 27] =-1.681341926869194e+02;
  NLX_coes[1][ 28][ 28] =-1.690894834343758e+02;
  NLX_coes[1][ 28][ 29] =-1.700090442373246e+02;
  NLX_coes[1][ 28][ 30] =-1.708943544012699e+02;
  NLX_coes[1][ 28][ 31] =-1.717483560170944e+02;
  NLX_coes[1][ 28][ 32] =-1.725749071763128e+02;
  NLX_coes[1][ 28][ 33] =-1.733783054026351e+02;
  NLX_coes[1][ 28][ 34] =-1.741629084879305e+02;
  NLX_coes[1][ 28][ 35] =-1.749328547773540e+02;
  NLX_coes[1][ 28][ 36] =-1.756918729062796e+02;
  NLX_coes[1][ 28][ 37] =-1.764431670039525e+02;
  NLX_coes[1][ 28][ 38] =-1.771893639953146e+02;
  NLX_coes[1][ 28][ 39] =-1.779325127919826e+02;
  NLX_coes[1][ 28][ 40] =-1.786741296752000e+02;
  NLX_coes[1][ 29][  0] =-1.501626094854460e+02;
  NLX_coes[1][ 29][  1] =-1.509544523730377e+02;
  NLX_coes[1][ 29][  2] =-1.517484476392995e+02;
  NLX_coes[1][ 29][  3] =-1.525459717477587e+02;
  NLX_coes[1][ 29][  4] =-1.533484179986125e+02;
  NLX_coes[1][ 29][  5] =-1.541572243901092e+02;
  NLX_coes[1][ 29][  6] =-1.549738934902573e+02;
  NLX_coes[1][ 29][  7] =-1.558000045646360e+02;
  NLX_coes[1][ 29][  8] =-1.566372169105629e+02;
  NLX_coes[1][ 29][  9] =-1.574872619706730e+02;
  NLX_coes[1][ 29][ 10] =-1.583519204509210e+02;
  NLX_coes[1][ 29][ 11] =-1.592329795942199e+02;
  NLX_coes[1][ 29][ 12] =-1.601321654443930e+02;
  NLX_coes[1][ 29][ 13] =-1.610510461556183e+02;
  NLX_coes[1][ 29][ 14] =-1.619909060986267e+02;
  NLX_coes[1][ 29][ 15] =-1.629525971550247e+02;
  NLX_coes[1][ 29][ 16] =-1.639363817019070e+02;
  NLX_coes[1][ 29][ 17] =-1.649417858459671e+02;
  NLX_coes[1][ 29][ 18] =-1.659674720725588e+02;
  NLX_coes[1][ 29][ 19] =-1.670111173435500e+02;
  NLX_coes[1][ 29][ 20] =-1.680692895794410e+02;
  NLX_coes[1][ 29][ 21] =-1.691373603636754e+02;
  NLX_coes[1][ 29][ 22] =-1.702095222491166e+02;
  NLX_coes[1][ 29][ 23] =-1.712790178981328e+02;
  NLX_coes[1][ 29][ 24] =-1.723386109321556e+02;
  NLX_coes[1][ 29][ 25] =-1.733812617164299e+02;
  NLX_coes[1][ 29][ 26] =-1.744008495230049e+02;
  NLX_coes[1][ 29][ 27] =-1.753927649508672e+02;
  NLX_coes[1][ 29][ 28] =-1.763542353657053e+02;
  NLX_coes[1][ 29][ 29] =-1.772843452063612e+02;
  NLX_coes[1][ 29][ 30] =-1.781838178314057e+02;
  NLX_coes[1][ 29][ 31] =-1.790546636051095e+02;
  NLX_coes[1][ 29][ 32] =-1.798997866081931e+02;
  NLX_coes[1][ 29][ 33] =-1.807226122812602e+02;
  NLX_coes[1][ 29][ 34] =-1.815267696573417e+02;
  NLX_coes[1][ 29][ 35] =-1.823158412286578e+02;
  NLX_coes[1][ 29][ 36] =-1.830931811208086e+02;
  NLX_coes[1][ 29][ 37] =-1.838617962108492e+02;
  NLX_coes[1][ 29][ 38] =-1.846242830831789e+02;
  NLX_coes[1][ 29][ 39] =-1.853828146766565e+02;
  NLX_coes[1][ 29][ 40] =-1.861391729986391e+02;
  NLX_coes[1][ 30][  0] =-1.572238463410416e+02;
  NLX_coes[1][ 30][  1] =-1.580280666814562e+02;
  NLX_coes[1][ 30][  2] =-1.588346797474167e+02;
  NLX_coes[1][ 30][  3] =-1.596451934438889e+02;
  NLX_coes[1][ 30][  4] =-1.604611029571591e+02;
  NLX_coes[1][ 30][  5] =-1.612839128025335e+02;
  NLX_coes[1][ 30][  6] =-1.621151504089939e+02;
  NLX_coes[1][ 30][  7] =-1.629563714701616e+02;
  NLX_coes[1][ 30][  8] =-1.638091561766257e+02;
  NLX_coes[1][ 30][  9] =-1.646750944097133e+02;
  NLX_coes[1][ 30][ 10] =-1.655557571691897e+02;
  NLX_coes[1][ 30][ 11] =-1.664526511401562e+02;
  NLX_coes[1][ 30][ 12] =-1.673671537045344e+02;
  NLX_coes[1][ 30][ 13] =-1.683004272726851e+02;
  NLX_coes[1][ 30][ 14] =-1.692533148318325e+02;
  NLX_coes[1][ 30][ 15] =-1.702262227915489e+02;
  NLX_coes[1][ 30][ 16] =-1.712190010531285e+02;
  NLX_coes[1][ 30][ 17] =-1.722308308223180e+02;
  NLX_coes[1][ 30][ 18] =-1.732601258109844e+02;
  NLX_coes[1][ 30][ 19] =-1.743044476983506e+02;
  NLX_coes[1][ 30][ 20] =-1.753604460845145e+02;
  NLX_coes[1][ 30][ 21] =-1.764238488800486e+02;
  NLX_coes[1][ 30][ 22] =-1.774895465845778e+02;
  NLX_coes[1][ 30][ 23] =-1.785518118031855e+02;
  NLX_coes[1][ 30][ 24] =-1.796046643956103e+02;
  NLX_coes[1][ 30][ 25] =-1.806423436910646e+02;
  NLX_coes[1][ 30][ 26] =-1.816598030414261e+02;
  NLX_coes[1][ 30][ 27] =-1.826531230268437e+02;
  NLX_coes[1][ 30][ 28] =-1.836197640515258e+02;
  NLX_coes[1][ 30][ 29] =-1.845586264327546e+02;
  NLX_coes[1][ 30][ 30] =-1.854699412244353e+02;
  NLX_coes[1][ 30][ 31] =-1.863550473490972e+02;
  NLX_coes[1][ 30][ 32] =-1.872161149200620e+02;
  NLX_coes[1][ 30][ 33] =-1.880558627646636e+02;
  NLX_coes[1][ 30][ 34] =-1.888773017264779e+02;
  NLX_coes[1][ 30][ 35] =-1.896835207936638e+02;
  NLX_coes[1][ 30][ 36] =-1.904775227217256e+02;
  NLX_coes[1][ 30][ 37] =-1.912621096094740e+02;
  NLX_coes[1][ 30][ 38] =-1.920398160323949e+02;
  NLX_coes[1][ 30][ 39] =-1.928128868813360e+02;
  NLX_coes[1][ 30][ 40] =-1.935832981147408e+02;
  NLX_coes[1][ 31][  0] =-1.643058000720945e+02;
  NLX_coes[1][ 31][  1] =-1.651215624773006e+02;
  NLX_coes[1][ 31][  2] =-1.659399066113991e+02;
  NLX_coes[1][ 31][  3] =-1.667624452203189e+02;
  NLX_coes[1][ 31][  4] =-1.675907520772121e+02;
  NLX_coes[1][ 31][  5] =-1.684263773508121e+02;
  NLX_coes[1][ 31][  6] =-1.692708549386058e+02;
  NLX_coes[1][ 31][  7] =-1.701257020587206e+02;
  NLX_coes[1][ 31][  8] =-1.709924103863303e+02;
  NLX_coes[1][ 31][  9] =-1.718724272612932e+02;
  NLX_coes[1][ 31][ 10] =-1.727671250849289e+02;
  NLX_coes[1][ 31][ 11] =-1.736777570883563e+02;
  NLX_coes[1][ 31][ 12] =-1.746053983554847e+02;
  NLX_coes[1][ 31][ 13] =-1.755508724626890e+02;
  NLX_coes[1][ 31][ 14] =-1.765146663280603e+02;
  NLX_coes[1][ 31][ 15] =-1.774968384276372e+02;
  NLX_coes[1][ 31][ 16] =-1.784969274799710e+02;
  NLX_coes[1][ 31][ 17] =-1.795138688991894e+02;
  NLX_coes[1][ 31][ 18] =-1.805459249629089e+02;
  NLX_coes[1][ 31][ 19] =-1.815906351372026e+02;
  NLX_coes[1][ 31][ 20] =-1.826447981771880e+02;
  NLX_coes[1][ 31][ 21] =-1.837045043219102e+02;
  NLX_coes[1][ 31][ 22] =-1.847652411071262e+02;
  NLX_coes[1][ 31][ 23] =-1.858220891952797e+02;
  NLX_coes[1][ 31][ 24] =-1.868700068560405e+02;
  NLX_coes[1][ 31][ 25] =-1.879041737755525e+02;
  NLX_coes[1][ 31][ 26] =-1.889203426334014e+02;
  NLX_coes[1][ 31][ 27] =-1.899151371565839e+02;
  NLX_coes[1][ 31][ 28] =-1.908862474814094e+02;
  NLX_coes[1][ 31][ 29] =-1.918324995608768e+02;
  NLX_coes[1][ 31][ 30] =-1.927538055052273e+02;
  NLX_coes[1][ 31][ 31] =-1.936510240014868e+02;
  NLX_coes[1][ 31][ 32] =-1.945257681978384e+02;
  NLX_coes[1][ 31][ 33] =-1.953801955385389e+02;
  NLX_coes[1][ 31][ 34] =-1.962168057073697e+02;
  NLX_coes[1][ 31][ 35] =-1.970382635903618e+02;
  NLX_coes[1][ 31][ 36] =-1.978472564239405e+02;
  NLX_coes[1][ 31][ 37] =-1.986463888933576e+02;
  NLX_coes[1][ 31][ 38] =-1.994381168561237e+02;
  NLX_coes[1][ 31][ 39] =-2.002247191805448e+02;
  NLX_coes[1][ 31][ 40] =-2.010083073676231e+02;
  NLX_coes[1][ 32][  0] =-1.714062762483822e+02;
  NLX_coes[1][ 32][  1] =-1.722326796576779e+02;
  NLX_coes[1][ 32][  2] =-1.730618104692779e+02;
  NLX_coes[1][ 32][  3] =-1.738953648311557e+02;
  NLX_coes[1][ 32][  4] =-1.747349770824431e+02;
  NLX_coes[1][ 32][  5] =-1.755822277140334e+02;
  NLX_coes[1][ 32][  6] =-1.764386446720839e+02;
  NLX_coes[1][ 32][  7] =-1.773056983029495e+02;
  NLX_coes[1][ 32][  8] =-1.781847892327928e+02;
  NLX_coes[1][ 32][  9] =-1.790772279524699e+02;
  NLX_coes[1][ 32][ 10] =-1.799842047731377e+02;
  NLX_coes[1][ 32][ 11] =-1.809067491518567e+02;
  NLX_coes[1][ 32][ 12] =-1.818456781891479e+02;
  NLX_coes[1][ 32][ 13] =-1.828015353711761e+02;
  NLX_coes[1][ 32][ 14] =-1.837745222471852e+02;
  NLX_coes[1][ 32][ 15] =-1.847644273753124e+02;
  NLX_coes[1][ 32][ 16] =-1.857705580528639e+02;
  NLX_coes[1][ 32][ 17] =-1.867916807601430e+02;
  NLX_coes[1][ 32][ 18] =-1.878259763779230e+02;
  NLX_coes[1][ 32][ 19] =-1.888710173922150e+02;
  NLX_coes[1][ 32][ 20] =-1.899237767165952e+02;
  NLX_coes[1][ 32][ 21] =-1.909806801931566e+02;
  NLX_coes[1][ 32][ 22] =-1.920377146208540e+02;
  NLX_coes[1][ 32][ 23] =-1.930905970071408e+02;
  NLX_coes[1][ 32][ 24] =-1.941349989911909e+02;
  NLX_coes[1][ 32][ 25] =-1.951668051616408e+02;
  NLX_coes[1][ 32][ 26] =-1.961823718590483e+02;
  NLX_coes[1][ 32][ 27] =-1.971787486834961e+02;
  NLX_coes[1][ 32][ 28] =-1.981538312864311e+02;
  NLX_coes[1][ 32][ 29] =-1.991064287781751e+02;
  NLX_coes[1][ 32][ 30] =-2.000362466131162e+02;
  NLX_coes[1][ 32][ 31] =-2.009438003707554e+02;
  NLX_coes[1][ 32][ 32] =-2.018302835350346e+02;
  NLX_coes[1][ 32][ 33] =-2.026974131955226e+02;
  NLX_coes[1][ 32][ 34] =-2.035472739512718e+02;
  NLX_coes[1][ 32][ 35] =-2.043821748457651e+02;
  NLX_coes[1][ 32][ 36] =-2.052045287858855e+02;
  NLX_coes[1][ 32][ 37] =-2.060167595991073e+02;
  NLX_coes[1][ 32][ 38] =-2.068212390190772e+02;
  NLX_coes[1][ 32][ 39] =-2.076202544356076e+02;
  NLX_coes[1][ 32][ 40] =-2.084160080516143e+02;
  NLX_coes[1][ 33][  0] =-1.785232107058112e+02;
  NLX_coes[1][ 33][  1] =-1.793593052432145e+02;
  NLX_coes[1][ 33][  2] =-1.801982392120562e+02;
  NLX_coes[1][ 33][  3] =-1.810417766884385e+02;
  NLX_coes[1][ 33][  4] =-1.818915999243011e+02;
  NLX_coes[1][ 33][  5] =-1.827493092128859e+02;
  NLX_coes[1][ 33][  6] =-1.836164186431196e+02;
  NLX_coes[1][ 33][  7] =-1.844943476620612e+02;
  NLX_coes[1][ 33][  8] =-1.853844074239749e+02;
  NLX_coes[1][ 33][  9] =-1.862877806589767e+02;
  NLX_coes[1][ 33][ 10] =-1.872054939760818e+02;
  NLX_coes[1][ 33][ 11] =-1.881383820595505e+02;
  NLX_coes[1][ 33][ 12] =-1.890870440585256e+02;
  NLX_coes[1][ 33][ 13] =-1.900517935502379e+02;
  NLX_coes[1][ 33][ 14] =-1.910326046617607e+02;
  NLX_coes[1][ 33][ 15] =-1.920290580681983e+02;
  NLX_coes[1][ 33][ 16] =-1.930402914331104e+02;
  NLX_coes[1][ 33][ 17] =-1.940649593704202e+02;
  NLX_coes[1][ 33][ 18] =-1.951012084642325e+02;
  NLX_coes[1][ 33][ 19] =-1.961466736378348e+02;
  NLX_coes[1][ 33][ 20] =-1.971985030405196e+02;
  NLX_coes[1][ 33][ 21] =-1.982534188659925e+02;
  NLX_coes[1][ 33][ 22] =-1.993078196824167e+02;
  NLX_coes[1][ 33][ 23] =-2.003579249971899e+02;
  NLX_coes[1][ 33][ 24] =-2.013999550705839e+02;
  NLX_coes[1][ 33][ 25] =-2.024303303210186e+02;
  NLX_coes[1][ 33][ 26] =-2.034458679260853e+02;
  NLX_coes[1][ 33][ 27] =-2.044439512114083e+02;
  NLX_coes[1][ 33][ 28] =-2.054226512157359e+02;
  NLX_coes[1][ 33][ 29] =-2.063807885103938e+02;
  NLX_coes[1][ 33][ 30] =-2.073179340315226e+02;
  NLX_coes[1][ 33][ 31] =-2.082343572181113e+02;
  NLX_coes[1][ 33][ 32] =-2.091309358318757e+02;
  NLX_coes[1][ 33][ 33] =-2.100090438508805e+02;
  NLX_coes[1][ 33][ 34] =-2.108704326237990e+02;
  NLX_coes[1][ 33][ 35] =-2.117171174429198e+02;
  NLX_coes[1][ 33][ 36] =-2.125512780974401e+02;
  NLX_coes[1][ 33][ 37] =-2.133751786636277e+02;
  NLX_coes[1][ 33][ 38] =-2.141911092737373e+02;
  NLX_coes[1][ 33][ 39] =-2.150013510803821e+02;
  NLX_coes[1][ 33][ 40] =-2.158081652918586e+02;
  NLX_coes[1][ 34][  0] =-1.856546679530974e+02;
  NLX_coes[1][ 34][  1] =-1.864994684689632e+02;
  NLX_coes[1][ 34][  2] =-1.873471982368361e+02;
  NLX_coes[1][ 34][  3] =-1.881996805234909e+02;
  NLX_coes[1][ 34][  4] =-1.890586379236950e+02;
  NLX_coes[1][ 34][  5] =-1.899256836322412e+02;
  NLX_coes[1][ 34][  6] =-1.908023127236208e+02;
  NLX_coes[1][ 34][  7] =-1.916898919290439e+02;
  NLX_coes[1][ 34][  8] =-1.925896462057072e+02;
  NLX_coes[1][ 34][  9] =-1.935026404873946e+02;
  NLX_coes[1][ 34][ 10] =-1.944297555567607e+02;
  NLX_coes[1][ 34][ 11] =-1.953716577405801e+02;
  NLX_coes[1][ 34][ 12] =-1.963287630028005e+02;
  NLX_coes[1][ 34][ 13] =-1.973011969444023e+02;
  NLX_coes[1][ 34][ 14] =-1.982887531468684e+02;
  NLX_coes[1][ 34][ 15] =-1.992908531208949e+02;
  NLX_coes[1][ 34][ 16] =-2.003065117586396e+02;
  NLX_coes[1][ 34][ 17] =-2.013343126351003e+02;
  NLX_coes[1][ 34][ 18] =-2.023723978558493e+02;
  NLX_coes[1][ 34][ 19] =-2.034184774379250e+02;
  NLX_coes[1][ 34][ 20] =-2.044698632376637e+02;
  NLX_coes[1][ 34][ 21] =-2.055235317293728e+02;
  NLX_coes[1][ 34][ 22] =-2.065762178618663e+02;
  NLX_coes[1][ 34][ 23] =-2.076245385034767e+02;
  NLX_coes[1][ 34][ 24] =-2.086651389463312e+02;
  NLX_coes[1][ 34][ 25] =-2.096948507611443e+02;
  NLX_coes[1][ 34][ 26] =-2.107108455149787e+02;
  NLX_coes[1][ 34][ 27] =-2.117107679432910e+02;
  NLX_coes[1][ 34][ 28] =-2.126928346455079e+02;
  NLX_coes[1][ 34][ 29] =-2.136558897368438e+02;
  NLX_coes[1][ 34][ 30] =-2.145994156430002e+02;
  NLX_coes[1][ 34][ 31] =-2.155235035863204e+02;
  NLX_coes[1][ 34][ 32] =-2.164287928796562e+02;
  NLX_coes[1][ 34][ 33] =-2.173163903034661e+02;
  NLX_coes[1][ 34][ 34] =-2.181877807897501e+02;
  NLX_coes[1][ 34][ 35] =-2.190447390339914e+02;
  NLX_coes[1][ 34][ 36] =-2.198892492471148e+02;
  NLX_coes[1][ 34][ 37] =-2.207234376662373e+02;
  NLX_coes[1][ 34][ 38] =-2.215495201478052e+02;
  NLX_coes[1][ 34][ 39] =-2.223697656201199e+02;
  NLX_coes[1][ 34][ 40] =-2.231864751689139e+02;
  NLX_coes[1][ 35][  0] =-1.927988319615140e+02;
  NLX_coes[1][ 35][  1] =-1.936513278122901e+02;
  NLX_coes[1][ 35][  2] =-1.945068345580297e+02;
  NLX_coes[1][ 35][  3] =-1.953672335885832e+02;
  NLX_coes[1][ 35][  4] =-1.962342842948575e+02;
  NLX_coes[1][ 35][  5] =-1.971096076378713e+02;
  NLX_coes[1][ 35][  6] =-1.979946751371776e+02;
  NLX_coes[1][ 35][  7] =-1.988907990469454e+02;
  NLX_coes[1][ 35][  8] =-1.997991210461454e+02;
  NLX_coes[1][ 35][  9] =-2.007205972495703e+02;
  NLX_coes[1][ 35][ 10] =-2.016559784369548e+02;
  NLX_coes[1][ 35][ 11] =-2.026057853909211e+02;
  NLX_coes[1][ 35][ 12] =-2.035702801271941e+02;
  NLX_coes[1][ 35][ 13] =-2.045494346058862e+02;
  NLX_coes[1][ 35][ 14] =-2.055428992299491e+02;
  NLX_coes[1][ 35][ 15] =-2.065499740390268e+02;
  NLX_coes[1][ 35][ 16] =-2.075695859630860e+02;
  NLX_coes[1][ 35][ 17] =-2.086002758029953e+02;
  NLX_coes[1][ 35][ 18] =-2.096401987576428e+02;
  NLX_coes[1][ 35][ 19] =-2.106871422675644e+02;
  NLX_coes[1][ 35][ 20] =-2.117385645363320e+02;
  NLX_coes[1][ 35][ 21] =-2.127916560619029e+02;
  NLX_coes[1][ 35][ 22] =-2.138434246181016e+02;
  NLX_coes[1][ 35][ 23] =-2.148908013425848e+02;
  NLX_coes[1][ 35][ 24] =-2.159307622396670e+02;
  NLX_coes[1][ 35][ 25] =-2.169604562081827e+02;
  NLX_coes[1][ 35][ 26] =-2.179773285411200e+02;
  NLX_coes[1][ 35][ 27] =-2.189792284722281e+02;
  NLX_coes[1][ 35][ 28] =-2.199644910558439e+02;
  NLX_coes[1][ 35][ 29] =-2.209319871486952e+02;
  NLX_coes[1][ 35][ 30] =-2.218811396972253e+02;
  NLX_coes[1][ 35][ 31] =-2.228119088886602e+02;
  NLX_coes[1][ 35][ 32] =-2.237247521201625e+02;
  NLX_coes[1][ 35][ 33] =-2.246205666874090e+02;
  NLX_coes[1][ 35][ 34] =-2.255006235405201e+02;
  NLX_coes[1][ 35][ 35] =-2.263664996587963e+02;
  NLX_coes[1][ 35][ 36] =-2.272200149292455e+02;
  NLX_coes[1][ 35][ 37] =-2.280631772385732e+02;
  NLX_coes[1][ 35][ 38] =-2.288981371131767e+02;
  NLX_coes[1][ 35][ 39] =-2.297271516448322e+02;
  NLX_coes[1][ 35][ 40] =-2.305525549024333e+02;
  NLX_coes[1][ 36][  0] =-1.999539888089194e+02;
  NLX_coes[1][ 36][  1] =-2.008131492246516e+02;
  NLX_coes[1][ 36][  2] =-2.016754131135772e+02;
  NLX_coes[1][ 36][  3] =-2.025427274225511e+02;
  NLX_coes[1][ 36][  4] =-2.034168858200221e+02;
  NLX_coes[1][ 36][  5] =-2.042995109267654e+02;
  NLX_coes[1][ 36][  6] =-2.051920441945553e+02;
  NLX_coes[1][ 36][  7] =-2.060957395975622e+02;
  NLX_coes[1][ 36][  8] =-2.070116564570539e+02;
  NLX_coes[1][ 36][  9] =-2.079406488081718e+02;
  NLX_coes[1][ 36][ 10] =-2.088833503380102e+02;
  NLX_coes[1][ 36][ 11] =-2.098401550990537e+02;
  NLX_coes[1][ 36][ 12] =-2.108111950393722e+02;
  NLX_coes[1][ 36][ 13] =-2.117963160349724e+02;
  NLX_coes[1][ 36][ 14] =-2.127950546311965e+02;
  NLX_coes[1][ 36][ 15] =-2.138066181131762e+02;
  NLX_coes[1][ 36][ 16] =-2.148298708170558e+02;
  NLX_coes[1][ 36][ 17] =-2.158633297437393e+02;
  NLX_coes[1][ 36][ 18] =-2.169051725140757e+02;
  NLX_coes[1][ 36][ 19] =-2.179532604459958e+02;
  NLX_coes[1][ 36][ 20] =-2.190051789355652e+02;
  NLX_coes[1][ 36][ 21] =-2.200582962595548e+02;
  NLX_coes[1][ 36][ 22] =-2.211098403143139e+02;
  NLX_coes[1][ 36][ 23] =-2.221569907303239e+02;
  NLX_coes[1][ 36][ 24] =-2.231969815228144e+02;
  NLX_coes[1][ 36][ 25] =-2.242272074052030e+02;
  NLX_coes[1][ 36][ 26] =-2.252453256234825e+02;
  NLX_coes[1][ 36][ 27] =-2.262493450793012e+02;
  NLX_coes[1][ 36][ 28] =-2.272376957479554e+02;
  NLX_coes[1][ 36][ 29] =-2.282092737730977e+02;
  NLX_coes[1][ 36][ 30] =-2.291634606542493e+02;
  NLX_coes[1][ 36][ 31] =-2.301001180198796e+02;
  NLX_coes[1][ 36][ 32] =-2.310195620365748e+02;
  NLX_coes[1][ 36][ 33] =-2.319225231771923e+02;
  NLX_coes[1][ 36][ 34] =-2.328100977279794e+02;
  NLX_coes[1][ 36][ 35] =-2.336836971051114e+02;
  NLX_coes[1][ 36][ 36] =-2.345449998882257e+02;
  NLX_coes[1][ 36][ 37] =-2.353959095401400e+02;
  NLX_coes[1][ 36][ 38] =-2.362385181814540e+02;
  NLX_coes[1][ 36][ 39] =-2.370750737374974e+02;
  NLX_coes[1][ 36][ 40] =-2.379079483390921e+02;
  NLX_coes[1][ 37][  0] =-2.071184994413892e+02;
  NLX_coes[1][ 37][  1] =-2.079832728793263e+02;
  NLX_coes[1][ 37][  2] =-2.088512849275615e+02;
  NLX_coes[1][ 37][  3] =-2.097245603481924e+02;
  NLX_coes[1][ 37][  4] =-2.106049197907690e+02;
  NLX_coes[1][ 37][  5] =-2.114939764512317e+02;
  NLX_coes[1][ 37][  6] =-2.123931303153388e+02;
  NLX_coes[1][ 37][  7] =-2.133035694296449e+02;
  NLX_coes[1][ 37][  8] =-2.142262686421981e+02;
  NLX_coes[1][ 37][  9] =-2.151619838252218e+02;
  NLX_coes[1][ 37][ 10] =-2.161112413059638e+02;
  NLX_coes[1][ 37][ 11] =-2.170743232766376e+02;
  NLX_coes[1][ 37][ 12] =-2.180512505759121e+02;
  NLX_coes[1][ 37][ 13] =-2.190417646446587e+02;
  NLX_coes[1][ 37][ 14] =-2.200453107761298e+02;
  NLX_coes[1][ 37][ 15] =-2.210610250264913e+02;
  NLX_coes[1][ 37][ 16] =-2.220877273038055e+02;
  NLX_coes[1][ 37][ 17] =-2.231239231779890e+02;
  NLX_coes[1][ 37][ 18] =-2.241678168092034e+02;
  NLX_coes[1][ 37][ 19] =-2.252173370241543e+02;
  NLX_coes[1][ 37][ 20] =-2.262701779166768e+02;
  NLX_coes[1][ 37][ 21] =-2.273238543605086e+02;
  NLX_coes[1][ 37][ 22] =-2.283757714996080e+02;
  NLX_coes[1][ 37][ 23] =-2.294233057181924e+02;
  NLX_coes[1][ 37][ 24] =-2.304638929953063e+02;
  NLX_coes[1][ 37][ 25] =-2.314951192114202e+02;
  NLX_coes[1][ 37][ 26] =-2.325148062117624e+02;
  NLX_coes[1][ 37][ 27] =-2.335210874797276e+02;
  NLX_coes[1][ 37][ 28] =-2.345124682100915e+02;
  NLX_coes[1][ 37][ 29] =-2.354878662720685e+02;
  NLX_coes[1][ 37][ 30] =-2.364466327215846e+02;
  NLX_coes[1][ 37][ 31] =-2.373885527774648e+02;
  NLX_coes[1][ 37][ 32] =-2.383138301524383e+02;
  NLX_coes[1][ 37][ 33] =-2.392230590653829e+02;
  NLX_coes[1][ 37][ 34] =-2.401171890280945e+02;
  NLX_coes[1][ 37][ 35] =-2.409974875612161e+02;
  NLX_coes[1][ 37][ 36] =-2.418655053043478e+02;
  NLX_coes[1][ 37][ 37] =-2.427230464435833e+02;
  NLX_coes[1][ 37][ 38] =-2.435721448636241e+02;
  NLX_coes[1][ 37][ 39] =-2.444150375547218e+02;
  NLX_coes[1][ 37][ 40] =-2.452541469797180e+02;
  NLX_coes[1][ 38][  0] =-2.142907586958956e+02;
  NLX_coes[1][ 38][  1] =-2.151600614686311e+02;
  NLX_coes[1][ 38][  2] =-2.160328482802003e+02;
  NLX_coes[1][ 38][  3] =-2.169112076968742e+02;
  NLX_coes[1][ 38][  4] =-2.177969734431123e+02;
  NLX_coes[1][ 38][  5] =-2.186917257963694e+02;
  NLX_coes[1][ 38][  6] =-2.195968046256690e+02;
  NLX_coes[1][ 38][  7] =-2.205133197994957e+02;
  NLX_coes[1][ 38][  8] =-2.214421564727591e+02;
  NLX_coes[1][ 38][  9] =-2.223839736493250e+02;
  NLX_coes[1][ 38][ 10] =-2.233391971328518e+02;
  NLX_coes[1][ 38][ 11] =-2.243080085490871e+02;
  NLX_coes[1][ 38][ 12] =-2.252903322061078e+02;
  NLX_coes[1][ 38][ 13] =-2.262858216547511e+02;
  NLX_coes[1][ 38][ 14] =-2.272938479458431e+02;
  NLX_coes[1][ 38][ 15] =-2.283134916995310e+02;
  NLX_coes[1][ 38][ 16] =-2.293435411532636e+02;
  NLX_coes[1][ 38][ 17] =-2.303824982939573e+02;
  NLX_coes[1][ 38][ 18] =-2.314285949646604e+02;
  NLX_coes[1][ 38][ 19] =-2.324798204260408e+02;
  NLX_coes[1][ 38][ 20] =-2.335339612146835e+02;
  NLX_coes[1][ 38][ 21] =-2.345886532619483e+02;
  NLX_coes[1][ 38][ 22] =-2.356414451505895e+02;
  NLX_coes[1][ 38][ 23] =-2.366898701845392e+02;
  NLX_coes[1][ 38][ 24] =-2.377315237877412e+02;
  NLX_coes[1][ 38][ 25] =-2.387641418314781e+02;
  NLX_coes[1][ 38][ 26] =-2.397856750159490e+02;
  NLX_coes[1][ 38][ 27] =-2.407943545436848e+02;
  NLX_coes[1][ 38][ 28] =-2.417887450583609e+02;
  NLX_coes[1][ 38][ 29] =-2.427677820973615e+02;
  NLX_coes[1][ 38][ 30] =-2.437307929276521e+02;
  NLX_coes[1][ 38][ 31] =-2.446775013516202e+02;
  NLX_coes[1][ 38][ 32] =-2.456080186376325e+02;
  NLX_coes[1][ 38][ 33] =-2.465228239673607e+02;
  NLX_coes[1][ 38][ 34] =-2.474227386145621e+02;
  NLX_coes[1][ 38][ 35] =-2.483088984890022e+02;
  NLX_coes[1][ 38][ 36] =-2.491827296923941e+02;
  NLX_coes[1][ 38][ 37] =-2.500459308688189e+02;
  NLX_coes[1][ 38][ 38] =-2.509004626547076e+02;
  NLX_coes[1][ 38][ 39] =-2.517485426356477e+02;
  NLX_coes[1][ 38][ 40] =-2.525926297092150e+02;
  NLX_coes[1][ 39][  0] =-2.214691483458714e+02;
  NLX_coes[1][ 39][  1] =-2.223418017770676e+02;
  NLX_coes[1][ 39][  2] =-2.232185027232269e+02;
  NLX_coes[1][ 39][  3] =-2.241011986110313e+02;
  NLX_coes[1][ 39][  4] =-2.249917314425981e+02;
  NLX_coes[1][ 39][  5] =-2.258916135806834e+02;
  NLX_coes[1][ 39][  6] =-2.268020966244110e+02;
  NLX_coes[1][ 39][  7] =-2.277241961876935e+02;
  NLX_coes[1][ 39][  8] =-2.286587009504405e+02;
  NLX_coes[1][ 39][  9] =-2.296061728478657e+02;
  NLX_coes[1][ 39][ 10] =-2.305669417840710e+02;
  NLX_coes[1][ 39][ 11] =-2.315410970110416e+02;
  NLX_coes[1][ 39][ 12] =-2.325284769721261e+02;
  NLX_coes[1][ 39][ 13] =-2.335286593624133e+02;
  NLX_coes[1][ 39][ 14] =-2.345409532061771e+02;
  NLX_coes[1][ 39][ 15] =-2.355643948010552e+02;
  NLX_coes[1][ 39][ 16] =-2.365977493716395e+02;
  NLX_coes[1][ 39][ 17] =-2.376395201677579e+02;
  NLX_coes[1][ 39][ 18] =-2.386879664989323e+02;
  NLX_coes[1][ 39][ 19] =-2.397411317876685e+02;
  NLX_coes[1][ 39][ 20] =-2.407968821344020e+02;
  NLX_coes[1][ 39][ 21] =-2.418529551228723e+02;
  NLX_coes[1][ 39][ 22] =-2.429070176975600e+02;
  NLX_coes[1][ 39][ 23] =-2.439567309977906e+02;
  NLX_coes[1][ 39][ 24] =-2.449998191582771e+02;
  NLX_coes[1][ 39][ 25] =-2.460341384268644e+02;
  NLX_coes[1][ 39][ 26] =-2.470577426420938e+02;
  NLX_coes[1][ 39][ 27] =-2.480689412475676e+02;
  NLX_coes[1][ 39][ 28] =-2.490663466168306e+02;
  NLX_coes[1][ 39][ 29] =-2.500489084586702e+02;
  NLX_coes[1][ 39][ 30] =-2.510159343338923e+02;
  NLX_coes[1][ 39][ 31] =-2.519670966656834e+02;
  NLX_coes[1][ 39][ 32] =-2.529024278927108e+02;
  NLX_coes[1][ 39][ 33] =-2.538223064646781e+02;
  NLX_coes[1][ 39][ 34] =-2.547274371550366e+02;
  NLX_coes[1][ 39][ 35] =-2.556188297067234e+02;
  NLX_coes[1][ 39][ 36] =-2.564977803468751e+02;
  NLX_coes[1][ 39][ 37] =-2.573658618906081e+02;
  NLX_coes[1][ 39][ 38] =-2.582249314828761e+02;
  NLX_coes[1][ 39][ 39] =-2.590771876223858e+02;
  NLX_coes[1][ 39][ 40] =-2.599249128812641e+02;
  NLX_coes[1][ 40][  0] =-2.286516584759781e+02;
  NLX_coes[1][ 40][  1] =-2.295268719620260e+02;
  NLX_coes[1][ 40][  2] =-2.304066011141933e+02;
  NLX_coes[1][ 40][  3] =-2.312930924432462e+02;
  NLX_coes[1][ 40][  4] =-2.321879787957897e+02;
  NLX_coes[1][ 40][  5] =-2.330926372555400e+02;
  NLX_coes[1][ 40][  6] =-2.340082033879061e+02;
  NLX_coes[1][ 40][  7] =-2.349355860499320e+02;
  NLX_coes[1][ 40][  8] =-2.358754725099188e+02;
  NLX_coes[1][ 40][  9] =-2.368283274053467e+02;
  NLX_coes[1][ 40][ 10] =-2.377943877773758e+02;
  NLX_coes[1][ 40][ 11] =-2.387736558666822e+02;
  NLX_coes[1][ 40][ 12] =-2.397658911732771e+02;
  NLX_coes[1][ 40][ 13] =-2.407706032763678e+02;
  NLX_coes[1][ 40][ 14] =-2.417870469546258e+02;
  NLX_coes[1][ 40][ 15] =-2.428142211749903e+02;
  NLX_coes[1][ 40][ 16] =-2.438508734856913e+02;
  NLX_coes[1][ 40][ 17] =-2.448955112234776e+02;
  NLX_coes[1][ 40][ 18] =-2.459464207000926e+02;
  NLX_coes[1][ 40][ 19] =-2.470016951508562e+02;
  NLX_coes[1][ 40][ 20] =-2.480592717032650e+02;
  NLX_coes[1][ 40][ 21] =-2.491169769699572e+02;
  NLX_coes[1][ 40][ 22] =-2.501725801288597e+02;
  NLX_coes[1][ 40][ 23] =-2.512238515943592e+02;
  NLX_coes[1][ 40][ 24] =-2.522686247027815e+02;
  NLX_coes[1][ 40][ 25] =-2.533048573407694e+02;
  NLX_coes[1][ 40][ 26] =-2.543306902336676e+02;
  NLX_coes[1][ 40][ 27] =-2.553444987462416e+02;
  NLX_coes[1][ 40][ 28] =-2.563449355406549e+02;
  NLX_coes[1][ 40][ 29] =-2.573309622375317e+02;
  NLX_coes[1][ 40][ 30] =-2.583018692342458e+02;
  NLX_coes[1][ 40][ 31] =-2.592572839188713e+02;
  NLX_coes[1][ 40][ 32] =-2.601971685453497e+02;
  NLX_coes[1][ 40][ 33] =-2.611218098991708e+02;
  NLX_coes[1][ 40][ 34] =-2.620318035276222e+02;
  NLX_coes[1][ 40][ 35] =-2.629280357503609e+02;
  NLX_coes[1][ 40][ 36] =-2.638116670236742e+02;
  NLX_coes[1][ 40][ 37] =-2.646841207300566e+02;
  NLX_coes[1][ 40][ 38] =-2.655470825259067e+02;
  NLX_coes[1][ 40][ 39] =-2.664024999574540e+02;
  NLX_coes[1][ 40][ 40] =-2.672529538553629e+02;
  NLX_coes[2][  0][  0] =+2.082903200050033e+01;
  NLX_coes[2][  0][  1] =+2.023496544914620e+01;
  NLX_coes[2][  0][  2] =+1.964057159335603e+01;
  NLX_coes[2][  0][  3] =+1.904632418060650e+01;
  NLX_coes[2][  0][  4] =+1.845312843443335e+01;
  NLX_coes[2][  0][  5] =+1.786219652264035e+01;
  NLX_coes[2][  0][  6] =+1.727494469939384e+01;
  NLX_coes[2][  0][  7] =+1.669281556576023e+01;
  NLX_coes[2][  0][  8] =+1.611701211846201e+01;
  NLX_coes[2][  0][  9] =+1.554828916049103e+01;
  NLX_coes[2][  0][ 10] =+1.498741419992337e+01;
  NLX_coes[2][  0][ 11] =+1.443644425477411e+01;
  NLX_coes[2][  0][ 12] =+1.389984820439746e+01;
  NLX_coes[2][  0][ 13] =+1.338533842480845e+01;
  NLX_coes[2][  0][ 14] =+1.290436780916383e+01;
  NLX_coes[2][  0][ 15] =+1.247219210067976e+01;
  NLX_coes[2][  0][ 16] =+1.210724259099642e+01;
  NLX_coes[2][  0][ 17] =+1.182809303186744e+01;
  NLX_coes[2][  0][ 18] =+1.164790167914445e+01;
  NLX_coes[2][  0][ 19] =+1.156985390427945e+01;
  NLX_coes[2][  0][ 20] =+1.158483675008324e+01;
  NLX_coes[2][  0][ 21] =+1.167257085205791e+01;
  NLX_coes[2][  0][ 22] =+1.180601980306299e+01;
  NLX_coes[2][  0][ 23] =+1.195341502533017e+01;
  NLX_coes[2][  0][ 24] =+1.209650687195110e+01;
  NLX_coes[2][  0][ 25] =+1.223896346535062e+01;
  NLX_coes[2][  0][ 26] =+1.239663475941089e+01;
  NLX_coes[2][  0][ 27] =+1.257851969488248e+01;
  NLX_coes[2][  0][ 28] =+1.277220168511590e+01;
  NLX_coes[2][  0][ 29] =+1.288519197900144e+01;
  NLX_coes[2][  0][ 30] =+1.285683218079087e+01;
  NLX_coes[2][  0][ 31] =+1.280897653509468e+01;
  NLX_coes[2][  0][ 32] =+1.270230178067344e+01;
  NLX_coes[2][  0][ 33] =+1.252463019511355e+01;
  NLX_coes[2][  0][ 34] =+1.225306633946050e+01;
  NLX_coes[2][  0][ 35] =+1.187028832190284e+01;
  NLX_coes[2][  0][ 36] =+1.142674395445054e+01;
  NLX_coes[2][  0][ 37] =+1.094422402825415e+01;
  NLX_coes[2][  0][ 38] =+1.043811021040767e+01;
  NLX_coes[2][  0][ 39] =+9.918778029190142e+00;
  NLX_coes[2][  0][ 40] =+9.393484506348184e+00;
  NLX_coes[2][  1][  0] =+1.808597297312070e+01;
  NLX_coes[2][  1][  1] =+1.749240412557409e+01;
  NLX_coes[2][  1][  2] =+1.689870620645688e+01;
  NLX_coes[2][  1][  3] =+1.630544807163523e+01;
  NLX_coes[2][  1][  4] =+1.571354393864096e+01;
  NLX_coes[2][  1][  5] =+1.512424326272831e+01;
  NLX_coes[2][  1][  6] =+1.453903372702622e+01;
  NLX_coes[2][  1][  7] =+1.395945268335102e+01;
  NLX_coes[2][  1][  8] =+1.338671809127150e+01;
  NLX_coes[2][  1][  9] =+1.282117904639744e+01;
  NLX_coes[2][  1][ 10] =+1.226221738840624e+01;
  NLX_coes[2][  1][ 11] =+1.171014289194396e+01;
  NLX_coes[2][  1][ 12] =+1.116817769438634e+01;
  NLX_coes[2][  1][ 13] =+1.064235345267540e+01;
  NLX_coes[2][  1][ 14] =+1.014295831778436e+01;
  NLX_coes[2][  1][ 15] =+9.683493526242959e+00;
  NLX_coes[2][  1][ 16] =+9.282331856546081e+00;
  NLX_coes[2][  1][ 17] =+8.960819846977749e+00;
  NLX_coes[2][  1][ 18] =+8.734323024018414e+00;
  NLX_coes[2][  1][ 19] =+8.606667985240438e+00;
  NLX_coes[2][  1][ 20] =+8.567732949259057e+00;
  NLX_coes[2][  1][ 21] =+8.594425335097529e+00;
  NLX_coes[2][  1][ 22] =+8.658896324336895e+00;
  NLX_coes[2][  1][ 23] =+8.722173243558768e+00;
  NLX_coes[2][  1][ 24] =+8.753006941276892e+00;
  NLX_coes[2][  1][ 25] =+8.753243983222456e+00;
  NLX_coes[2][  1][ 26] =+8.764464272922529e+00;
  NLX_coes[2][  1][ 27] =+8.789913735026355e+00;
  NLX_coes[2][  1][ 28] =+8.858663841982182e+00;
  NLX_coes[2][  1][ 29] =+8.912584848880718e+00;
  NLX_coes[2][  1][ 30] =+8.737687183710074e+00;
  NLX_coes[2][  1][ 31] =+8.551992211959920e+00;
  NLX_coes[2][  1][ 32] =+8.353641052800628e+00;
  NLX_coes[2][  1][ 33] =+8.159188758274318e+00;
  NLX_coes[2][  1][ 34] =+7.851759408307752e+00;
  NLX_coes[2][  1][ 35] =+7.464319331559876e+00;
  NLX_coes[2][  1][ 36] =+7.014586136857767e+00;
  NLX_coes[2][  1][ 37] =+6.527838230692441e+00;
  NLX_coes[2][  1][ 38] =+6.018882307440778e+00;
  NLX_coes[2][  1][ 39] =+5.497527059114696e+00;
  NLX_coes[2][  1][ 40] =+4.970456169216484e+00;
  NLX_coes[2][  2][  0] =+1.534155936572446e+01;
  NLX_coes[2][  2][  1] =+1.474853766643083e+01;
  NLX_coes[2][  2][  2] =+1.415560924880681e+01;
  NLX_coes[2][  2][  3] =+1.356335843222603e+01;
  NLX_coes[2][  2][  4] =+1.297272813514049e+01;
  NLX_coes[2][  2][  5] =+1.238499496011819e+01;
  NLX_coes[2][  2][  6] =+1.180175368684953e+01;
  NLX_coes[2][  2][  7] =+1.122485764170524e+01;
  NLX_coes[2][  2][  8] =+1.065612608288064e+01;
  NLX_coes[2][  2][  9] =+1.009636813539027e+01;
  NLX_coes[2][  2][ 10] =+9.543194268249351e+00;
  NLX_coes[2][  2][ 11] =+8.993626492632130e+00;
  NLX_coes[2][  2][ 12] =+8.449893611707052e+00;
  NLX_coes[2][  2][ 13] =+7.917365686456916e+00;
  NLX_coes[2][  2][ 14] =+7.404724131163702e+00;
  NLX_coes[2][  2][ 15] =+6.920783910346184e+00;
  NLX_coes[2][  2][ 16] =+6.482152049044799e+00;
  NLX_coes[2][  2][ 17] =+6.115642072921748e+00;
  NLX_coes[2][  2][ 18] =+5.843507535275625e+00;
  NLX_coes[2][  2][ 19] =+5.668732460041341e+00;
  NLX_coes[2][  2][ 20] =+5.576849202455085e+00;
  NLX_coes[2][  2][ 21] =+5.546294485361662e+00;
  NLX_coes[2][  2][ 22] =+5.544712889466473e+00;
  NLX_coes[2][  2][ 23] =+5.552270430963864e+00;
  NLX_coes[2][  2][ 24] =+5.475635890834178e+00;
  NLX_coes[2][  2][ 25] =+5.302422477417436e+00;
  NLX_coes[2][  2][ 26] =+5.158890432727230e+00;
  NLX_coes[2][  2][ 27] =+5.075841717313366e+00;
  NLX_coes[2][  2][ 28] =+4.846922265030766e+00;
  NLX_coes[2][  2][ 29] =+4.564762260959379e+00;
  NLX_coes[2][  2][ 30] =+4.375562779188914e+00;
  NLX_coes[2][  2][ 31] =+4.319523159728407e+00;
  NLX_coes[2][  2][ 32] =+4.118558667357449e+00;
  NLX_coes[2][  2][ 33] =+3.821372815555506e+00;
  NLX_coes[2][  2][ 34] =+3.478167197978442e+00;
  NLX_coes[2][  2][ 35] =+3.061938201513915e+00;
  NLX_coes[2][  2][ 36] =+2.602427008094397e+00;
  NLX_coes[2][  2][ 37] =+2.111871440380150e+00;
  NLX_coes[2][  2][ 38] =+1.600535830032542e+00;
  NLX_coes[2][  2][ 39] =+1.076913486342834e+00;
  NLX_coes[2][  2][ 40] =+5.474707349375869e-01;
  NLX_coes[2][  3][  0] =+1.259446189980934e+01;
  NLX_coes[2][  3][  1] =+1.200209031127760e+01;
  NLX_coes[2][  3][  2] =+1.141000163468779e+01;
  NLX_coes[2][  3][  3] =+1.081885491601879e+01;
  NLX_coes[2][  3][  4] =+1.022961209733221e+01;
  NLX_coes[2][  3][  5] =+9.643533929223493e+00;
  NLX_coes[2][  3][  6] =+9.062240273585422e+00;
  NLX_coes[2][  3][  7] =+8.487881079342062e+00;
  NLX_coes[2][  3][  8] =+7.923353750443054e+00;
  NLX_coes[2][  3][  9] =+7.371683264761300e+00;
  NLX_coes[2][  3][ 10] =+6.830616182802810e+00;
  NLX_coes[2][  3][ 11] =+6.289302311080347e+00;
  NLX_coes[2][  3][ 12] =+5.745975849584474e+00;
  NLX_coes[2][  3][ 13] =+5.216744148013208e+00;
  NLX_coes[2][  3][ 14] =+4.702295022885258e+00;
  NLX_coes[2][  3][ 15] =+4.204316313739569e+00;
  NLX_coes[2][  3][ 16] =+3.733549978422893e+00;
  NLX_coes[2][  3][ 17] =+3.318727798046443e+00;
  NLX_coes[2][  3][ 18] =+3.007676882829020e+00;
  NLX_coes[2][  3][ 19] =+2.792038616246486e+00;
  NLX_coes[2][  3][ 20] =+2.639712648333333e+00;
  NLX_coes[2][  3][ 21] =+2.548910437892553e+00;
  NLX_coes[2][  3][ 22] =+2.451116957265528e+00;
  NLX_coes[2][  3][ 23] =+2.456477248036217e+00;
  NLX_coes[2][  3][ 24] =+2.300593916732893e+00;
  NLX_coes[2][  3][ 25] =+1.985621878954434e+00;
  NLX_coes[2][  3][ 26] =+1.619803946048255e+00;
  NLX_coes[2][  3][ 27] =+1.185439228520908e+00;
  NLX_coes[2][  3][ 28] =+7.140334569757963e-01;
  NLX_coes[2][  3][ 29] =+4.301776064973675e-01;
  NLX_coes[2][  3][ 30] =+3.065205528539588e-01;
  NLX_coes[2][  3][ 31] =+7.676260426223155e-02;
  NLX_coes[2][  3][ 32] =-2.026627168583295e-01;
  NLX_coes[2][  3][ 33] =-5.310924833547916e-01;
  NLX_coes[2][  3][ 34] =-9.117422350525962e-01;
  NLX_coes[2][  3][ 35] =-1.336790657352012e+00;
  NLX_coes[2][  3][ 36] =-1.802369169412963e+00;
  NLX_coes[2][  3][ 37] =-2.297103726843534e+00;
  NLX_coes[2][  3][ 38] =-2.811554476525801e+00;
  NLX_coes[2][  3][ 39] =-3.338285176242093e+00;
  NLX_coes[2][  3][ 40] =-3.871329977437692e+00;
  NLX_coes[2][  4][  0] =+9.843032937283651e+00;
  NLX_coes[2][  4][  1] =+9.251353416132115e+00;
  NLX_coes[2][  4][  2] =+8.660204429624894e+00;
  NLX_coes[2][  4][  3] =+8.070360287304002e+00;
  NLX_coes[2][  4][  4] =+7.482826677674924e+00;
  NLX_coes[2][  4][  5] =+6.898830713962799e+00;
  NLX_coes[2][  4][  6] =+6.319859183144787e+00;
  NLX_coes[2][  4][  7] =+5.747832537442727e+00;
  NLX_coes[2][  4][  8] =+5.185766759040621e+00;
  NLX_coes[2][  4][  9] =+4.639685277657215e+00;
  NLX_coes[2][  4][ 10] =+4.120336435787562e+00;
  NLX_coes[2][  4][ 11] =+3.597187861322500e+00;
  NLX_coes[2][  4][ 12] =+3.045826486780316e+00;
  NLX_coes[2][  4][ 13] =+2.540900901550757e+00;
  NLX_coes[2][  4][ 14] =+2.041624978754525e+00;
  NLX_coes[2][  4][ 15] =+1.552599755418243e+00;
  NLX_coes[2][  4][ 16] =+1.075332908562365e+00;
  NLX_coes[2][  4][ 17] =+6.104315604241621e-01;
  NLX_coes[2][  4][ 18] =+2.799078160288949e-01;
  NLX_coes[2][  4][ 19] =+3.953251043011142e-02;
  NLX_coes[2][  4][ 20] =-1.950481833164525e-01;
  NLX_coes[2][  4][ 21] =-4.022524314796998e-01;
  NLX_coes[2][  4][ 22] =-5.677053371502566e-01;
  NLX_coes[2][  4][ 23] =-7.013561399825242e-01;
  NLX_coes[2][  4][ 24] =-8.866568118542680e-01;
  NLX_coes[2][  4][ 25] =-1.135787412552344e+00;
  NLX_coes[2][  4][ 26] =-1.479630942413904e+00;
  NLX_coes[2][  4][ 27] =-1.925646541411500e+00;
  NLX_coes[2][  4][ 28] =-2.612588033746403e+00;
  NLX_coes[2][  4][ 29] =-3.276121777576871e+00;
  NLX_coes[2][  4][ 30] =-3.718694564181374e+00;
  NLX_coes[2][  4][ 31] =-4.095996694746201e+00;
  NLX_coes[2][  4][ 32] =-4.465429805570720e+00;
  NLX_coes[2][  4][ 33] =-4.852005289041951e+00;
  NLX_coes[2][  4][ 34] =-5.267586978271035e+00;
  NLX_coes[2][  4][ 35] =-5.713386005495331e+00;
  NLX_coes[2][  4][ 36] =-6.189424572172038e+00;
  NLX_coes[2][  4][ 37] =-6.690316331205244e+00;
  NLX_coes[2][  4][ 38] =-7.209573211025329e+00;
  NLX_coes[2][  4][ 39] =-7.741065751598715e+00;
  NLX_coes[2][  4][ 40] =-8.279387101906403e+00;
  NLX_coes[2][  5][  0] =+7.085205210183470e+00;
  NLX_coes[2][  5][  1] =+6.494180176243458e+00;
  NLX_coes[2][  5][  2] =+5.904048007739794e+00;
  NLX_coes[2][  5][  3] =+5.315758835825223e+00;
  NLX_coes[2][  5][  4] =+4.730430124544906e+00;
  NLX_coes[2][  5][  5] =+4.149332395407274e+00;
  NLX_coes[2][  5][  6] =+3.573903554648089e+00;
  NLX_coes[2][  5][  7] =+3.005775005817094e+00;
  NLX_coes[2][  5][  8] =+2.446753824219315e+00;
  NLX_coes[2][  5][  9] =+1.899562563990016e+00;
  NLX_coes[2][  5][ 10] =+1.378130007127984e+00;
  NLX_coes[2][  5][ 11] =+8.649618327086757e-01;
  NLX_coes[2][  5][ 12] =+3.650809982924080e-01;
  NLX_coes[2][  5][ 13] =-1.307794668676507e-01;
  NLX_coes[2][  5][ 14] =-6.059783636005738e-01;
  NLX_coes[2][  5][ 15] =-1.060503439173133e+00;
  NLX_coes[2][  5][ 16] =-1.485086568426853e+00;
  NLX_coes[2][  5][ 17] =-1.857719171207951e+00;
  NLX_coes[2][  5][ 18] =-2.195237075626475e+00;
  NLX_coes[2][  5][ 19] =-2.546508747332521e+00;
  NLX_coes[2][  5][ 20] =-2.941290008014072e+00;
  NLX_coes[2][  5][ 21] =-3.306118020427013e+00;
  NLX_coes[2][  5][ 22] =-3.619730592572075e+00;
  NLX_coes[2][  5][ 23] =-3.853645659369914e+00;
  NLX_coes[2][  5][ 24] =-4.072838132631767e+00;
  NLX_coes[2][  5][ 25] =-4.347156795308099e+00;
  NLX_coes[2][  5][ 26] =-4.628794859584707e+00;
  NLX_coes[2][  5][ 27] =-5.427084903205077e+00;
  NLX_coes[2][  5][ 28] =-6.230600462336709e+00;
  NLX_coes[2][  5][ 29] =-7.048914917942551e+00;
  NLX_coes[2][  5][ 30] =-7.699857097426865e+00;
  NLX_coes[2][  5][ 31] =-8.221887645096171e+00;
  NLX_coes[2][  5][ 32] =-8.685007973624312e+00;
  NLX_coes[2][  5][ 33] =-9.131755403037241e+00;
  NLX_coes[2][  5][ 34] =-9.585167980426853e+00;
  NLX_coes[2][  5][ 35] =-1.005562840262276e+01;
  NLX_coes[2][  5][ 36] =-1.054672916289578e+01;
  NLX_coes[2][  5][ 37] =-1.105751723094438e+01;
  NLX_coes[2][  5][ 38] =-1.158446075797418e+01;
  NLX_coes[2][  5][ 39] =-1.212310947647214e+01;
  NLX_coes[2][  5][ 40] =-1.266884099681869e+01;
  NLX_coes[2][  6][  0] =+4.318478373697429e+00;
  NLX_coes[2][  6][  1] =+3.727939903663733e+00;
  NLX_coes[2][  6][  2] =+3.138796321018871e+00;
  NLX_coes[2][  6][  3] =+2.552249238509628e+00;
  NLX_coes[2][  6][  4] =+1.969627121880692e+00;
  NLX_coes[2][  6][  5] =+1.392383427005974e+00;
  NLX_coes[2][  6][  6] =+8.221228976934771e-01;
  NLX_coes[2][  6][  7] =+2.607059544689007e-01;
  NLX_coes[2][  6][  8] =-2.893932334823333e-01;
  NLX_coes[2][  6][  9] =-8.244529185248280e-01;
  NLX_coes[2][  6][ 10] =-1.341305094444287e+00;
  NLX_coes[2][  6][ 11] =-1.833978422030890e+00;
  NLX_coes[2][  6][ 12] =-2.294870641988149e+00;
  NLX_coes[2][  6][ 13] =-2.749993023100481e+00;
  NLX_coes[2][  6][ 14] =-3.176303542554150e+00;
  NLX_coes[2][  6][ 15] =-3.570755423976413e+00;
  NLX_coes[2][  6][ 16] =-3.934850719790370e+00;
  NLX_coes[2][  6][ 17] =-4.282255558810005e+00;
  NLX_coes[2][  6][ 18] =-4.643457745226915e+00;
  NLX_coes[2][  6][ 19] =-4.888027243473885e+00;
  NLX_coes[2][  6][ 20] =-5.249174187967974e+00;
  NLX_coes[2][  6][ 21] =-5.813914314723054e+00;
  NLX_coes[2][  6][ 22] =-6.416394437541284e+00;
  NLX_coes[2][  6][ 23] =-7.016295159356807e+00;
  NLX_coes[2][  6][ 24] =-7.572405927487254e+00;
  NLX_coes[2][  6][ 25] =-8.167976451195699e+00;
  NLX_coes[2][  6][ 26] =-8.407534060781698e+00;
  NLX_coes[2][  6][ 27] =-9.172301430518946e+00;
  NLX_coes[2][  6][ 28] =-1.016487887532517e+01;
  NLX_coes[2][  6][ 29] =-1.103189796958188e+01;
  NLX_coes[2][  6][ 30] =-1.174737501597697e+01;
  NLX_coes[2][  6][ 31] =-1.234482319743539e+01;
  NLX_coes[2][  6][ 32] =-1.287316226349865e+01;
  NLX_coes[2][  6][ 33] =-1.336998614511121e+01;
  NLX_coes[2][  6][ 34] =-1.385958914922244e+01;
  NLX_coes[2][  6][ 35] =-1.435580252830987e+01;
  NLX_coes[2][  6][ 36] =-1.486511593146456e+01;
  NLX_coes[2][  6][ 37] =-1.538932141240370e+01;
  NLX_coes[2][  6][ 38] =-1.592718339388376e+01;
  NLX_coes[2][  6][ 39] =-1.647580537584544e+01;
  NLX_coes[2][  6][ 40] =-1.703141943518329e+01;
  NLX_coes[2][  7][  0] =+1.539945764266804e+00;
  NLX_coes[2][  7][  1] =+9.495323468276092e-01;
  NLX_coes[2][  7][  2] =+3.611390033529734e-01;
  NLX_coes[2][  7][  3] =-2.237163079053083e-01;
  NLX_coes[2][  7][  4] =-8.034039652039564e-01;
  NLX_coes[2][  7][  5] =-1.376142189289343e+00;
  NLX_coes[2][  7][  6] =-1.939910547994318e+00;
  NLX_coes[2][  7][  7] =-2.492282472464993e+00;
  NLX_coes[2][  7][  8] =-3.030135967507781e+00;
  NLX_coes[2][  7][  9] =-3.549269119122277e+00;
  NLX_coes[2][  7][ 10] =-4.045081346900786e+00;
  NLX_coes[2][  7][ 11] =-4.510766018422740e+00;
  NLX_coes[2][  7][ 12] =-4.943614299964646e+00;
  NLX_coes[2][  7][ 13] =-5.347847763387029e+00;
  NLX_coes[2][  7][ 14] =-5.715762466722840e+00;
  NLX_coes[2][  7][ 15] =-6.042654952777539e+00;
  NLX_coes[2][  7][ 16] =-6.332707388939460e+00;
  NLX_coes[2][  7][ 17] =-6.604519100033766e+00;
  NLX_coes[2][  7][ 18] =-6.889564572505217e+00;
  NLX_coes[2][  7][ 19] =-7.201049421443595e+00;
  NLX_coes[2][  7][ 20] =-7.576316144344487e+00;
  NLX_coes[2][  7][ 21] =-8.055296871774962e+00;
  NLX_coes[2][  7][ 22] =-8.638865183194536e+00;
  NLX_coes[2][  7][ 23] =-9.567824445893805e+00;
  NLX_coes[2][  7][ 24] =-1.057098379626969e+01;
  NLX_coes[2][  7][ 25] =-1.172274951013762e+01;
  NLX_coes[2][  7][ 26] =-1.228183902280786e+01;
  NLX_coes[2][  7][ 27] =-1.335030138225320e+01;
  NLX_coes[2][  7][ 28] =-1.434564819983781e+01;
  NLX_coes[2][  7][ 29] =-1.514542785088344e+01;
  NLX_coes[2][  7][ 30] =-1.584567978153555e+01;
  NLX_coes[2][  7][ 31] =-1.646644417379961e+01;
  NLX_coes[2][  7][ 32] =-1.703131539204320e+01;
  NLX_coes[2][  7][ 33] =-1.756470024298747e+01;
  NLX_coes[2][  7][ 34] =-1.808580580379259e+01;
  NLX_coes[2][  7][ 35] =-1.860749124755828e+01;
  NLX_coes[2][  7][ 36] =-1.913706148771998e+01;
  NLX_coes[2][  7][ 37] =-1.967767771927701e+01;
  NLX_coes[2][  7][ 38] =-2.022958121236606e+01;
  NLX_coes[2][  7][ 39] =-2.079107913644188e+01;
  NLX_coes[2][  7][ 40] =-2.135919447079407e+01;
  NLX_coes[2][  8][  0] =-1.253622864407534e+00;
  NLX_coes[2][  8][  1] =-1.844524023102558e+00;
  NLX_coes[2][  8][  2] =-2.432713940177101e+00;
  NLX_coes[2][  8][  3] =-3.016330768384552e+00;
  NLX_coes[2][  8][  4] =-3.593407858077430e+00;
  NLX_coes[2][  8][  5] =-4.161789989266452e+00;
  NLX_coes[2][  8][  6] =-4.719007695111358e+00;
  NLX_coes[2][  8][  7] =-5.262093040861794e+00;
  NLX_coes[2][  8][  8] =-5.787351186459887e+00;
  NLX_coes[2][  8][  9] =-6.290231157592014e+00;
  NLX_coes[2][  8][ 10] =-6.765396265587776e+00;
  NLX_coes[2][  8][ 11] =-7.207120689510315e+00;
  NLX_coes[2][  8][ 12] =-7.611835371848722e+00;
  NLX_coes[2][  8][ 13] =-7.976247929998422e+00;
  NLX_coes[2][  8][ 14] =-8.295800877347112e+00;
  NLX_coes[2][  8][ 15] =-8.566070075291750e+00;
  NLX_coes[2][  8][ 16] =-8.788947139103062e+00;
  NLX_coes[2][  8][ 17] =-8.980301488348998e+00;
  NLX_coes[2][  8][ 18] =-9.185597599193027e+00;
  NLX_coes[2][  8][ 19] =-9.512846343504480e+00;
  NLX_coes[2][  8][ 20] =-9.892758274088592e+00;
  NLX_coes[2][  8][ 21] =-1.045325279111675e+01;
  NLX_coes[2][  8][ 22] =-1.130551380517764e+01;
  NLX_coes[2][  8][ 23] =-1.251607429992636e+01;
  NLX_coes[2][  8][ 24] =-1.387281070726968e+01;
  NLX_coes[2][  8][ 25] =-1.509525811294619e+01;
  NLX_coes[2][  8][ 26] =-1.626804660744622e+01;
  NLX_coes[2][  8][ 27] =-1.747369875718147e+01;
  NLX_coes[2][  8][ 28] =-1.845659737105190e+01;
  NLX_coes[2][  8][ 29] =-1.923488438320486e+01;
  NLX_coes[2][  8][ 30] =-1.992561487512662e+01;
  NLX_coes[2][  8][ 31] =-2.055716944015724e+01;
  NLX_coes[2][  8][ 32] =-2.114603719449291e+01;
  NLX_coes[2][  8][ 33] =-2.170817845694273e+01;
  NLX_coes[2][  8][ 34] =-2.225753690383052e+01;
  NLX_coes[2][  8][ 35] =-2.280460239796738e+01;
  NLX_coes[2][  8][ 36] =-2.335617968762884e+01;
  NLX_coes[2][  8][ 37] =-2.391586271373649e+01;
  NLX_coes[2][  8][ 38] =-2.448471586551290e+01;
  NLX_coes[2][  8][ 39] =-2.506189972821661e+01;
  NLX_coes[2][  8][ 40] =-2.564512286566001e+01;
  NLX_coes[2][  9][  0] =-4.065643527744651e+00;
  NLX_coes[2][  9][  1] =-4.657930576224475e+00;
  NLX_coes[2][  9][  2] =-5.246833295719305e+00;
  NLX_coes[2][  9][  3] =-5.830171879655085e+00;
  NLX_coes[2][  9][  4] =-6.405681776330812e+00;
  NLX_coes[2][  9][  5] =-6.970901622374662e+00;
  NLX_coes[2][  9][  6] =-7.523038036233777e+00;
  NLX_coes[2][  9][  7] =-8.058809403707034e+00;
  NLX_coes[2][  9][  8] =-8.574303580727923e+00;
  NLX_coes[2][  9][  9] =-9.064947113353497e+00;
  NLX_coes[2][  9][ 10] =-9.525621235286108e+00;
  NLX_coes[2][  9][ 11] =-9.951261777898150e+00;
  NLX_coes[2][  9][ 12] =-1.033751550038385e+01;
  NLX_coes[2][  9][ 13] =-1.068016544432188e+01;
  NLX_coes[2][  9][ 14] =-1.097566397820449e+01;
  NLX_coes[2][  9][ 15] =-1.122199201576049e+01;
  NLX_coes[2][  9][ 16] =-1.142187642704608e+01;
  NLX_coes[2][  9][ 17] =-1.158611808749863e+01;
  NLX_coes[2][  9][ 18] =-1.174974265495035e+01;
  NLX_coes[2][  9][ 19] =-1.210836427206583e+01;
  NLX_coes[2][  9][ 20] =-1.266515932720089e+01;
  NLX_coes[2][  9][ 21] =-1.354596423270858e+01;
  NLX_coes[2][  9][ 22] =-1.468581182574940e+01;
  NLX_coes[2][  9][ 23] =-1.598802741748149e+01;
  NLX_coes[2][  9][ 24] =-1.745520545736278e+01;
  NLX_coes[2][  9][ 25] =-1.889819057271050e+01;
  NLX_coes[2][  9][ 26] =-2.022813635146064e+01;
  NLX_coes[2][  9][ 27] =-2.141025140555594e+01;
  NLX_coes[2][  9][ 28] =-2.239164978507776e+01;
  NLX_coes[2][  9][ 29] =-2.321317582271028e+01;
  NLX_coes[2][  9][ 30] =-2.392997159058057e+01;
  NLX_coes[2][  9][ 31] =-2.458450387068871e+01;
  NLX_coes[2][  9][ 32] =-2.519959125124777e+01;
  NLX_coes[2][  9][ 33] =-2.579022877589722e+01;
  NLX_coes[2][  9][ 34] =-2.636795925896039e+01;
  NLX_coes[2][  9][ 35] =-2.694158371356320e+01;
  NLX_coes[2][  9][ 36] =-2.751722791323012e+01;
  NLX_coes[2][  9][ 37] =-2.809854433286452e+01;
  NLX_coes[2][  9][ 38] =-2.868706542665225e+01;
  NLX_coes[2][  9][ 39] =-2.928257323202607e+01;
  NLX_coes[2][  9][ 40] =-2.988338351530350e+01;
  NLX_coes[2][ 10][  0] =-6.899540422519509e+00;
  NLX_coes[2][ 10][  1] =-7.494395492155425e+00;
  NLX_coes[2][ 10][  2] =-8.085300114379027e+00;
  NLX_coes[2][ 10][  3] =-8.669833575536073e+00;
  NLX_coes[2][ 10][  4] =-9.245534643587915e+00;
  NLX_coes[2][ 10][  5] =-9.809780449764144e+00;
  NLX_coes[2][ 10][  6] =-1.035966632856103e+01;
  NLX_coes[2][ 10][  7] =-1.089189737405966e+01;
  NLX_coes[2][ 10][  8] =-1.140272654064828e+01;
  NLX_coes[2][ 10][  9] =-1.188799482162709e+01;
  NLX_coes[2][ 10][ 10] =-1.234333031378620e+01;
  NLX_coes[2][ 10][ 11] =-1.276458348078400e+01;
  NLX_coes[2][ 10][ 12] =-1.314819296312547e+01;
  NLX_coes[2][ 10][ 13] =-1.349154622350835e+01;
  NLX_coes[2][ 10][ 14] =-1.379390881151456e+01;
  NLX_coes[2][ 10][ 15] =-1.405829222585358e+01;
  NLX_coes[2][ 10][ 16] =-1.429585115720678e+01;
  NLX_coes[2][ 10][ 17] =-1.453625477595986e+01;
  NLX_coes[2][ 10][ 18] =-1.484958601721171e+01;
  NLX_coes[2][ 10][ 19] =-1.535403737692132e+01;
  NLX_coes[2][ 10][ 20] =-1.604669769384819e+01;
  NLX_coes[2][ 10][ 21] =-1.701483324193038e+01;
  NLX_coes[2][ 10][ 22] =-1.826701187170526e+01;
  NLX_coes[2][ 10][ 23] =-1.971029428619619e+01;
  NLX_coes[2][ 10][ 24] =-2.124631153719858e+01;
  NLX_coes[2][ 10][ 25] =-2.281051047204742e+01;
  NLX_coes[2][ 10][ 26] =-2.412947551385947e+01;
  NLX_coes[2][ 10][ 27] =-2.526444047096632e+01;
  NLX_coes[2][ 10][ 28] =-2.623971865381891e+01;
  NLX_coes[2][ 10][ 29] =-2.708808450876453e+01;
  NLX_coes[2][ 10][ 30] =-2.784124944889351e+01;
  NLX_coes[2][ 10][ 31] =-2.853100195919655e+01;
  NLX_coes[2][ 10][ 32] =-2.917976476750703e+01;
  NLX_coes[2][ 10][ 33] =-2.980296303159377e+01;
  NLX_coes[2][ 10][ 34] =-3.041173936935244e+01;
  NLX_coes[2][ 10][ 35] =-3.101430028246553e+01;
  NLX_coes[2][ 10][ 36] =-3.161646361551866e+01;
  NLX_coes[2][ 10][ 37] =-3.222195540801447e+01;
  NLX_coes[2][ 10][ 38] =-3.283266983162645e+01;
  NLX_coes[2][ 10][ 39] =-3.344891012918284e+01;
  NLX_coes[2][ 10][ 40] =-3.406957038483639e+01;
  NLX_coes[2][ 11][  0] =-9.758553652707105e+00;
  NLX_coes[2][ 11][  1] =-1.035740356611683e+01;
  NLX_coes[2][ 11][  2] =-1.095191686646487e+01;
  NLX_coes[2][ 11][  3] =-1.153954534968206e+01;
  NLX_coes[2][ 11][  4] =-1.211776790524294e+01;
  NLX_coes[2][ 11][  5] =-1.268397679985790e+01;
  NLX_coes[2][ 11][  6] =-1.323538685965516e+01;
  NLX_coes[2][ 11][  7] =-1.376898004847104e+01;
  NLX_coes[2][ 11][  8] =-1.428151354254890e+01;
  NLX_coes[2][ 11][  9] =-1.476962900130284e+01;
  NLX_coes[2][ 11][ 10] =-1.523010629659669e+01;
  NLX_coes[2][ 11][ 11] =-1.566026976057240e+01;
  NLX_coes[2][ 11][ 12] =-1.605850737373560e+01;
  NLX_coes[2][ 11][ 13] =-1.642509529718479e+01;
  NLX_coes[2][ 11][ 14] =-1.676363635848138e+01;
  NLX_coes[2][ 11][ 15] =-1.708373993769800e+01;
  NLX_coes[2][ 11][ 16] =-1.740577736731327e+01;
  NLX_coes[2][ 11][ 17] =-1.776815233424346e+01;
  NLX_coes[2][ 11][ 18] =-1.823311434743986e+01;
  NLX_coes[2][ 11][ 19] =-1.886419168735673e+01;
  NLX_coes[2][ 11][ 20] =-1.969987263899715e+01;
  NLX_coes[2][ 11][ 21] =-2.077535112052705e+01;
  NLX_coes[2][ 11][ 22] =-2.208396715258178e+01;
  NLX_coes[2][ 11][ 23] =-2.355575741088311e+01;
  NLX_coes[2][ 11][ 24] =-2.510080231170794e+01;
  NLX_coes[2][ 11][ 25] =-2.665984389524468e+01;
  NLX_coes[2][ 11][ 26] =-2.791639086264122e+01;
  NLX_coes[2][ 11][ 27] =-2.901846888755064e+01;
  NLX_coes[2][ 11][ 28] =-2.999768394365405e+01;
  NLX_coes[2][ 11][ 29] =-3.087295682534610e+01;
  NLX_coes[2][ 11][ 30] =-3.166480067817765e+01;
  NLX_coes[2][ 11][ 31] =-3.239525213461401e+01;
  NLX_coes[2][ 11][ 32] =-3.308327298758600e+01;
  NLX_coes[2][ 11][ 33] =-3.374341828761430e+01;
  NLX_coes[2][ 11][ 34] =-3.438658071144388e+01;
  NLX_coes[2][ 11][ 35] =-3.502084212501479e+01;
  NLX_coes[2][ 11][ 36] =-3.565203056291535e+01;
  NLX_coes[2][ 11][ 37] =-3.628406878974404e+01;
  NLX_coes[2][ 11][ 38] =-3.691922371830590e+01;
  NLX_coes[2][ 11][ 39] =-3.755830024913471e+01;
  NLX_coes[2][ 11][ 40] =-3.820077704306838e+01;
  NLX_coes[2][ 12][  0] =-1.264555970930489e+01;
  NLX_coes[2][ 12][  1] =-1.325000555453367e+01;
  NLX_coes[2][ 12][  2] =-1.384995304176577e+01;
  NLX_coes[2][ 12][  3] =-1.444285687492799e+01;
  NLX_coes[2][ 12][  4] =-1.502628119802764e+01;
  NLX_coes[2][ 12][  5] =-1.559780496182827e+01;
  NLX_coes[2][ 12][  6] =-1.615496658985067e+01;
  NLX_coes[2][ 12][  7] =-1.669525955547725e+01;
  NLX_coes[2][ 12][  8] =-1.721620111695147e+01;
  NLX_coes[2][ 12][  9] =-1.771550356336344e+01;
  NLX_coes[2][ 12][ 10] =-1.819137991690469e+01;
  NLX_coes[2][ 12][ 11] =-1.864300610755334e+01;
  NLX_coes[2][ 12][ 12] =-1.907119352593089e+01;
  NLX_coes[2][ 12][ 13] =-1.947941630967647e+01;
  NLX_coes[2][ 12][ 14] =-1.987540099831256e+01;
  NLX_coes[2][ 12][ 15] =-2.027354161562920e+01;
  NLX_coes[2][ 12][ 16] =-2.069810798923817e+01;
  NLX_coes[2][ 12][ 17] =-2.118607705403909e+01;
  NLX_coes[2][ 12][ 18] =-2.178521053959624e+01;
  NLX_coes[2][ 12][ 19] =-2.254045857574647e+01;
  NLX_coes[2][ 12][ 20] =-2.348770053639179e+01;
  NLX_coes[2][ 12][ 21] =-2.463906492878763e+01;
  NLX_coes[2][ 12][ 22] =-2.597551503892711e+01;
  NLX_coes[2][ 12][ 23] =-2.743809037935578e+01;
  NLX_coes[2][ 12][ 24] =-2.895224461680103e+01;
  NLX_coes[2][ 12][ 25] =-3.039493967706911e+01;
  NLX_coes[2][ 12][ 26] =-3.157390072861871e+01;
  NLX_coes[2][ 12][ 27] =-3.267364067606967e+01;
  NLX_coes[2][ 12][ 28] =-3.367102078105821e+01;
  NLX_coes[2][ 12][ 29] =-3.457814790995683e+01;
  NLX_coes[2][ 12][ 30] =-3.541012979270148e+01;
  NLX_coes[2][ 12][ 31] =-3.618341593479111e+01;
  NLX_coes[2][ 12][ 32] =-3.691351080413801e+01;
  NLX_coes[2][ 12][ 33] =-3.761344293290564e+01;
  NLX_coes[2][ 12][ 34] =-3.829355612434946e+01;
  NLX_coes[2][ 12][ 35] =-3.896180569051741e+01;
  NLX_coes[2][ 12][ 36] =-3.962411201409871e+01;
  NLX_coes[2][ 12][ 37] =-4.028465083167364e+01;
  NLX_coes[2][ 12][ 38] =-4.094607747021781e+01;
  NLX_coes[2][ 12][ 39] =-4.160969787161660e+01;
  NLX_coes[2][ 12][ 40] =-4.227558808242790e+01;
  NLX_coes[2][ 13][  0] =-1.556292724529561e+01;
  NLX_coes[2][ 13][  1] =-1.617465500591999e+01;
  NLX_coes[2][ 13][  2] =-1.678195958430890e+01;
  NLX_coes[2][ 13][  3] =-1.738242759010761e+01;
  NLX_coes[2][ 13][  4] =-1.797384047847112e+01;
  NLX_coes[2][ 13][  5] =-1.855410592235234e+01;
  NLX_coes[2][ 13][  6] =-1.912123839088795e+01;
  NLX_coes[2][ 13][  7] =-1.967339913115582e+01;
  NLX_coes[2][ 13][  8] =-2.020901348517378e+01;
  NLX_coes[2][ 13][  9] =-2.072698909438549e+01;
  NLX_coes[2][ 13][ 10] =-2.122706138573745e+01;
  NLX_coes[2][ 13][ 11] =-2.171029662861679e+01;
  NLX_coes[2][ 13][ 12] =-2.217980431231512e+01;
  NLX_coes[2][ 13][ 13] =-2.264173033810772e+01;
  NLX_coes[2][ 13][ 14] =-2.310659071858669e+01;
  NLX_coes[2][ 13][ 15] =-2.359088746790319e+01;
  NLX_coes[2][ 13][ 16] =-2.411857271579195e+01;
  NLX_coes[2][ 13][ 17] =-2.472115182260131e+01;
  NLX_coes[2][ 13][ 18] =-2.543431221273971e+01;
  NLX_coes[2][ 13][ 19] =-2.629146704106564e+01;
  NLX_coes[2][ 13][ 20] =-2.731810603916112e+01;
  NLX_coes[2][ 13][ 21] =-2.851590826131807e+01;
  NLX_coes[2][ 13][ 22] =-2.984720557412055e+01;
  NLX_coes[2][ 13][ 23] =-3.126958330102791e+01;
  NLX_coes[2][ 13][ 24] =-3.271584510221496e+01;
  NLX_coes[2][ 13][ 25] =-3.401952123089367e+01;
  NLX_coes[2][ 13][ 26] =-3.515331886256578e+01;
  NLX_coes[2][ 13][ 27] =-3.625709077701745e+01;
  NLX_coes[2][ 13][ 28] =-3.727720394699019e+01;
  NLX_coes[2][ 13][ 29] =-3.821750316947822e+01;
  NLX_coes[2][ 13][ 30] =-3.908919192143269e+01;
  NLX_coes[2][ 13][ 31] =-3.990501662370563e+01;
  NLX_coes[2][ 13][ 32] =-4.067758698915831e+01;
  NLX_coes[2][ 13][ 33] =-4.141823503842833e+01;
  NLX_coes[2][ 13][ 34] =-4.213649249420799e+01;
  NLX_coes[2][ 13][ 35] =-4.284002781982425e+01;
  NLX_coes[2][ 13][ 36] =-4.353477988254895e+01;
  NLX_coes[2][ 13][ 37] =-4.422513960180708e+01;
  NLX_coes[2][ 13][ 38] =-4.491412259551178e+01;
  NLX_coes[2][ 13][ 39] =-4.560351507635723e+01;
  NLX_coes[2][ 13][ 40] =-4.629398599257801e+01;
  NLX_coes[2][ 14][  0] =-1.851242153913976e+01;
  NLX_coes[2][ 14][  1] =-1.913310725942222e+01;
  NLX_coes[2][ 14][  2] =-1.974966597450950e+01;
  NLX_coes[2][ 14][  3] =-2.035992939422892e+01;
  NLX_coes[2][ 14][  4] =-2.096200140838235e+01;
  NLX_coes[2][ 14][  5] =-2.155421813575699e+01;
  NLX_coes[2][ 14][  6] =-2.213516073331681e+01;
  NLX_coes[2][ 14][  7] =-2.270372970062276e+01;
  NLX_coes[2][ 14][  8] =-2.325929473232711e+01;
  NLX_coes[2][ 14][  9] =-2.380193781161043e+01;
  NLX_coes[2][ 14][ 10] =-2.433280894083138e+01;
  NLX_coes[2][ 14][ 11] =-2.485461547600802e+01;
  NLX_coes[2][ 14][ 12] =-2.537226599246188e+01;
  NLX_coes[2][ 14][ 13] =-2.589367203371950e+01;
  NLX_coes[2][ 14][ 14] =-2.643065142092783e+01;
  NLX_coes[2][ 14][ 15] =-2.699972707239852e+01;
  NLX_coes[2][ 14][ 16] =-2.762233964624405e+01;
  NLX_coes[2][ 14][ 17] =-2.832365401294840e+01;
  NLX_coes[2][ 14][ 18] =-2.912924086712373e+01;
  NLX_coes[2][ 14][ 19] =-3.006070998094661e+01;
  NLX_coes[2][ 14][ 20] =-3.113303516216387e+01;
  NLX_coes[2][ 14][ 21] =-3.235429809148433e+01;
  NLX_coes[2][ 14][ 22] =-3.366032517984865e+01;
  NLX_coes[2][ 14][ 23] =-3.501125286582599e+01;
  NLX_coes[2][ 14][ 24] =-3.634127229109956e+01;
  NLX_coes[2][ 14][ 25] =-3.755241063726517e+01;
  NLX_coes[2][ 14][ 26] =-3.870357106748911e+01;
  NLX_coes[2][ 14][ 27] =-3.980076443949415e+01;
  NLX_coes[2][ 14][ 28] =-4.083663955781184e+01;
  NLX_coes[2][ 14][ 29] =-4.180646174667971e+01;
  NLX_coes[2][ 14][ 30] =-4.271494071969565e+01;
  NLX_coes[2][ 14][ 31] =-4.357092313151810e+01;
  NLX_coes[2][ 14][ 32] =-4.438437344746833e+01;
  NLX_coes[2][ 14][ 33] =-4.516490099747848e+01;
  NLX_coes[2][ 14][ 34] =-4.592104827821545e+01;
  NLX_coes[2][ 14][ 35] =-4.666001555493664e+01;
  NLX_coes[2][ 14][ 36] =-4.738762147013375e+01;
  NLX_coes[2][ 14][ 37] =-4.810837256447571e+01;
  NLX_coes[2][ 14][ 38] =-4.882557199458991e+01;
  NLX_coes[2][ 14][ 39] =-4.954143368607415e+01;
  NLX_coes[2][ 14][ 40] =-5.025718651872673e+01;
  NLX_coes[2][ 15][  0] =-2.149516142202306e+01;
  NLX_coes[2][ 15][  1] =-2.212638268851610e+01;
  NLX_coes[2][ 15][  2] =-2.275395728775733e+01;
  NLX_coes[2][ 15][  3] =-2.337605009826338e+01;
  NLX_coes[2][ 15][  4] =-2.399115891489580e+01;
  NLX_coes[2][ 15][  5] =-2.459810229942364e+01;
  NLX_coes[2][ 15][  6] =-2.519605817083607e+01;
  NLX_coes[2][ 15][  7] =-2.578466028729817e+01;
  NLX_coes[2][ 15][  8] =-2.636416263732027e+01;
  NLX_coes[2][ 15][  9] =-2.693568289796624e+01;
  NLX_coes[2][ 15][ 10] =-2.750153481080314e+01;
  NLX_coes[2][ 15][ 11] =-2.806565437918232e+01;
  NLX_coes[2][ 15][ 12] =-2.863411102694510e+01;
  NLX_coes[2][ 15][ 13] =-2.921566152002775e+01;
  NLX_coes[2][ 15][ 14] =-2.982223579632946e+01;
  NLX_coes[2][ 15][ 15] =-3.046912460690995e+01;
  NLX_coes[2][ 15][ 16] =-3.117448448387153e+01;
  NLX_coes[2][ 15][ 17] =-3.195769174332974e+01;
  NLX_coes[2][ 15][ 18] =-3.283633485506462e+01;
  NLX_coes[2][ 15][ 19] =-3.382228184526461e+01;
  NLX_coes[2][ 15][ 20] =-3.491812514347749e+01;
  NLX_coes[2][ 15][ 21] =-3.613037012826558e+01;
  NLX_coes[2][ 15][ 22] =-3.739346829439034e+01;
  NLX_coes[2][ 15][ 23] =-3.866623116604638e+01;
  NLX_coes[2][ 15][ 24] =-3.988382196473846e+01;
  NLX_coes[2][ 15][ 25] =-4.105002378138159e+01;
  NLX_coes[2][ 15][ 26] =-4.220655974267179e+01;
  NLX_coes[2][ 15][ 27] =-4.331213419837421e+01;
  NLX_coes[2][ 15][ 28] =-4.436385455801100e+01;
  NLX_coes[2][ 15][ 29] =-4.535894981919345e+01;
  NLX_coes[2][ 15][ 30] =-4.629969543521298e+01;
  NLX_coes[2][ 15][ 31] =-4.719192036196314e+01;
  NLX_coes[2][ 15][ 32] =-4.804316384883691e+01;
  NLX_coes[2][ 15][ 33] =-4.886132373118794e+01;
  NLX_coes[2][ 15][ 34] =-4.965384598664536e+01;
  NLX_coes[2][ 15][ 35] =-5.042731200159621e+01;
  NLX_coes[2][ 15][ 36] =-5.118727410724682e+01;
  NLX_coes[2][ 15][ 37] =-5.193823051869986e+01;
  NLX_coes[2][ 15][ 38] =-5.268367175706381e+01;
  NLX_coes[2][ 15][ 39] =-5.342616019051405e+01;
  NLX_coes[2][ 15][ 40] =-5.416742320771057e+01;
  NLX_coes[2][ 16][  0] =-2.451162467438818e+01;
  NLX_coes[2][ 16][  1] =-2.515478692442254e+01;
  NLX_coes[2][ 16][  2] =-2.579491804137953e+01;
  NLX_coes[2][ 16][  3] =-2.643057471029308e+01;
  NLX_coes[2][ 16][  4] =-2.706068560628039e+01;
  NLX_coes[2][ 16][  5] =-2.768456353433385e+01;
  NLX_coes[2][ 16][  6] =-2.830196196325603e+01;
  NLX_coes[2][ 16][  7] =-2.891318136872093e+01;
  NLX_coes[2][ 16][  8] =-2.951923132348105e+01;
  NLX_coes[2][ 16][  9] =-3.012205293993940e+01;
  NLX_coes[2][ 16][ 10] =-3.072480192106159e+01;
  NLX_coes[2][ 16][ 11] =-3.133218302393066e+01;
  NLX_coes[2][ 16][ 12] =-3.195080806705592e+01;
  NLX_coes[2][ 16][ 13] =-3.258951584029187e+01;
  NLX_coes[2][ 16][ 14] =-3.325953735666321e+01;
  NLX_coes[2][ 16][ 15] =-3.397431437949378e+01;
  NLX_coes[2][ 16][ 16] =-3.474870423599577e+01;
  NLX_coes[2][ 16][ 17] =-3.559729184892056e+01;
  NLX_coes[2][ 16][ 18] =-3.653168452164552e+01;
  NLX_coes[2][ 16][ 19] =-3.755701893402456e+01;
  NLX_coes[2][ 16][ 20] =-3.866813528249362e+01;
  NLX_coes[2][ 16][ 21] =-3.984402545559955e+01;
  NLX_coes[2][ 16][ 22] =-4.104791697723311e+01;
  NLX_coes[2][ 16][ 23] =-4.222425800301249e+01;
  NLX_coes[2][ 16][ 24] =-4.337345494862062e+01;
  NLX_coes[2][ 16][ 25] =-4.454307319082674e+01;
  NLX_coes[2][ 16][ 26] =-4.569175956952402e+01;
  NLX_coes[2][ 16][ 27] =-4.680332616448612e+01;
  NLX_coes[2][ 16][ 28] =-4.786904409779598e+01;
  NLX_coes[2][ 16][ 29] =-4.888562435494089e+01;
  NLX_coes[2][ 16][ 30] =-4.985390424934112e+01;
  NLX_coes[2][ 16][ 31] =-5.077774805895886e+01;
  NLX_coes[2][ 16][ 32] =-5.166278157849744e+01;
  NLX_coes[2][ 16][ 33] =-5.251534811358560e+01;
  NLX_coes[2][ 16][ 34] =-5.334177574723515e+01;
  NLX_coes[2][ 16][ 35] =-5.414792801981295e+01;
  NLX_coes[2][ 16][ 36] =-5.493896811118982e+01;
  NLX_coes[2][ 16][ 37] =-5.571926522709703e+01;
  NLX_coes[2][ 16][ 38] =-5.649238875092198e+01;
  NLX_coes[2][ 16][ 39] =-5.726115489917474e+01;
  NLX_coes[2][ 16][ 40] =-5.802770637330714e+01;
  NLX_coes[2][ 17][  0] =-2.756169286209034e+01;
  NLX_coes[2][ 17][  1] =-2.821797532803085e+01;
  NLX_coes[2][ 17][  2] =-2.887192453494475e+01;
  NLX_coes[2][ 17][  3] =-2.952251813415963e+01;
  NLX_coes[2][ 17][  4] =-3.016912158676794e+01;
  NLX_coes[2][ 17][  5] =-3.081151947372273e+01;
  NLX_coes[2][ 17][  6] =-3.144998205247790e+01;
  NLX_coes[2][ 17][  7] =-3.208537069871965e+01;
  NLX_coes[2][ 17][  8] =-3.271928445791731e+01;
  NLX_coes[2][ 17][  9] =-3.335424669613342e+01;
  NLX_coes[2][ 17][ 10] =-3.399392460480876e+01;
  NLX_coes[2][ 17][ 11] =-3.464336339242917e+01;
  NLX_coes[2][ 17][ 12] =-3.530919918473927e+01;
  NLX_coes[2][ 17][ 13] =-3.599978812488320e+01;
  NLX_coes[2][ 17][ 14] =-3.672515465052779e+01;
  NLX_coes[2][ 17][ 15] =-3.749662612692434e+01;
  NLX_coes[2][ 17][ 16] =-3.832599581954143e+01;
  NLX_coes[2][ 17][ 17] =-3.922404330129054e+01;
  NLX_coes[2][ 17][ 18] =-4.019818634997289e+01;
  NLX_coes[2][ 17][ 19] =-4.124872200049501e+01;
  NLX_coes[2][ 17][ 20] =-4.236195805431284e+01;
  NLX_coes[2][ 17][ 21] =-4.350409516312800e+01;
  NLX_coes[2][ 17][ 22] =-4.463417205015228e+01;
  NLX_coes[2][ 17][ 23] =-4.572593037769319e+01;
  NLX_coes[2][ 17][ 24] =-4.686702500827235e+01;
  NLX_coes[2][ 17][ 25] =-4.802691111222489e+01;
  NLX_coes[2][ 17][ 26] =-4.917128773778362e+01;
  NLX_coes[2][ 17][ 27] =-5.028586386772303e+01;
  NLX_coes[2][ 17][ 28] =-5.136174599630957e+01;
  NLX_coes[2][ 17][ 29] =-5.239523212730794e+01;
  NLX_coes[2][ 17][ 30] =-5.338608049162962e+01;
  NLX_coes[2][ 17][ 31] =-5.433666270449420e+01;
  NLX_coes[2][ 17][ 32] =-5.525105943297071e+01;
  NLX_coes[2][ 17][ 33] =-5.613425138185320e+01;
  NLX_coes[2][ 17][ 34] =-5.699149495684960e+01;
  NLX_coes[2][ 17][ 35] =-5.782789331962585e+01;
  NLX_coes[2][ 17][ 36] =-5.864813485145136e+01;
  NLX_coes[2][ 17][ 37] =-5.945635887880277e+01;
  NLX_coes[2][ 17][ 38] =-6.025611161432511e+01;
  NLX_coes[2][ 17][ 39] =-6.105036486615342e+01;
  NLX_coes[2][ 17][ 40] =-6.184158096950878e+01;
  NLX_coes[2][ 18][  0] =-3.064472422348853e+01;
  NLX_coes[2][ 18][  1] =-3.131504694210467e+01;
  NLX_coes[2][ 18][  2] =-3.198376645137065e+01;
  NLX_coes[2][ 18][  3] =-3.265028413447471e+01;
  NLX_coes[2][ 18][  4] =-3.331438308910995e+01;
  NLX_coes[2][ 18][  5] =-3.397627287130272e+01;
  NLX_coes[2][ 18][  6] =-3.463665908130022e+01;
  NLX_coes[2][ 18][  7] =-3.529683978669801e+01;
  NLX_coes[2][ 18][  8] =-3.595882802210757e+01;
  NLX_coes[2][ 18][  9] =-3.662549526783167e+01;
  NLX_coes[2][ 18][ 10] =-3.730072398866550e+01;
  NLX_coes[2][ 18][ 11] =-3.798954704672130e+01;
  NLX_coes[2][ 18][ 12] =-3.869823737016579e+01;
  NLX_coes[2][ 18][ 13] =-3.943429298733697e+01;
  NLX_coes[2][ 18][ 14] =-4.020624353089148e+01;
  NLX_coes[2][ 18][ 15] =-4.102319402496021e+01;
  NLX_coes[2][ 18][ 16] =-4.189403912815309e+01;
  NLX_coes[2][ 18][ 17] =-4.282632474497807e+01;
  NLX_coes[2][ 18][ 18] =-4.382460568371765e+01;
  NLX_coes[2][ 18][ 19] =-4.488727327334054e+01;
  NLX_coes[2][ 18][ 20] =-4.600064109834780e+01;
  NLX_coes[2][ 18][ 21] =-4.712073180581353e+01;
  NLX_coes[2][ 18][ 22] =-4.815386088244181e+01;
  NLX_coes[2][ 18][ 23] =-4.923007902023339e+01;
  NLX_coes[2][ 18][ 24] =-5.036967056749950e+01;
  NLX_coes[2][ 18][ 25] =-5.151673533853779e+01;
  NLX_coes[2][ 18][ 26] =-5.265360229953905e+01;
  NLX_coes[2][ 18][ 27] =-5.376704266941593e+01;
  NLX_coes[2][ 18][ 28] =-5.484873303786117e+01;
  NLX_coes[2][ 18][ 29] =-5.589440337019522e+01;
  NLX_coes[2][ 18][ 30] =-5.690289200637034e+01;
  NLX_coes[2][ 18][ 31] =-5.787536761999441e+01;
  NLX_coes[2][ 18][ 32] =-5.881462551234321e+01;
  NLX_coes[2][ 18][ 33] =-5.972445328699968e+01;
  NLX_coes[2][ 18][ 34] =-6.060910994691390e+01;
  NLX_coes[2][ 18][ 35] =-6.147293819625143e+01;
  NLX_coes[2][ 18][ 36] =-6.232010381823098e+01;
  NLX_coes[2][ 18][ 37] =-6.315444236723136e+01;
  NLX_coes[2][ 18][ 38] =-6.397939052163522e+01;
  NLX_coes[2][ 18][ 39] =-6.479798305653551e+01;
  NLX_coes[2][ 18][ 40] =-6.561290302152192e+01;
  NLX_coes[2][ 19][  0] =-3.375964301649149e+01;
  NLX_coes[2][ 19][  1] =-3.444465368741817e+01;
  NLX_coes[2][ 19][  2] =-3.512878039012612e+01;
  NLX_coes[2][ 19][  3] =-3.581182960516784e+01;
  NLX_coes[2][ 19][  4] =-3.649396494976541e+01;
  NLX_coes[2][ 19][  5] =-3.717575980494656e+01;
  NLX_coes[2][ 19][  6] =-3.785826446671125e+01;
  NLX_coes[2][ 19][  7] =-3.854308844973971e+01;
  NLX_coes[2][ 19][  8] =-3.923249510079040e+01;
  NLX_coes[2][ 19][  9] =-3.992950099038654e+01;
  NLX_coes[2][ 19][ 10] =-4.063796608167015e+01;
  NLX_coes[2][ 19][ 11] =-4.136265204933476e+01;
  NLX_coes[2][ 19][ 12] =-4.210921548855829e+01;
  NLX_coes[2][ 19][ 13] =-4.288409137790120e+01;
  NLX_coes[2][ 19][ 14] =-4.369421275732346e+01;
  NLX_coes[2][ 19][ 15] =-4.454650718522592e+01;
  NLX_coes[2][ 19][ 16] =-4.544710705983432e+01;
  NLX_coes[2][ 19][ 17] =-4.640024734119275e+01;
  NLX_coes[2][ 19][ 18] =-4.740724872754043e+01;
  NLX_coes[2][ 19][ 19] =-4.846759339110911e+01;
  NLX_coes[2][ 19][ 20] =-4.958219980553774e+01;
  NLX_coes[2][ 19][ 21] =-5.067338010578647e+01;
  NLX_coes[2][ 19][ 22] =-5.166840675499048e+01;
  NLX_coes[2][ 19][ 23] =-5.276572027343045e+01;
  NLX_coes[2][ 19][ 24] =-5.388569790788367e+01;
  NLX_coes[2][ 19][ 25] =-5.501627697163759e+01;
  NLX_coes[2][ 19][ 26] =-5.614202605626158e+01;
  NLX_coes[2][ 19][ 27] =-5.725066599800684e+01;
  NLX_coes[2][ 19][ 28] =-5.833409013826527e+01;
  NLX_coes[2][ 19][ 29] =-5.938758421584799e+01;
  NLX_coes[2][ 19][ 30] =-6.040918176278166e+01;
  NLX_coes[2][ 19][ 31] =-6.139903788571895e+01;
  NLX_coes[2][ 19][ 32] =-6.235886132870785e+01;
  NLX_coes[2][ 19][ 33] =-6.329140055701013e+01;
  NLX_coes[2][ 19][ 34] =-6.420000743216364e+01;
  NLX_coes[2][ 19][ 35] =-6.508829498446912e+01;
  NLX_coes[2][ 19][ 36] =-6.595989147095469e+01;
  NLX_coes[2][ 19][ 37] =-6.681828252987967e+01;
  NLX_coes[2][ 19][ 38] =-6.766672887391684e+01;
  NLX_coes[2][ 19][ 39] =-6.850824746035089e+01;
  NLX_coes[2][ 19][ 40] =-6.934564763036958e+01;
  NLX_coes[2][ 20][  0] =-3.690503501643215e+01;
  NLX_coes[2][ 20][  1] =-3.760511273310706e+01;
  NLX_coes[2][ 20][  2] =-3.830498152270298e+01;
  NLX_coes[2][ 20][  3] =-3.900481872268784e+01;
  NLX_coes[2][ 20][  4] =-3.970512035855630e+01;
  NLX_coes[2][ 20][  5] =-4.040675683619622e+01;
  NLX_coes[2][ 20][  6] =-4.111103393056654e+01;
  NLX_coes[2][ 20][  7] =-4.181975868834938e+01;
  NLX_coes[2][ 20][  8] =-4.253530622149538e+01;
  NLX_coes[2][ 20][  9] =-4.326067890225436e+01;
  NLX_coes[2][ 20][ 10] =-4.399954390830231e+01;
  NLX_coes[2][ 20][ 11] =-4.475622839553160e+01;
  NLX_coes[2][ 20][ 12] =-4.553564422461788e+01;
  NLX_coes[2][ 20][ 13] =-4.634310721664342e+01;
  NLX_coes[2][ 20][ 14] =-4.718401156539485e+01;
  NLX_coes[2][ 20][ 15] =-4.806332361858209e+01;
  NLX_coes[2][ 20][ 16] =-4.898487930567303e+01;
  NLX_coes[2][ 20][ 17] =-4.995045929401071e+01;
  NLX_coes[2][ 20][ 18] =-5.095814998136977e+01;
  NLX_coes[2][ 20][ 19] =-5.199749246203015e+01;
  NLX_coes[2][ 20][ 20] =-5.303968349359970e+01;
  NLX_coes[2][ 20][ 21] =-5.423067436528510e+01;
  NLX_coes[2][ 20][ 22] =-5.524338872784753e+01;
  NLX_coes[2][ 20][ 23] =-5.630932953799927e+01;
  NLX_coes[2][ 20][ 24] =-5.740889988112531e+01;
  NLX_coes[2][ 20][ 25] =-5.852271029601378e+01;
  NLX_coes[2][ 20][ 26] =-5.963580901775608e+01;
  NLX_coes[2][ 20][ 27] =-6.073732256511514e+01;
  NLX_coes[2][ 20][ 28] =-6.181947562667241e+01;
  NLX_coes[2][ 20][ 29] =-6.287728708678665e+01;
  NLX_coes[2][ 20][ 30] =-6.390816876894942e+01;
  NLX_coes[2][ 20][ 31] =-6.491146020148697e+01;
  NLX_coes[2][ 20][ 32] =-6.588797247626944e+01;
  NLX_coes[2][ 20][ 33] =-6.683956842505985e+01;
  NLX_coes[2][ 20][ 34] =-6.776879734936917e+01;
  NLX_coes[2][ 20][ 35] =-6.867859763028480e+01;
  NLX_coes[2][ 20][ 36] =-6.957207249204248e+01;
  NLX_coes[2][ 20][ 37] =-7.045233699881389e+01;
  NLX_coes[2][ 20][ 38] =-7.132243024713026e+01;
  NLX_coes[2][ 20][ 39] =-7.218528580896069e+01;
  NLX_coes[2][ 20][ 40] =-7.304375501739550e+01;
  NLX_coes[2][ 21][  0] =-4.007924096738005e+01;
  NLX_coes[2][ 21][  1] =-4.079451293460173e+01;
  NLX_coes[2][ 21][  2] =-4.151018367241750e+01;
  NLX_coes[2][ 21][  3] =-4.222675724868395e+01;
  NLX_coes[2][ 21][  4] =-4.294500906254478e+01;
  NLX_coes[2][ 21][  5] =-4.366604080134110e+01;
  NLX_coes[2][ 21][  6] =-4.439133317569561e+01;
  NLX_coes[2][ 21][  7] =-4.512279541117109e+01;
  NLX_coes[2][ 21][  8] =-4.586280707896634e+01;
  NLX_coes[2][ 21][  9] =-4.661424398505562e+01;
  NLX_coes[2][ 21][ 10] =-4.738047545572358e+01;
  NLX_coes[2][ 21][ 11] =-4.816531565893833e+01;
  NLX_coes[2][ 21][ 12] =-4.897290718612201e+01;
  NLX_coes[2][ 21][ 13] =-4.980751214191994e+01;
  NLX_coes[2][ 21][ 14] =-5.067318587860779e+01;
  NLX_coes[2][ 21][ 15] =-5.157331041899359e+01;
  NLX_coes[2][ 21][ 16] =-5.250996071331257e+01;
  NLX_coes[2][ 21][ 17] =-5.348308506164653e+01;
  NLX_coes[2][ 21][ 18] =-5.448979375898723e+01;
  NLX_coes[2][ 21][ 19] =-5.552570814231436e+01;
  NLX_coes[2][ 21][ 20] =-5.659257552925530e+01;
  NLX_coes[2][ 21][ 21] =-5.769942454765259e+01;
  NLX_coes[2][ 21][ 22] =-5.877330069007900e+01;
  NLX_coes[2][ 21][ 23] =-5.984302763049526e+01;
  NLX_coes[2][ 21][ 24] =-6.093063459859440e+01;
  NLX_coes[2][ 21][ 25] =-6.203025697529580e+01;
  NLX_coes[2][ 21][ 26] =-6.313166938984927e+01;
  NLX_coes[2][ 21][ 27] =-6.422558196488208e+01;
  NLX_coes[2][ 21][ 28] =-6.530485491004218e+01;
  NLX_coes[2][ 21][ 29] =-6.636456359379899e+01;
  NLX_coes[2][ 21][ 30] =-6.740178031027064e+01;
  NLX_coes[2][ 21][ 31] =-6.841526770602043e+01;
  NLX_coes[2][ 21][ 32] =-6.940514182965053e+01;
  NLX_coes[2][ 21][ 33] =-7.037254243998032e+01;
  NLX_coes[2][ 21][ 34] =-7.131933442799659e+01;
  NLX_coes[2][ 21][ 35] =-7.224785555051891e+01;
  NLX_coes[2][ 21][ 36] =-7.316071837090179e+01;
  NLX_coes[2][ 21][ 37] =-7.406066845919496e+01;
  NLX_coes[2][ 21][ 38] =-7.495049712267884e+01;
  NLX_coes[2][ 21][ 39] =-7.583300532816421e+01;
  NLX_coes[2][ 21][ 40] =-7.671101567436094e+01;
  NLX_coes[2][ 22][  0] =-4.328044218601049e+01;
  NLX_coes[2][ 22][  1] =-4.401080933412695e+01;
  NLX_coes[2][ 22][  2] =-4.474210207292524e+01;
  NLX_coes[2][ 22][  3] =-4.547510224372156e+01;
  NLX_coes[2][ 22][  4] =-4.621081142222074e+01;
  NLX_coes[2][ 22][  5] =-4.695050242223647e+01;
  NLX_coes[2][ 22][  6] =-4.769576302938953e+01;
  NLX_coes[2][ 22][  7] =-4.844853064116207e+01;
  NLX_coes[2][ 22][  8] =-4.921111360793815e+01;
  NLX_coes[2][ 22][  9] =-4.998619202095621e+01;
  NLX_coes[2][ 22][ 10] =-5.077678758915314e+01;
  NLX_coes[2][ 22][ 11] =-5.158618943164406e+01;
  NLX_coes[2][ 22][ 12] =-5.241782075171486e+01;
  NLX_coes[2][ 22][ 13] =-5.327503150532151e+01;
  NLX_coes[2][ 22][ 14] =-5.416080603540912e+01;
  NLX_coes[2][ 22][ 15] =-5.507738670638222e+01;
  NLX_coes[2][ 22][ 16] =-5.602585070795163e+01;
  NLX_coes[2][ 22][ 17] =-5.700577666137485e+01;
  NLX_coes[2][ 22][ 18] =-5.801532782913503e+01;
  NLX_coes[2][ 22][ 19] =-5.905228574868119e+01;
  NLX_coes[2][ 22][ 20] =-6.011730881800906e+01;
  NLX_coes[2][ 22][ 21] =-6.119686674848961e+01;
  NLX_coes[2][ 22][ 22] =-6.227911421329166e+01;
  NLX_coes[2][ 22][ 23] =-6.335715485300421e+01;
  NLX_coes[2][ 22][ 24] =-6.444200223725056e+01;
  NLX_coes[2][ 22][ 25] =-6.553345760547091e+01;
  NLX_coes[2][ 22][ 26] =-6.662612608578941e+01;
  NLX_coes[2][ 22][ 27] =-6.771339463931005e+01;
  NLX_coes[2][ 22][ 28] =-6.878937485469369e+01;
  NLX_coes[2][ 22][ 29] =-6.984958245928759e+01;
  NLX_coes[2][ 22][ 30] =-7.089105327219968e+01;
  NLX_coes[2][ 22][ 31] =-7.191222538470829e+01;
  NLX_coes[2][ 22][ 32] =-7.291272897352077e+01;
  NLX_coes[2][ 22][ 33] =-7.389314841819017e+01;
  NLX_coes[2][ 22][ 34] =-7.485479063944747e+01;
  NLX_coes[2][ 22][ 35] =-7.579947939542136e+01;
  NLX_coes[2][ 22][ 36] =-7.672938672710154e+01;
  NLX_coes[2][ 22][ 37] =-7.764690685060658e+01;
  NLX_coes[2][ 22][ 38] =-7.855457390733315e+01;
  NLX_coes[2][ 22][ 39] =-7.945502283181744e+01;
  NLX_coes[2][ 22][ 40] =-8.035099182953395e+01;
  NLX_coes[2][ 23][  0] =-4.650673480223362e+01;
  NLX_coes[2][ 23][  1] =-4.725190251135439e+01;
  NLX_coes[2][ 23][  2] =-4.799843638562709e+01;
  NLX_coes[2][ 23][  3] =-4.874734632651154e+01;
  NLX_coes[2][ 23][  4] =-4.949981008093541e+01;
  NLX_coes[2][ 23][  5] =-5.025721956604684e+01;
  NLX_coes[2][ 23][  6] =-5.102121581584482e+01;
  NLX_coes[2][ 23][  7] =-5.179371119291304e+01;
  NLX_coes[2][ 23][  8] =-5.257689527352864e+01;
  NLX_coes[2][ 23][  9] =-5.337321864076249e+01;
  NLX_coes[2][ 23][ 10] =-5.418534692584178e+01;
  NLX_coes[2][ 23][ 11] =-5.501607624771366e+01;
  NLX_coes[2][ 23][ 12] =-5.586820145554890e+01;
  NLX_coes[2][ 23][ 13] =-5.674433156165077e+01;
  NLX_coes[2][ 23][ 14] =-5.764665482244093e+01;
  NLX_coes[2][ 23][ 15] =-5.857667339506331e+01;
  NLX_coes[2][ 23][ 16] =-5.953496043505280e+01;
  NLX_coes[2][ 23][ 17] =-6.052104245707708e+01;
  NLX_coes[2][ 23][ 18] =-6.153354538302203e+01;
  NLX_coes[2][ 23][ 19] =-6.257056339241548e+01;
  NLX_coes[2][ 23][ 20] =-6.362904714496995e+01;
  NLX_coes[2][ 23][ 21] =-6.470030574568393e+01;
  NLX_coes[2][ 23][ 22] =-6.577903795904514e+01;
  NLX_coes[2][ 23][ 23] =-6.685960806079318e+01;
  NLX_coes[2][ 23][ 24] =-6.794335298029472e+01;
  NLX_coes[2][ 23][ 25] =-6.903016955521035e+01;
  NLX_coes[2][ 23][ 26] =-7.011691873497304e+01;
  NLX_coes[2][ 23][ 27] =-7.119907188255058e+01;
  NLX_coes[2][ 23][ 28] =-7.227208358855856e+01;
  NLX_coes[2][ 23][ 29] =-7.333215020930943e+01;
  NLX_coes[2][ 23][ 30] =-7.437651516161066e+01;
  NLX_coes[2][ 23][ 31] =-7.540351280185635e+01;
  NLX_coes[2][ 23][ 32] =-7.641247884389261e+01;
  NLX_coes[2][ 23][ 33] =-7.740360132339639e+01;
  NLX_coes[2][ 23][ 34] =-7.837775427850278e+01;
  NLX_coes[2][ 23][ 35] =-7.933633873695581e+01;
  NLX_coes[2][ 23][ 36] =-8.028114542758537e+01;
  NLX_coes[2][ 23][ 37] =-8.121424720157168e+01;
  NLX_coes[2][ 23][ 38] =-8.213792495340188e+01;
  NLX_coes[2][ 23][ 39] =-8.305462821395555e+01;
  NLX_coes[2][ 23][ 40] =-8.396697010604501e+01;
  NLX_coes[2][ 24][  0] =-4.975619105106272e+01;
  NLX_coes[2][ 24][  1] =-5.051570180802789e+01;
  NLX_coes[2][ 24][  2] =-5.127693404271280e+01;
  NLX_coes[2][ 24][  3] =-5.204107824561491e+01;
  NLX_coes[2][ 24][  4] =-5.280944358812213e+01;
  NLX_coes[2][ 24][  5] =-5.358349815344356e+01;
  NLX_coes[2][ 24][  6] =-5.436489589000882e+01;
  NLX_coes[2][ 24][  7] =-5.515548908050716e+01;
  NLX_coes[2][ 24][  8] =-5.595732354651845e+01;
  NLX_coes[2][ 24][  9] =-5.677261244478408e+01;
  NLX_coes[2][ 24][ 10] =-5.760368362559138e+01;
  NLX_coes[2][ 24][ 11] =-5.845289555402280e+01;
  NLX_coes[2][ 24][ 12] =-5.932251842407676e+01;
  NLX_coes[2][ 24][ 13] =-6.021458133756047e+01;
  NLX_coes[2][ 24][ 14] =-6.113069460605156e+01;
  NLX_coes[2][ 24][ 15] =-6.207186944855880e+01;
  NLX_coes[2][ 24][ 16] =-6.303837392612069e+01;
  NLX_coes[2][ 24][ 17] =-6.402967194790688e+01;
  NLX_coes[2][ 24][ 18] =-6.504445498991080e+01;
  NLX_coes[2][ 24][ 19] =-6.608061993738929e+01;
  NLX_coes[2][ 24][ 20] =-6.713454304374963e+01;
  NLX_coes[2][ 24][ 21] =-6.820114648893801e+01;
  NLX_coes[2][ 24][ 22] =-6.927591850598446e+01;
  NLX_coes[2][ 24][ 23] =-7.035522086656123e+01;
  NLX_coes[2][ 24][ 24] =-7.143727053758354e+01;
  NLX_coes[2][ 24][ 25] =-7.252080353680145e+01;
  NLX_coes[2][ 24][ 26] =-7.360339691227117e+01;
  NLX_coes[2][ 24][ 27] =-7.468174184593438e+01;
  NLX_coes[2][ 24][ 28] =-7.575233907115712e+01;
  NLX_coes[2][ 24][ 29] =-7.681206150785691e+01;
  NLX_coes[2][ 24][ 30] =-7.785847460708726e+01;
  NLX_coes[2][ 24][ 31] =-7.888995950679200e+01;
  NLX_coes[2][ 24][ 32] =-7.990570848849735e+01;
  NLX_coes[2][ 24][ 33] =-8.090564911003912e+01;
  NLX_coes[2][ 24][ 34] =-8.189033591016883e+01;
  NLX_coes[2][ 24][ 35] =-8.286083499408444e+01;
  NLX_coes[2][ 24][ 36] =-8.381861748103672e+01;
  NLX_coes[2][ 24][ 37] =-8.476547149371217e+01;
  NLX_coes[2][ 24][ 38] =-8.570343808850539e+01;
  NLX_coes[2][ 24][ 39] =-8.663477363386899e+01;
  NLX_coes[2][ 24][ 40] =-8.756193919474229e+01;
  NLX_coes[2][ 25][  0] =-5.302690749519152e+01;
  NLX_coes[2][ 25][  1] =-5.380017304694219e+01;
  NLX_coes[2][ 25][  2] =-5.457543561135918e+01;
  NLX_coes[2][ 25][  3] =-5.535402302722900e+01;
  NLX_coes[2][ 25][  4] =-5.613733742262367e+01;
  NLX_coes[2][ 25][  5] =-5.692688903484279e+01;
  NLX_coes[2][ 25][  6] =-5.772431621998751e+01;
  NLX_coes[2][ 25][  7] =-5.853139069330745e+01;
  NLX_coes[2][ 25][  8] =-5.935000605796736e+01;
  NLX_coes[2][ 25][  9] =-6.018214697309829e+01;
  NLX_coes[2][ 25][ 10] =-6.102983614423643e+01;
  NLX_coes[2][ 25][ 11] =-6.189505705119046e+01;
  NLX_coes[2][ 25][ 12] =-6.277965242160279e+01;
  NLX_coes[2][ 25][ 13] =-6.368520239915101e+01;
  NLX_coes[2][ 25][ 14] =-6.461289235549575e+01;
  NLX_coes[2][ 25][ 15] =-6.556338747060785e+01;
  NLX_coes[2][ 25][ 16] =-6.653673582823282e+01;
  NLX_coes[2][ 25][ 17] =-6.753231463889134e+01;
  NLX_coes[2][ 25][ 18] =-6.854879957838880e+01;
  NLX_coes[2][ 25][ 19] =-6.958407100768133e+01;
  NLX_coes[2][ 25][ 20] =-7.063502151584126e+01;
  NLX_coes[2][ 25][ 21] =-7.169801524371795e+01;
  NLX_coes[2][ 25][ 22] =-7.276938114803531e+01;
  NLX_coes[2][ 25][ 23] =-7.384609810980470e+01;
  NLX_coes[2][ 25][ 24] =-7.492576949563139e+01;
  NLX_coes[2][ 25][ 25] =-7.600642399182405e+01;
  NLX_coes[2][ 25][ 26] =-7.708579851989239e+01;
  NLX_coes[2][ 25][ 27] =-7.816122275228777e+01;
  NLX_coes[2][ 25][ 28] =-7.922990257314741e+01;
  NLX_coes[2][ 25][ 29] =-8.028926217472576e+01;
  NLX_coes[2][ 25][ 30] =-8.133719721224573e+01;
  NLX_coes[2][ 25][ 31] =-8.237221088231942e+01;
  NLX_coes[2][ 25][ 32] =-8.339345329077963e+01;
  NLX_coes[2][ 25][ 33] =-8.440069558370661e+01;
  NLX_coes[2][ 25][ 34] =-8.539426708963062e+01;
  NLX_coes[2][ 25][ 35] =-8.637497696649645e+01;
  NLX_coes[2][ 25][ 36] =-8.734403544275558e+01;
  NLX_coes[2][ 25][ 37] =-8.830298462864874e+01;
  NLX_coes[2][ 25][ 38] =-8.925364502109529e+01;
  NLX_coes[2][ 25][ 39] =-9.019808097490647e+01;
  NLX_coes[2][ 25][ 40] =-9.113858627108387e+01;
  NLX_coes[2][ 26][  0] =-5.631704104733860e+01;
  NLX_coes[2][ 26][  1] =-5.710337235545614e+01;
  NLX_coes[2][ 26][  2] =-5.789190475255238e+01;
  NLX_coes[2][ 26][  3] =-5.868406555152132e+01;
  NLX_coes[2][ 26][  4] =-5.948131790530832e+01;
  NLX_coes[2][ 26][  5] =-6.028518829281536e+01;
  NLX_coes[2][ 26][  6] =-6.109728065518072e+01;
  NLX_coes[2][ 26][  7] =-6.191927650085649e+01;
  NLX_coes[2][ 26][  8] =-6.275291978144941e+01;
  NLX_coes[2][ 26][  9] =-6.359998509972600e+01;
  NLX_coes[2][ 26][ 10] =-6.446222806170182e+01;
  NLX_coes[2][ 26][ 11] =-6.534131756771779e+01;
  NLX_coes[2][ 26][ 12] =-6.623875177537896e+01;
  NLX_coes[2][ 26][ 13] =-6.715576243579605e+01;
  NLX_coes[2][ 26][ 14] =-6.809321595315696e+01;
  NLX_coes[2][ 26][ 15] =-6.905152258577965e+01;
  NLX_coes[2][ 26][ 16] =-7.003056497700963e+01;
  NLX_coes[2][ 26][ 17] =-7.102964980684408e+01;
  NLX_coes[2][ 26][ 18] =-7.204747099673685e+01;
  NLX_coes[2][ 26][ 19] =-7.308206742746727e+01;
  NLX_coes[2][ 26][ 20] =-7.413086082611319e+01;
  NLX_coes[2][ 26][ 21] =-7.519092124663658e+01;
  NLX_coes[2][ 26][ 22] =-7.625924653421156e+01;
  NLX_coes[2][ 26][ 23] =-7.733311153492575e+01;
  NLX_coes[2][ 26][ 24] =-7.841009834971224e+01;
  NLX_coes[2][ 26][ 25] =-7.948801611457068e+01;
  NLX_coes[2][ 26][ 26] =-8.056466351428070e+01;
  NLX_coes[2][ 26][ 27] =-8.163771859784600e+01;
  NLX_coes[2][ 26][ 28] =-8.270484248160615e+01;
  NLX_coes[2][ 26][ 29] =-8.376386932087978e+01;
  NLX_coes[2][ 26][ 30] =-8.481298137398056e+01;
  NLX_coes[2][ 26][ 31] =-8.585082578803242e+01;
  NLX_coes[2][ 26][ 32] =-8.687656817861034e+01;
  NLX_coes[2][ 26][ 33] =-8.788989533095733e+01;
  NLX_coes[2][ 26][ 34] =-8.889098389641929e+01;
  NLX_coes[2][ 26][ 35] =-8.988045071239505e+01;
  NLX_coes[2][ 26][ 36] =-9.085929718370093e+01;
  NLX_coes[2][ 26][ 37] =-9.182885675144081e+01;
  NLX_coes[2][ 26][ 38] =-9.279075145666540e+01;
  NLX_coes[2][ 26][ 39] =-9.374686110154499e+01;
  NLX_coes[2][ 26][ 40] =-9.469930643696785e+01;
  NLX_coes[2][ 27][  0] =-5.962483423017313e+01;
  NLX_coes[2][ 27][  1] =-6.042346818788215e+01;
  NLX_coes[2][ 27][  2] =-6.122444564486690e+01;
  NLX_coes[2][ 27][  3] =-6.202926136203178e+01;
  NLX_coes[2][ 27][  4] =-6.283941388284729e+01;
  NLX_coes[2][ 27][  5] =-6.365642698565748e+01;
  NLX_coes[2][ 27][  6] =-6.448185889659410e+01;
  NLX_coes[2][ 27][  7] =-6.531729884466496e+01;
  NLX_coes[2][ 27][  8] =-6.616435035263726e+01;
  NLX_coes[2][ 27][  9] =-6.702460070186595e+01;
  NLX_coes[2][ 27][ 10] =-6.789957644260281e+01;
  NLX_coes[2][ 27][ 11] =-6.879068577696508e+01;
  NLX_coes[2][ 27][ 12] =-6.969915017717695e+01;
  NLX_coes[2][ 27][ 13] =-7.062592959765773e+01;
  NLX_coes[2][ 27][ 14] =-7.157164764760029e+01;
  NLX_coes[2][ 27][ 15] =-7.253652419307346e+01;
  NLX_coes[2][ 27][ 16] =-7.352032179034703e+01;
  NLX_coes[2][ 27][ 17] =-7.452230854390362e+01;
  NLX_coes[2][ 27][ 18] =-7.554123677209222e+01;
  NLX_coes[2][ 27][ 19] =-7.657534741862084e+01;
  NLX_coes[2][ 27][ 20] =-7.762244805867726e+01;
  NLX_coes[2][ 27][ 21] =-7.868006901409413e+01;
  NLX_coes[2][ 27][ 22] =-7.974565660168136e+01;
  NLX_coes[2][ 27][ 23] =-8.081674627522791e+01;
  NLX_coes[2][ 27][ 24] =-8.189103217833622e+01;
  NLX_coes[2][ 27][ 25] =-8.296634072007710e+01;
  NLX_coes[2][ 27][ 26] =-8.404055982739362e+01;
  NLX_coes[2][ 27][ 27] =-8.511159311541265e+01;
  NLX_coes[2][ 27][ 28] =-8.617740493794459e+01;
  NLX_coes[2][ 27][ 29] =-8.723612647779306e+01;
  NLX_coes[2][ 27][ 30] =-8.828617076228799e+01;
  NLX_coes[2][ 27][ 31] =-8.932632244951105e+01;
  NLX_coes[2][ 27][ 32] =-9.035578989769188e+01;
  NLX_coes[2][ 27][ 33] =-9.137422122995586e+01;
  NLX_coes[2][ 27][ 34] =-9.238169271914195e+01;
  NLX_coes[2][ 27][ 35] =-9.337867951302215e+01;
  NLX_coes[2][ 27][ 36] =-9.436601789800764e+01;
  NLX_coes[2][ 27][ 37] =-9.534486646513389e+01;
  NLX_coes[2][ 27][ 38] =-9.631667147173449e+01;
  NLX_coes[2][ 27][ 39] =-9.728313971065825e+01;
  NLX_coes[2][ 27][ 40] =-9.824622038421572e+01;
  NLX_coes[2][ 28][  0] =-6.294863134655329e+01;
  NLX_coes[2][ 28][  1] =-6.375875374823015e+01;
  NLX_coes[2][ 28][  2] =-6.457131064236114e+01;
  NLX_coes[2][ 28][  3] =-6.538783808349990e+01;
  NLX_coes[2][ 28][  4] =-6.620985015521210e+01;
  NLX_coes[2][ 28][  5] =-6.703885478173146e+01;
  NLX_coes[2][ 28][  6] =-6.787635878152599e+01;
  NLX_coes[2][ 28][  7] =-6.872386197518446e+01;
  NLX_coes[2][ 28][  8] =-6.958284017042686e+01;
  NLX_coes[2][ 28][  9] =-7.045471703227751e+01;
  NLX_coes[2][ 28][ 10] =-7.134082531295395e+01;
  NLX_coes[2][ 28][ 11] =-7.224235872357032e+01;
  NLX_coes[2][ 28][ 12] =-7.316031685361061e+01;
  NLX_coes[2][ 28][ 13] =-7.409544682247144e+01;
  NLX_coes[2][ 28][ 14] =-7.504818643563311e+01;
  NLX_coes[2][ 28][ 15] =-7.601861402409139e+01;
  NLX_coes[2][ 28][ 16] =-7.700640950555621e+01;
  NLX_coes[2][ 28][ 17] =-7.801083002548830e+01;
  NLX_coes[2][ 28][ 18] =-7.903070433652357e+01;
  NLX_coes[2][ 28][ 19] =-8.006445699502726e+01;
  NLX_coes[2][ 28][ 20] =-8.111017778690446e+01;
  NLX_coes[2][ 28][ 21] =-8.216572884843349e+01;
  NLX_coes[2][ 28][ 22] =-8.322887334256389e+01;
  NLX_coes[2][ 28][ 23] =-8.429738351916083e+01;
  NLX_coes[2][ 28][ 24] =-8.536910606529592e+01;
  NLX_coes[2][ 28][ 25] =-8.644197375828232e+01;
  NLX_coes[2][ 28][ 26] =-8.751399471056581e+01;
  NLX_coes[2][ 28][ 27] =-8.858324817275587e+01;
  NLX_coes[2][ 28][ 28] =-8.964791608345102e+01;
  NLX_coes[2][ 28][ 29] =-9.070634832240293e+01;
  NLX_coes[2][ 28][ 30] =-9.175713910641832e+01;
  NLX_coes[2][ 28][ 31] =-9.279919334910008e+01;
  NLX_coes[2][ 28][ 32] =-9.383177138678153e+01;
  NLX_coes[2][ 28][ 33] =-9.485450951156416e+01;
  NLX_coes[2][ 28][ 34] =-9.586741947997896e+01;
  NLX_coes[2][ 28][ 35] =-9.687087276024982e+01;
  NLX_coes[2][ 28][ 36] =-9.786557575499501e+01;
  NLX_coes[2][ 28][ 37] =-9.885254153322997e+01;
  NLX_coes[2][ 28][ 38] =-9.983306236372775e+01;
  NLX_coes[2][ 28][ 39] =-1.008086859151911e+02;
  NLX_coes[2][ 28][ 40] =-1.017811965313647e+02;
  NLX_coes[2][ 29][  0] =-6.628688722814178e+01;
  NLX_coes[2][ 29][  1] =-6.710765187111208e+01;
  NLX_coes[2][ 29][  2] =-6.793090059950484e+01;
  NLX_coes[2][ 29][  3] =-6.875819021849328e+01;
  NLX_coes[2][ 29][  4] =-6.959103565852651e+01;
  NLX_coes[2][ 29][  5] =-7.043092053588616e+01;
  NLX_coes[2][ 29][  6] =-7.127929861146566e+01;
  NLX_coes[2][ 29][  7] =-7.213758615828841e+01;
  NLX_coes[2][ 29][  8] =-7.300714534477859e+01;
  NLX_coes[2][ 29][  9] =-7.388925896367704e+01;
  NLX_coes[2][ 29][ 10] =-7.478509725796249e+01;
  NLX_coes[2][ 29][ 11] =-7.569567823199074e+01;
  NLX_coes[2][ 29][ 12] =-7.662182363487196e+01;
  NLX_coes[2][ 29][ 13] =-7.756411362550779e+01;
  NLX_coes[2][ 29][ 14] =-7.852284376013577e+01;
  NLX_coes[2][ 29][ 15] =-7.949798816968534e+01;
  NLX_coes[2][ 29][ 16] =-8.048917260100541e+01;
  NLX_coes[2][ 29][ 17] =-8.149566086817224e+01;
  NLX_coes[2][ 29][ 18] =-8.251635920306262e+01;
  NLX_coes[2][ 29][ 19] =-8.354984488180985e+01;
  NLX_coes[2][ 29][ 20] =-8.459442252796670e+01;
  NLX_coes[2][ 29][ 21] =-8.564820280664725e+01;
  NLX_coes[2][ 29][ 22] =-8.670919116074226e+01;
  NLX_coes[2][ 29][ 23] =-8.777536629972795e+01;
  NLX_coes[2][ 29][ 24] =-8.884473633280015e+01;
  NLX_coes[2][ 29][ 25] =-8.991536826412877e+01;
  NLX_coes[2][ 29][ 26] =-9.098540292268389e+01;
  NLX_coes[2][ 29][ 27] =-9.205307106555645e+01;
  NLX_coes[2][ 29][ 28] =-9.311672343668580e+01;
  NLX_coes[2][ 29][ 29] =-9.417487708384905e+01;
  NLX_coes[2][ 29][ 30] =-9.522626901804689e+01;
  NLX_coes[2][ 29][ 31] =-9.626990557202811e+01;
  NLX_coes[2][ 29][ 32] =-9.730509912150187e+01;
  NLX_coes[2][ 29][ 33] =-9.833148874164375e+01;
  NLX_coes[2][ 29][ 34] =-9.934904538410704e+01;
  NLX_coes[2][ 29][ 35] =-1.003580645556414e+02;
  NLX_coes[2][ 29][ 36] =-1.013591504341893e+02;
  NLX_coes[2][ 29][ 37] =-1.023531953138459e+02;
  NLX_coes[2][ 29][ 38] =-1.033413576367486e+02;
  NLX_coes[2][ 29][ 39] =-1.043250409337294e+02;
  NLX_coes[2][ 29][ 40] =-1.053058749160517e+02;
  NLX_coes[2][ 30][  0] =-6.963817007081761e+01;
  NLX_coes[2][ 30][  1] =-7.046871412953324e+01;
  NLX_coes[2][ 30][  2] =-7.130175984993839e+01;
  NLX_coes[2][ 30][  3] =-7.213886945397856e+01;
  NLX_coes[2][ 30][  4] =-7.298154855014735e+01;
  NLX_coes[2][ 30][  5] =-7.383125175858301e+01;
  NLX_coes[2][ 30][  6] =-7.468938095642845e+01;
  NLX_coes[2][ 30][  7] =-7.555727632501953e+01;
  NLX_coes[2][ 30][  8] =-7.643620046622951e+01;
  NLX_coes[2][ 30][  9] =-7.732731604812233e+01;
  NLX_coes[2][ 30][ 10] =-7.823165778713947e+01;
  NLX_coes[2][ 30][ 11] =-7.915010006460241e+01;
  NLX_coes[2][ 30][ 12] =-8.008332204949680e+01;
  NLX_coes[2][ 30][ 13] =-8.103177275356212e+01;
  NLX_coes[2][ 30][ 14] =-8.199563885505752e+01;
  NLX_coes[2][ 30][ 15] =-8.297481830973899e+01;
  NLX_coes[2][ 30][ 16] =-8.396890276807686e+01;
  NLX_coes[2][ 30][ 17] =-8.497717186013746e+01;
  NLX_coes[2][ 30][ 18] =-8.599860264580424e+01;
  NLX_coes[2][ 30][ 19] =-8.703189714456806e+01;
  NLX_coes[2][ 30][ 20] =-8.807552802788366e+01;
  NLX_coes[2][ 30][ 21] =-8.912779847139474e+01;
  NLX_coes[2][ 30][ 22] =-9.018690769764643e+01;
  NLX_coes[2][ 30][ 23] =-9.125101206308497e+01;
  NLX_coes[2][ 30][ 24] =-9.231827419613218e+01;
  NLX_coes[2][ 30][ 25] =-9.338689847306458e+01;
  NLX_coes[2][ 30][ 26] =-9.445515736497066e+01;
  NLX_coes[2][ 30][ 27] =-9.552141662076443e+01;
  NLX_coes[2][ 30][ 28] =-9.658416554368341e+01;
  NLX_coes[2][ 30][ 29] =-9.764205421968514e+01;
  NLX_coes[2][ 30][ 30] =-9.869393418127915e+01;
  NLX_coes[2][ 30][ 31] =-9.973889642691424e+01;
  NLX_coes[2][ 30][ 32] =-1.007763014601392e+02;
  NLX_coes[2][ 30][ 33] =-1.018057983966316e+02;
  NLX_coes[2][ 30][ 34] =-1.028273326747479e+02;
  NLX_coes[2][ 30][ 35] =-1.038411437221355e+02;
  NLX_coes[2][ 30][ 36] =-1.048477549040088e+02;
  NLX_coes[2][ 30][ 37] =-1.058479583364328e+02;
  NLX_coes[2][ 30][ 38] =-1.068427969058305e+02;
  NLX_coes[2][ 30][ 39] =-1.078335452875198e+02;
  NLX_coes[2][ 30][ 40] =-1.088216910303501e+02;
  NLX_coes[2][ 31][  0] =-7.300115961636774e+01;
  NLX_coes[2][ 31][  1] =-7.384061557993789e+01;
  NLX_coes[2][ 31][  2] =-7.468256735240163e+01;
  NLX_coes[2][ 31][  3] =-7.552857201639139e+01;
  NLX_coes[2][ 31][  4] =-7.638011963746474e+01;
  NLX_coes[2][ 31][  5] =-7.723863414097897e+01;
  NLX_coes[2][ 31][  6] =-7.810546857743506e+01;
  NLX_coes[2][ 31][  7] =-7.898189509760537e+01;
  NLX_coes[2][ 31][  8] =-7.986908994461085e+01;
  NLX_coes[2][ 31][  9] =-8.076811389799798e+01;
  NLX_coes[2][ 31][ 10] =-8.167988888135099e+01;
  NLX_coes[2][ 31][ 11] =-8.260517183271664e+01;
  NLX_coes[2][ 31][ 12] =-8.354452737206306e+01;
  NLX_coes[2][ 31][ 13] =-8.449830120243665e+01;
  NLX_coes[2][ 31][ 14] =-8.546659647662267e+01;
  NLX_coes[2][ 31][ 15] =-8.644925550916412e+01;
  NLX_coes[2][ 31][ 16] =-8.744584923451846e+01;
  NLX_coes[2][ 31][ 17] =-8.845567675104911e+01;
  NLX_coes[2][ 31][ 18] =-8.947777704246538e+01;
  NLX_coes[2][ 31][ 19] =-9.051095405343750e+01;
  NLX_coes[2][ 31][ 20] =-9.155381438292544e+01;
  NLX_coes[2][ 31][ 21] =-9.260481448869898e+01;
  NLX_coes[2][ 31][ 22] =-9.366231205848743e+01;
  NLX_coes[2][ 31][ 23] =-9.472461575519743e+01;
  NLX_coes[2][ 31][ 24] =-9.579002887097157e+01;
  NLX_coes[2][ 31][ 25] =-9.685688562235316e+01;
  NLX_coes[2][ 31][ 26] =-9.792358184514006e+01;
  NLX_coes[2][ 31][ 27] =-9.898860377358618e+01;
  NLX_coes[2][ 31][ 28] =-1.000505580593406e+02;
  NLX_coes[2][ 31][ 29] =-1.011082040947509e+02;
  NLX_coes[2][ 31][ 30] =-1.021604871177758e+02;
  NLX_coes[2][ 31][ 31] =-1.032065689140709e+02;
  NLX_coes[2][ 31][ 32] =-1.042458528652134e+02;
  NLX_coes[2][ 31][ 33] =-1.052780011582657e+02;
  NLX_coes[2][ 31][ 34] =-1.063029433992083e+02;
  NLX_coes[2][ 31][ 35] =-1.073208771054328e+02;
  NLX_coes[2][ 31][ 36] =-1.083322613485179e+02;
  NLX_coes[2][ 31][ 37] =-1.093378051606589e+02;
  NLX_coes[2][ 31][ 38] =-1.103384522984229e+02;
  NLX_coes[2][ 31][ 39] =-1.113353636945513e+02;
  NLX_coes[2][ 31][ 40] =-1.123298985202762e+02;
  NLX_coes[2][ 32][  0] =-7.637464164493532e+01;
  NLX_coes[2][ 32][  1] =-7.722214617248126e+01;
  NLX_coes[2][ 32][  2] =-7.807212505983566e+01;
  NLX_coes[2][ 32][  3] =-7.892612410153464e+01;
  NLX_coes[2][ 32][  4] =-7.978561505387782e+01;
  NLX_coes[2][ 32][  5] =-8.065199177985089e+01;
  NLX_coes[2][ 32][  6] =-8.152656268249166e+01;
  NLX_coes[2][ 32][  7] =-8.241053980811154e+01;
  NLX_coes[2][ 32][  8] =-8.330502481537586e+01;
  NLX_coes[2][ 32][  9] =-8.421099207784221e+01;
  NLX_coes[2][ 32][ 10] =-8.512926942485991e+01;
  NLX_coes[2][ 32][ 11] =-8.606051736386037e+01;
  NLX_coes[2][ 32][ 12] =-8.700520799062259e+01;
  NLX_coes[2][ 32][ 13] =-8.796360511547536e+01;
  NLX_coes[2][ 32][ 14] =-8.893574736148447e+01;
  NLX_coes[2][ 32][ 15] =-8.992143609850545e+01;
  NLX_coes[2][ 32][ 16] =-9.092023005821959e+01;
  NLX_coes[2][ 32][ 17] =-9.193144830837367e+01;
  NLX_coes[2][ 32][ 18] =-9.295418284365704e+01;
  NLX_coes[2][ 32][ 19] =-9.398732119889041e+01;
  NLX_coes[2][ 32][ 20] =-9.502957821952479e+01;
  NLX_coes[2][ 32][ 21] =-9.607953469412381e+01;
  NLX_coes[2][ 32][ 22] =-9.713567944897180e+01;
  NLX_coes[2][ 32][ 23] =-9.819645131423349e+01;
  NLX_coes[2][ 32][ 24] =-9.926027818902597e+01;
  NLX_coes[2][ 32][ 25] =-1.003256120820724e+02;
  NLX_coes[2][ 32][ 26] =-1.013909606605274e+02;
  NLX_coes[2][ 32][ 27] =-1.024549168648852e+02;
  NLX_coes[2][ 32][ 28] =-1.035161880726048e+02;
  NLX_coes[2][ 32][ 29] =-1.045736253079615e+02;
  NLX_coes[2][ 32][ 30] =-1.056262517075602e+02;
  NLX_coes[2][ 32][ 31] =-1.066732884904826e+02;
  NLX_coes[2][ 32][ 32] =-1.077141764599396e+02;
  NLX_coes[2][ 32][ 33] =-1.087485915118048e+02;
  NLX_coes[2][ 32][ 34] =-1.097764534039911e+02;
  NLX_coes[2][ 32][ 35] =-1.107979278159993e+02;
  NLX_coes[2][ 32][ 36] =-1.118134223014256e+02;
  NLX_coes[2][ 32][ 37] =-1.128235770485582e+02;
  NLX_coes[2][ 32][ 38] =-1.138292514502398e+02;
  NLX_coes[2][ 32][ 39] =-1.148315074156274e+02;
  NLX_coes[2][ 32][ 40] =-1.158315902311813e+02;
  NLX_coes[2][ 33][  0] =-7.975749942182431e+01;
  NLX_coes[2][ 33][  1] =-8.061219946401026e+01;
  NLX_coes[2][ 33][  2] =-8.146934413802617e+01;
  NLX_coes[2][ 33][  3] =-8.233046597816084e+01;
  NLX_coes[2][ 33][  4] =-8.319701869846966e+01;
  NLX_coes[2][ 33][  5] =-8.407036843657306e+01;
  NLX_coes[2][ 33][  6] =-8.495178354818742e+01;
  NLX_coes[2][ 33][  7] =-8.584242313510521e+01;
  NLX_coes[2][ 33][  8] =-8.674332418448137e+01;
  NLX_coes[2][ 33][  9] =-8.765538727477137e+01;
  NLX_coes[2][ 33][ 10] =-8.857936106369668e+01;
  NLX_coes[2][ 33][ 11] =-8.951582612383555e+01;
  NLX_coes[2][ 33][ 12] =-9.046517903761814e+01;
  NLX_coes[2][ 33][ 13] =-9.142761794687415e+01;
  NLX_coes[2][ 33][ 14] =-9.240313093783701e+01;
  NLX_coes[2][ 33][ 15] =-9.339148871239297e+01;
  NLX_coes[2][ 33][ 16] =-9.439224293748664e+01;
  NLX_coes[2][ 33][ 17] =-9.540473144999041e+01;
  NLX_coes[2][ 33][ 18] =-9.642809106995361e+01;
  NLX_coes[2][ 33][ 19] =-9.746127810245986e+01;
  NLX_coes[2][ 33][ 20] =-9.850309575057307e+01;
  NLX_coes[2][ 33][ 21] =-9.955222679051312e+01;
  NLX_coes[2][ 33][ 22] =-1.006072692621810e+02;
  NLX_coes[2][ 33][ 23] =-1.016667728331374e+02;
  NLX_coes[2][ 33][ 24] =-1.027292739754712e+02;
  NLX_coes[2][ 33][ 25] =-1.037933289723500e+02;
  NLX_coes[2][ 33][ 26] =-1.048575446925227e+02;
  NLX_coes[2][ 33][ 27] =-1.059206076546397e+02;
  NLX_coes[2][ 33][ 28] =-1.069813119539243e+02;
  NLX_coes[2][ 33][ 29] =-1.080385861814267e+02;
  NLX_coes[2][ 33][ 30] =-1.090915188195472e+02;
  NLX_coes[2][ 33][ 31] =-1.101393810675081e+02;
  NLX_coes[2][ 33][ 32] =-1.111816458625804e+02;
  NLX_coes[2][ 33][ 33] =-1.122180020430179e+02;
  NLX_coes[2][ 33][ 34] =-1.132483630078578e+02;
  NLX_coes[2][ 33][ 35] =-1.142728696847162e+02;
  NLX_coes[2][ 33][ 36] =-1.152918879847081e+02;
  NLX_coes[2][ 33][ 37] =-1.163060011500696e+02;
  NLX_coes[2][ 33][ 38] =-1.173159975091547e+02;
  NLX_coes[2][ 33][ 39] =-1.183228541984572e+02;
  NLX_coes[2][ 33][ 40] =-1.193277174955533e+02;
  NLX_coes[2][ 34][  0] =-8.314870239164200e+01;
  NLX_coes[2][ 34][  1] =-8.400975885924659e+01;
  NLX_coes[2][ 34][  2] =-8.487322924451337e+01;
  NLX_coes[2][ 34][  3] =-8.574063501700611e+01;
  NLX_coes[2][ 34][  4] =-8.661341470643896e+01;
  NLX_coes[2][ 34][  5] =-8.749291002377595e+01;
  NLX_coes[2][ 34][  6] =-8.838035352209617e+01;
  NLX_coes[2][ 34][  7] =-8.927685711236072e+01;
  NLX_coes[2][ 34][  8] =-9.018340077606909e+01;
  NLX_coes[2][ 34][  9] =-9.110082096642883e+01;
  NLX_coes[2][ 34][ 10] =-9.202979858032269e+01;
  NLX_coes[2][ 34][ 11] =-9.297084681083258e+01;
  NLX_coes[2][ 34][ 12] =-9.392429955745529e+01;
  NLX_coes[2][ 34][ 13] =-9.489030133844300e+01;
  NLX_coes[2][ 34][ 14] =-9.586879980398592e+01;
  NLX_coes[2][ 34][ 15] =-9.685954198643606e+01;
  NLX_coes[2][ 34][ 16] =-9.786207533930350e+01;
  NLX_coes[2][ 34][ 17] =-9.887575439752378e+01;
  NLX_coes[2][ 34][ 18] =-9.989975352235679e+01;
  NLX_coes[2][ 34][ 19] =-1.009330856848649e+02;
  NLX_coes[2][ 34][ 20] =-1.019746266534621e+02;
  NLX_coes[2][ 34][ 21] =-1.030231434008637e+02;
  NLX_coes[2][ 34][ 22] =-1.040773251862775e+02;
  NLX_coes[2][ 34][ 23] =-1.051358157094866e+02;
  NLX_coes[2][ 34][ 24] =-1.061972449993194e+02;
  NLX_coes[2][ 34][ 25] =-1.072602601813769e+02;
  NLX_coes[2][ 34][ 26] =-1.083235547867656e+02;
  NLX_coes[2][ 34][ 27] =-1.093858966142712e+02;
  NLX_coes[2][ 34][ 28] =-1.104461542357782e+02;
  NLX_coes[2][ 34][ 29] =-1.115033220504030e+02;
  NLX_coes[2][ 34][ 30] =-1.125565434767809e+02;
  NLX_coes[2][ 34][ 31] =-1.136051315921335e+02;
  NLX_coes[2][ 34][ 32] =-1.146485864053076e+02;
  NLX_coes[2][ 34][ 33] =-1.156866080226556e+02;
  NLX_coes[2][ 34][ 34] =-1.167191051738470e+02;
  NLX_coes[2][ 34][ 35] =-1.177461988105735e+02;
  NLX_coes[2][ 34][ 36] =-1.187682206875345e+02;
  NLX_coes[2][ 34][ 37] =-1.197857069440322e+02;
  NLX_coes[2][ 34][ 38] =-1.207993867574455e+02;
  NLX_coes[2][ 34][ 39] =-1.218101662480097e+02;
  NLX_coes[2][ 34][ 40] =-1.228191078752520e+02;
  NLX_coes[2][ 35][  0] =-8.654729199714767e+01;
  NLX_coes[2][ 35][  1] =-8.741388111959087e+01;
  NLX_coes[2][ 35][  2] =-8.828286064709626e+01;
  NLX_coes[2][ 35][  3] =-8.915574764490080e+01;
  NLX_coes[2][ 35][  4] =-9.003397011629436e+01;
  NLX_coes[2][ 35][  5] =-9.091884853332523e+01;
  NLX_coes[2][ 35][  6] =-9.181158253788659e+01;
  NLX_coes[2][ 35][  7] =-9.271324046746169e+01;
  NLX_coes[2][ 35][  8] =-9.362475033376491e+01;
  NLX_coes[2][ 35][  9] =-9.454689115922271e+01;
  NLX_coes[2][ 35][ 10] =-9.548028425592071e+01;
  NLX_coes[2][ 35][ 11] =-9.642538458648932e+01;
  NLX_coes[2][ 35][ 12] =-9.738247274493131e+01;
  NLX_coes[2][ 35][ 13] =-9.835164834392559e+01;
  NLX_coes[2][ 35][ 14] =-9.933282571432775e+01;
  NLX_coes[2][ 35][ 15] =-1.003257328286273e+02;
  NLX_coes[2][ 35][ 16] =-1.013299142600670e+02;
  NLX_coes[2][ 35][ 17] =-1.023447387838466e+02;
  NLX_coes[2][ 35][ 18] =-1.033694119206719e+02;
  NLX_coes[2][ 35][ 19] =-1.044029933380470e+02;
  NLX_coes[2][ 35][ 20] =-1.054444186107930e+02;
  NLX_coes[2][ 35][ 21] =-1.064925244755146e+02;
  NLX_coes[2][ 35][ 22] =-1.075460764766634e+02;
  NLX_coes[2][ 35][ 23] =-1.086037978508646e+02;
  NLX_coes[2][ 35][ 24] =-1.096643986348482e+02;
  NLX_coes[2][ 35][ 25] =-1.107266042500518e+02;
  NLX_coes[2][ 35][ 26] =-1.117891831120916e+02;
  NLX_coes[2][ 35][ 27] =-1.128509730304510e+02;
  NLX_coes[2][ 35][ 28] =-1.139109062398528e+02;
  NLX_coes[2][ 35][ 29] =-1.149680328412978e+02;
  NLX_coes[2][ 35][ 30] =-1.160215422873740e+02;
  NLX_coes[2][ 35][ 31] =-1.170707824087919e+02;
  NLX_coes[2][ 35][ 32] =-1.181152754157726e+02;
  NLX_coes[2][ 35][ 33] =-1.191547303428395e+02;
  NLX_coes[2][ 35][ 34] =-1.201890515119966e+02;
  NLX_coes[2][ 35][ 35] =-1.212183427066974e+02;
  NLX_coes[2][ 35][ 36] =-1.222429068120742e+02;
  NLX_coes[2][ 35][ 37] =-1.232632406477454e+02;
  NLX_coes[2][ 35][ 38] =-1.242800246138057e+02;
  NLX_coes[2][ 35][ 39] =-1.252941069033966e+02;
  NLX_coes[2][ 35][ 40] =-1.263064818475133e+02;
  NLX_coes[2][ 36][  0] =-8.995236393042055e+01;
  NLX_coes[2][ 36][  1] =-9.082367617893277e+01;
  NLX_coes[2][ 36][  2] =-9.169737352201409e+01;
  NLX_coes[2][ 36][  3] =-9.257498013367044e+01;
  NLX_coes[2][ 36][  4] =-9.345791800385220e+01;
  NLX_coes[2][ 36][  5] =-9.434748782720393e+01;
  NLX_coes[2][ 36][  6] =-9.524485652319345e+01;
  NLX_coes[2][ 36][  7] =-9.615104951043423e+01;
  NLX_coes[2][ 36][  8] =-9.706694489886149e+01;
  NLX_coes[2][ 36][  9] =-9.799326804996016e+01;
  NLX_coes[2][ 36][ 10] =-9.893058596775525e+01;
  NLX_coes[2][ 36][ 11] =-9.987930166108397e+01;
  NLX_coes[2][ 36][ 12] =-1.008396490061543e+02;
  NLX_coes[2][ 36][ 13] =-1.018116888338573e+02;
  NLX_coes[2][ 36][ 14] =-1.027953070322586e+02;
  NLX_coes[2][ 36][ 15] =-1.037902154243610e+02;
  NLX_coes[2][ 36][ 16] =-1.047959560688043e+02;
  NLX_coes[2][ 36][ 17] =-1.058119094435946e+02;
  NLX_coes[2][ 36][ 18] =-1.068373067204862e+02;
  NLX_coes[2][ 36][ 19] =-1.078712460421609e+02;
  NLX_coes[2][ 36][ 20] =-1.089127124127807e+02;
  NLX_coes[2][ 36][ 21] =-1.099606005532011e+02;
  NLX_coes[2][ 36][ 22] =-1.110137399020768e+02;
  NLX_coes[2][ 36][ 23] =-1.120709208928329e+02;
  NLX_coes[2][ 36][ 24] =-1.131309217021642e+02;
  NLX_coes[2][ 36][ 25] =-1.141925348132764e+02;
  NLX_coes[2][ 36][ 26] =-1.152545929089240e+02;
  NLX_coes[2][ 36][ 27] =-1.163159937473008e+02;
  NLX_coes[2][ 36][ 28] =-1.173757237407504e+02;
  NLX_coes[2][ 36][ 29] =-1.184328799500878e+02;
  NLX_coes[2][ 36][ 30] =-1.194866901541869e+02;
  NLX_coes[2][ 36][ 31] =-1.205365305993826e+02;
  NLX_coes[2][ 36][ 32] =-1.215819410142721e+02;
  NLX_coes[2][ 36][ 33] =-1.226226365076818e+02;
  NLX_coes[2][ 36][ 34] =-1.236585160350086e+02;
  NLX_coes[2][ 36][ 35] =-1.246896671758007e+02;
  NLX_coes[2][ 36][ 36] =-1.257163669472895e+02;
  NLX_coes[2][ 36][ 37] =-1.267390782131170e+02;
  NLX_coes[2][ 36][ 38] =-1.277584409216849e+02;
  NLX_coes[2][ 36][ 39] =-1.287752570372416e+02;
  NLX_coes[2][ 36][ 40] =-1.297904692473710e+02;
  NLX_coes[2][ 37][  0] =-9.336304524400840e+01;
  NLX_coes[2][ 37][  1] =-9.423828097179675e+01;
  NLX_coes[2][ 37][  2] =-9.511593360310475e+01;
  NLX_coes[2][ 37][  3] =-9.599754832214522e+01;
  NLX_coes[2][ 37][  4] =-9.688454179124211e+01;
  NLX_coes[2][ 37][  5] =-9.777819216837250e+01;
  NLX_coes[2][ 37][  6] =-9.867962946562443e+01;
  NLX_coes[2][ 37][  7] =-9.958983309336251e+01;
  NLX_coes[2][ 37][  8] =-1.005096302283641e+02;
  NLX_coes[2][ 37][  9] =-1.014396936703163e+02;
  NLX_coes[2][ 37][ 10] =-1.023805389577578e+02;
  NLX_coes[2][ 37][ 11] =-1.033325211255869e+02;
  NLX_coes[2][ 37][ 12] =-1.042958317578455e+02;
  NLX_coes[2][ 37][ 13] =-1.052704970722709e+02;
  NLX_coes[2][ 37][ 14] =-1.062563777621010e+02;
  NLX_coes[2][ 37][ 15] =-1.072531712499513e+02;
  NLX_coes[2][ 37][ 16] =-1.082604168861773e+02;
  NLX_coes[2][ 37][ 17] =-1.092775044535671e+02;
  NLX_coes[2][ 37][ 18] =-1.103036861300425e+02;
  NLX_coes[2][ 37][ 19] =-1.113380918285834e+02;
  NLX_coes[2][ 37][ 20] =-1.123797476050122e+02;
  NLX_coes[2][ 37][ 21] =-1.134275966305064e+02;
  NLX_coes[2][ 37][ 22] =-1.144805220947379e+02;
  NLX_coes[2][ 37][ 23] =-1.155373713534543e+02;
  NLX_coes[2][ 37][ 24] =-1.165969806587137e+02;
  NLX_coes[2][ 37][ 25] =-1.176581998898131e+02;
  NLX_coes[2][ 37][ 26] =-1.187199168049012e+02;
  NLX_coes[2][ 37][ 27] =-1.197810804237065e+02;
  NLX_coes[2][ 37][ 28] =-1.208407232081410e+02;
  NLX_coes[2][ 37][ 29] =-1.218979817256562e+02;
  NLX_coes[2][ 37][ 30] =-1.229521154734509e+02;
  NLX_coes[2][ 37][ 31] =-1.240025235333757e+02;
  NLX_coes[2][ 37][ 32] =-1.250487587400535e+02;
  NLX_coes[2][ 37][ 33] =-1.260905390897999e+02;
  NLX_coes[2][ 37][ 34] =-1.271277561886022e+02;
  NLX_coes[2][ 37][ 35] =-1.281604806000331e+02;
  NLX_coes[2][ 37][ 36] =-1.291889639357205e+02;
  NLX_coes[2][ 37][ 37] =-1.302136373296222e+02;
  NLX_coes[2][ 37][ 38] =-1.312351054661149e+02;
  NLX_coes[2][ 37][ 39] =-1.322541328604768e+02;
  NLX_coes[2][ 37][ 40] =-1.332716267630251e+02;
  NLX_coes[2][ 38][  0] =-9.677846327306179e+01;
  NLX_coes[2][ 38][  1] =-9.765682211713622e+01;
  NLX_coes[2][ 38][  2] =-9.853770932738466e+01;
  NLX_coes[2][ 38][  3] =-9.942268701655102e+01;
  NLX_coes[2][ 38][  4] =-1.003131624001635e+02;
  NLX_coes[2][ 38][  5] =-1.012103791678303e+02;
  NLX_coes[2][ 38][  6] =-1.021154203990690e+02;
  NLX_coes[2][ 38][  7] =-1.030292124392174e+02;
  NLX_coes[2][ 38][  8] =-1.039525277714789e+02;
  NLX_coes[2][ 38][  9] =-1.048859856782208e+02;
  NLX_coes[2][ 38][ 10] =-1.058300513021371e+02;
  NLX_coes[2][ 38][ 11] =-1.067850340215497e+02;
  NLX_coes[2][ 38][ 12] =-1.077510859857449e+02;
  NLX_coes[2][ 38][ 13] =-1.087282015628682e+02;
  NLX_coes[2][ 38][ 14] =-1.097162183658233e+02;
  NLX_coes[2][ 38][ 15] =-1.107148204224197e+02;
  NLX_coes[2][ 38][ 16] =-1.117235439317264e+02;
  NLX_coes[2][ 38][ 17] =-1.127417858957439e+02;
  NLX_coes[2][ 38][ 18] =-1.137688157391524e+02;
  NLX_coes[2][ 38][ 19] =-1.148037898426185e+02;
  NLX_coes[2][ 38][ 20] =-1.158457687353566e+02;
  NLX_coes[2][ 38][ 21] =-1.168937365408284e+02;
  NLX_coes[2][ 38][ 22] =-1.179466221632006e+02;
  NLX_coes[2][ 38][ 23] =-1.190033216509161e+02;
  NLX_coes[2][ 38][ 24] =-1.200627211755811e+02;
  NLX_coes[2][ 38][ 25] =-1.211237201062857e+02;
  NLX_coes[2][ 38][ 26] =-1.221852537214089e+02;
  NLX_coes[2][ 38][ 27] =-1.232463151614017e+02;
  NLX_coes[2][ 38][ 28] =-1.243059762726422e+02;
  NLX_coes[2][ 38][ 29] =-1.253634070202492e+02;
  NLX_coes[2][ 38][ 30] =-1.264178931630547e+02;
  NLX_coes[2][ 38][ 31] =-1.274688518999430e+02;
  NLX_coes[2][ 38][ 32] =-1.285158452278843e+02;
  NLX_coes[2][ 38][ 33] =-1.295585908094065e+02;
  NLX_coes[2][ 38][ 34] =-1.305969702372830e+02;
  NLX_coes[2][ 38][ 35] =-1.316310347071421e+02;
  NLX_coes[2][ 38][ 36] =-1.326610082328748e+02;
  NLX_coes[2][ 38][ 37] =-1.336872884616737e+02;
  NLX_coes[2][ 38][ 38] =-1.347104442146656e+02;
  NLX_coes[2][ 38][ 39] =-1.357312082930196e+02;
  NLX_coes[2][ 38][ 40] =-1.367504586041483e+02;
  NLX_coes[2][ 39][  0] =-1.001977108134847e+02;
  NLX_coes[2][ 39][  1] =-1.010783482431170e+02;
  NLX_coes[2][ 39][  2] =-1.019618395815538e+02;
  NLX_coes[2][ 39][  3] =-1.028496344852572e+02;
  NLX_coes[2][ 39][  4] =-1.037431317661032e+02;
  NLX_coes[2][ 39][  5] =-1.046435196936784e+02;
  NLX_coes[2][ 39][  6] =-1.055518170275656e+02;
  NLX_coes[2][ 39][  7] =-1.064688867708846e+02;
  NLX_coes[2][ 39][  8] =-1.073954415989561e+02;
  NLX_coes[2][ 39][  9] =-1.083320454118116e+02;
  NLX_coes[2][ 39][ 10] =-1.092791130942154e+02;
  NLX_coes[2][ 39][ 11] =-1.102369096613143e+02;
  NLX_coes[2][ 39][ 12] =-1.112055496339590e+02;
  NLX_coes[2][ 39][ 13] =-1.121849973275897e+02;
  NLX_coes[2][ 39][ 14] =-1.131750686307171e+02;
  NLX_coes[2][ 39][ 15] =-1.141754347476724e+02;
  NLX_coes[2][ 39][ 16] =-1.151856282662649e+02;
  NLX_coes[2][ 39][ 17] =-1.162050517785513e+02;
  NLX_coes[2][ 39][ 18] =-1.172329891352754e+02;
  NLX_coes[2][ 39][ 19] =-1.182686192612027e+02;
  NLX_coes[2][ 39][ 20] =-1.193110323130541e+02;
  NLX_coes[2][ 39][ 21] =-1.203592478385051e+02;
  NLX_coes[2][ 39][ 22] =-1.214122345056702e+02;
  NLX_coes[2][ 39][ 23] =-1.224689309238486e+02;
  NLX_coes[2][ 39][ 24] =-1.235282670667702e+02;
  NLX_coes[2][ 39][ 25] =-1.245891858307410e+02;
  NLX_coes[2][ 39][ 26] =-1.256506642990579e+02;
  NLX_coes[2][ 39][ 27] =-1.267117343274791e+02;
  NLX_coes[2][ 39][ 28] =-1.277715021036551e+02;
  NLX_coes[2][ 39][ 29] =-1.288291663626917e+02;
  NLX_coes[2][ 39][ 30] =-1.298840349643917e+02;
  NLX_coes[2][ 39][ 31] =-1.309355395629456e+02;
  NLX_coes[2][ 39][ 32] =-1.319832481366226e+02;
  NLX_coes[2][ 39][ 33] =-1.330268752030110e+02;
  NLX_coes[2][ 39][ 34] =-1.340662896348259e+02;
  NLX_coes[2][ 39][ 35] =-1.351015201298204e+02;
  NLX_coes[2][ 39][ 36] =-1.361327586277257e+02;
  NLX_coes[2][ 39][ 37] =-1.371603624999526e+02;
  NLX_coes[2][ 39][ 38] =-1.381848576144515e+02;
  NLX_coes[2][ 39][ 39] =-1.392069521006570e+02;
  NLX_coes[2][ 39][ 40] =-1.402274389855678e+02;
  NLX_coes[2][ 40][  0] =-1.036195929014140e+02;
  NLX_coes[2][ 40][  1] =-1.045019362590390e+02;
  NLX_coes[2][ 40][  2] =-1.053873988952547e+02;
  NLX_coes[2][ 40][  3] =-1.062776175534461e+02;
  NLX_coes[2][ 40][  4] =-1.071738380699572e+02;
  NLX_coes[2][ 40][  5] =-1.080771493656470e+02;
  NLX_coes[2][ 40][  6] =-1.089884879768674e+02;
  NLX_coes[2][ 40][  7] =-1.099086452887680e+02;
  NLX_coes[2][ 40][  8] =-1.108382702978146e+02;
  NLX_coes[2][ 40][  9] =-1.117778700773706e+02;
  NLX_coes[2][ 40][ 10] =-1.127278092292708e+02;
  NLX_coes[2][ 40][ 11] =-1.136883091728524e+02;
  NLX_coes[2][ 40][ 12] =-1.146594479137637e+02;
  NLX_coes[2][ 40][ 13] =-1.156411608267601e+02;
  NLX_coes[2][ 40][ 14] =-1.166332429080786e+02;
  NLX_coes[2][ 40][ 15] =-1.176353528730861e+02;
  NLX_coes[2][ 40][ 16] =-1.186470193815010e+02;
  NLX_coes[2][ 40][ 17] =-1.196676495630066e+02;
  NLX_coes[2][ 40][ 18] =-1.206965398942398e+02;
  NLX_coes[2][ 40][ 19] =-1.217328893520804e+02;
  NLX_coes[2][ 40][ 20] =-1.227758146485948e+02;
  NLX_coes[2][ 40][ 21] =-1.238243672509119e+02;
  NLX_coes[2][ 40][ 22] =-1.248775518135975e+02;
  NLX_coes[2][ 40][ 23] =-1.259343456063813e+02;
  NLX_coes[2][ 40][ 24] =-1.269937185056010e+02;
  NLX_coes[2][ 40][ 25] =-1.280546531277412e+02;
  NLX_coes[2][ 40][ 26] =-1.291161647089798e+02;
  NLX_coes[2][ 40][ 27] =-1.301773203663210e+02;
  NLX_coes[2][ 40][ 28] =-1.312372574065106e+02;
  NLX_coes[2][ 40][ 29] =-1.322952003753664e+02;
  NLX_coes[2][ 40][ 30] =-1.333504765636539e+02;
  NLX_coes[2][ 40][ 31] =-1.344025297107281e+02;
  NLX_coes[2][ 40][ 32] =-1.354509316799225e+02;
  NLX_coes[2][ 40][ 33] =-1.364953919262867e+02;
  NLX_coes[2][ 40][ 34] =-1.375357646443507e+02;
  NLX_coes[2][ 40][ 35] =-1.385720535824389e+02;
  NLX_coes[2][ 40][ 36] =-1.396044146687568e+02;
  NLX_coes[2][ 40][ 37] =-1.406331568631710e+02;
  NLX_coes[2][ 40][ 38] =-1.416587421358769e+02;
  NLX_coes[2][ 40][ 39] =-1.426817804461402e+02;
  NLX_coes[2][ 40][ 40] =-1.437031430308185e+02;
}

