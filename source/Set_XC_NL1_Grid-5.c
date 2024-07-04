#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

static double mask_func_q(int m_flag, double q, double alpha_para);
static void chi_func(int bf_flag, double rho, double drho, double chi[3]);
static void gx_kernel(int mu, double rho, double drho, double gx[3]);

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

  static int firsttime=1;
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
  double alpha_para,G2,fmu;
  int spin,spinsize,N2D,GN,GNs,BN_CB,BN_AB,mu,Nmu;

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

  Nmu = 3;
  alpha_para = 6.0;

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
    drho = 0.100;

    d = rho/10000.0;

    k = 0;
    i = 2;

    chi_func(k, rho, drho, chi);
    
    printf("A d_chi=%18.15f\n",chi[i]);

    if (i==1){
      chi_func(k, rho+d, drho, chi);
      t1 = chi[0];
      chi_func(k, rho-d, drho, chi);
      t2 = chi[0];
    }
    else if (i==2){
      chi_func(k, rho, drho+d, chi);
      t1 = chi[0];
      chi_func(k, rho, drho-d, chi);
      t2 = chi[0];
    }

    printf("N d_chi=%18.15f\n",(t1-t2)/(2.0*d));
    MPI_Finalize();
    exit(0);
  }

  MPI_Finalize();
  exit(0);

  */
  

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
    i = 1;

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

    for (mu=0; mu<Nmu; mu++){

      /* calculate gx and the derivatives of gx with respect to rho and drho */

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

	rho = Density_Grid_B[spin][BN_AB] + PCCDensity_Grid_B[spin][BN_AB];

	drho = sqrt( Re_drho_r[spin][0][BN_AB]*Re_drho_r[spin][0][BN_AB] + Im_drho_r[spin][0][BN_AB]*Im_drho_r[spin][0][BN_AB]
                   + Re_drho_r[spin][1][BN_AB]*Re_drho_r[spin][1][BN_AB] + Im_drho_r[spin][1][BN_AB]*Im_drho_r[spin][1][BN_AB]
                   + Re_drho_r[spin][2][BN_AB]*Re_drho_r[spin][2][BN_AB] + Im_drho_r[spin][2][BN_AB]*Im_drho_r[spin][2][BN_AB] );

        /* gx and its derivatives */

        gx_kernel(mu, 2.0*rho, 2.0*drho, gx);

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

      /*
      MPI_Finalize();
      exit(0);
      */

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


void gx_kernel(int mu, double rho, double drho, double gx[3])
{
  int n,Nchi;
  int Eid[300],Rid[300],Zid[300];
  double sum,coes[5][30];
  double Achi[3];
  double NLX_coes[3][300];
  double x,y,chi,dchi0,dchi1;
  double id,jd,kd,pe,pe1,pr,pr1,pZ,pZ1,w,tmp;
  double a,b,c,g,dg,f,df,e,de0,de1,dr0,dr1,dZ0,dZ1;
  double r,Z,xy,yx,ln;
  double threshold=1.0e-12;

  /* avoid the numerical instability if rho and drho are small. */

  if ( rho<threshold ){
    gx[0] = 0.0;
    gx[1] = 0.0;
    gx[2] = 0.0;
    return;
  }
  else if ( drho<(0.1*threshold) ){
    drho = 0.1*threshold;
  }

  /* setting of parameters */

  Nchi = 190;

NLX_coes[0][  0]=-0.1038934170405;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
NLX_coes[0][  1]=+0.0016113114326;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
NLX_coes[0][  2]=-0.0439177164046;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
NLX_coes[0][  3]=-0.0595428958025;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
NLX_coes[0][  4]=+0.0112156364964;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
NLX_coes[0][  5]=+0.0975503596878;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
NLX_coes[0][  6]=-0.0448170563395;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
NLX_coes[0][  7]=+0.0690311501921;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
NLX_coes[0][  8]=-0.0648041575410;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
NLX_coes[0][  9]=-0.0628329390724;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
NLX_coes[0][ 10]=-0.1045993299701;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
NLX_coes[0][ 11]=+0.0281825065842;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
NLX_coes[0][ 12]=+0.0401049979348;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
NLX_coes[0][ 13]=+0.1395257003615;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
NLX_coes[0][ 14]=-0.1650782520141;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
NLX_coes[0][ 15]=+0.0050740438811;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
NLX_coes[0][ 16]=-0.1497796572879;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
NLX_coes[0][ 17]=+0.1196119490019;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
NLX_coes[0][ 18]=-0.1694772065223;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
NLX_coes[0][ 19]=+0.2359484519801;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
NLX_coes[0][ 20]=+0.0540244971392;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
NLX_coes[0][ 21]=-0.0192823077482;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
NLX_coes[0][ 22]=-0.0993946276719;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
NLX_coes[0][ 23]=+0.0624973692084;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
NLX_coes[0][ 24]=+0.0701297071205;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
NLX_coes[0][ 25]=+0.1353391324083;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
NLX_coes[0][ 26]=-0.1665921445601;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
NLX_coes[0][ 27]=-0.0556533882199;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
NLX_coes[0][ 28]=-0.1395754952693;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
NLX_coes[0][ 29]=-0.3040323837064;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
NLX_coes[0][ 30]=+0.2397256731155;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
NLX_coes[0][ 31]=-0.1655742734415;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
NLX_coes[0][ 32]=+0.2684442089703;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
NLX_coes[0][ 33]=+0.0154310883954;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
NLX_coes[0][ 34]=-0.0803399744290;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
NLX_coes[0][ 35]=+0.0912705086986;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
NLX_coes[0][ 36]=+0.0635120793723;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
NLX_coes[0][ 37]=+0.0395720942764;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
NLX_coes[0][ 38]=-0.1062552916310;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
NLX_coes[0][ 39]=-0.0825670271981;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
NLX_coes[0][ 40]=-0.0652271039435;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
NLX_coes[0][ 41]=-0.0091273679138;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
NLX_coes[0][ 42]=+0.4535207919478;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
NLX_coes[0][ 43]=-0.2852878862528;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
NLX_coes[0][ 44]=-0.0598142707394;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
NLX_coes[0][ 45]=+0.0329368549085;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
NLX_coes[0][ 46]=-0.0639282642619;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
NLX_coes[0][ 47]=+0.0871498305958;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
NLX_coes[0][ 48]=-0.0121311785375;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
NLX_coes[0][ 49]=-0.0605255734408;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
NLX_coes[0][ 50]=-0.0340943068493;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
NLX_coes[0][ 51]=-0.0422147293960;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
NLX_coes[0][ 52]=+0.1004266047731;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
NLX_coes[0][ 53]=+0.0008714273998;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
NLX_coes[0][ 54]=+0.3082482349541;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
NLX_coes[0][ 55]=-0.2784361299764;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
NLX_coes[0][ 56]=-0.2161703448510;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
NLX_coes[0][ 57]=+0.0333757839398;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
NLX_coes[0][ 58]=-0.0597435529395;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
NLX_coes[0][ 59]=+0.0459810504358;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
NLX_coes[0][ 60]=-0.1180758656988;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
NLX_coes[0][ 61]=-0.1067051306888;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
NLX_coes[0][ 62]=+0.0275300800903;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
NLX_coes[0][ 63]=+0.0052099711004;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
NLX_coes[0][ 64]=+0.2103805047809;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
NLX_coes[0][ 65]=-0.0048893608546;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
NLX_coes[0][ 66]=+0.0635642122642;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
NLX_coes[0][ 67]=-0.2057355943844;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
NLX_coes[0][ 68]=-0.1343588057504;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
NLX_coes[0][ 69]=+0.0211592006500;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
NLX_coes[0][ 70]=-0.0635015031291;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
NLX_coes[0][ 71]=-0.0072490550051;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
NLX_coes[0][ 72]=-0.1803951888542;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
NLX_coes[0][ 73]=-0.0981191963074;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
NLX_coes[0][ 74]=+0.0655581266273;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
NLX_coes[0][ 75]=+0.0176126704505;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
NLX_coes[0][ 76]=+0.1944148114322;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
NLX_coes[0][ 77]=+0.0593432778709;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
NLX_coes[0][ 78]=-0.1161803468186;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
NLX_coes[0][ 79]=-0.0949330558798;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
NLX_coes[0][ 80]=+0.0366672400960;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
NLX_coes[0][ 81]=+0.0030223492795;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
NLX_coes[0][ 82]=-0.0652419603949;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
NLX_coes[0][ 83]=-0.0429101702210;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
NLX_coes[0][ 84]=-0.1437541794702;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
NLX_coes[0][ 85]=-0.0566287140186;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
NLX_coes[0][ 86]=+0.0748065985693;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
NLX_coes[0][ 87]=-0.0087688671027;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
NLX_coes[0][ 88]=+0.0803477794689;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
NLX_coes[0][ 89]=+0.1086666822866;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
NLX_coes[0][ 90]=-0.2011122143140;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
NLX_coes[0][ 91]=+0.0182883905507;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
NLX_coes[0][ 92]=+0.1578545785010;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
NLX_coes[0][ 93]=-0.0141340881804;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
NLX_coes[0][ 94]=-0.0577558984303;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
NLX_coes[0][ 95]=-0.0458889406406;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
NLX_coes[0][ 96]=-0.0111320512951;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
NLX_coes[0][ 97]=-0.0040169772692;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
NLX_coes[0][ 98]=+0.0589194961330;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
NLX_coes[0][ 99]=-0.0511471570311;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
NLX_coes[0][100]=-0.0522829596828;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
NLX_coes[0][101]=+0.0747749629445;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
NLX_coes[0][102]=-0.2081540063887;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
NLX_coes[0][103]=+0.1021388176304;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
NLX_coes[0][104]=+0.1595565422140;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
NLX_coes[0][105]=-0.0251265869738;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
NLX_coes[0][106]=-0.0409201580766;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
NLX_coes[0][107]=-0.0218696477310;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
NLX_coes[0][108]=+0.1517469184939;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
NLX_coes[0][109]=+0.0419097619440;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
NLX_coes[0][110]=+0.0273602969353;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
NLX_coes[0][111]=-0.0788717077027;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
NLX_coes[0][112]=-0.1252638464939;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
NLX_coes[0][113]=-0.0199981840015;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
NLX_coes[0][114]=-0.1484879040269;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
NLX_coes[0][115]=+0.1431681318440;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
NLX_coes[0][116]=+0.0472997316490;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
NLX_coes[0][117]=-0.0273810581986;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
NLX_coes[0][118]=-0.0213572913359;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
NLX_coes[0][119]=+0.0085971660692;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
NLX_coes[0][120]=+0.2429507169018;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
NLX_coes[0][121]=+0.0651722401381;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
NLX_coes[0][122]=-0.0094030647169;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
NLX_coes[0][123]=-0.0708930078819;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
NLX_coes[0][124]=-0.1029853677326;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
NLX_coes[0][125]=-0.0916327510420;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
NLX_coes[0][126]=-0.0291117240555;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
NLX_coes[0][127]=+0.1450966696893;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
NLX_coes[0][128]=-0.1105372398306;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
NLX_coes[0][129]=-0.0207419916987;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
NLX_coes[0][130]=-0.0084247666530;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
NLX_coes[0][131]=+0.0240178720705;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
NLX_coes[0][132]=+0.1801763708862;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
NLX_coes[0][133]=+0.0542500541742;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
NLX_coes[0][134]=-0.0423937379673;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
NLX_coes[0][135]=-0.0284499656235;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
NLX_coes[0][136]=-0.0158737201346;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
NLX_coes[0][137]=-0.0708598948186;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
NLX_coes[0][138]=+0.1148365677338;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
NLX_coes[0][139]=+0.1128386031981;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
NLX_coes[0][140]=-0.2105377709245;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
NLX_coes[0][141]=-0.0072813451841;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
NLX_coes[0][142]=-0.0078435586388;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
NLX_coes[0][143]=+0.0185298131409;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
NLX_coes[0][144]=-0.0342530534863;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
NLX_coes[0][145]=+0.0131530848448;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
NLX_coes[0][146]=-0.0620723831592;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
NLX_coes[0][147]=+0.0218408082166;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
NLX_coes[0][148]=+0.0464535910748;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
NLX_coes[0][149]=+0.0238351476521;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
NLX_coes[0][150]=+0.1926005255944;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
NLX_coes[0][151]=+0.0350635449912;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
NLX_coes[0][152]=-0.1629368515860;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
NLX_coes[0][153]=+0.0080222614342;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
NLX_coes[0][154]=-0.0166204309101;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
NLX_coes[0][155]=+0.0108095519855;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
NLX_coes[0][156]=-0.2576815380193;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
NLX_coes[0][157]=-0.0264721628220;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
NLX_coes[0][158]=-0.0544641144261;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
NLX_coes[0][159]=+0.0398293066583;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
NLX_coes[0][160]=-0.0071538313878;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
NLX_coes[0][161]=+0.0704883762555;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
NLX_coes[0][162]=+0.0872672826135;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
NLX_coes[0][163]=-0.1060920314101;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
NLX_coes[0][164]=+0.0372116671384;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
NLX_coes[0][165]=+0.0166309969278;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
NLX_coes[0][166]=-0.0230405771190;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
NLX_coes[0][167]=+0.0285917859782;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
NLX_coes[0][168]=-0.2319762153729;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
NLX_coes[0][169]=-0.0162890586071;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
NLX_coes[0][170]=-0.0084826386129;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
NLX_coes[0][171]=+0.0036300555856;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
NLX_coes[0][172]=-0.1250663982861;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
NLX_coes[0][173]=-0.0362410443279;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
NLX_coes[0][174]=-0.1698880292388;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
NLX_coes[0][175]=-0.2431116524733;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
NLX_coes[0][176]=+0.2243946120579;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
NLX_coes[0][177]=+0.0120739756530;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
NLX_coes[0][178]=-0.0175773656300;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
NLX_coes[0][179]=+0.0409994872542;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
NLX_coes[0][180]=+0.2153521843129;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
NLX_coes[0][181]=+0.0209170377071;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
NLX_coes[0][182]=+0.0334752722962;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
NLX_coes[0][183]=-0.0379291994609;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
NLX_coes[0][184]=+0.1732427125462;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
NLX_coes[0][185]=-0.0015700712602;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
NLX_coes[0][186]=+0.0753924336870;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
NLX_coes[0][187]=+0.0544230418714;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
NLX_coes[0][188]=-0.0104160304306;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
NLX_coes[0][189]=+1.0000000218316;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
NLX_coes[1][  0]=+0.1513438319503;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
NLX_coes[1][  1]=-0.0510789658998;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
NLX_coes[1][  2]=+0.0717152181580;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
NLX_coes[1][  3]=-0.0734696544973;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
NLX_coes[1][  4]=+0.0064063893328;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
NLX_coes[1][  5]=-0.3654287818287;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
NLX_coes[1][  6]=-0.0636579031994;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
NLX_coes[1][  7]=-0.9323261811474;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
NLX_coes[1][  8]=+0.2439974368105;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
NLX_coes[1][  9]=+0.3293411785964;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
NLX_coes[1][ 10]=+0.0985634081532;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
NLX_coes[1][ 11]=-0.0681453905615;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
NLX_coes[1][ 12]=-0.0189427411211;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
NLX_coes[1][ 13]=+0.0065795405968;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
NLX_coes[1][ 14]=+0.2677200261288;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
NLX_coes[1][ 15]=+0.3664496857585;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
NLX_coes[1][ 16]=+0.4883086270953;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
NLX_coes[1][ 17]=+0.0541190607132;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
NLX_coes[1][ 18]=-0.2216092088193;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
NLX_coes[1][ 19]=-0.6342600578766;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
NLX_coes[1][ 20]=-0.2506707615659;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
NLX_coes[1][ 21]=+0.2560834475486;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
NLX_coes[1][ 22]=+0.0202060210720;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
NLX_coes[1][ 23]=-0.1841169793679;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
NLX_coes[1][ 24]=-0.1975487664460;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
NLX_coes[1][ 25]=+0.0044314400492;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
NLX_coes[1][ 26]=+0.2797277790087;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
NLX_coes[1][ 27]=+0.2050805944928;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
NLX_coes[1][ 28]=+0.3008101594074;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
NLX_coes[1][ 29]=-0.0042597301484;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
NLX_coes[1][ 30]=+0.1507159953974;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
NLX_coes[1][ 31]=+0.3053787637522;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
NLX_coes[1][ 32]=+1.1020888576610;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
NLX_coes[1][ 33]=+0.1971602671307;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
NLX_coes[1][ 34]=+0.0014952214769;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
NLX_coes[1][ 35]=-0.1944568401250;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
NLX_coes[1][ 36]=-0.1318430141092;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
NLX_coes[1][ 37]=-0.0369895250280;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
NLX_coes[1][ 38]=+0.1913102150411;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
NLX_coes[1][ 39]=-0.0370842143561;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
NLX_coes[1][ 40]=-0.1959498256527;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
NLX_coes[1][ 41]=-0.0762365466950;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
NLX_coes[1][ 42]=+0.1473642309611;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
NLX_coes[1][ 43]=+0.3265994040494;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
NLX_coes[1][ 44]=+0.6904762329625;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
NLX_coes[1][ 45]=+0.1578120328424;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
NLX_coes[1][ 46]=+0.0249371089879;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
NLX_coes[1][ 47]=-0.1128007677488;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
NLX_coes[1][ 48]=+0.1064139064011;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
NLX_coes[1][ 49]=-0.0625219558796;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
NLX_coes[1][ 50]=+0.1306080075865;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
NLX_coes[1][ 51]=-0.1224162343419;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
NLX_coes[1][ 52]=-0.3957132283376;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
NLX_coes[1][ 53]=-0.1186031592770;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
NLX_coes[1][ 54]=+0.0557257067167;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
NLX_coes[1][ 55]=+0.1586271143060;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
NLX_coes[1][ 56]=+0.0259392875754;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
NLX_coes[1][ 57]=+0.1287009526480;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
NLX_coes[1][ 58]=+0.0518269129905;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
NLX_coes[1][ 59]=-0.0215178435519;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
NLX_coes[1][ 60]=+0.2848625075839;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
NLX_coes[1][ 61]=-0.0668995108360;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
NLX_coes[1][ 62]=+0.1070931318192;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
NLX_coes[1][ 63]=-0.0785317749845;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
NLX_coes[1][ 64]=-0.3011869533610;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
NLX_coes[1][ 65]=-0.1107232908253;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
NLX_coes[1][ 66]=-0.0138695180419;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
NLX_coes[1][ 67]=+0.0434100215566;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
NLX_coes[1][ 68]=-0.3147684186842;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
NLX_coes[1][ 69]=+0.1027254232450;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
NLX_coes[1][ 70]=+0.0607443391049;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
NLX_coes[1][ 71]=+0.0293380314385;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
NLX_coes[1][ 72]=+0.2841033345907;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
NLX_coes[1][ 73]=-0.0592716292686;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
NLX_coes[1][ 74]=+0.1022078130327;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
NLX_coes[1][ 75]=+0.0078547902085;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
NLX_coes[1][ 76]=-0.1029444548261;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
NLX_coes[1][ 77]=-0.0682619787829;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
NLX_coes[1][ 78]=-0.0401430574641;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
NLX_coes[1][ 79]=+0.0071130752069;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
NLX_coes[1][ 80]=-0.3335334050039;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
NLX_coes[1][ 81]=+0.0774571255859;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
NLX_coes[1][ 82]=+0.0497549655629;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
NLX_coes[1][ 83]=+0.0334383993500;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
NLX_coes[1][ 84]=+0.1250468690551;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
NLX_coes[1][ 85]=-0.0497321592423;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
NLX_coes[1][ 86]=+0.0993091293226;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
NLX_coes[1][ 87]=+0.0732286698251;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
NLX_coes[1][ 88]=+0.0565686261692;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
NLX_coes[1][ 89]=-0.0150503320550;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
NLX_coes[1][ 90]=-0.0318911227669;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
NLX_coes[1][ 91]=+0.0213303394521;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
NLX_coes[1][ 92]=-0.1718266230508;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
NLX_coes[1][ 93]=+0.0535922201382;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
NLX_coes[1][ 94]=+0.0282319127287;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
NLX_coes[1][ 95]=+0.0091718368920;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
NLX_coes[1][ 96]=-0.0929306408783;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
NLX_coes[1][ 97]=-0.0446025533562;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
NLX_coes[1][ 98]=+0.0904488441984;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
NLX_coes[1][ 99]=+0.0903734679414;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
NLX_coes[1][100]=+0.1209811726204;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
NLX_coes[1][101]=+0.0285792806513;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
NLX_coes[1][102]=-0.0083007420866;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
NLX_coes[1][103]=+0.0489975539261;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
NLX_coes[1][104]=+0.0320561008042;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
NLX_coes[1][105]=+0.0327067137888;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
NLX_coes[1][106]=+0.0084030757123;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
NLX_coes[1][107]=-0.0198160464890;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
NLX_coes[1][108]=-0.2612440574048;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
NLX_coes[1][109]=-0.0452410254912;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
NLX_coes[1][110]=+0.0760739815646;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
NLX_coes[1][111]=+0.0621091182102;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
NLX_coes[1][112]=+0.0998545438117;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
NLX_coes[1][113]=+0.0509014229523;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
NLX_coes[1][114]=+0.0114132200185;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
NLX_coes[1][115]=+0.0620992942108;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
NLX_coes[1][116]=+0.1787383135470;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
NLX_coes[1][117]=+0.0155669684338;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
NLX_coes[1][118]=-0.0008305693901;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
NLX_coes[1][119]=-0.0374971603433;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
NLX_coes[1][120]=-0.3011658098324;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
NLX_coes[1][121]=-0.0488314261291;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
NLX_coes[1][122]=+0.0615596663167;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
NLX_coes[1][123]=+0.0101946202467;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
NLX_coes[1][124]=+0.0402453704983;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
NLX_coes[1][125]=+0.0504120015423;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
NLX_coes[1][126]=+0.0136660387269;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
NLX_coes[1][127]=+0.0499659642504;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
NLX_coes[1][128]=+0.2167734654090;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
NLX_coes[1][129]=+0.0014586499578;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
NLX_coes[1][130]=+0.0028876427488;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
NLX_coes[1][131]=-0.0414885831287;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
NLX_coes[1][132]=-0.1832334293177;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
NLX_coes[1][133]=-0.0501437706558;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
NLX_coes[1][134]=+0.0524963861024;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
NLX_coes[1][135]=-0.0368428376033;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
NLX_coes[1][136]=-0.0011533351351;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
NLX_coes[1][137]=+0.0345564308276;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
NLX_coes[1][138]=-0.0069572982108;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
NLX_coes[1][139]=+0.0215658674961;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
NLX_coes[1][140]=+0.1430890058573;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
NLX_coes[1][141]=-0.0112496157798;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
NLX_coes[1][142]=+0.0151841600574;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
NLX_coes[1][143]=-0.0417111899378;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
NLX_coes[1][144]=+0.0647725451914;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
NLX_coes[1][145]=-0.0437230568093;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
NLX_coes[1][146]=+0.0499859633310;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
NLX_coes[1][147]=-0.0554113472490;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
NLX_coes[1][148]=+0.0132870108529;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
NLX_coes[1][149]=+0.0157769919998;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
NLX_coes[1][150]=-0.0468157236177;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
NLX_coes[1][151]=+0.0029715496691;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
NLX_coes[1][152]=-0.0006489413687;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
NLX_coes[1][153]=-0.0235437059500;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
NLX_coes[1][154]=+0.0274601247690;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
NLX_coes[1][155]=-0.0519170488971;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
NLX_coes[1][156]=+0.3327784846488;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
NLX_coes[1][157]=-0.0261904694188;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
NLX_coes[1][158]=+0.0473060670119;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
NLX_coes[1][159]=-0.0370499789494;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
NLX_coes[1][160]=+0.0698904849192;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
NLX_coes[1][161]=+0.0056777955881;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
NLX_coes[1][162]=-0.0928847318751;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
NLX_coes[1][163]=+0.0285055161710;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
NLX_coes[1][164]=-0.1376194854767;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
NLX_coes[1][165]=-0.0336946959169;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
NLX_coes[1][166]=+0.0316441445643;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
NLX_coes[1][167]=-0.0717009484756;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
NLX_coes[1][168]=+0.3709501295425;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
NLX_coes[1][169]=+0.0013062917241;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
NLX_coes[1][170]=+0.0300258607141;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
NLX_coes[1][171]=+0.0053927600229;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
NLX_coes[1][172]=+0.0622031343307;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
NLX_coes[1][173]=+0.0096586973101;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
NLX_coes[1][174]=-0.1215011683217;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
NLX_coes[1][175]=+0.1209395483632;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
NLX_coes[1][176]=-0.1797490703346;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
NLX_coes[1][177]=-0.0360444308214;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
NLX_coes[1][178]=+0.0253582899616;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
NLX_coes[1][179]=-0.0572534801234;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
NLX_coes[1][180]=-0.3360116205325;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
NLX_coes[1][181]=+0.0298853493335;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
NLX_coes[1][182]=-0.0167489942530;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
NLX_coes[1][183]=+0.0407603792458;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
NLX_coes[1][184]=-0.2611561176980;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
NLX_coes[1][185]=+0.0270204813672;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
NLX_coes[1][186]=-0.0930137430850;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
NLX_coes[1][187]=+0.2526078248855;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
NLX_coes[1][188]=-0.0817900874036;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
NLX_coes[1][189]=+0.0000000009571;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
NLX_coes[2][  0]=-0.0356910997130;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
NLX_coes[2][  1]=+0.0069371404050;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
NLX_coes[2][  2]=-0.0824575413242;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
NLX_coes[2][  3]=+0.1661447336424;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
NLX_coes[2][  4]=+0.3122755795367;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
NLX_coes[2][  5]=+0.0058056350087;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
NLX_coes[2][  6]=-0.0134314792982;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
NLX_coes[2][  7]=-0.1162937059996;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
NLX_coes[2][  8]=+0.0041894445345;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
NLX_coes[2][  9]=+0.0715447905578;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
NLX_coes[2][ 10]=-0.0422479358926;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
NLX_coes[2][ 11]=+0.0395288584225;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
NLX_coes[2][ 12]=+0.1639421654387;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
NLX_coes[2][ 13]=-0.0335168476255;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
NLX_coes[2][ 14]=+0.0385600526503;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
NLX_coes[2][ 15]=-0.0551866444601;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
NLX_coes[2][ 16]=+0.1046716863393;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
NLX_coes[2][ 17]=-0.1022614247758;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
NLX_coes[2][ 18]=+0.0710190129852;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
NLX_coes[2][ 19]=-0.0328920449405;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
NLX_coes[2][ 20]=-0.2150390574055;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
NLX_coes[2][ 21]=+0.0595846671032;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
NLX_coes[2][ 22]=-0.0266679275042;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
NLX_coes[2][ 23]=+0.0131905999191;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
NLX_coes[2][ 24]=-0.0795713295006;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
NLX_coes[2][ 25]=-0.0367285169038;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
NLX_coes[2][ 26]=+0.0581419703271;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
NLX_coes[2][ 27]=-0.0771901879001;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
NLX_coes[2][ 28]=-0.0601102696197;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
NLX_coes[2][ 29]=+0.3754451288469;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
NLX_coes[2][ 30]=-0.0649626146324;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
NLX_coes[2][ 31]=-0.0625874599804;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
NLX_coes[2][ 32]=-0.0129958375491;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
NLX_coes[2][ 33]=+0.0452467644601;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
NLX_coes[2][ 34]=-0.0130393502693;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
NLX_coes[2][ 35]=+0.0339403831470;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
NLX_coes[2][ 36]=+0.0567750505423;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
NLX_coes[2][ 37]=-0.0108271271001;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
NLX_coes[2][ 38]=+0.0469929690199;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
NLX_coes[2][ 39]=-0.0216130829379;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
NLX_coes[2][ 40]=-0.0028389397564;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
NLX_coes[2][ 41]=+0.0868209847674;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
NLX_coes[2][ 42]=-0.1281532626244;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
NLX_coes[2][ 43]=+0.1551918830370;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
NLX_coes[2][ 44]=-0.2581119605238;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
NLX_coes[2][ 45]=+0.0393867138115;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
NLX_coes[2][ 46]=-0.0125906198392;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
NLX_coes[2][ 47]=+0.0053856299799;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
NLX_coes[2][ 48]=+0.0361623837099;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
NLX_coes[2][ 49]=+0.0247695742175;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
NLX_coes[2][ 50]=-0.0012239318598;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
NLX_coes[2][ 51]=+0.0051615111897;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
NLX_coes[2][ 52]=+0.1889259418921;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
NLX_coes[2][ 53]=-0.1389997582497;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
NLX_coes[2][ 54]=-0.1500975707970;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
NLX_coes[2][ 55]=+0.1838417231411;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
NLX_coes[2][ 56]=+0.0355092371670;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
NLX_coes[2][ 57]=+0.0441004156724;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
NLX_coes[2][ 58]=-0.0114720500980;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
NLX_coes[2][ 59]=-0.0354263493642;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
NLX_coes[2][ 60]=-0.1196984290278;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
NLX_coes[2][ 61]=+0.0579632901297;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
NLX_coes[2][ 62]=-0.0303617873344;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
NLX_coes[2][ 63]=-0.0161142570514;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
NLX_coes[2][ 64]=+0.1005866235322;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
NLX_coes[2][ 65]=-0.2095191922117;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
NLX_coes[2][ 66]=-0.0998493030818;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
NLX_coes[2][ 67]=+0.0405753293883;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
NLX_coes[2][ 68]=+0.0868375336251;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
NLX_coes[2][ 69]=+0.0515234874877;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
NLX_coes[2][ 70]=-0.0057975555381;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
NLX_coes[2][ 71]=-0.0375781686960;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
NLX_coes[2][ 72]=-0.1820895462780;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
NLX_coes[2][ 73]=+0.0682632439733;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
NLX_coes[2][ 74]=-0.0240331618183;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
NLX_coes[2][ 75]=-0.0242934672817;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
NLX_coes[2][ 76]=-0.0783051393943;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
NLX_coes[2][ 77]=-0.1539574949465;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
NLX_coes[2][ 78]=+0.0248184177925;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
NLX_coes[2][ 79]=-0.0784322095227;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
NLX_coes[2][ 80]=-0.0293891460041;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
NLX_coes[2][ 81]=+0.0544883341072;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
NLX_coes[2][ 82]=-0.0017688261929;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
NLX_coes[2][ 83]=-0.0047727108729;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
NLX_coes[2][ 84]=-0.0858292010466;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
NLX_coes[2][ 85]=+0.0454239043832;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
NLX_coes[2][ 86]=+0.0018514262053;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
NLX_coes[2][ 87]=-0.0042765473010;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
NLX_coes[2][ 88]=-0.1607201966762;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
NLX_coes[2][ 89]=-0.0401095481429;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
NLX_coes[2][ 90]=+0.1512717064135;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
NLX_coes[2][ 91]=-0.1158236818233;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
NLX_coes[2][ 92]=-0.1160299849108;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
NLX_coes[2][ 93]=+0.0505276292102;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
NLX_coes[2][ 94]=-0.0044216091036;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
NLX_coes[2][ 95]=+0.0312781288929;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
NLX_coes[2][ 96]=+0.0848949263412;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
NLX_coes[2][ 97]=-0.0013247800707;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
NLX_coes[2][ 98]=+0.0288826661984;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
NLX_coes[2][ 99]=+0.0247036071620;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
NLX_coes[2][100]=-0.1086058636421;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
NLX_coes[2][101]=+0.0668576855281;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
NLX_coes[2][102]=+0.2065112307595;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
NLX_coes[2][103]=-0.0935419308338;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
NLX_coes[2][104]=-0.0973981020666;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
NLX_coes[2][105]=+0.0408407244332;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
NLX_coes[2][106]=-0.0126806953922;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
NLX_coes[2][107]=+0.0435860948353;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
NLX_coes[2][108]=+0.1993530825927;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
NLX_coes[2][109]=-0.0488510707371;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
NLX_coes[2][110]=+0.0466597927744;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
NLX_coes[2][111]=+0.0368043158490;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
NLX_coes[2][112]=+0.0137164656902;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
NLX_coes[2][113]=+0.1193076335405;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
NLX_coes[2][114]=+0.1556898696225;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
NLX_coes[2][115]=-0.0473588806958;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
NLX_coes[2][116]=+0.0107574655397;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
NLX_coes[2][117]=+0.0281763424791;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
NLX_coes[2][118]=-0.0201729279698;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
NLX_coes[2][119]=+0.0268437550981;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
NLX_coes[2][120]=+0.1709476100975;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
NLX_coes[2][121]=-0.0714902448066;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
NLX_coes[2][122]=+0.0528305869621;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
NLX_coes[2][123]=+0.0178713503975;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
NLX_coes[2][124]=+0.1079193399640;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
NLX_coes[2][125]=+0.0956121511190;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
NLX_coes[2][126]=+0.0145667761913;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
NLX_coes[2][127]=-0.0037689014737;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
NLX_coes[2][128]=+0.1354642994403;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
NLX_coes[2][129]=+0.0152180054997;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
NLX_coes[2][130]=-0.0196291220734;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
NLX_coes[2][131]=-0.0018714493829;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
NLX_coes[2][132]=+0.0112659682196;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
NLX_coes[2][133]=-0.0543022677045;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
NLX_coes[2][134]=+0.0494960676230;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
NLX_coes[2][135]=-0.0240865921444;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
NLX_coes[2][136]=+0.1026714707880;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
NLX_coes[2][137]=+0.0119187137540;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
NLX_coes[2][138]=-0.1469711585923;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
NLX_coes[2][139]=+0.0291411795820;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
NLX_coes[2][140]=+0.1788188212168;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
NLX_coes[2][141]=+0.0040833945828;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
NLX_coes[2][142]=-0.0087390135480;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
NLX_coes[2][143]=-0.0160196653058;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
NLX_coes[2][144]=-0.1589692697959;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
NLX_coes[2][145]=-0.0056684304036;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
NLX_coes[2][146]=+0.0381532481821;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
NLX_coes[2][147]=-0.0583941084203;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
NLX_coes[2][148]=-0.0011140564479;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
NLX_coes[2][149]=-0.0688991240108;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
NLX_coes[2][150]=-0.2165579237718;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
NLX_coes[2][151]=+0.0636055654600;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
NLX_coes[2][152]=+0.0676560939914;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
NLX_coes[2][153]=-0.0029765643991;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
NLX_coes[2][154]=+0.0056711844048;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
NLX_coes[2][155]=-0.0004279970341;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
NLX_coes[2][156]=-0.1750714627881;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
NLX_coes[2][157]=+0.0374896407175;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
NLX_coes[2][158]=+0.0161895486071;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
NLX_coes[2][159]=-0.0467816378777;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
NLX_coes[2][160]=-0.1134431906958;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
NLX_coes[2][161]=-0.0553733990074;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
NLX_coes[2][162]=-0.0929279732412;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
NLX_coes[2][163]=+0.1143382979906;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
NLX_coes[2][164]=-0.1702724425457;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
NLX_coes[2][165]=-0.0030120400956;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
NLX_coes[2][166]=+0.0099156928023;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
NLX_coes[2][167]=+0.0292200528653;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
NLX_coes[2][168]=+0.0187954715886;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
NLX_coes[2][169]=+0.0283864363226;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
NLX_coes[2][170]=-0.0167522174995;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
NLX_coes[2][171]=+0.0185083568854;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
NLX_coes[2][172]=-0.0895101729360;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
NLX_coes[2][173]=+0.0736195315345;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
NLX_coes[2][174]=+0.1640791930389;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
NLX_coes[2][175]=+0.1306330214325;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
NLX_coes[2][176]=-0.2977544634035;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
NLX_coes[2][177]=+0.0035610715663;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
NLX_coes[2][178]=-0.0009366392235;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
NLX_coes[2][179]=+0.0257366608281;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
NLX_coes[2][180]=+0.0790949565865;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
NLX_coes[2][181]=-0.0114371372093;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
NLX_coes[2][182]=-0.0234597507223;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
NLX_coes[2][183]=+0.0452102091974;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
NLX_coes[2][184]=+0.1467295639143;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
NLX_coes[2][185]=-0.0157788386040;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
NLX_coes[2][186]=-0.0149860715659;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
NLX_coes[2][187]=-0.1808570761643;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
NLX_coes[2][188]=+0.2746850185382;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
NLX_coes[2][189]=-0.0000000017564;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;

  /* convert rho and drho to r, Z, and e */

  x = rho;
  y = fabs(drho);

  if (x<1.0e-20) x = 1.0e-20;
  if (y<1.0e-20) y = 1.0e-20;

  xy = x/y;
  yx = y/x;
  ln = -log(8.0*PI) - 3.0*log(pow(x,4.0/3.0)/y);

  r = xy*ln;
  dr0 = (-4.0 + ln)/y;
  dr1 = (3.0 - ln)*xy/y;

  w = 1.0e-20;
  tmp = 1.0/(x*x+w);
  Z = 0.50*x*y*tmp;
  dZ0 = -x*x*y*tmp*tmp + 0.5*y*tmp; 
  dZ1 = 0.5*x*tmp;

  /*
  if (2.0<Z){
    gx[0] = 0.0;
    gx[1] = 0.0;
    gx[2] = 0.0;
    return;
  }
  */

  a = 0.25;
  e = exp(-a*Z*r);
  de0 = -a*e*Z*dr0 - a*e*r*dZ0;
  de1 = -a*e*Z*dr1 - a*e*r*dZ1;

  b = 1.0;
  c =-2.0;
  f = 0.5*(erf(b*(r-c))+1.0);
  df = b*exp(-b*b*(r-c)*(r-c))/sqrt(PI);
  g = 1.0 - f;
  dg = -df;

  /* calculate gx[0,1,2] */

  gx[0] = 0.0;
  gx[1] = 0.0;
  gx[2] = 0.0;

  for (n=0; n<=(Nchi-2); n++){

    id = (double)Eid[n];
    jd = (double)Rid[n];
    kd = (double)Zid[n];

    pe1 = pow(e,id-1.0);
    pe = pe1*e;

    pr1 = pow(r,jd-1.0);
    pr = pr1*r;

    pZ1 = pow(Z,kd-1.0);
    pZ = pZ1*Z;

    chi = f*pe*pr*pZ;
    dchi0 = df*dr0*pe*pr*pZ + f*(id*pe1*de0*pr*pZ + jd*pr1*dr0*pe*pZ + kd*pZ1*dZ0*pe*pr);
    dchi1 = df*dr1*pe*pr*pZ + f*(id*pe1*de1*pr*pZ + jd*pr1*dr1*pe*pZ + kd*pZ1*dZ1*pe*pr);

    gx[0] += NLX_coes[mu][n]*chi;
    gx[1] += NLX_coes[mu][n]*dchi0;
    gx[2] += NLX_coes[mu][n]*dchi1;

    //printf("VVV1 n=%2d x=%15.12f y=%15.12f r=%15.12f Z=%15.12f e=%15.12f   %15.12f %15.12f %15.12f\n",n,x,y,r,Z,e,gx[0],gx[1],gx[2]);
    
  }

  //printf("VVV3 x=%15.12f y=%15.12f r=%15.12f Z=%15.12f e=%15.12f\n",x,y,r,Z,e);

  /* exchange by GGA */

  chi = (-3.0/4.0*pow(3.0/PI*x,1.0/3.0)*(1.0 + 10.0/81.0*yx*yx/(4.0*pow(3.0*PI*PI*x,2.0/3.0))));
  dchi0 = dg*dr0*chi
    + g*(-pow(3.0/PI,1.0/3.0)/(4.0*pow(x,2.0/3.0)) + 35.0*yx*yx/(648.0*pow(3.0,1.0/3.0)*pow(PI,5.0/3.0)*pow(x,4.0/3.0)));
  dchi1 = dg*dr1*chi
    + g*(-5.0*yx/(108.0*pow(3.0,1.0/3.0)*pow(PI,5.0/3.0)*pow(x,4.0/3.0)));

  n = Nchi - 1;

  gx[0] += NLX_coes[mu][n]*chi*g;
  gx[1] += NLX_coes[mu][n]*dchi0;
  gx[2] += NLX_coes[mu][n]*dchi1;

}


void chi_func(int bf_flag, double rho, double drho, double chi[3])
{
  double r,Z,e,x,y,minval;
  double ln,ln2,ln3,ln4,ln5,x2,x3,x4,x5,y2,y3,y4,y5;

  x = rho;
  y = drho;

  minval = 1.0e-9;
  if (x<minval) x = minval;
  if (y<minval) y = minval;

  ln = -(log(8.0*PI) + 3.0*log(pow(x,4.0/3.0)/y));
  ln2 = ln*ln;
  ln3 = ln2*ln;
  ln4 = ln2*ln2;
  ln5 = ln2*ln3;
  x2 = x*x;
  x3 = x2*x;
  x4 = x2*x2;
  x5 = x3*x2;
  y2 = y*y;
  y3 = y2*y;
  y4 = y2*y2;
  y5 = y3*y2;
  r = x/y*ln;
  Z = 0.5*y/x;
  e = exp(-2.0*Z*r);

  /*
  if (r<0.0){
    printf("r is negative. r=%18.15f\n",r);
  }
  */

  /*
  printf("r=%18.15f\n",r);
  */

  switch (bf_flag){

  case 0:
 
    /* H-like */


    if (fabs(r)<1.0e-5){
      chi[0] = -2.0*Z + 2.0*Z*Z*r - 4.0/3.0*Z*Z*Z*r*r + 2.0/3.0*Z*Z*Z*Z*r*r*r;
      chi[1] = -y/x2 + 5.0*y*ln/(6.0*x2) - y*ln2/(3.0*x2) - y*ln3/(24.0*x);
      chi[2] = 1.0/(2.0*x) -ln/(2.0*x) + 5.0*ln2/(24.0*x) + ln3/(24.0*x);
    }
    else{
      chi[0] = -1.0/r + e/r;
      chi[1] = 32.0*PI*x2/(y2*ln2) - 4.0*y/(x2*ln2) + 24.0*PI*x2/(y2*ln) + y/(x2*ln);
      chi[2] = 3.0/(x*ln2) - 24.0*PI*x3/(y3*ln2) - 1.0/(x*ln) - 16.0*PI*x3/(y3*ln);
    }

    /*
    printf("WWW %15.12f %15.12f  %15.12f %15.12f %15.12f %15.12f  %15.12f %15.12f %15.12f\n",
           x,y, ln,r,Z,e, chi[0],chi[1],chi[2]);
    */

    /*
    chi[0] = (1.0e+2)*x*x;
    chi[1] = (1.0e+2)*2.0*x;
    chi[2] = 0.0;
    */

    /*
    chi[0] = (1.0e+2)*x*x*y*y*y;
    chi[1] = (1.0e+2)*2.0*x*y*y*y;
    chi[2] = (1.0e+2)*3.0*x*x*y*y;
    */

    /*
    chi[0] = x/(y+1.0);
    chi[1] = 1.0/(y+1.0);
    chi[2] = -x/(y+1)/(y+1);
    */

    break;

  case 1:

    chi[0] = Z*e;
    chi[1] = 12.0*PI*x2/y2;
    chi[2] = -8.0*PI*x3/y3;

    break;

  case 2:

    chi[0] = r*Z*Z*e; 
    chi[1] = -8.0*PI*x2/y2 + 6.0*PI*x2*ln/y2;
    chi[2] = 6.0*PI*x3/y3 - 4.0*PI*x3*ln/y3;

    break;

  case 3:

    chi[0] = r*r*Z*Z*Z*e;  
    chi[1] = -8.0*PI*x2*ln/y2 + 3.0*PI*x2*ln2/y2;
    chi[2] = 6.0*PI*x3*ln/y3 - 2.0*PI*x3*ln2/y3;

    break;

  case 4:

    chi[0] = r*r*r*Z*Z*Z*Z*e;
    chi[1] = -6.0*PI*x2*ln2/y2 + 3.0*PI*x2*ln3/(2.0*y2);
    chi[2] = 9.0*PI*x3*ln2/(2.0*y3) - PI*x3*ln3/y3;

    break;

  case 5:

    chi[0] = r*r*r*r*Z*Z*Z*Z*Z*e;
    chi[1] = -4.0*PI*x2*ln3/y2 + 3.0*PI*x2*ln4/(4.0*y2);
    chi[2] = 3.0*PI*x3*ln3/y3 - PI*x3*ln4/(2.0*y3);

    break;

  case 6:

    chi[0] = r*r*r*r*r*Z*Z*Z*Z*Z*Z*e;
    chi[1] = -5.0*PI*x2*ln4/(2.0*y2) + 3.0*PI*x2*ln5/(8.0*y2);
    chi[2] = 15.0*PI*x3*ln4/(8.0*y3) - PI*x3*ln5/(4.0*y3);


    break;

  case 7:

    chi[0] = Z*Z*Z*e;   // proportinal to rho
    chi[1] = PI;
    chi[2] = 0.0;

    break;

  case 8:

    chi[0] = r*Z*Z*Z*Z*e;
    chi[1] = -2.0*PI + 0.5*PI*ln;
    chi[2] = 3.0*PI*x/(2.0*y);

    break;

  case 9:

    chi[0] = r*r*Z*Z*Z*Z*Z*e;
    chi[1] = -2.0*PI*ln + 0.25*PI*ln2;
    chi[2] = 3.0*PI*x*ln/(2.0*y);

    break;

  case 10:

    chi[0] = r*r*r*Z*Z*Z*Z*Z*Z*e;
    chi[1] = -1.5*PI*ln2 + 1.0/8.0*PI*ln3;
    chi[2] = 9.0*PI*x*ln2/(8.0*y);

    break;

  case 11:

    chi[0] = r*r*r*r*Z*Z*Z*Z*Z*Z*Z*e;
    chi[1] = -PI*ln3 + 1.0/16.0*PI*ln4;
    chi[2] = 3.0*PI*x*ln3/(4.0*y);

    break;

  case 12:

    chi[0] = Z*Z*e;
    chi[1] = 4.0*PI*x/y;
    chi[2] = -2.0*PI*x2/y2;

    break;

  case 13:

    chi[0] = r*Z*Z*Z*e;
    chi[1] = -4.0*PI*x/y + 2.0*PI*x*ln/y;
    chi[2] = 3.0*PI*x2/y2 - PI*x2*ln/y2;

    break;

  case 14:

    chi[0] = r*rho;
    chi[1] = -4.0*x/y + 2.0*x*ln/y;
    chi[2] = 3.0*x2/y2 - x2*ln/y2;

    break;

  case 15:

    chi[0] = r*r*rho;
    chi[1] = -8.0*x2*ln/y2 + 3.0*x2*ln2/y2;
    chi[2] = 6.0*x3*ln/y3 - 2.0*x3*ln2/y3;

    break;

  case 16:

    chi[0] = r*drho;
    chi[1] = -4.0 + ln; 
    chi[2] = 3.0*x/y;

    break;

  case 17:

    chi[0] = r*r*drho;
    chi[1] = -8.0*x*ln/y + 2.0*x*ln2/y;
    chi[2] = 6.0*x2*ln/y2 - x2*ln2/y2;

    break;

  case 18:

    chi[0] = pow(e,0.50);
    chi[1] = 4.0*sqrt(2.0*PI)*x3/(sqrt(x4/y3)*y3);
    chi[2] = -3.0*sqrt(2.0*PI)*x4/(sqrt(x4/y3)*y4);

    break;

  case 19:

    chi[0] = pow(e,1.00);
    chi[1] = 32.0*PI*x3/y3;
    chi[2] = -24.0*PI*x4/y4;

    break;

  case 20:

    chi[0] = pow(e,1.50);
    chi[1] = 96.0*PI*x3*sqrt(2.0*PI*x4/y3)/y3;
    chi[2] = -72.0*PI*x4*sqrt(2.0*PI*x4/y3)/y4;

    break;

  case 21:

    chi[0] = e*pow(Z,0.50);
    chi[1] = -2.0*sqrt(2)*PI*x2/(y2*sqrt(y/x)) + 16.0*sqrt(2)*PI*x3*sqrt(y/x)/y3;
    chi[2] = 2.0*sqrt(2)*PI*x3/(y3*sqrt(y/x)) - 12.0*sqrt(2)*PI*x4*sqrt(y/x)/y4;

    break;

  case 22:

    chi[0] = rho*e;
    chi[1] = 40.0*PI*x4/y3;
    chi[2] = -24.0*PI*x5/y4;

    break;

  case 23:

    chi[0] = e*r*r*Z;
    chi[1] = -32.0*PI*x4*ln/y4 + 20.0*PI*x4*ln2/y4;
    chi[2] = 24.0*PI*x5*ln/y5 - 16.0*PI*x5*ln2/y5;

    break;

  case 24:

    chi[0] = Z*e*r;
    chi[1] = -16.0*PI*x3/y3 + 16.0*PI*x3*ln/y3;
    chi[2] = 12.0*PI*x4/y4 - 12.0*PI*x4*ln/y4;


    break;

  default:
    printf("bf_flag is invalid.\n"); exit(0);

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




