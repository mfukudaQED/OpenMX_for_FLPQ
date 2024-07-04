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

  NLX_coes[0][  0]=-0.1510321032997;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[0][  1]=+0.0309327724648;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[0][  2]=-0.0354924377232;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[0][  3]=-0.0383669291900;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[0][  4]=-0.0849688863096;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[0][  5]=+0.0629584394071;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[0][  6]=-0.0288060521086;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[0][  7]=+0.0974884830201;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[0][  8]=-0.0507518610772;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[0][  9]=+0.0422397455425;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[0][ 10]=-0.1450218980397;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[0][ 11]=+0.0086439492683;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[0][ 12]=+0.0326414835510;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[0][ 13]=-0.0275193955910;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[0][ 14]=+0.0634979638466;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[0][ 15]=+0.0352111826025;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[0][ 16]=-0.0653672127482;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[0][ 17]=-0.0055390583005;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[0][ 18]=+0.1789326244313;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[0][ 19]=-0.3724399337396;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[0][ 20]=-0.0252585739917;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[0][ 21]=+0.0194668614993;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[0][ 22]=-0.1413931904189;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[0][ 23]=+0.0543911807269;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[0][ 24]=+0.0387114194954;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[0][ 25]=-0.0091791452214;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[0][ 26]=+0.1023022737432;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[0][ 27]=-0.0914440930991;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[0][ 28]=-0.2598952148528;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[0][ 29]=-0.0017338495634;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[0][ 30]=-0.1960022636252;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[0][ 31]=+0.0199500221960;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[0][ 32]=+0.6483826884651;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[0][ 33]=-0.0031472845003;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[0][ 34]=-0.1036287403594;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[0][ 35]=+0.1083988873413;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[0][ 36]=+0.0628918373025;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[0][ 37]=+0.0259528300646;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[0][ 38]=+0.0711113221586;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[0][ 39]=-0.1857986333869;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[0][ 40]=-0.1191086941399;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[0][ 41]=+0.0135852725816;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[0][ 42]=-0.0539380139627;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[0][ 43]=+0.2003111381972;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[0][ 44]=+0.0753719044683;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[0][ 45]=-0.0141232856087;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[0][ 46]=-0.0597580966745;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[0][ 47]=+0.1024119316537;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[0][ 48]=+0.0227283081570;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[0][ 49]=+0.0203187987208;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[0][ 50]=+0.0161542556642;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[0][ 51]=-0.1494623627897;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[0][ 52]=+0.1600875100747;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[0][ 53]=-0.0089495814934;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[0][ 54]=+0.0765985589430;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[0][ 55]=+0.2142676771598;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[0][ 56]=-0.3016575044021;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[0][ 57]=-0.0122789908201;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[0][ 58]=-0.0329800709743;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[0][ 59]=+0.0415519545940;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[0][ 60]=-0.0675619048460;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[0][ 61]=-0.0014832344239;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[0][ 62]=-0.0269124900764;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[0][ 63]=-0.0629127753959;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[0][ 64]=+0.2900113947690;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[0][ 65]=-0.0316829773001;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[0][ 66]=+0.0819831010981;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[0][ 67]=+0.1203245676772;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[0][ 68]=-0.3490906590536;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[0][ 69]=-0.0029750853334;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[0][ 70]=-0.0271631313392;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[0][ 71]=-0.0248459480105;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[0][ 72]=-0.1440617783446;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[0][ 73]=-0.0163560639594;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[0][ 74]=-0.0430056025532;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[0][ 75]=+0.0111915066159;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[0][ 76]=+0.2122980912761;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[0][ 77]=-0.0128921303830;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[0][ 78]=+0.0341086333433;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[0][ 79]=+0.0127186006703;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[0][ 80]=-0.2274262990977;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[0][ 81]=+0.0073553558905;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[0][ 82]=-0.0352983828993;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[0][ 83]=-0.0556936713723;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[0][ 84]=-0.1472002986236;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[0][ 85]=-0.0185522872234;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[0][ 86]=-0.0369246192939;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[0][ 87]=+0.0466724351611;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[0][ 88]=+0.0287292820358;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[0][ 89]=+0.0232138302144;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[0][ 90]=-0.0150538266652;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[0][ 91]=-0.0530740522203;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[0][ 92]=-0.0547684661104;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[0][ 93]=+0.0145955969836;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[0][ 94]=-0.0469822988048;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[0][ 95]=-0.0418253117536;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[0][ 96]=-0.0604853816549;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[0][ 97]=-0.0097493349094;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[0][ 98]=-0.0199821757889;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[0][ 99]=+0.0405172003935;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[0][100]=-0.1253122953119;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[0][101]=+0.0382840106204;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[0][102]=-0.0530891686611;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[0][103]=-0.0774198498970;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[0][104]=+0.0951736970601;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[0][105]=+0.0170990292827;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[0][106]=-0.0531270754489;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[0][107]=-0.0030428144279;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[0][108]=+0.0789172623642;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[0][109]=+0.0062518356283;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[0][110]=+0.0001307721774;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[0][111]=+0.0054703381824;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[0][112]=-0.1621408164210;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[0][113]=+0.0237985423676;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[0][114]=-0.0696100852836;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[0][115]=-0.0811844670249;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[0][116]=+0.1694379878096;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[0][117]=+0.0146053893666;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[0][118]=-0.0492029327361;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[0][119]=+0.0286531129951;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[0][120]=+0.1919381886437;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[0][121]=+0.0213004669058;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[0][122]=+0.0192903265296;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[0][123]=-0.0357876554600;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[0][124]=-0.0749128760209;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[0][125]=-0.0019863284930;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[0][126]=-0.0414982033265;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[0][127]=-0.0680720555743;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[0][128]=+0.1443799002882;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[0][129]=+0.0080385652836;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[0][130]=-0.0366639121227;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[0][131]=+0.0301537141419;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[0][132]=+0.1910601415298;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[0][133]=+0.0223994380344;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[0][134]=+0.0312107426315;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[0][135]=-0.0601988350969;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[0][136]=+0.0612464258156;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[0][137]=-0.0219905337945;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[0][138]=+0.0394267929271;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[0][139]=-0.0199203678940;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[0][140]=+0.0425334793263;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[0][141]=+0.0004594547435;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[0][142]=-0.0215343865988;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[0][143]=+0.0047001135257;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[0][144]=+0.0345761819190;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[0][145]=+0.0016045738012;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[0][146]=+0.0259196507492;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[0][147]=-0.0559497138608;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[0][148]=+0.1184911994343;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[0][149]=-0.0332467662184;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[0][150]=+0.1168970751078;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[0][151]=+0.0652997309360;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[0][152]=-0.0722589119032;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[0][153]=-0.0032167634927;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[0][154]=-0.0106258409366;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[0][155]=-0.0135972785094;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[0][156]=-0.1993455407203;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[0][157]=-0.0247167406963;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[0][158]=+0.0011517609159;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[0][159]=-0.0287195331401;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[0][160]=-0.0041671620420;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[0][161]=-0.0200065310664;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[0][162]=+0.0597815827217;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[0][163]=+0.1166362963405;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[0][164]=-0.1324983638993;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[0][165]=-0.0004511529575;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[0][166]=-0.0075286661317;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[0][167]=+0.0145789598504;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[0][168]=-0.2636441019313;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[0][169]=-0.0119218047621;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[0][170]=-0.0185687951045;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[0][171]=+0.0027793655117;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[0][172]=-0.2040568633540;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[0][173]=+0.0592592942010;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[0][174]=-0.1739381078276;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[0][175]=+0.0088963507976;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[0][176]=-0.1018732012599;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[0][177]=+0.0025262202016;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[0][178]=-0.0099192883373;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[0][179]=+0.0440732822562;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[0][180]=+0.2004943799918;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[0][181]=+0.0273785157073;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[0][182]=-0.0004905220622;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[0][183]=+0.0063169618472;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[0][184]=+0.1644378194542;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[0][185]=-0.0223034095043;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[0][186]=+0.0818208945945;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[0][187]=-0.0872882757963;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[0][188]=+0.1392082686775;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[0][189]=+1.0000000337867;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[1][  0]=+0.3817515871066;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[1][  1]=-0.0669837909445;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[1][  2]=+0.0364169776486;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[1][  3]=-0.3083997486079;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[1][  4]=+0.0604123812613;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[1][  5]=-0.3727490340053;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[1][  6]=+0.0983358722806;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[1][  7]=-0.5977710484563;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[1][  8]=+0.1789661459706;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[1][  9]=-0.0711395242592;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[1][ 10]=+0.3214025098969;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[1][ 11]=-0.0632674842830;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[1][ 12]=+0.0889952427452;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[1][ 13]=+0.0113893372478;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[1][ 14]=-0.0079279165969;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[1][ 15]=+0.3942317322470;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[1][ 16]=+0.4946221342224;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[1][ 17]=+0.0960794195323;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[1][ 18]=-0.4873394471168;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[1][ 19]=-0.7927112142706;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[1][ 20]=-0.4730670993931;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[1][ 21]=-0.0809281793088;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[1][ 22]=+0.2276858078001;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[1][ 23]=-0.1906756507124;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[1][ 24]=-0.2215936871677;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[1][ 25]=+0.0579220266324;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[1][ 26]=+0.0446477982059;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[1][ 27]=+0.2969421931666;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[1][ 28]=+0.3676279754379;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[1][ 29]=-0.0042685746619;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[1][ 30]=+0.0365970337044;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[1][ 31]=+0.1967523143516;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[1][ 32]=+0.9278881574210;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[1][ 33]=-0.0833712501613;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[1][ 34]=+0.1831541560241;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[1][ 35]=-0.2196365660202;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[1][ 36]=-0.2108603144574;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[1][ 37]=+0.0293734207280;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[1][ 38]=-0.0285134722954;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[1][ 39]=+0.0687163222860;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[1][ 40]=-0.1197655507808;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[1][ 41]=-0.0521849532082;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[1][ 42]=+0.1447725100879;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[1][ 43]=+0.2910609203864;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[1][ 44]=+0.7266107289543;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[1][ 45]=-0.0690468354161;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[1][ 46]=+0.1820002357599;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[1][ 47]=-0.1531746338969;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[1][ 48]=+0.0503367981354;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[1][ 49]=-0.0017485227360;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[1][ 50]=-0.0774545895983;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[1][ 51]=-0.0324708221924;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[1][ 52]=-0.3589859967630;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[1][ 53]=-0.1513967097742;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[1][ 54]=+0.0946213617951;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[1][ 55]=+0.0891290043900;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[1][ 56]=+0.0883493299377;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[1][ 57]=-0.0489081337480;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[1][ 58]=+0.1876359092394;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[1][ 59]=-0.0626553178881;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[1][ 60]=+0.2804969559944;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[1][ 61]=-0.0179664114798;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[1][ 62]=-0.0766950635016;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[1][ 63]=+0.0049359793680;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[1][ 64]=-0.3315174239133;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[1][ 65]=-0.2186125633417;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[1][ 66]=+0.0416960240248;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[1][ 67]=-0.0767357385703;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[1][ 68]=-0.3312936185367;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[1][ 69]=-0.0328687616057;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[1][ 70]=+0.1744168897776;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[1][ 71]=+0.0017084175511;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[1][ 72]=+0.3224291175924;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[1][ 73]=-0.0271251313236;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[1][ 74]=-0.0500619386003;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[1][ 75]=+0.1038118715514;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[1][ 76]=-0.1726917891863;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[1][ 77]=-0.1919332524104;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[1][ 78]=+0.0408133480770;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[1][ 79]=-0.1147516009188;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[1][ 80]=-0.4044538853619;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[1][ 81]=-0.0234669556556;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[1][ 82]=+0.1373506677764;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[1][ 83]=+0.0228778171841;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[1][ 84]=+0.1708241369135;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[1][ 85]=-0.0417284292374;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[1][ 86]=-0.0277909544046;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[1][ 87]=+0.1843488482750;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[1][ 88]=-0.0027647890718;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[1][ 89]=-0.0688250527792;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[1][ 90]=+0.0789560785670;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[1][ 91]=-0.0529745983422;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[1][ 92]=-0.2067035539670;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[1][ 93]=-0.0172943274762;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[1][ 94]=+0.0879356992436;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[1][ 95]=+0.0088509256411;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[1][ 96]=-0.0740878924686;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[1][ 97]=-0.0657031681048;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[1][ 98]=-0.0248514923223;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[1][ 99]=+0.1981164538083;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[1][100]=+0.0997413443678;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[1][101]=+0.0981754549038;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[1][102]=+0.1062469050833;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[1][103]=+0.0242059595196;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[1][104]=+0.0989824038041;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[1][105]=-0.0099748904204;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[1][106]=+0.0443126344690;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[1][107]=-0.0188801626937;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[1][108]=-0.2713953389928;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[1][109]=-0.0895395232168;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[1][110]=-0.0340894045869;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[1][111]=+0.1410306745119;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[1][112]=+0.1069307441272;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[1][113]=+0.2203879922205;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[1][114]=+0.0709008401086;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[1][115]=+0.0335045741031;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[1][116]=+0.3322924597363;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[1][117]=-0.0011086821559;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[1][118]=+0.0203872576542;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[1][119]=-0.0386738598588;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[1][120]=-0.3049943915589;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[1][121]=-0.0941846405467;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[1][122]=-0.0331085270839;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[1][123]=+0.0490462750677;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[1][124]=+0.0445057690996;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[1][125]=+0.2095721434531;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[1][126]=-0.0458418512168;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[1][127]=-0.0560542476923;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[1][128]=+0.3646143578429;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[1][129]=+0.0040248071318;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[1][130]=+0.0180883501544;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[1][131]=-0.0407942477598;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[1][132]=-0.1450020063863;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[1][133]=-0.0634937458359;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[1][134]=-0.0024057101332;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[1][135]=-0.0228603487039;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[1][136]=-0.0221111837919;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[1][137]=+0.0396833634842;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[1][138]=-0.1971624236579;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[1][139]=-0.1861257275798;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[1][140]=+0.1812354621288;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[1][141]=-0.0019149464457;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[1][142]=+0.0266438663545;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[1][143]=-0.0329948653493;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[1][144]=+0.1208493121855;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[1][145]=-0.0031579699695;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[1][146]=+0.0508939965293;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[1][147]=-0.0344716656982;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[1][148]=-0.0258516221368;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[1][149]=-0.1785535824077;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[1][150]=-0.2562707902029;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[1][151]=-0.2065060120616;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[1][152]=-0.0830717290392;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[1][153]=-0.0193045863879;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[1][154]=+0.0308951574348;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[1][155]=-0.0342405774865;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[1][156]=+0.3004316916314;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[1][157]=+0.0437800256700;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[1][158]=+0.0779327947845;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[1][159]=+0.0048557919821;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[1][160]=+0.0430172632657;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[1][161]=-0.1808012848087;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[1][162]=-0.0707285212338;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[1][163]=+0.0404009668964;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[1][164]=-0.1837484751375;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[1][165]=-0.0331673232324;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[1][166]=+0.0253747081040;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[1][167]=-0.0518185691171;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[1][168]=+0.1929949281017;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[1][169]=+0.0150904286095;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[1][170]=+0.0157631111933;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[1][171]=+0.0308030081505;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[1][172]=+0.0745995136046;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[1][173]=+0.1977688893688;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[1][174]=+0.2932815335838;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[1][175]=+0.4370938571837;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[1][176]=+0.0138084382173;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[1][177]=-0.0240015880409;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[1][178]=+0.0197534046638;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[1][179]=-0.0483922882904;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[1][180]=-0.2232961142363;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[1][181]=-0.0424090143833;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[1][182]=-0.0689072417610;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[1][183]=+0.0158018737939;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[1][184]=-0.1724010938297;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[1][185]=-0.0340015783516;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[1][186]=-0.1475656240268;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[1][187]=-0.1032724876213;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[1][188]=-0.0661613627794;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[1][189]=+0.0000000045885;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[2][  0]=+0.0132179726341;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[2][  1]=+0.1647406803271;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[2][  2]=-0.0069181631941;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[2][  3]=-0.0578081766624;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[2][  4]=+0.1534547410419;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[2][  5]=-0.0354662003392;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[2][  6]=-0.0023694963830;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[2][  7]=-0.0605141251203;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[2][  8]=+0.0089217690660;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[2][  9]=-0.0167413711208;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[2][ 10]=+0.0116598877769;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[2][ 11]=-0.0013951748570;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[2][ 12]=-0.1010764428795;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[2][ 13]=-0.0420643054349;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[2][ 14]=-0.0180411085723;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[2][ 15]=+0.0069195023804;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[2][ 16]=+0.0252537616939;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[2][ 17]=-0.1514053322363;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[2][ 18]=+0.1562486913343;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[2][ 19]=+0.0056995258163;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[2][ 20]=-0.1534762870749;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[2][ 21]=-0.0139621856770;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[2][ 22]=+0.0168576785846;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[2][ 23]=+0.0257327965242;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[2][ 24]=+0.1219463837950;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[2][ 25]=-0.0240818934735;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[2][ 26]=+0.0054154801369;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[2][ 27]=-0.0514676470686;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[2][ 28]=-0.1192822403446;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[2][ 29]=+0.5592710940437;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[2][ 30]=-0.1484519423978;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[2][ 31]=-0.1721879892433;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[2][ 32]=+0.0388584760845;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[2][ 33]=-0.0186289556267;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[2][ 34]=+0.0254137967791;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[2][ 35]=+0.0133012108193;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[2][ 36]=+0.2066425943964;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[2][ 37]=+0.0030789244606;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[2][ 38]=+0.0211125831403;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[2][ 39]=-0.0086853727386;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[2][ 40]=+0.2190560068759;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[2][ 41]=+0.0231809019041;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[2][ 42]=-0.0889984426331;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[2][ 43]=+0.0668128344268;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[2][ 44]=-0.1766599156957;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[2][ 45]=-0.0204967769427;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[2][ 46]=+0.0250899893130;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[2][ 47]=-0.0492955808137;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[2][ 48]=-0.0644300599391;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[2][ 49]=+0.0317135668424;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[2][ 50]=-0.0081799309098;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[2][ 51]=+0.0009352490466;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[2][ 52]=+0.1984202869630;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[2][ 53]=-0.3613535085566;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[2][ 54]=-0.1413937937339;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[2][ 55]=+0.1190535880448;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[2][ 56]=+0.0597854322246;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[2][ 57]=-0.0156372933286;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[2][ 58]=+0.0266784098158;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[2][ 59]=-0.0625278318787;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[2][ 60]=-0.2446563401915;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[2][ 61]=+0.0649850096435;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[2][ 62]=-0.0268970845271;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[2][ 63]=-0.0108670612450;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[2][ 64]=-0.0969010884569;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[2][ 65]=-0.3235405926948;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[2][ 66]=-0.0995060718413;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[2][ 67]=+0.0312903672168;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[2][ 68]=-0.0002809499969;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[2][ 69]=-0.0084691622457;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[2][ 70]=+0.0280795813423;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[2][ 71]=-0.0132597858672;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[2][ 72]=-0.1813324645311;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[2][ 73]=+0.0717777542895;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[2][ 74]=-0.0291067078252;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[2][ 75]=+0.0041882129744;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[2][ 76]=-0.2564795907114;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[2][ 77]=-0.0645844603593;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[2][ 78]=+0.0540197083806;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[2][ 79]=-0.0014842305202;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[2][ 80]=-0.1066465564360;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[2][ 81]=-0.0011725689678;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[2][ 82]=+0.0234287568897;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[2][ 83]=+0.0454658406202;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[2][ 84]=-0.0052722963726;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[2][ 85]=+0.0394956848409;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[2][ 86]=-0.0296928067936;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[2][ 87]=+0.0329212076920;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[2][ 88]=-0.1801463417365;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[2][ 89]=+0.1624531564033;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[2][ 90]=+0.1908105199810;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[2][ 91]=+0.0072081041654;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[2][ 92]=-0.1048771031295;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[2][ 93]=+0.0064611367738;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[2][ 94]=+0.0126339692470;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[2][ 95]=+0.0665841220761;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[2][ 96]=+0.1268151380143;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[2][ 97]=-0.0136860034265;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[2][ 98]=-0.0308954641502;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[2][ 99]=+0.0500237789751;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[2][100]=+0.0178120967600;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[2][101]=+0.2155013123975;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[2][102]=+0.2114032232819;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[2][103]=-0.0040438642488;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[2][104]=-0.0245229339808;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[2][105]=+0.0136361964270;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[2][106]=+0.0003760120293;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[2][107]=+0.0395986854573;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[2][108]=+0.1498381642018;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[2][109]=-0.0552239315850;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[2][110]=-0.0244972022449;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[2][111]=+0.0440803959505;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[2][112]=+0.1810812993856;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[2][113]=+0.0965091695356;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[2][114]=+0.1030997379439;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[2][115]=-0.0516212743249;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[2][116]=+0.0697581262695;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[2][117]=+0.0175601923760;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[2][118]=-0.0077065589505;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[2][119]=-0.0120014292725;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[2][120]=+0.0874215442341;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[2][121]=-0.0601589375293;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[2][122]=-0.0047549043898;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[2][123]=+0.0180285119939;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[2][124]=+0.2039918153308;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[2][125]=-0.0734673395421;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[2][126]=-0.0644590635610;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[2][127]=-0.1023998648483;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[2][128]=+0.1299418695355;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[2][129]=+0.0153608415348;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[2][130]=-0.0080176680331;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[2][131]=-0.0493428330233;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[2][132]=+0.0004026361843;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[2][133]=-0.0277178273687;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[2][134]=+0.0223653372557;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[2][135]=-0.0153299914639;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[2][136]=+0.0737619991185;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[2][137]=-0.1333315733913;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[2][138]=-0.1760397044263;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[2][139]=-0.1017800080912;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[2][140]=+0.1238297186334;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[2][141]=+0.0072609808838;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[2][142]=-0.0010282657812;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[2][143]=-0.0435089371476;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[2][144]=-0.0635186314923;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[2][145]=+0.0151448195653;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[2][146]=+0.0388095698598;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[2][147]=-0.0367616523193;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[2][148]=-0.1181659392961;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[2][149]=-0.0167409997902;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[2][150]=-0.1388215028838;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[2][151]=-0.0155974410258;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[2][152]=+0.0333200157097;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[2][153]=-0.0015226829238;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[2][154]=+0.0076803639744;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[2][155]=+0.0010410980813;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[2][156]=-0.0969769119791;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[2][157]=+0.0304463713301;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[2][158]=+0.0257486365407;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[2][159]=-0.0262876628473;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[2][160]=-0.2062011264764;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[2][161]=+0.1127927436061;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[2][162]=+0.0269970451893;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[2][163]=+0.1196543393805;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[2][164]=-0.1221562813029;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[2][165]=-0.0033145222102;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[2][166]=+0.0090287079616;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[2][167]=+0.0393065923516;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[2][168]=-0.0938744274254;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[2][169]=+0.0066284011260;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[2][170]=-0.0131084893482;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[2][171]=+0.0193743478739;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[2][172]=-0.0595390873583;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[2][173]=-0.0488016540669;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[2][174]=+0.1302903421347;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[2][175]=+0.1562557225947;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[2][176]=-0.2095113718074;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[2][177]=+0.0028808656122;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[2][178]=-0.0003273735068;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[2][179]=+0.0225605926799;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[2][180]=+0.1028459591441;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[2][181]=-0.0100808099605;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[2][182]=-0.0241129081506;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[2][183]=+0.0393652085837;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[2][184]=+0.1604616853676;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[2][185]=+0.0069143873102;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[2][186]=-0.0416383001970;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[2][187]=-0.1441519518908;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[2][188]=+0.1920684242827;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[2][189]=-0.0000000066234;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;

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




