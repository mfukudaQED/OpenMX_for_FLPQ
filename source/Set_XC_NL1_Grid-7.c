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

  NLX_coes[0][  0]=-0.1259940682499;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[0][  1]=+0.0279858919854;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[0][  2]=-0.0419384894233;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[0][  3]=-0.0559705402382;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[0][  4]=-0.0697251451937;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[0][  5]=+0.0396250636755;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[0][  6]=-0.0271879058468;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[0][  7]=+0.0976868019602;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[0][  8]=-0.0457315067517;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[0][  9]=+0.0277248227074;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[0][ 10]=-0.1231818259804;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[0][ 11]=-0.0058101604997;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[0][ 12]=+0.0591133369680;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[0][ 13]=-0.0045760695344;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[0][ 14]=+0.0431938374831;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[0][ 15]=+0.0479668241052;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[0][ 16]=+0.0204258618087;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[0][ 17]=+0.0263659448974;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[0][ 18]=+0.0964018721370;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[0][ 19]=-0.3130056228201;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[0][ 20]=-0.0681682630633;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[0][ 21]=+0.0130215266057;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[0][ 22]=-0.1248576159758;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[0][ 23]=+0.0260843596115;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[0][ 24]=+0.0059091386751;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[0][ 25]=-0.0037827662891;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[0][ 26]=+0.0563885511269;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[0][ 27]=-0.0607711461871;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[0][ 28]=-0.2415613748279;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[0][ 29]=-0.0995575695414;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[0][ 30]=-0.0576827951583;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[0][ 31]=+0.1177191265307;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[0][ 32]=+0.5763382972218;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[0][ 33]=+0.0034625371918;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[0][ 34]=-0.0966122986268;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[0][ 35]=+0.0830105800964;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[0][ 36]=+0.0371053669248;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[0][ 37]=+0.0081954114127;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[0][ 38]=+0.0193796830266;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[0][ 39]=-0.1420779999306;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[0][ 40]=-0.1536578920786;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[0][ 41]=+0.0647716406579;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[0][ 42]=-0.0266833299960;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[0][ 43]=+0.1401531844790;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[0][ 44]=+0.0457811786013;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[0][ 45]=+0.0021438539875;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[0][ 46]=-0.0639860104633;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[0][ 47]=+0.0883028301754;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[0][ 48]=+0.0403090193406;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[0][ 49]=+0.0102322752355;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[0][ 50]=-0.0050395624048;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[0][ 51]=-0.1059197354517;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[0][ 52]=+0.1084675901526;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[0][ 53]=+0.0732166214323;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[0][ 54]=+0.0039897216444;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[0][ 55]=+0.0512869253790;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[0][ 56]=-0.2666892316774;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[0][ 57]=+0.0041336631402;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[0][ 58]=-0.0462532584315;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[0][ 59]=+0.0408483849029;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[0][ 60]=-0.0206589467441;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[0][ 61]=+0.0081454588070;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[0][ 62]=-0.0078611872589;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[0][ 63]=-0.0368983949067;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[0][ 64]=+0.2354159380775;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[0][ 65]=-0.0336251854897;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[0][ 66]=+0.0130717812181;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[0][ 67]=-0.0206164650629;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[0][ 68]=-0.2480453142760;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[0][ 69]=+0.0051654430406;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[0][ 70]=-0.0434051879732;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[0][ 71]=-0.0120910982383;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[0][ 72]=-0.0982116027080;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[0][ 73]=+0.0020085115684;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[0][ 74]=+0.0000952418408;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[0][ 75]=+0.0081162114397;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[0][ 76]=+0.1606653050975;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[0][ 77]=-0.0917707551433;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[0][ 78]=+0.0400452661558;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[0][ 79]=-0.0323674477014;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[0][ 80]=-0.1057809464328;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[0][ 81]=+0.0044287292694;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[0][ 82]=-0.0472357656431;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[0][ 83]=-0.0318126210129;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[0][ 84]=-0.1278574401716;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[0][ 85]=-0.0124224771648;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[0][ 86]=+0.0043021997756;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[0][ 87]=+0.0168795569930;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[0][ 88]=-0.0054797617195;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[0][ 89]=-0.0383799774791;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[0][ 90]=+0.0718341765442;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[0][ 91]=+0.0024964656251;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[0][ 92]=+0.0284815407085;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[0][ 93]=+0.0037902427067;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[0][ 94]=-0.0496091187125;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[0][ 95]=-0.0136157474716;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[0][ 96]=-0.0798890992301;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[0][ 97]=-0.0294710410817;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[0][ 98]=-0.0003057732068;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[0][ 99]=-0.0010232836230;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[0][100]=-0.1229151069403;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[0][101]=+0.0606755098848;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[0][102]=+0.0515508100302;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[0][103]=+0.0275070021991;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[0][104]=+0.1026189415206;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[0][105]=+0.0045692103376;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[0][106]=-0.0459738114132;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[0][107]=+0.0189496902992;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[0][108]=+0.0238810200278;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[0][109]=-0.0315529730477;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[0][110]=-0.0046051766685;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[0][111]=-0.0279390514128;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[0][112]=-0.1153546165151;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[0][113]=+0.0937011314054;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[0][114]=-0.0449559098932;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[0][115]=-0.0030632724495;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[0][116]=+0.1000640437723;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[0][117]=+0.0049700649973;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[0][118]=-0.0364033471258;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[0][119]=+0.0348349661682;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[0][120]=+0.1212330454691;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[0][121]=-0.0063409584400;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[0][122]=+0.0040712980882;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[0][123]=-0.0458924067938;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[0][124]=-0.0000502047911;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[0][125]=+0.0160561536484;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[0][126]=-0.1463292860871;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[0][127]=-0.0648915396418;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[0][128]=+0.0370039151603;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[0][129]=+0.0016641582770;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[0][130]=-0.0245651403012;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[0][131]=+0.0179426111457;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[0][132]=+0.1408050332735;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[0][133]=+0.0299827388215;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[0][134]=+0.0223146869472;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[0][135]=-0.0459801246046;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[0][136]=+0.1260366098140;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[0][137]=-0.0831942140649;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[0][138]=-0.1106546550828;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[0][139]=-0.0610216172247;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[0][140]=-0.0314194602408;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[0][141]=-0.0047270569209;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[0][142]=-0.0143902781147;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[0][143]=-0.0181844673933;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[0][144]=+0.0480555029280;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[0][145]=+0.0323905455007;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[0][146]=+0.0230427325784;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[0][147]=-0.0347527300640;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[0][148]=+0.1339123792124;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[0][149]=-0.0495622641563;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[0][150]=+0.1119297069744;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[0][151]=+0.0735539462530;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[0][152]=-0.0482213715804;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[0][153]=-0.0065521497163;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[0][154]=-0.0070600420640;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[0][155]=-0.0329627424369;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[0][156]=-0.1038183262217;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[0][157]=-0.0189946246040;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[0][158]=-0.0133865888489;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[0][159]=-0.0239510946898;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[0][160]=-0.0431376130437;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[0][161]=+0.0910576780370;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[0][162]=+0.2374187591151;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[0][163]=+0.1846533509190;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[0][164]=-0.0449341313004;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[0][165]=+0.0010281187152;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[0][166]=-0.0031902683473;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[0][167]=+0.0067901461236;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[0][168]=-0.1518835507959;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[0][169]=-0.0339429601736;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[0][170]=-0.0358568956740;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[0][171]=-0.0029861720774;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[0][172]=-0.2459245872804;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[0][173]=-0.0361404323156;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[0][174]=-0.2253907953434;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[0][175]=-0.0844181648952;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[0][176]=-0.1415133331466;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[0][177]=+0.0036096817057;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[0][178]=-0.0070175743944;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[0][179]=+0.0426657892223;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[0][180]=+0.0919779492352;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[0][181]=+0.0533891164711;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[0][182]=+0.0171434574869;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[0][183]=+0.0247983688436;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[0][184]=+0.1412473928212;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[0][185]=+0.0034716484661;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[0][186]=+0.0746642692955;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[0][187]=-0.0372475135202;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[0][188]=+0.1233382174315;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[0][189]=+1.0000000408039;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[1][  0]=+0.3302013956454;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[1][  1]=-0.0527479336170;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[1][  2]=+0.0514847545313;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[1][  3]=-0.2530611008182;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[1][  4]=+0.0365881208564;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[1][  5]=-0.3513792728895;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[1][  6]=+0.1308454112941;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[1][  7]=-0.5704646811785;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[1][  8]=+0.1672892649713;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[1][  9]=-0.0359483943121;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[1][ 10]=+0.2755981516711;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[1][ 11]=-0.0455224257745;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[1][ 12]=+0.0411693941742;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[1][ 13]=-0.0109383561502;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[1][ 14]=+0.0485571348202;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[1][ 15]=+0.3750437540786;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[1][ 16]=+0.4196251051729;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[1][ 17]=+0.2471894154143;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[1][ 18]=-0.5951252296330;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[1][ 19]=-0.7520267438783;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[1][ 20]=-0.4894789650759;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[1][ 21]=-0.0584152912414;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[1][ 22]=+0.1874000826987;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[1][ 23]=-0.1580918326961;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[1][ 24]=-0.1894277034312;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[1][ 25]=+0.0714829229713;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[1][ 26]=+0.0897516896785;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[1][ 27]=+0.2686048956266;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[1][ 28]=+0.3381581905187;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[1][ 29]=-0.0235495687635;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[1][ 30]=-0.1163922938065;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[1][ 31]=+0.2384630830177;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[1][ 32]=+0.9918380255566;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[1][ 33]=-0.0707887732340;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[1][ 34]=+0.1518554773866;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[1][ 35]=-0.1878996704545;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[1][ 36]=-0.1600143128244;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[1][ 37]=+0.0736818029942;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[1][ 38]=+0.0053156543199;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[1][ 39]=+0.0260574919656;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[1][ 40]=-0.0879919022408;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[1][ 41]=-0.0951375782685;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[1][ 42]=+0.0617653093177;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[1][ 43]=+0.3442889068301;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[1][ 44]=+0.7699624006612;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[1][ 45]=-0.0630308392998;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[1][ 46]=+0.1622427241253;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[1][ 47]=-0.1333364356610;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[1][ 48]=+0.0495516681626;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[1][ 49]=+0.0517443823050;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[1][ 50]=-0.0596060374955;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[1][ 51]=-0.0799660936991;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[1][ 52]=-0.3017837902672;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[1][ 53]=-0.1624476354587;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[1][ 54]=+0.1028278948554;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[1][ 55]=+0.1624870117689;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[1][ 56]=+0.0797370903571;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[1][ 57]=-0.0449429711617;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[1][ 58]=+0.1788400379373;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[1][ 59]=-0.0573051565443;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[1][ 60]=+0.2193491908927;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[1][ 61]=+0.0241546079496;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[1][ 62]=-0.0784405419128;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[1][ 63]=-0.0327722580687;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[1][ 64]=-0.2753638417340;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[1][ 65]=-0.1806905447119;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[1][ 66]=+0.1186326616446;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[1][ 67]=+0.0039181037980;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[1][ 68]=-0.3988836609530;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[1][ 69]=-0.0260190965304;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[1][ 70]=+0.1725156165498;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[1][ 71]=-0.0059223264108;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[1][ 72]=+0.2396163018738;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[1][ 73]=-0.0083737519558;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[1][ 74]=-0.0708542758002;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[1][ 75]=+0.0876151185673;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[1][ 76]=-0.1254342161177;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[1][ 77]=-0.1208688046664;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[1][ 78]=+0.1420502015679;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[1][ 79]=-0.0471701154585;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[1][ 80]=-0.5020992037096;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[1][ 81]=-0.0096601054439;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[1][ 82]=+0.1364643587222;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[1][ 83]=+0.0048852801774;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[1][ 84]=+0.1187160428803;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[1][ 85]=-0.0455683957981;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[1][ 86]=-0.0609135863051;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[1][ 87]=+0.1919367846394;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[1][ 88]=+0.0319681479767;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[1][ 89]=-0.0106078903155;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[1][ 90]=+0.1494002372197;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[1][ 91]=-0.0270748440653;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[1][ 92]=-0.2929898344940;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[1][ 93]=+0.0049454791342;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[1][ 94]=+0.0828221027885;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[1][ 95]=-0.0154395563295;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[1][ 96]=-0.0590156604957;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[1][ 97]=-0.0785474612249;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[1][ 98]=-0.0563842634365;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[1][ 99]=+0.2226704737186;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[1][100]=+0.1136519618780;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[1][101]=+0.0889099740993;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[1][102]=+0.1035172463079;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[1][103]=-0.0172239153605;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[1][104]=+0.0527082213384;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[1][105]=+0.0181108604815;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[1][106]=+0.0321327798301;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[1][107]=-0.0429525220989;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[1][108]=-0.1851255785283;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[1][109]=-0.0921295055391;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[1][110]=-0.0466011472521;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[1][111]=+0.1708854304690;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[1][112]=+0.0897280058420;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[1][113]=+0.1213140763361;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[1][114]=-0.0028111349666;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[1][115]=-0.0731714143458;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[1][116]=+0.3368444409995;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[1][117]=+0.0264290347763;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[1][118]=+0.0014278937683;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[1][119]=-0.0540197430180;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[1][120]=-0.1826161532418;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[1][121]=-0.0757875208947;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[1][122]=-0.0155466970755;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[1][123]=+0.0713274035888;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[1][124]=-0.0077395418723;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[1][125]=+0.0721072534385;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[1][126]=-0.1238499246813;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[1][127]=-0.1772588769937;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[1][128]=+0.4200163531110;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[1][129]=+0.0239188456025;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[1][130]=-0.0042474031538;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[1][131]=-0.0396061692732;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[1][132]=-0.0538169599800;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[1][133]=-0.0361608660836;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[1][134]=+0.0380791418256;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[1][135]=-0.0199542646272;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[1][136]=-0.0986259512827;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[1][137]=-0.0121756444449;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[1][138]=-0.1740157741679;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[1][139]=-0.2328626353174;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[1][140]=+0.2777629992444;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[1][141]=+0.0077465543505;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[1][142]=+0.0067862463139;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[1][143]=-0.0126717286311;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[1][144]=+0.1048032172434;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[1][145]=-0.0001460307423;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[1][146]=+0.0866047034592;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[1][147]=-0.0573238914125;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[1][148]=-0.0958076021746;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[1][149]=-0.0492590960905;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[1][150]=-0.0922838433916;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[1][151]=-0.1205496175993;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[1][152]=+0.0158453020181;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[1][153]=-0.0149589344380;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[1][154]=+0.0191932840651;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[1][155]=-0.0021763782192;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[1][156]=+0.1458398593260;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[1][157]=+0.0041852473625;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[1][158]=+0.0828551891095;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[1][159]=-0.0337167218071;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[1][160]=+0.0277398445134;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[1][161]=-0.0083700360804;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[1][162]=+0.0521412950721;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[1][163]=+0.1677374619630;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[1][164]=-0.1771911216911;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[1][165]=-0.0276240180501;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[1][166]=+0.0217074696420;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[1][167]=-0.0270820791053;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[1][168]=+0.0054378549360;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[1][169]=-0.0121595693819;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[1][170]=+0.0103258523359;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[1][171]=+0.0126223808356;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[1][172]=+0.1495909277070;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[1][173]=+0.0053232712051;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[1][174]=+0.0331551689311;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[1][175]=+0.3524756922556;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[1][176]=-0.1533169577090;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[1][177]=-0.0218242183012;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[1][178]=+0.0156321799521;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[1][179]=-0.0465796979324;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[1][180]=-0.0465664426291;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[1][181]=-0.0021671111898;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[1][182]=-0.0368237351786;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[1][183]=+0.0271784676210;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[1][184]=-0.1050383043761;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[1][185]=+0.0087194848729;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[1][186]=-0.0393485647414;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[1][187]=-0.1305961324679;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[1][188]=+0.0542844998889;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[1][189]=+0.0000000054192;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[2][  0]=+0.0105822635854;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[2][  1]=+0.0937000466268;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[2][  2]=-0.0078094793642;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[2][  3]=-0.0194251882471;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[2][  4]=+0.0817381223588;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[2][  5]=-0.0128637553324;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[2][  6]=-0.0140484176917;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[2][  7]=-0.0417889654204;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[2][  8]=+0.0042514061415;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[2][  9]=+0.0052473489069;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[2][ 10]=+0.0104508280028;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[2][ 11]=+0.0191585996298;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[2][ 12]=-0.0989549883093;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[2][ 13]=-0.0134863965415;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[2][ 14]=-0.0383477866606;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[2][ 15]=+0.0250950519891;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[2][ 16]=-0.0337028797139;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[2][ 17]=-0.0887807693064;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[2][ 18]=+0.1677289857291;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[2][ 19]=-0.0356081497197;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[2][ 20]=-0.1014105259836;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[2][ 21]=+0.0106453100376;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[2][ 22]=+0.0113015569780;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[2][ 23]=+0.0374298588737;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[2][ 24]=+0.1434680045759;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[2][ 25]=-0.0125204084371;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[2][ 26]=+0.0036845489894;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[2][ 27]=-0.0321009863622;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[2][ 28]=-0.0467446958405;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[2][ 29]=+0.3599703958688;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[2][ 30]=-0.0638423438439;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[2][ 31]=-0.2163949979049;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[2][ 32]=+0.0392195116338;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[2][ 33]=+0.0052717898668;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[2][ 34]=+0.0162921136964;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[2][ 35]=+0.0104230039554;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[2][ 36]=+0.1714163192015;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[2][ 37]=+0.0011486203133;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[2][ 38]=+0.0290217134197;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[2][ 39]=-0.0026716195158;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[2][ 40]=+0.2490900657549;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[2][ 41]=-0.1224548054609;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[2][ 42]=-0.0603236719992;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[2][ 43]=+0.0141455443906;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[2][ 44]=-0.1050515299458;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[2][ 45]=-0.0005603810529;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[2][ 46]=+0.0159927443075;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[2][ 47]=-0.0523457157869;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[2][ 48]=-0.0930458422025;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[2][ 49]=+0.0215653558003;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[2][ 50]=+0.0162460744824;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[2][ 51]=-0.0020126703576;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[2][ 52]=+0.1797107189919;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[2][ 53]=-0.2853549351592;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[2][ 54]=-0.0926419147887;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[2][ 55]=+0.0522690883591;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[2][ 56]=+0.0583721350092;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[2][ 57]=-0.0034777257587;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[2][ 58]=+0.0179893052613;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[2][ 59]=-0.0614897509488;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[2][ 60]=-0.2249173940876;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[2][ 61]=+0.0378658895666;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[2][ 62]=+0.0088211228286;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[2][ 63]=-0.0111873197051;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[2][ 64]=-0.1040189710261;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[2][ 65]=-0.1074531815825;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[2][ 66]=-0.0200036357108;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[2][ 67]=+0.0126838454296;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[2][ 68]=-0.0027670153680;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[2][ 69]=-0.0053114936635;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[2][ 70]=+0.0199296252130;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[2][ 71]=-0.0131233945085;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[2][ 72]=-0.1328467600443;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[2][ 73]=+0.0278638627281;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[2][ 74]=+0.0055078049560;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[2][ 75]=-0.0020190384154;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[2][ 76]=-0.2462691045607;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[2][ 77]=+0.1148458808906;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[2][ 78]=+0.0959880112170;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[2][ 79]=+0.0196971027164;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[2][ 80]=-0.0578503256316;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[2][ 81]=-0.0059042595526;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[2][ 82]=+0.0186157876210;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[2][ 83]=+0.0442381413987;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[2][ 84]=+0.0321846056451;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[2][ 85]=-0.0038715432125;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[2][ 86]=-0.0012062696336;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[2][ 87]=+0.0120780344624;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[2][ 88]=-0.1844403018071;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[2][ 89]=+0.1788836032984;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[2][ 90]=+0.1403508838702;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[2][ 91]=+0.0300975976514;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[2][ 92]=-0.0479823584015;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[2][ 93]=-0.0047781544456;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[2][ 94]=+0.0143842832066;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[2][ 95]=+0.0699378594940;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[2][ 96]=+0.1287997851512;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[2][ 97]=-0.0335811270022;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[2][ 98]=-0.0070719477738;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[2][ 99]=+0.0205904305811;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[2][100]=-0.0131276268887;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[2][101]=+0.0690983095243;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[2][102]=+0.0827919087555;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[2][103]=+0.0129138684889;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[2][104]=-0.0198827644671;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[2][105]=-0.0028801741257;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[2][106]=+0.0089102300081;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[2][107]=+0.0515836151175;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[2][108]=+0.1156216995003;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[2][109]=-0.0408899334393;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[2][110]=-0.0071254345496;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[2][111]=+0.0220314910725;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[2][112]=+0.1497716524759;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[2][113]=-0.0907496074572;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[2][114]=-0.0264328750312;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[2][115]=-0.0104591767974;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[2][116]=+0.0089113334841;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[2][117]=-0.0014957648688;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[2][118]=+0.0040204600936;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[2][119]=+0.0034197446305;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[2][120]=+0.0337336336639;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[2][121]=-0.0225380019087;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[2][122]=-0.0039775455355;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[2][123]=+0.0156314284765;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[2][124]=+0.2138171606455;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[2][125]=-0.1423674008032;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[2][126]=-0.1105771677380;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[2][127]=-0.0094808637145;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[2][128]=+0.0429228315156;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[2][129]=-0.0005802213261;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[2][130]=+0.0017759661726;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[2][131]=-0.0436143539960;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[2][132]=-0.0450129161019;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[2][133]=+0.0079785106392;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[2][134]=-0.0043545289426;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[2][135]=+0.0013509109921;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[2][136]=+0.1397202712648;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[2][137]=-0.0204868147177;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[2][138]=-0.1195789347556;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[2][139]=+0.0137396142109;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[2][140]=+0.0663168005631;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[2][141]=+0.0011127630511;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[2][142]=+0.0033807695641;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[2][143]=-0.0574081661312;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[2][144]=-0.0681304945488;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[2][145]=+0.0310007492978;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[2][146]=-0.0095124222568;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[2][147]=-0.0128691179447;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[2][148]=-0.0397910936147;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[2][149]=+0.1519235161845;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[2][150]=-0.0546429967391;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[2][151]=+0.0213166434484;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[2][152]=+0.0343760335476;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[2][153]=+0.0039059910089;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[2][154]=+0.0067051518041;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[2][155]=-0.0244187961506;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[2][156]=-0.0357960556343;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[2][157]=+0.0317713987778;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[2][158]=-0.0114735938204;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[2][159]=-0.0091419890957;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[2][160]=-0.2084910872959;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[2][161]=+0.1100923279114;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[2][162]=+0.0530594174020;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[2][163]=-0.0102903342456;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[2][164]=-0.0734002521229;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[2][165]=+0.0061345868830;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[2][166]=+0.0055684928526;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[2][167]=+0.0308965848734;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[2][168]=+0.0032037365639;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[2][169]=+0.0040381262925;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[2][170]=-0.0057937929902;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[2][171]=+0.0212728907906;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[2][172]=-0.1858922865777;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[2][173]=-0.2097724422622;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[2][174]=+0.1601647175385;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[2][175]=-0.0236246467907;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[2][176]=-0.1423872289660;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[2][177]=+0.0072056609483;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[2][178]=-0.0025943395950;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[2][179]=+0.0381227799292;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[2][180]=+0.0219157360599;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[2][181]=-0.0232543878464;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[2][182]=-0.0033824770851;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[2][183]=+0.0323565445675;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[2][184]=+0.1963601227141;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[2][185]=+0.0682563276749;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[2][186]=-0.0836750315206;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[2][187]=-0.0207735347499;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[2][188]=+0.1444808694504;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[2][189]=-0.0000000080508;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;

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




