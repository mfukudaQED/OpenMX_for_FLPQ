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

  double XC[2],X[2],C[2];
  double G[3],My_Ex[2],Ex[2],My_Ec[2],Ec[2],TotEx,re,im;
  double ***Re_a_r,***Im_a_r;
  double **Re_g_r, **Im_g_r, **Re_g_q, **Im_g_q;
  double **Re_rho_r, **Im_rho_r, **Re_rho_q, **Im_rho_q;
  double **dg_drho_r, dg_ddrho_r, ***Re_drho_r, ***Im_drho_r;
  double *ReTmpq0,*ImTmpq0,*ReTmpq1,*ImTmpq1,*ReTmpq2,*ImTmpq2;
  double *ReTmpr0,*ImTmpr0,*ReTmpr1,*ImTmpr1,*ReTmpr2,*ImTmpr2;

  double Re_dx,Re_dy,Re_dz,Im_dx,Im_dy,Im_dz;
  double sk1,sk2,sk3,time,tmpr,tmpi;
  double rho,drho,rho0,rho1;
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

    My_Ex[spin] = 0.0; 

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

        My_Ex[spin] += ( Re_rho_q[spin][BN_CB]*Re_g_q[spin][BN_CB]
                       + Im_rho_q[spin][BN_CB]*Im_g_q[spin][BN_CB])*fmu;

      } /* BN_CB */

    } /* mu */

    My_Ex[spin] *= (GridVol/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3);

  } /* spin */

  if (SpinP_switch==0) My_Ex[1] = My_Ex[0];
  MPI_Allreduce(&My_Ex[0], &Ex[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Ex[1], &Ex[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (myid==Host_ID) printf("Ex %18.15f %18.15f\n",Ex[0],Ex[1]);

  /*
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    if (0.01<fabs(Vxc0[BN_AB])) printf("ABC %2d %18.15f\n",BN_AB,Vxc0[BN_AB]);
  }
  */

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

  NLX_coes[0][  0]=-0.2181211103059;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[0][  1]=+0.0599423824822;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[0][  2]=-0.0129485612630;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[0][  3]=-0.0441507478726;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[0][  4]=+0.0729716853660;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[0][  5]=+0.0425012538610;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[0][  6]=-0.0323690806017;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[0][  7]=+0.1191667670336;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[0][  8]=-0.1788603643971;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[0][  9]=+0.0482500065919;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[0][ 10]=-0.2133360116543;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[0][ 11]=+0.0546368095542;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[0][ 12]=+0.0069952055903;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[0][ 13]=+0.0589209020534;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[0][ 14]=+0.0424747232944;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[0][ 15]=-0.1277117144484;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[0][ 16]=-0.1865493532771;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[0][ 17]=+0.0676602371794;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[0][ 18]=-0.0762575734426;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[0][ 19]=-0.2734514901915;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[0][ 20]=+0.6253550531237;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[0][ 21]=+0.0350211847869;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[0][ 22]=-0.1710854372390;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[0][ 23]=+0.1227884978376;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[0][ 24]=+0.0690729855711;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[0][ 25]=+0.0053302792693;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[0][ 26]=+0.0454118116119;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[0][ 27]=-0.1825345061860;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[0][ 28]=-0.2487621706277;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[0][ 29]=-0.1813822439219;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[0][ 30]=+0.3233905907335;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[0][ 31]=+0.1471033225782;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[0][ 32]=-0.0450370974267;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[0][ 33]=+0.0279468637940;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[0][ 34]=-0.1288687075262;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[0][ 35]=+0.1496123894805;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[0][ 36]=+0.0587210027068;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[0][ 37]=-0.0505588593590;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[0][ 38]=+0.0160400570681;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[0][ 39]=-0.1500361121401;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[0][ 40]=+0.0439220991999;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[0][ 41]=-0.0424699364660;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[0][ 42]=+0.0982153628437;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[0][ 43]=+0.0413389090553;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[0][ 44]=-0.1755511935623;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[0][ 45]=+0.0233746943877;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[0][ 46]=-0.0970269486810;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[0][ 47]=+0.1092211457636;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[0][ 48]=-0.0426871355142;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[0][ 49]=-0.0506247444731;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[0][ 50]=+0.0267750027311;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[0][ 51]=-0.1047973010862;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[0][ 52]=+0.2494470614488;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[0][ 53]=+0.0985474937493;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[0][ 54]=-0.0685895424349;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[0][ 55]=-0.1057621717215;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[0][ 56]=-0.1738722514737;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[0][ 57]=+0.0130397644388;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[0][ 58]=-0.0757474936205;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[0][ 59]=+0.0400737266862;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[0][ 60]=-0.1443044260079;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[0][ 61]=-0.0289463064383;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[0][ 62]=+0.0557060327044;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[0][ 63]=-0.0679431018313;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[0][ 64]=+0.2770125380508;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[0][ 65]=+0.1069647149858;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[0][ 66]=-0.0874278275548;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[0][ 67]=-0.1021474616170;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[0][ 68]=-0.1145410104280;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[0][ 69]=-0.0031880442418;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[0][ 70]=-0.0615514270600;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[0][ 71]=-0.0191309309182;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[0][ 72]=-0.1618486617581;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[0][ 73]=-0.0115697818655;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[0][ 74]=+0.0699804564795;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[0][ 75]=-0.0487241432969;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[0][ 76]=+0.1831688018310;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[0][ 77]=+0.0170419352181;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[0][ 78]=-0.0486804416767;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[0][ 79]=-0.0044436894227;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[0][ 80]=-0.0408340875494;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[0][ 81]=-0.0196725023967;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[0][ 82]=-0.0500621817049;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[0][ 83]=-0.0479712785805;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[0][ 84]=-0.0835532646508;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[0][ 85]=-0.0041824772884;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[0][ 86]=+0.0575230948208;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[0][ 87]=-0.0451178681208;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[0][ 88]=+0.0461262227371;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[0][ 89]=-0.0697589510390;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[0][ 90]=-0.0135919863844;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[0][ 91]=+0.0994422493782;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[0][ 92]=+0.0049327751378;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[0][ 93]=-0.0301971705219;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[0][ 94]=-0.0385118438287;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[0][ 95]=-0.0438875192423;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[0][ 96]=+0.0450906944662;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[0][ 97]=-0.0027479857444;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[0][ 98]=+0.0243367295927;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[0][ 99]=-0.0472038418294;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[0][100]=-0.0717260371287;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[0][101]=-0.0843446597066;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[0][102]=-0.0022060233509;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[0][103]=+0.1556119367829;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[0][104]=+0.0037825830340;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[0][105]=-0.0313509849318;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[0][106]=-0.0269465307252;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[0][107]=-0.0169428670841;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[0][108]=+0.1467826493440;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[0][109]=-0.0011016715162;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[0][110]=-0.0147163421445;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[0][111]=-0.0436849901367;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[0][112]=-0.1326773263345;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[0][113]=-0.0249498488022;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[0][114]=-0.0142152536876;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[0][115]=+0.1459203511905;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[0][116]=-0.0365005831322;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[0][117]=-0.0236763305762;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[0][118]=-0.0176957367032;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[0][119]=+0.0166066369357;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[0][120]=+0.1564012181725;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[0][121]=+0.0053831755425;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[0][122]=-0.0439358114813;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[0][123]=-0.0281291970977;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[0][124]=-0.1244015101444;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[0][125]=+0.0564476425673;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[0][126]=-0.0381595240724;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[0][127]=+0.0807046991655;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[0][128]=-0.0861902799058;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[0][129]=-0.0110829517589;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[0][130]=-0.0136766205674;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[0][131]=+0.0404949170736;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[0][132]=+0.0600432585907;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[0][133]=+0.0171924385506;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[0][134]=-0.0530696836980;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[0][135]=-0.0033070762452;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[0][136]=-0.0602191387937;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[0][137]=+0.0920291134074;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[0][138]=-0.0524335546289;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[0][139]=-0.0080802133013;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[0][140]=-0.1035105105015;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[0][141]=+0.0011066650270;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[0][142]=-0.0157712739065;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[0][143]=+0.0450291246711;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[0][144]=-0.0798601385628;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[0][145]=+0.0284678971205;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[0][146]=-0.0415706641123;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[0][147]=+0.0179431527446;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[0][148]=+0.0226872723297;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[0][149]=+0.0447468312085;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[0][150]=-0.0294445818299;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[0][151]=-0.0769416743958;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[0][152]=-0.0539997044721;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[0][153]=+0.0087970178576;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[0][154]=-0.0208916752061;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[0][155]=+0.0318663938941;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[0][156]=-0.1469452882883;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[0][157]=+0.0274907156362;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[0][158]=-0.0194856690820;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[0][159]=+0.0183791184802;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[0][160]=+0.0731430764406;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[0][161]=-0.0513780450116;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[0][162]=+0.0446689969149;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[0][163]=-0.0881944587204;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[0][164]=+0.0563915807167;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[0][165]=+0.0113260245929;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[0][166]=-0.0226385554261;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[0][167]=+0.0147669210128;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[0][168]=-0.0436150829242;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[0][169]=+0.0057880610643;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[0][170]=-0.0028985179433;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[0][171]=-0.0078675173463;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[0][172]=+0.0499466529149;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[0][173]=-0.0854481777799;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[0][174]=+0.1172812488817;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[0][175]=-0.0365658592705;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[0][176]=+0.1333309692843;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[0][177]=+0.0115961412278;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[0][178]=-0.0174083963237;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[0][179]=+0.0115366680762;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[0][180]=+0.0685298178237;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[0][181]=-0.0125586378434;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[0][182]=-0.0015573634799;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[0][183]=-0.0260210830611;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[0][184]=-0.0258904757318;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[0][185]=+0.0427005767480;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[0][186]=-0.0461657764731;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[0][187]=+0.0157737736587;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[0][188]=-0.0636890061555;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[0][189]=+1.0000000010247;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[1][  0]=+0.2819291835356;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[1][  1]=-0.0786851671663;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[1][  2]=+0.0319669341036;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[1][  3]=-0.2136875227828;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[1][  4]=+0.2731200650411;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[1][  5]=-0.2339679932881;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[1][  6]=-0.0530397801953;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[1][  7]=-0.8654412587924;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[1][  8]=-0.0584522237329;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[1][  9]=+0.1585268277551;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[1][ 10]=+0.1943404808730;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[1][ 11]=-0.1441988235915;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[1][ 12]=-0.0182375419566;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[1][ 13]=+0.0100740433289;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[1][ 14]=+0.0102293387278;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[1][ 15]=+0.4115548007044;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[1][ 16]=+0.4938997993649;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[1][ 17]=+0.2123338223798;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[1][ 18]=+0.0339921126940;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[1][ 19]=-0.5221404498043;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[1][ 20]=-0.0195745901488;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[1][ 21]=+0.1468384423131;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[1][ 22]=+0.1047225003919;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[1][ 23]=-0.2445112359456;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[1][ 24]=-0.1412675144811;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[1][ 25]=-0.0098234666887;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[1][ 26]=+0.0620790038935;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[1][ 27]=+0.2671900759337;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[1][ 28]=+0.2145701538555;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[1][ 29]=+0.1253581314444;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[1][ 30]=+0.1006433038944;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[1][ 31]=-0.0080400795603;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[1][ 32]=+0.8977634967241;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[1][ 33]=+0.1322162577840;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[1][ 34]=+0.0647857737556;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[1][ 35]=-0.2385587949286;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[1][ 36]=-0.0537262913374;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[1][ 37]=-0.0308532465301;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[1][ 38]=+0.0532691047757;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[1][ 39]=+0.0645727576269;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[1][ 40]=-0.2367124167211;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[1][ 41]=+0.0426389757132;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[1][ 42]=+0.0549829414688;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[1][ 43]=+0.0369219149045;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[1][ 44]=+0.6219715564928;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[1][ 45]=+0.1193372415466;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[1][ 46]=+0.0628574075341;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[1][ 47]=-0.1494980390814;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[1][ 48]=+0.1124925177424;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[1][ 49]=-0.0343311518164;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[1][ 50]=+0.0529661463951;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[1][ 51]=-0.0288393499825;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[1][ 52]=-0.4231593606555;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[1][ 53]=-0.0403044562353;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[1][ 54]=-0.0267948855645;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[1][ 55]=-0.0376633422147;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[1][ 56]=+0.1402927733037;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[1][ 57]=+0.1042676065714;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[1][ 58]=+0.0707512165390;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[1][ 59]=-0.0492918346597;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[1][ 60]=+0.2122966995830;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[1][ 61]=-0.0199500726987;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[1][ 62]=+0.0696117440853;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[1][ 63]=-0.0219930629613;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[1][ 64]=-0.3456444111465;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[1][ 65]=-0.0964191247754;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[1][ 66]=-0.0802094661614;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[1][ 67]=-0.0857434177228;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[1][ 68]=-0.1549200817278;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[1][ 69]=+0.0841845274918;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[1][ 70]=+0.0701686861974;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[1][ 71]=+0.0134786368873;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[1][ 72]=+0.1943695136543;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[1][ 73]=+0.0013097535321;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[1][ 74]=+0.0904004689354;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[1][ 75]=+0.0294213279541;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[1][ 76]=-0.1492279976975;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[1][ 77]=-0.1169042566145;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[1][ 78]=-0.0811720086335;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[1][ 79]=-0.0702607824923;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[1][ 80]=-0.2229576401745;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[1][ 81]=+0.0599643222241;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[1][ 82]=+0.0564723170053;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[1][ 83]=+0.0266039401412;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[1][ 84]=+0.0846842259598;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[1][ 85]=+0.0169470052331;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[1][ 86]=+0.1006819202169;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[1][ 87]=+0.0764286689771;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[1][ 88]=+0.0340807623355;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[1][ 89]=-0.1072812821843;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[1][ 90]=-0.0382653268346;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[1][ 91]=-0.0066258649151;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[1][ 92]=-0.1348222838153;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[1][ 93]=+0.0354017578923;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[1][ 94]=+0.0346756259365;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[1][ 95]=+0.0038175519289;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[1][ 96]=-0.0539899685533;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[1][ 97]=+0.0186121749035;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[1][ 98]=+0.0925733457782;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[1][ 99]=+0.0915650743802;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[1][100]=+0.1309496294151;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[1][101]=-0.0754721339813;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[1][102]=+0.0265289767359;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[1][103]=+0.0707196314195;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[1][104]=+0.0170229644642;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[1][105]=+0.0151460512585;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[1][106]=+0.0138455959189;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[1][107]=-0.0304323971963;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[1][108]=-0.1598498364884;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[1][109]=+0.0039531351946;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[1][110]=+0.0671059825486;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[1][111]=+0.0693957350047;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[1][112]=+0.1283278779797;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[1][113]=-0.0282683768264;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[1][114]=+0.0880532935552;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[1][115]=+0.1269398915697;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[1][116]=+0.1519537192305;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[1][117]=+0.0025713842344;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[1][118]=+0.0021249415187;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[1][119]=-0.0547159714254;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[1][120]=-0.1922482290256;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[1][121]=-0.0229865958888;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[1][122]=+0.0331361690860;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[1][123]=+0.0218316263354;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[1][124]=+0.0596897600371;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[1][125]=+0.0260371154098;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[1][126]=+0.1204447132060;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[1][127]=+0.1351965470454;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[1][128]=+0.2129083481747;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[1][129]=-0.0018787774737;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[1][130]=+0.0031686621633;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[1][131]=-0.0597407565785;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[1][132]=-0.1319730435917;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[1][133]=-0.0517403506142;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[1][134]=+0.0046595072683;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[1][135]=-0.0284663838139;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[1][136]=-0.0149449730695;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[1][137]=+0.0728928791859;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[1][138]=+0.0987196867470;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[1][139]=+0.0823630870383;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[1][140]=+0.1719440427763;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[1][141]=-0.0014534660677;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[1][142]=+0.0144561937251;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[1][143]=-0.0517077991402;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[1][144]=+0.0175220039192;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[1][145]=-0.0663831670593;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[1][146]=-0.0037030381714;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[1][147]=-0.0562088243753;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[1][148]=-0.0367571175402;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[1][149]=+0.0890546338153;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[1][150]=+0.0063600579061;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[1][151]=-0.0211948807547;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[1][152]=+0.0374037983703;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[1][153]=-0.0023566320488;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[1][154]=+0.0278827980520;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[1][155]=-0.0476302639484;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[1][156]=+0.2048985233730;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[1][157]=-0.0500266048712;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[1][158]=+0.0152086025322;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[1][159]=-0.0439101726140;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[1][160]=+0.0133573732802;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[1][161]=+0.0511563892607;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[1][162]=-0.1421500096734;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[1][163]=-0.1215341945184;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[1][164]=-0.1340963793022;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[1][165]=-0.0101682726449;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[1][166]=+0.0333379666337;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[1][167]=-0.0583930709433;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[1][168]=+0.2692469909863;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[1][169]=+0.0016776690094;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[1][170]=+0.0458394941379;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[1][171]=+0.0039690679258;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[1][172]=+0.0583592974346;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[1][173]=-0.0262273890540;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[1][174]=-0.2483899685577;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[1][175]=-0.0822404396390;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[1][176]=-0.2202639224196;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[1][177]=-0.0209479176212;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[1][178]=+0.0265409822674;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[1][179]=-0.0560152921002;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[1][180]=-0.2078261111474;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[1][181]=+0.0472565920313;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[1][182]=+0.0212066042372;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[1][183]=+0.0419277664891;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[1][184]=-0.1557784768533;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[1][185]=+0.0134798243204;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[1][186]=-0.0185753260643;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[1][187]=+0.3876443616393;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[1][188]=-0.0110405230098;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[1][189]=-0.0000000000201;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[2][  0]=+0.0014018507344;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[2][  1]=-0.0415204604156;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[2][  2]=+0.0030527088995;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[2][  3]=+0.0168360908184;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[2][  4]=-0.0268030818262;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[2][  5]=+0.0218279019054;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[2][  6]=-0.0164308372283;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[2][  7]=-0.1257196365760;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[2][  8]=-0.0160214395071;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[2][  9]=+0.0575109961798;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[2][ 10]=+0.0081256335364;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[2][ 11]=+0.0868615531042;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[2][ 12]=-0.0792361441821;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[2][ 13]=+0.0486532309659;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[2][ 14]=+0.0422593412359;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[2][ 15]=-0.0441537810054;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[2][ 16]=-0.1433900091916;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[2][ 17]=-0.0978094406147;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[2][ 18]=+0.0393381267309;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[2][ 19]=-0.0944931060880;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[2][ 20]=+0.1006154792036;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[2][ 21]=+0.0467476041323;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[2][ 22]=+0.0174806604416;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[2][ 23]=+0.0660648326790;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[2][ 24]=+0.1222228552556;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[2][ 25]=+0.0003226044231;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[2][ 26]=+0.0337161029983;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[2][ 27]=-0.0202674866939;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[2][ 28]=+0.2068683122218;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[2][ 29]=+0.0876430454653;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[2][ 30]=+0.2170537203846;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[2][ 31]=+0.1319921129549;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[2][ 32]=-0.1459433827935;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[2][ 33]=+0.0422695772004;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[2][ 34]=+0.0199998989281;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[2][ 35]=+0.0094010831627;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[2][ 36]=+0.0625461968550;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[2][ 37]=-0.0287189368423;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[2][ 38]=+0.0055813757394;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[2][ 39]=-0.0075043338011;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[2][ 40]=+0.2127737073393;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[2][ 41]=+0.0572455179858;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[2][ 42]=+0.0153635395745;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[2][ 43]=+0.0567567077167;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[2][ 44]=+0.0255638008763;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[2][ 45]=+0.0413042974931;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[2][ 46]=+0.0228268599345;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[2][ 47]=-0.0288227635043;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[2][ 48]=-0.0417566268403;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[2][ 49]=-0.0186795830416;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[2][ 50]=+0.0064892089748;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[2][ 51]=-0.0137059028445;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[2][ 52]=+0.0827437073898;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[2][ 53]=+0.0505774894378;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[2][ 54]=-0.0706218058208;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[2][ 55]=-0.0458804672773;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[2][ 56]=-0.0037210206776;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[2][ 57]=+0.0380902238485;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[2][ 58]=+0.0251959647313;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[2][ 59]=-0.0325455318935;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[2][ 60]=-0.1224863722856;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[2][ 61]=-0.0012294512591;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[2][ 62]=+0.0189097798281;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[2][ 63]=-0.0017707737612;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[2][ 64]=-0.0436323688065;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[2][ 65]=+0.0206737190499;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[2][ 66]=-0.0625018108064;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[2][ 67]=-0.0399592633315;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[2][ 68]=-0.0854256741025;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[2][ 69]=+0.0325531024375;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[2][ 70]=+0.0225781328273;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[2][ 71]=-0.0133021428728;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[2][ 72]=-0.1301114160798;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[2][ 73]=+0.0118134374532;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[2][ 74]=+0.0235978346642;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[2][ 75]=+0.0235405386688;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[2][ 76]=-0.0969565606385;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[2][ 77]=-0.0343099748479;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[2][ 78]=-0.0424166093737;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[2][ 79]=+0.0020329736881;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[2][ 80]=-0.1054262427106;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[2][ 81]=+0.0270340708506;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[2][ 82]=+0.0136803059309;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[2][ 83]=+0.0101018727091;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[2][ 84]=-0.0569941363033;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[2][ 85]=+0.0200354968431;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[2][ 86]=+0.0195563450722;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[2][ 87]=+0.0459970860662;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[2][ 88]=-0.0811840612723;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[2][ 89]=-0.0755409345137;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[2][ 90]=-0.0224933299610;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[2][ 91]=+0.0258543837549;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[2][ 92]=-0.0664051441567;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[2][ 93]=+0.0230046601265;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[2][ 94]=+0.0008113749799;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[2][ 95]=+0.0243675848375;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[2][ 96]=+0.0569501185115;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[2][ 97]=+0.0223190001456;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[2][ 98]=+0.0126840401880;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[2][ 99]=+0.0538729223129;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[2][100]=-0.0305388587560;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[2][101]=-0.0744978800287;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[2][102]=+0.0056921289999;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[2][103]=+0.0258884965966;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[2][104]=+0.0069740166990;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[2][105]=+0.0205178973629;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[2][106]=-0.0115878961120;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[2][107]=+0.0256511415382;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[2][108]=+0.1401624859777;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[2][109]=+0.0169435521359;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[2][110]=+0.0078331159708;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[2][111]=+0.0419745329621;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[2][112]=+0.0170187497289;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[2][113]=-0.0295207766548;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[2][114]=+0.0363579795031;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[2][115]=+0.0111067720101;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[2][116]=+0.0864495183332;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[2][117]=+0.0186835104787;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[2][118]=-0.0187361815142;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[2][119]=+0.0183736714112;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[2][120]=+0.1316737540301;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[2][121]=+0.0054636322078;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[2][122]=+0.0078045724767;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[2][123]=+0.0131325479059;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[2][124]=+0.0350588237138;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[2][125]=+0.0349185966773;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[2][126]=+0.0513163271882;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[2][127]=-0.0117332467183;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[2][128]=+0.1360898260169;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[2][129]=+0.0161126287154;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[2][130]=-0.0177272001108;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[2][131]=+0.0110322118629;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[2][132]=+0.0239113463683;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[2][133]=-0.0062545874742;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[2][134]=+0.0137180914891;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[2][135]=-0.0205758988305;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[2][136]=+0.0189425567788;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[2][137]=+0.0764550629414;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[2][138]=+0.0375557652160;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[2][139]=-0.0344065697193;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[2][140]=+0.1205293091722;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[2][141]=+0.0114851093296;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[2][142]=-0.0096004397147;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[2][143]=+0.0100655108130;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[2][144]=-0.1096768745267;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[2][145]=-0.0116250312035;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[2][146]=+0.0234923951598;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[2][147]=-0.0405219795387;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[2][148]=-0.0106583303226;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[2][149]=+0.0549879990848;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[2][150]=+0.0045058481579;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[2][151]=-0.0385401890507;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[2][152]=+0.0274207966283;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[2][153]=+0.0046825958663;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[2][154]=+0.0000267291799;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[2][155]=+0.0143916570041;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[2][156]=-0.1418333881217;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[2][157]=-0.0097834425558;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[2][158]=+0.0293010177671;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[2][159]=-0.0310184875206;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[2][160]=-0.0174509919354;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[2][161]=-0.0303113832932;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[2][162]=-0.0133207269476;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[2][163]=+0.0007101739427;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[2][164]=-0.1020949068139;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[2][165]=-0.0018687528751;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[2][166]=+0.0037019229056;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[2][167]=+0.0149785522558;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[2][168]=-0.0008351870886;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[2][169]=-0.0083647419361;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[2][170]=+0.0199964011448;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[2][171]=+0.0035498215426;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[2][172]=+0.0184858890205;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[2][173]=-0.0934332174066;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[2][174]=+0.0055089834446;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[2][175]=+0.0728545554288;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[2][176]=-0.1447177944833;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[2][177]=-0.0044914990149;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[2][178]=-0.0001306427000;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[2][179]=+0.0067172795007;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[2][180]=+0.0651307173838;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[2][181]=-0.0063598271101;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[2][182]=+0.0034737467187;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[2][183]=+0.0166031204492;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[2][184]=+0.0290811764733;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[2][185]=+0.0668009179786;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[2][186]=-0.0644117850086;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[2][187]=+0.0033544352982;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[2][188]=+0.1109228811618;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[2][189]=+0.0000000000061;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;

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

  w = 1.0e-14;
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




