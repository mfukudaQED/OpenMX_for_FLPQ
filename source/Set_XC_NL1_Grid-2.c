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

      MPI_Finalize();
      exit(0);


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
   
  NLX_coes[0][  0]=-0.2635246046260;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[0][  1]=+0.0817565394549;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[0][  2]=+0.0175735855988;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[0][  3]=+0.1729346467452;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[0][  4]=+0.0160992379126;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[0][  5]=-0.0897585627095;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[0][  6]=+0.0884906225776;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[0][  7]=-0.3305578760326;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[0][  8]=-0.1229505932008;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[0][  9]=+0.0549987080047;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[0][ 10]=-0.2455718241287;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[0][ 11]=+0.0879869131895;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[0][ 12]=-0.0193009544560;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[0][ 13]=-0.0572263823927;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[0][ 14]=+0.1451130375317;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[0][ 15]=-0.1954538378359;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[0][ 16]=-0.0797170675989;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[0][ 17]=-0.0372171096384;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[0][ 18]=-0.0489763026743;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[0][ 19]=+0.2489958096848;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[0][ 20]=+0.9077238370126;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[0][ 21]=+0.0254997560627;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[0][ 22]=-0.1795023132789;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[0][ 23]=+0.1247192049234;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[0][ 24]=+0.0213843885101;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[0][ 25]=-0.0329432422851;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[0][ 26]=+0.0952023892630;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[0][ 27]=-0.2884084177587;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[0][ 28]=-0.1641552699100;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[0][ 29]=+0.0133874547185;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[0][ 30]=-0.1228105692390;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[0][ 31]=+0.1528212307709;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[0][ 32]=-0.1302292429970;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[0][ 33]=+0.0061270560900;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[0][ 34]=-0.1175757048192;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[0][ 35]=+0.1460717372499;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[0][ 36]=+0.0443422003143;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[0][ 37]=-0.0035468365085;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[0][ 38]=+0.0602557247350;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[0][ 39]=-0.1910716175847;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[0][ 40]=+0.0218075568716;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[0][ 41]=+0.0940687253369;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[0][ 42]=-0.1044908132170;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[0][ 43]=+0.1390641085770;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[0][ 44]=-0.5182240404847;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[0][ 45]=-0.0044999238689;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[0][ 46]=-0.0851557148130;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[0][ 47]=+0.1120385233563;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[0][ 48]=+0.0138807393595;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[0][ 49]=+0.0177917620255;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[0][ 50]=+0.0246978361295;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[0][ 51]=-0.0752648533709;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[0][ 52]=+0.2336114360406;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[0][ 53]=+0.0684937429817;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[0][ 54]=-0.0655520174951;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[0][ 55]=+0.1396928055182;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[0][ 56]=-0.3587301637838;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[0][ 57]=-0.0066523876529;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[0][ 58]=-0.0746775715198;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[0][ 59]=+0.0428116484915;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[0][ 60]=-0.0628193064309;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[0][ 61]=+0.0289153573826;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[0][ 62]=-0.0056228769476;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[0][ 63]=-0.0197609395852;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[0][ 64]=+0.2893188554140;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[0][ 65]=-0.0265023135566;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[0][ 66]=-0.0222455381558;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[0][ 67]=+0.0956312820181;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[0][ 68]=-0.0931862692101;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[0][ 69]=-0.0029591697338;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[0][ 70]=-0.0706465706207;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[0][ 71]=-0.0190217565148;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[0][ 72]=-0.1300416228826;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[0][ 73]=+0.0282323267899;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[0][ 74]=-0.0210249480005;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[0][ 75]=-0.0190943440075;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[0][ 76]=+0.1849894978280;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[0][ 77]=-0.1046548183070;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[0][ 78]=+0.0287320034614;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[0][ 79]=+0.0268267067476;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[0][ 80]=+0.0638278076738;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[0][ 81]=+0.0024830007568;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[0][ 82]=-0.0633250972469;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[0][ 83]=-0.0440862797014;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[0][ 84]=-0.1357099599986;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[0][ 85]=+0.0146350684224;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[0][ 86]=-0.0201150775347;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[0][ 87]=-0.0403743705052;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[0][ 88]=+0.0135382325518;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[0][ 89]=-0.1092193934162;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[0][ 90]=+0.0780422407296;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[0][ 91]=-0.0297962346470;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[0][ 92]=+0.0875489713433;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[0][ 93]=+0.0062191930723;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[0][ 94]=-0.0509554034547;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[0][ 95]=-0.0289175415894;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[0][ 96]=-0.0649137589588;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[0][ 97]=-0.0086278409783;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[0][ 98]=-0.0087188242324;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[0][ 99]=-0.0561536741331;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[0][100]=-0.1219912654170;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[0][101]=-0.0392353546617;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[0][102]=+0.1014353311901;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[0][103]=-0.0560339043719;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[0][104]=+0.0308330430688;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[0][105]=+0.0066067860049;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[0][106]=-0.0367547962806;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[0][107]=+0.0085046952162;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[0][108]=+0.0476449571413;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[0][109]=-0.0319008147435;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[0][110]=+0.0055340483465;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[0][111]=-0.0544327611532;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[0][112]=-0.1588033812816;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[0][113]=+0.0538688338229;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[0][114]=+0.0754504220518;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[0][115]=-0.0541275967935;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[0][116]=-0.0427941471833;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[0][117]=+0.0037098186044;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[0][118]=-0.0250447206143;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[0][119]=+0.0417945438602;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[0][120]=+0.1347959845283;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[0][121]=-0.0419587772959;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[0][122]=+0.0168504979023;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[0][123]=-0.0373487621851;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[0][124]=-0.0934921094995;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[0][125]=+0.1046193859520;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[0][126]=-0.0020249692409;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[0][127]=-0.0341348313248;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[0][128]=-0.0860823450905;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[0][129]=-0.0012723885746;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[0][130]=-0.0184520206341;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[0][131]=+0.0509699135973;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[0][132]=+0.1350509023656;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[0][133]=-0.0293754892819;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[0][134]=+0.0216140206966;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[0][135]=-0.0157722936420;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[0][136]=+0.0217352775424;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[0][137]=+0.0733082527344;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[0][138]=-0.0965352436448;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[0][139]=-0.0020407999127;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[0][140]=-0.0777215277386;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[0][141]=-0.0063001036896;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[0][142]=-0.0164718560236;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[0][143]=+0.0337482010415;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[0][144]=+0.0370397230958;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[0][145]=+0.0019303299682;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[0][146]=+0.0168137413172;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[0][147]=-0.0020867068414;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[0][148]=+0.1031146497960;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[0][149]=-0.0162169481246;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[0][150]=-0.1381839540688;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[0][151]=+0.0426403172249;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[0][152]=-0.0273861918029;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[0][153]=-0.0086419418298;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[0][154]=-0.0162185629739;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[0][155]=+0.0082312823670;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[0][156]=-0.0900983517735;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[0][157]=+0.0292569865272;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[0][158]=-0.0003025639892;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[0][159]=-0.0022364391174;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[0][160]=+0.0812409557837;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[0][161]=-0.0720401110880;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[0][162]=-0.0572917253006;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[0][163]=+0.0931427264519;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[0][164]=+0.0241446103925;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[0][165]=-0.0053599543307;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[0][166]=-0.0147115876354;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[0][167]=+0.0011457207814;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[0][168]=-0.1153942036743;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[0][169]=+0.0222385477825;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[0][170]=-0.0261915184112;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[0][171]=-0.0096401035175;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[0][172]=-0.0311555556641;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[0][173]=-0.0074029244888;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[0][174]=+0.1133831238955;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[0][175]=+0.0970667421295;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[0][176]=+0.0226749517227;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[0][177]=+0.0028909534036;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[0][178]=-0.0121709286647;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[0][179]=+0.0175227034224;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[0][180]=+0.0410584805263;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[0][181]=-0.0069596506543;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[0][182]=-0.0310634266161;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[0][183]=-0.0083409779312;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[0][184]=-0.0607921705000;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[0][185]=+0.0091764762632;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[0][186]=+0.0143185551508;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[0][187]=-0.1205957074898;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[0][188]=-0.0351779198709;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[0][189]=+1.0000000006815;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[1][  0]=+0.4121834836618;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[1][  1]=-0.0956177452986;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[1][  2]=-0.0407875083525;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[1][  3]=-0.2349804760463;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[1][  4]=+0.4848684029436;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[1][  5]=-0.1508766608991;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[1][  6]=-0.0571456658416;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[1][  7]=-0.9357838760316;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[1][  8]=-0.3764194831370;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[1][  9]=+0.1930581188362;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[1][ 10]=+0.2961713104112;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[1][ 11]=-0.1901198876216;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[1][ 12]=+0.1334930831341;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[1][ 13]=+0.0894908287673;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[1][ 14]=-0.0464004018532;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[1][ 15]=+0.4020245780164;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[1][ 16]=+0.4833282611012;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[1][ 17]=+0.2286654786624;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[1][ 18]=+0.0909621155682;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[1][ 19]=-0.5465584914770;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[1][ 20]=+0.6782726924260;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[1][ 21]=+0.1913680140336;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[1][ 22]=+0.2176600013566;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[1][ 23]=-0.2617537262371;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[1][ 24]=-0.1517251857401;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[1][ 25]=+0.0482302529088;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[1][ 26]=+0.0469257099492;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[1][ 27]=+0.2281365396727;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[1][ 28]=+0.0703240208145;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[1][ 29]=-0.0005387674646;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[1][ 30]=+0.2139137858734;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[1][ 31]=-0.2352388638460;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[1][ 32]=+0.6381483543786;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[1][ 33]=+0.1739726226365;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[1][ 34]=+0.1778417049239;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[1][ 35]=-0.2355770515707;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[1][ 36]=-0.1341342496709;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[1][ 37]=-0.0059001707591;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[1][ 38]=+0.0870353519236;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[1][ 39]=+0.1024280030646;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[1][ 40]=-0.2429230112122;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[1][ 41]=-0.1218011679663;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[1][ 42]=+0.2020761469071;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[1][ 43]=-0.1084336074612;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[1][ 44]=+0.3537610058261;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[1][ 45]=+0.1493922217972;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[1][ 46]=+0.1602388828235;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[1][ 47]=-0.1588653903956;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[1][ 48]=+0.0458557590817;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[1][ 49]=-0.0475116432895;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[1][ 50]=+0.1077606641823;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[1][ 51]=+0.0497714726088;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[1][ 52]=-0.3376163522717;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[1][ 53]=-0.1445968878027;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[1][ 54]=+0.1257012020573;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[1][ 55]=-0.0629286556486;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[1][ 56]=+0.0825888650321;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[1][ 57]=+0.1208427198502;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[1][ 58]=+0.1484551310387;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[1][ 59]=-0.0798424567953;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[1][ 60]=+0.1978820835487;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[1][ 61]=-0.0680667257705;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[1][ 62]=+0.1191456762595;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[1][ 63]=+0.0507529833871;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[1][ 64]=-0.2759236844886;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[1][ 65]=-0.1008075050190;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[1][ 66]=+0.0399634205496;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[1][ 67]=-0.0271396207411;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[1][ 68]=-0.0675944011223;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[1][ 69]=+0.0906417601943;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[1][ 70]=+0.1331094995745;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[1][ 71]=-0.0266564650598;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[1][ 72]=+0.2318777015078;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[1][ 73]=-0.0687590117314;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[1][ 74]=+0.1206860133203;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[1][ 75]=+0.0730438755015;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[1][ 76]=-0.1445393550514;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[1][ 77]=-0.0356839839102;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[1][ 78]=-0.0309585378656;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[1][ 79]=+0.0156226444604;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[1][ 80]=-0.0930760415441;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[1][ 81]=+0.0614095457614;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[1][ 82]=+0.1123577615119;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[1][ 83]=-0.0075330099963;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[1][ 84]=+0.1500006853210;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[1][ 85]=-0.0545926463157;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[1][ 86]=+0.1114173552954;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[1][ 87]=+0.0906863468463;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[1][ 88]=-0.0125629246033;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[1][ 89]=+0.0136291034913;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[1][ 90]=-0.0777748051444;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[1][ 91]=+0.0573670388794;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[1][ 92]=-0.0393999658828;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[1][ 93]=+0.0357027069952;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[1][ 94]=+0.0892859957123;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[1][ 95]=-0.0169455360422;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[1][ 96]=+0.0096438841499;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[1][ 97]=-0.0323304150789;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[1][ 98]=+0.0929749671794;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[1][ 99]=+0.0905020671500;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[1][100]=+0.0792744191034;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[1][101]=+0.0302079721439;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[1][102]=-0.0951166086651;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[1][103]=+0.0876276630021;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[1][104]=+0.0416143366498;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[1][105]=+0.0153815822478;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[1][106]=+0.0687394716761;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[1][107]=-0.0424330738117;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[1][108]=-0.1180349397884;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[1][109]=-0.0098164080943;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[1][110]=+0.0690522201313;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[1][111]=+0.0709899834884;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[1][112]=+0.1149378326633;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[1][113]=+0.0172535485425;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[1][114]=-0.0793280499677;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[1][115]=+0.1006641308115;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[1][116]=+0.1097121385018;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[1][117]=+0.0012537950814;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[1][118]=+0.0548402692766;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[1][119]=-0.0697758950383;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[1][120]=-0.1770924691251;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[1][121]=+0.0046632376493;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[1][122]=+0.0442484977714;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[1][123]=+0.0390477246566;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[1][124]=+0.0953814230936;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[1][125]=-0.0067148009252;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[1][126]=-0.0320469196847;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[1][127]=+0.0944616785535;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[1][128]=+0.1373002234264;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[1][129]=-0.0070764627352;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[1][130]=+0.0492113020958;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[1][131]=-0.0865324850874;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[1][132]=-0.1416791045114;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[1][133]=+0.0046388088791;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[1][134]=+0.0234725508034;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[1][135]=+0.0072869557361;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[1][136]=+0.0330322904038;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[1][137]=-0.0190259855164;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[1][138]=+0.0333744807111;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[1][139]=+0.0684816575536;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[1][140]=+0.1058619111375;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[1][141]=-0.0112180912766;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[1][142]=+0.0498475493142;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[1][143]=-0.0846434437995;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[1][144]=-0.0228021420462;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[1][145]=-0.0109520188826;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[1][146]=+0.0114287699398;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[1][147]=-0.0082904901932;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[1][148]=-0.0487044804224;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[1][149]=-0.0050660783223;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[1][150]=+0.0853628860731;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[1][151]=+0.0259568812781;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[1][152]=+0.0082579276534;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[1][153]=-0.0137255056977;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[1][154]=+0.0507611239683;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[1][155]=-0.0636850652496;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[1][156]=+0.1258218638003;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[1][157]=-0.0326248288741;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[1][158]=+0.0102980149346;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[1][159]=+0.0069469189363;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[1][160]=-0.1115100684619;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[1][161]=+0.0292651284275;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[1][162]=+0.0716037078155;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[1][163]=-0.0125962192462;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[1][164]=-0.1322961912273;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[1][165]=-0.0166537084163;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[1][166]=+0.0436905532125;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[1][167]=-0.0351173937905;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[1][168]=+0.1849092291300;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[1][169]=-0.0358910449280;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[1][170]=+0.0116309320595;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[1][171]=+0.0532034333871;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[1][172]=-0.0987823321027;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[1][173]=+0.0578433525004;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[1][174]=-0.0719397641299;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[1][175]=+0.0259501912603;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[1][176]=-0.2110640167595;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[1][177]=-0.0165746108771;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[1][178]=+0.0251605537202;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[1][179]=-0.0263023409413;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[1][180]=-0.0879854105785;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[1][181]=+0.0178908118126;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[1][182]=-0.0251374078432;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[1][183]=+0.0837859595886;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[1][184]=+0.0573750303969;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[1][185]=+0.0578869661185;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[1][186]=-0.3865313006638;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[1][187]=+0.3426860790928;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[1][188]=+0.0670859018261;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[1][189]=-0.0000000000124;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;
  NLX_coes[2][  0]=+0.0587059604952;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=+1;
  NLX_coes[2][  1]=+0.0053775098183;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=+2;
  NLX_coes[2][  2]=-0.0122863655725;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+3;
  NLX_coes[2][  3]=+0.0958316307588;   Eid[  3]= 0; Rid[  3]=1; Zid[  3]=+1;
  NLX_coes[2][  4]=-0.0542526189362;   Eid[  4]= 0; Rid[  4]=1; Zid[  4]=+2;
  NLX_coes[2][  5]=-0.0212388718810;   Eid[  5]= 0; Rid[  5]=1; Zid[  5]=+3;
  NLX_coes[2][  6]=-0.1526883732094;   Eid[  6]= 0; Rid[  6]=2; Zid[  6]=+1;
  NLX_coes[2][  7]=-0.2671237689770;   Eid[  7]= 0; Rid[  7]=2; Zid[  7]=+2;
  NLX_coes[2][  8]=-0.0018643983080;   Eid[  8]= 0; Rid[  8]=2; Zid[  8]=+3;
  NLX_coes[2][  9]=+0.0563139634875;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+0;
  NLX_coes[2][ 10]=+0.0673342658691;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+1;
  NLX_coes[2][ 11]=+0.0811019487503;   Eid[ 11]= 1; Rid[ 11]=0; Zid[ 11]=+2;
  NLX_coes[2][ 12]=-0.0458254644135;   Eid[ 12]= 1; Rid[ 12]=0; Zid[ 12]=+3;
  NLX_coes[2][ 13]=+0.0171645968134;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[2][ 14]=+0.0585630456561;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[2][ 15]=-0.0235682886230;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[2][ 16]=-0.0740141158145;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[2][ 17]=-0.0777147755142;   Eid[ 17]= 1; Rid[ 17]=2; Zid[ 17]=+0;
  NLX_coes[2][ 18]=+0.1555725744467;   Eid[ 18]= 1; Rid[ 18]=2; Zid[ 18]=+1;
  NLX_coes[2][ 19]=+0.2659152105536;   Eid[ 19]= 1; Rid[ 19]=2; Zid[ 19]=+2;
  NLX_coes[2][ 20]=+0.0422620108708;   Eid[ 20]= 1; Rid[ 20]=2; Zid[ 20]=+3;
  NLX_coes[2][ 21]=+0.0477760005677;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+0;
  NLX_coes[2][ 22]=+0.0754889302693;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+1;
  NLX_coes[2][ 23]=+0.0412079251422;   Eid[ 23]= 2; Rid[ 23]=0; Zid[ 23]=+2;
  NLX_coes[2][ 24]=+0.1494492837178;   Eid[ 24]= 2; Rid[ 24]=0; Zid[ 24]=+3;
  NLX_coes[2][ 25]=-0.0045958327482;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[2][ 26]=+0.0194427397323;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[2][ 27]=-0.0387395967361;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[2][ 28]=+0.2705117953858;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[2][ 29]=+0.0498707780652;   Eid[ 29]= 2; Rid[ 29]=2; Zid[ 29]=+0;
  NLX_coes[2][ 30]=+0.0207310218890;   Eid[ 30]= 2; Rid[ 30]=2; Zid[ 30]=+1;
  NLX_coes[2][ 31]=+0.0453497485370;   Eid[ 31]= 2; Rid[ 31]=2; Zid[ 31]=+2;
  NLX_coes[2][ 32]=-0.0111917435575;   Eid[ 32]= 2; Rid[ 32]=2; Zid[ 32]=+3;
  NLX_coes[2][ 33]=+0.0430572584948;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+0;
  NLX_coes[2][ 34]=+0.0825421185166;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+1;
  NLX_coes[2][ 35]=-0.0144592896635;   Eid[ 35]= 3; Rid[ 35]=0; Zid[ 35]=+2;
  NLX_coes[2][ 36]=+0.0048096117360;   Eid[ 36]= 3; Rid[ 36]=0; Zid[ 36]=+3;
  NLX_coes[2][ 37]=-0.0015709049146;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[2][ 38]=+0.0212736652219;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[2][ 39]=-0.0102655170037;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[2][ 40]=+0.1448877614664;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[2][ 41]=+0.1052711868827;   Eid[ 41]= 3; Rid[ 41]=2; Zid[ 41]=+0;
  NLX_coes[2][ 42]=-0.0276177833928;   Eid[ 42]= 3; Rid[ 42]=2; Zid[ 42]=+1;
  NLX_coes[2][ 43]=-0.0278720264150;   Eid[ 43]= 3; Rid[ 43]=2; Zid[ 43]=+2;
  NLX_coes[2][ 44]=-0.1847368624979;   Eid[ 44]= 3; Rid[ 44]=2; Zid[ 44]=+3;
  NLX_coes[2][ 45]=+0.0371971180567;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+0;
  NLX_coes[2][ 46]=+0.0792485677625;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+1;
  NLX_coes[2][ 47]=-0.0307583507410;   Eid[ 47]= 4; Rid[ 47]=0; Zid[ 47]=+2;
  NLX_coes[2][ 48]=-0.0991030264923;   Eid[ 48]= 4; Rid[ 48]=0; Zid[ 48]=+3;
  NLX_coes[2][ 49]=+0.0036397425215;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[2][ 50]=+0.0237847185078;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[2][ 51]=+0.0350459953180;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[2][ 52]=-0.0005863950319;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[2][ 53]=+0.0770068690132;   Eid[ 53]= 4; Rid[ 53]=2; Zid[ 53]=+0;
  NLX_coes[2][ 54]=-0.0090630682773;   Eid[ 54]= 4; Rid[ 54]=2; Zid[ 54]=+1;
  NLX_coes[2][ 55]=+0.0136949501167;   Eid[ 55]= 4; Rid[ 55]=2; Zid[ 55]=+2;
  NLX_coes[2][ 56]=-0.1503991415514;   Eid[ 56]= 4; Rid[ 56]=2; Zid[ 56]=+3;
  NLX_coes[2][ 57]=+0.0320978476825;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+0;
  NLX_coes[2][ 58]=+0.0669145742955;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+1;
  NLX_coes[2][ 59]=-0.0221228092759;   Eid[ 59]= 5; Rid[ 59]=0; Zid[ 59]=+2;
  NLX_coes[2][ 60]=-0.0947519551727;   Eid[ 60]= 5; Rid[ 60]=0; Zid[ 60]=+3;
  NLX_coes[2][ 61]=+0.0025678515811;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[2][ 62]=+0.0157563747497;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[2][ 63]=+0.0590805806349;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[2][ 64]=-0.0545400785032;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[2][ 65]=-0.0094182664485;   Eid[ 65]= 5; Rid[ 65]=2; Zid[ 65]=+0;
  NLX_coes[2][ 66]=+0.0157565304779;   Eid[ 66]= 5; Rid[ 66]=2; Zid[ 66]=+1;
  NLX_coes[2][ 67]=+0.0431873695201;   Eid[ 67]= 5; Rid[ 67]=2; Zid[ 67]=+2;
  NLX_coes[2][ 68]=-0.0431860924702;   Eid[ 68]= 5; Rid[ 68]=2; Zid[ 68]=+3;
  NLX_coes[2][ 69]=+0.0294580959882;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+0;
  NLX_coes[2][ 70]=+0.0516294231257;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+1;
  NLX_coes[2][ 71]=-0.0084672676110;   Eid[ 71]= 6; Rid[ 71]=0; Zid[ 71]=+2;
  NLX_coes[2][ 72]=-0.0435132805609;   Eid[ 72]= 6; Rid[ 72]=0; Zid[ 72]=+3;
  NLX_coes[2][ 73]=-0.0028833309420;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[2][ 74]=+0.0059726145941;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[2][ 75]=+0.0604948515062;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[2][ 76]=-0.0487571724985;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[2][ 77]=-0.0963884273148;   Eid[ 77]= 6; Rid[ 77]=2; Zid[ 77]=+0;
  NLX_coes[2][ 78]=+0.0250045692627;   Eid[ 78]= 6; Rid[ 78]=2; Zid[ 78]=+1;
  NLX_coes[2][ 79]=+0.0290869322930;   Eid[ 79]= 6; Rid[ 79]=2; Zid[ 79]=+2;
  NLX_coes[2][ 80]=+0.0218155965374;   Eid[ 80]= 6; Rid[ 80]=2; Zid[ 80]=+3;
  NLX_coes[2][ 81]=+0.0285322484333;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+0;
  NLX_coes[2][ 82]=+0.0374413741378;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+1;
  NLX_coes[2][ 83]=+0.0009314845471;   Eid[ 83]= 7; Rid[ 83]=0; Zid[ 83]=+2;
  NLX_coes[2][ 84]=+0.0022024563213;   Eid[ 84]= 7; Rid[ 84]=0; Zid[ 84]=+3;
  NLX_coes[2][ 85]=-0.0087054455165;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[2][ 86]=+0.0028811891940;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[2][ 87]=+0.0523785168988;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[2][ 88]=-0.0265576512623;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[2][ 89]=-0.1311036206632;   Eid[ 89]= 7; Rid[ 89]=2; Zid[ 89]=+0;
  NLX_coes[2][ 90]=+0.0197306021856;   Eid[ 90]= 7; Rid[ 90]=2; Zid[ 90]=+1;
  NLX_coes[2][ 91]=-0.0050661701048;   Eid[ 91]= 7; Rid[ 91]=2; Zid[ 91]=+2;
  NLX_coes[2][ 92]=+0.0376052493035;   Eid[ 92]= 7; Rid[ 92]=2; Zid[ 92]=+3;
  NLX_coes[2][ 93]=+0.0273299974411;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+0;
  NLX_coes[2][ 94]=+0.0255284512212;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+1;
  NLX_coes[2][ 95]=+0.0058001212093;   Eid[ 95]= 8; Rid[ 95]=0; Zid[ 95]=+2;
  NLX_coes[2][ 96]=+0.0260136184143;   Eid[ 96]= 8; Rid[ 96]=0; Zid[ 96]=+3;
  NLX_coes[2][ 97]=-0.0120836084835;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[2][ 98]=+0.0078652800532;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[2][ 99]=+0.0429252078685;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[2][100]=-0.0156098707451;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[2][101]=-0.0941719052785;   Eid[101]= 8; Rid[101]=2; Zid[101]=+0;
  NLX_coes[2][102]=+0.0064121803547;   Eid[102]= 8; Rid[102]=2; Zid[102]=+1;
  NLX_coes[2][103]=-0.0321363677521;   Eid[103]= 8; Rid[103]=2; Zid[103]=+2;
  NLX_coes[2][104]=+0.0380005605185;   Eid[104]= 8; Rid[104]=2; Zid[104]=+3;
  NLX_coes[2][105]=+0.0242215695430;   Eid[105]= 9; Rid[105]=0; Zid[105]=+0;
  NLX_coes[2][106]=+0.0160267394705;   Eid[106]= 9; Rid[106]=0; Zid[106]=+1;
  NLX_coes[2][107]=+0.0104914031969;   Eid[107]= 9; Rid[107]=0; Zid[107]=+2;
  NLX_coes[2][108]=+0.0339193877214;   Eid[108]= 9; Rid[108]=0; Zid[108]=+3;
  NLX_coes[2][109]=-0.0117145289123;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[2][110]=+0.0173717166929;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[2][111]=+0.0328626598122;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[2][112]=-0.0250805961307;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[2][113]=-0.0044352499635;   Eid[113]= 9; Rid[113]=2; Zid[113]=+0;
  NLX_coes[2][114]=-0.0092989352279;   Eid[114]= 9; Rid[114]=2; Zid[114]=+1;
  NLX_coes[2][115]=-0.0407016173320;   Eid[115]= 9; Rid[115]=2; Zid[115]=+2;
  NLX_coes[2][116]=+0.0436752592972;   Eid[116]= 9; Rid[116]=2; Zid[116]=+3;
  NLX_coes[2][117]=+0.0186549914299;   Eid[117]=10; Rid[117]=0; Zid[117]=+0;
  NLX_coes[2][118]=+0.0092629133265;   Eid[118]=10; Rid[118]=0; Zid[118]=+1;
  NLX_coes[2][119]=+0.0191854328279;   Eid[119]=10; Rid[119]=0; Zid[119]=+2;
  NLX_coes[2][120]=+0.0351724576012;   Eid[120]=10; Rid[120]=0; Zid[120]=+3;
  NLX_coes[2][121]=-0.0076645613751;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[2][122]=+0.0267972021052;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[2][123]=+0.0209641117259;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[2][124]=-0.0440886033073;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[2][125]=+0.0891094917856;   Eid[125]=10; Rid[125]=2; Zid[125]=+0;
  NLX_coes[2][126]=-0.0220868682668;   Eid[126]=10; Rid[126]=2; Zid[126]=+1;
  NLX_coes[2][127]=-0.0323456022288;   Eid[127]=10; Rid[127]=2; Zid[127]=+2;
  NLX_coes[2][128]=+0.0504243102319;   Eid[128]=10; Rid[128]=2; Zid[128]=+3;
  NLX_coes[2][129]=+0.0111328765122;   Eid[129]=11; Rid[129]=0; Zid[129]=+0;
  NLX_coes[2][130]=+0.0055791316768;   Eid[130]=11; Rid[130]=0; Zid[130]=+1;
  NLX_coes[2][131]=+0.0310381091386;   Eid[131]=11; Rid[131]=0; Zid[131]=+2;
  NLX_coes[2][132]=+0.0286097239885;   Eid[132]=11; Rid[132]=0; Zid[132]=+3;
  NLX_coes[2][133]=-0.0015012905067;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[2][134]=+0.0325387209577;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[2][135]=+0.0088114315778;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[2][136]=-0.0451693853116;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[2][137]=+0.1258913323315;   Eid[137]=11; Rid[137]=2; Zid[137]=+0;
  NLX_coes[2][138]=-0.0249394208735;   Eid[138]=11; Rid[138]=2; Zid[138]=+1;
  NLX_coes[2][139]=-0.0117486810821;   Eid[139]=11; Rid[139]=2; Zid[139]=+2;
  NLX_coes[2][140]=+0.0436309299948;   Eid[140]=11; Rid[140]=2; Zid[140]=+3;
  NLX_coes[2][141]=+0.0029375594449;   Eid[141]=12; Rid[141]=0; Zid[141]=+0;
  NLX_coes[2][142]=+0.0045210030132;   Eid[142]=12; Rid[142]=0; Zid[142]=+1;
  NLX_coes[2][143]=+0.0378396608751;   Eid[143]=12; Rid[143]=0; Zid[143]=+2;
  NLX_coes[2][144]=+0.0018397744164;   Eid[144]=12; Rid[144]=0; Zid[144]=+3;
  NLX_coes[2][145]=+0.0032794111185;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[2][146]=+0.0317282003410;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[2][147]=+0.0006504873752;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[2][148]=-0.0001196552916;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[2][149]=+0.0666055856034;   Eid[149]=12; Rid[149]=2; Zid[149]=+0;
  NLX_coes[2][150]=-0.0096770687539;   Eid[150]=12; Rid[150]=2; Zid[150]=+1;
  NLX_coes[2][151]=+0.0202043390343;   Eid[151]=12; Rid[151]=2; Zid[151]=+2;
  NLX_coes[2][152]=+0.0196695800025;   Eid[152]=12; Rid[152]=2; Zid[152]=+3;
  NLX_coes[2][153]=-0.0039453083959;   Eid[153]=13; Rid[153]=0; Zid[153]=+0;
  NLX_coes[2][154]=+0.0046837009112;   Eid[154]=13; Rid[154]=0; Zid[154]=+1;
  NLX_coes[2][155]=+0.0273897829742;   Eid[155]=13; Rid[155]=0; Zid[155]=+2;
  NLX_coes[2][156]=-0.0523249468122;   Eid[156]=13; Rid[156]=0; Zid[156]=+3;
  NLX_coes[2][157]=+0.0017050912958;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[2][158]=+0.0215674485424;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[2][159]=-0.0011681776688;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[2][160]=+0.0871644173623;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[2][161]=-0.0653665478925;   Eid[161]=13; Rid[161]=2; Zid[161]=+0;
  NLX_coes[2][162]=+0.0241245179249;   Eid[162]=13; Rid[162]=2; Zid[162]=+1;
  NLX_coes[2][163]=+0.0617477820880;   Eid[163]=13; Rid[163]=2; Zid[163]=+2;
  NLX_coes[2][164]=-0.0021268474714;   Eid[164]=13; Rid[164]=2; Zid[164]=+3;
  NLX_coes[2][165]=-0.0070578343878;   Eid[165]=14; Rid[165]=0; Zid[165]=+0;
  NLX_coes[2][166]=+0.0049046115692;   Eid[166]=14; Rid[166]=0; Zid[166]=+1;
  NLX_coes[2][167]=-0.0025631903821;   Eid[167]=14; Rid[167]=0; Zid[167]=+2;
  NLX_coes[2][168]=-0.0907753266537;   Eid[168]=14; Rid[168]=0; Zid[168]=+3;
  NLX_coes[2][169]=-0.0077012725395;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[2][170]=+0.0023890255003;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[2][171]=-0.0007520744883;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[2][172]=+0.1314965576101;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[2][173]=-0.1417883531075;   Eid[173]=14; Rid[173]=2; Zid[173]=+0;
  NLX_coes[2][174]=+0.0417558323454;   Eid[174]=14; Rid[174]=2; Zid[174]=+1;
  NLX_coes[2][175]=+0.0817546349820;   Eid[175]=14; Rid[175]=2; Zid[175]=+2;
  NLX_coes[2][176]=+0.0037643852169;   Eid[176]=14; Rid[176]=2; Zid[176]=+3;
  NLX_coes[2][177]=-0.0058037325875;   Eid[177]=15; Rid[177]=0; Zid[177]=+0;
  NLX_coes[2][178]=+0.0053275109385;   Eid[178]=15; Rid[178]=0; Zid[178]=+1;
  NLX_coes[2][179]=-0.0166647574519;   Eid[179]=15; Rid[179]=0; Zid[179]=+2;
  NLX_coes[2][180]=+0.0648345039248;   Eid[180]=15; Rid[180]=0; Zid[180]=+3;
  NLX_coes[2][181]=-0.0072218838943;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[2][182]=-0.0069918305377;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[2][183]=-0.0022425653689;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[2][184]=-0.0910322119851;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[2][185]=+0.1081306518508;   Eid[185]=15; Rid[185]=2; Zid[185]=+0;
  NLX_coes[2][186]=-0.0912566589131;   Eid[186]=15; Rid[186]=2; Zid[186]=+1;
  NLX_coes[2][187]=-0.0541589281072;   Eid[187]=15; Rid[187]=2; Zid[187]=+2;
  NLX_coes[2][188]=-0.0080694059854;   Eid[188]=15; Rid[188]=2; Zid[188]=+3;
  NLX_coes[2][189]=-0.0000000000157;   Eid[189]= 0; Rid[189]=0; Zid[189]=+0;

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

  w = 1.0e-8;
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

  printf("VVV3 x=%15.12f y=%15.12f r=%15.12f Z=%15.12f e=%15.12f\n",x,y,r,Z,e);

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


  /*
  coes[0][0] = 1.0;
  coes[0][1] = 1.0;
  coes[0][2] = 1.0;
  coes[0][3] = 1.0;

  coes[1][0] =-0.6;
  coes[1][1] = 2.0;
  coes[1][2] =-1.0;
  coes[1][3] = 2.5;

  Nchi = 1;
  gx[0] = 0.0;
  gx[1] = 0.0;
  gx[2] = 0.0;

  for (i=0; i<Nchi; i++){
    chi_func(i,rho,drho,Achi);  
    gx[0] += coes[mu][i]*Achi[0];
    gx[1] += coes[mu][i]*Achi[1];
    gx[2] += coes[mu][i]*Achi[2];
  } 
  */

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




