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
  double id,jd,kd,pe,pe1,pr,pr1,pZ,pZ1;
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

  Nchi = 186;
   
  NLX_coes[0][  0]=-0.0342052594605;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=-2;
  NLX_coes[0][  1]=-0.0188414001123;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=-1;
  NLX_coes[0][  2]=+0.1148034998390;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+1;
  NLX_coes[0][  3]=+0.2225404057194;   Eid[  3]= 0; Rid[  3]=0; Zid[  3]=+2;
  NLX_coes[0][  4]=-0.0387561112195;   Eid[  4]= 0; Rid[  4]=0; Zid[  4]=+3;
  NLX_coes[0][  5]=+0.0012352304781;   Eid[  5]= 1; Rid[  5]=0; Zid[  5]=-2;
  NLX_coes[0][  6]=-0.0104997597479;   Eid[  6]= 1; Rid[  6]=0; Zid[  6]=-1;
  NLX_coes[0][  7]=+0.1047504029185;   Eid[  7]= 1; Rid[  7]=0; Zid[  7]=+0;
  NLX_coes[0][  8]=-0.5110056015242;   Eid[  8]= 1; Rid[  8]=0; Zid[  8]=+1;
  NLX_coes[0][  9]=-0.0195780200858;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+2;
  NLX_coes[0][ 10]=+0.0332012986547;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+3;
  NLX_coes[0][ 11]=+0.1027809318927;   Eid[ 11]= 1; Rid[ 11]=1; Zid[ 11]=-2;
  NLX_coes[0][ 12]=-0.2605775381168;   Eid[ 12]= 1; Rid[ 12]=1; Zid[ 12]=-1;
  NLX_coes[0][ 13]=+0.3542376122190;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[0][ 14]=-0.0448304665563;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[0][ 15]=+1.0027361094387;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[0][ 16]=+0.0826155451769;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[0][ 17]=+0.0507536796137;   Eid[ 17]= 2; Rid[ 17]=0; Zid[ 17]=-2;
  NLX_coes[0][ 18]=-0.0760220316481;   Eid[ 18]= 2; Rid[ 18]=0; Zid[ 18]=-1;
  NLX_coes[0][ 19]=+0.1595787296362;   Eid[ 19]= 2; Rid[ 19]=0; Zid[ 19]=+0;
  NLX_coes[0][ 20]=-0.5370447836559;   Eid[ 20]= 2; Rid[ 20]=0; Zid[ 20]=+1;
  NLX_coes[0][ 21]=+0.1109242598105;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+2;
  NLX_coes[0][ 22]=+0.1023587551888;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+3;
  NLX_coes[0][ 23]=-0.0758786171627;   Eid[ 23]= 2; Rid[ 23]=1; Zid[ 23]=-2;
  NLX_coes[0][ 24]=-0.1051201486731;   Eid[ 24]= 2; Rid[ 24]=1; Zid[ 24]=-1;
  NLX_coes[0][ 25]=+0.1880365440441;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[0][ 26]=-0.3075329788353;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[0][ 27]=-0.4707748082286;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[0][ 28]=-0.7296671973325;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[0][ 29]=+0.0519555881912;   Eid[ 29]= 3; Rid[ 29]=0; Zid[ 29]=-2;
  NLX_coes[0][ 30]=-0.1137603872457;   Eid[ 30]= 3; Rid[ 30]=0; Zid[ 30]=-1;
  NLX_coes[0][ 31]=+0.2260572715850;   Eid[ 31]= 3; Rid[ 31]=0; Zid[ 31]=+0;
  NLX_coes[0][ 32]=-0.3836197737991;   Eid[ 32]= 3; Rid[ 32]=0; Zid[ 32]=+1;
  NLX_coes[0][ 33]=+0.2398662095143;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+2;
  NLX_coes[0][ 34]=+0.1028363725618;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+3;
  NLX_coes[0][ 35]=+0.0077682422864;   Eid[ 35]= 3; Rid[ 35]=1; Zid[ 35]=-2;
  NLX_coes[0][ 36]=+0.0808692408395;   Eid[ 36]= 3; Rid[ 36]=1; Zid[ 36]=-1;
  NLX_coes[0][ 37]=+0.1244798888556;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[0][ 38]=-0.1878742723394;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[0][ 39]=-0.6344145996056;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[0][ 40]=-0.2308017836244;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[0][ 41]=+0.0214453137749;   Eid[ 41]= 4; Rid[ 41]=0; Zid[ 41]=-2;
  NLX_coes[0][ 42]=-0.1390328660321;   Eid[ 42]= 4; Rid[ 42]=0; Zid[ 42]=-1;
  NLX_coes[0][ 43]=+0.2471648120658;   Eid[ 43]= 4; Rid[ 43]=0; Zid[ 43]=+0;
  NLX_coes[0][ 44]=-0.2496762420045;   Eid[ 44]= 4; Rid[ 44]=0; Zid[ 44]=+1;
  NLX_coes[0][ 45]=+0.2300235202073;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+2;
  NLX_coes[0][ 46]=-0.0421084244582;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+3;
  NLX_coes[0][ 47]=+0.0008869864013;   Eid[ 47]= 4; Rid[ 47]=1; Zid[ 47]=-2;
  NLX_coes[0][ 48]=+0.1390798160589;   Eid[ 48]= 4; Rid[ 48]=1; Zid[ 48]=-1;
  NLX_coes[0][ 49]=+0.0728907095577;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[0][ 50]=+0.0148807156397;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[0][ 51]=-0.4199131086425;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[0][ 52]=+0.2563463760288;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[0][ 53]=-0.0075203962890;   Eid[ 53]= 5; Rid[ 53]=0; Zid[ 53]=-2;
  NLX_coes[0][ 54]=-0.1468599504172;   Eid[ 54]= 5; Rid[ 54]=0; Zid[ 54]=-1;
  NLX_coes[0][ 55]=+0.2245654841406;   Eid[ 55]= 5; Rid[ 55]=0; Zid[ 55]=+0;
  NLX_coes[0][ 56]=-0.1724895443442;   Eid[ 56]= 5; Rid[ 56]=0; Zid[ 56]=+1;
  NLX_coes[0][ 57]=+0.1426857145498;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+2;
  NLX_coes[0][ 58]=-0.1841836466570;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+3;
  NLX_coes[0][ 59]=-0.0536606845173;   Eid[ 59]= 5; Rid[ 59]=1; Zid[ 59]=-2;
  NLX_coes[0][ 60]=+0.0787141564317;   Eid[ 60]= 5; Rid[ 60]=1; Zid[ 60]=-1;
  NLX_coes[0][ 61]=-0.0164734260505;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[0][ 62]=+0.1504874004946;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[0][ 63]=-0.2097373933322;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[0][ 64]=+0.3869088402675;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[0][ 65]=-0.0146475093091;   Eid[ 65]= 6; Rid[ 65]=0; Zid[ 65]=-2;
  NLX_coes[0][ 66]=-0.1290984440177;   Eid[ 66]= 6; Rid[ 66]=0; Zid[ 66]=-1;
  NLX_coes[0][ 67]=+0.1794613563122;   Eid[ 67]= 6; Rid[ 67]=0; Zid[ 67]=+0;
  NLX_coes[0][ 68]=-0.1378512008442;   Eid[ 68]= 6; Rid[ 68]=0; Zid[ 68]=+1;
  NLX_coes[0][ 69]=+0.0558974619417;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+2;
  NLX_coes[0][ 70]=-0.2048655457995;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+3;
  NLX_coes[0][ 71]=-0.0594868708192;   Eid[ 71]= 6; Rid[ 71]=1; Zid[ 71]=-2;
  NLX_coes[0][ 72]=-0.0210646886017;   Eid[ 72]= 6; Rid[ 72]=1; Zid[ 72]=-1;
  NLX_coes[0][ 73]=-0.1253828930737;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[0][ 74]=+0.1970263309341;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[0][ 75]=-0.0807488567264;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[0][ 76]=+0.2859049861968;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[0][ 77]=-0.0001705818280;   Eid[ 77]= 7; Rid[ 77]=0; Zid[ 77]=-2;
  NLX_coes[0][ 78]=-0.0863005139628;   Eid[ 78]= 7; Rid[ 78]=0; Zid[ 78]=-1;
  NLX_coes[0][ 79]=+0.1312760622354;   Eid[ 79]= 7; Rid[ 79]=0; Zid[ 79]=+0;
  NLX_coes[0][ 80]=-0.1263045879358;   Eid[ 80]= 7; Rid[ 80]=0; Zid[ 80]=+1;
  NLX_coes[0][ 81]=+0.0039024328796;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+2;
  NLX_coes[0][ 82]=-0.1101887717856;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+3;
  NLX_coes[0][ 83]=+0.0112997615955;   Eid[ 83]= 7; Rid[ 83]=1; Zid[ 83]=-2;
  NLX_coes[0][ 84]=-0.0840817971903;   Eid[ 84]= 7; Rid[ 84]=1; Zid[ 84]=-1;
  NLX_coes[0][ 85]=-0.2089165203128;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[0][ 86]=+0.1811766742532;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[0][ 87]=-0.0213755426779;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[0][ 88]=+0.1309419721839;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[0][ 89]=+0.0211358292454;   Eid[ 89]= 8; Rid[ 89]=0; Zid[ 89]=-2;
  NLX_coes[0][ 90]=-0.0301614159796;   Eid[ 90]= 8; Rid[ 90]=0; Zid[ 90]=-1;
  NLX_coes[0][ 91]=+0.0908936617388;   Eid[ 91]= 8; Rid[ 91]=0; Zid[ 91]=+0;
  NLX_coes[0][ 92]=-0.1231903833432;   Eid[ 92]= 8; Rid[ 92]=0; Zid[ 92]=+1;
  NLX_coes[0][ 93]=-0.0121544875376;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+2;
  NLX_coes[0][ 94]=+0.0204368935925;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+3;
  NLX_coes[0][ 95]=+0.1058970192793;   Eid[ 95]= 8; Rid[ 95]=1; Zid[ 95]=-2;
  NLX_coes[0][ 96]=-0.0737018654708;   Eid[ 96]= 8; Rid[ 96]=1; Zid[ 96]=-1;
  NLX_coes[0][ 97]=-0.2270865843190;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[0][ 98]=+0.1408862465769;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[0][ 99]=-0.0096298095558;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[0][100]=+0.0043355758905;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[0][101]=+0.0314543245722;   Eid[101]= 9; Rid[101]=0; Zid[101]=-2;
  NLX_coes[0][102]=+0.0204821708108;   Eid[102]= 9; Rid[102]=0; Zid[102]=-1;
  NLX_coes[0][103]=+0.0601449344249;   Eid[103]= 9; Rid[103]=0; Zid[103]=+0;
  NLX_coes[0][104]=-0.1194097236345;   Eid[104]= 9; Rid[104]=0; Zid[104]=+1;
  NLX_coes[0][105]=-0.0016987110735;   Eid[105]= 9; Rid[105]=0; Zid[105]=+2;
  NLX_coes[0][106]=+0.1093337864173;   Eid[106]= 9; Rid[106]=0; Zid[106]=+3;
  NLX_coes[0][107]=+0.1353755343958;   Eid[107]= 9; Rid[107]=1; Zid[107]=-2;
  NLX_coes[0][108]=-0.0019305240673;   Eid[108]= 9; Rid[108]=1; Zid[108]=-1;
  NLX_coes[0][109]=-0.1654042385572;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[0][110]=+0.1082118185104;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[0][111]=-0.0228138965157;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[0][112]=-0.0814369352569;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[0][113]=+0.0210745651550;   Eid[113]=10; Rid[113]=0; Zid[113]=-2;
  NLX_coes[0][114]=+0.0476221495261;   Eid[114]=10; Rid[114]=0; Zid[114]=-1;
  NLX_coes[0][115]=+0.0349977874796;   Eid[115]=10; Rid[115]=0; Zid[115]=+0;
  NLX_coes[0][116]=-0.1116345759836;   Eid[116]=10; Rid[116]=0; Zid[116]=+1;
  NLX_coes[0][117]=+0.0252094617843;   Eid[117]=10; Rid[117]=0; Zid[117]=+2;
  NLX_coes[0][118]=+0.1210748031850;   Eid[118]=10; Rid[118]=0; Zid[118]=+3;
  NLX_coes[0][119]=+0.0437105380631;   Eid[119]=10; Rid[119]=1; Zid[119]=-2;
  NLX_coes[0][120]=+0.0817374584887;   Eid[120]=10; Rid[120]=1; Zid[120]=-1;
  NLX_coes[0][121]=-0.0459332942991;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[0][122]=+0.0960325945649;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[0][123]=-0.0380853392066;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[0][124]=-0.1263084620478;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[0][125]=-0.0052640193952;   Eid[125]=11; Rid[125]=0; Zid[125]=-2;
  NLX_coes[0][126]=+0.0433590315340;   Eid[126]=11; Rid[126]=0; Zid[126]=-1;
  NLX_coes[0][127]=+0.0116362571494;   Eid[127]=11; Rid[127]=0; Zid[127]=+0;
  NLX_coes[0][128]=-0.1014385717129;   Eid[128]=11; Rid[128]=0; Zid[128]=+1;
  NLX_coes[0][129]=+0.0569145916335;   Eid[129]=11; Rid[129]=0; Zid[129]=+2;
  NLX_coes[0][130]=+0.0647206973314;   Eid[130]=11; Rid[130]=0; Zid[130]=+3;
  NLX_coes[0][131]=-0.1281452455952;   Eid[131]=11; Rid[131]=1; Zid[131]=-2;
  NLX_coes[0][132]=+0.1164231422088;   Eid[132]=11; Rid[132]=1; Zid[132]=-1;
  NLX_coes[0][133]=+0.0739934245492;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[0][134]=+0.0888684159622;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[0][135]=-0.0411955424937;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[0][136]=-0.1170504964517;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[0][137]=-0.0297966832570;   Eid[137]=12; Rid[137]=0; Zid[137]=-2;
  NLX_coes[0][138]=+0.0167688524230;   Eid[138]=12; Rid[138]=0; Zid[138]=-1;
  NLX_coes[0][139]=-0.0065176546267;   Eid[139]=12; Rid[139]=0; Zid[139]=+0;
  NLX_coes[0][140]=-0.0916580046423;   Eid[140]=12; Rid[140]=0; Zid[140]=+1;
  NLX_coes[0][141]=+0.0779831697885;   Eid[141]=12; Rid[141]=0; Zid[141]=+2;
  NLX_coes[0][142]=-0.0223746262553;   Eid[142]=12; Rid[142]=0; Zid[142]=+3;
  NLX_coes[0][143]=-0.2227781422807;   Eid[143]=12; Rid[143]=1; Zid[143]=-2;
  NLX_coes[0][144]=+0.0650513516857;   Eid[144]=12; Rid[144]=1; Zid[144]=-1;
  NLX_coes[0][145]=+0.1210272194937;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[0][146]=+0.0467615603697;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[0][147]=-0.0375175891354;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[0][148]=-0.0400344422710;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[0][149]=-0.0322248013755;   Eid[149]=13; Rid[149]=0; Zid[149]=-2;
  NLX_coes[0][150]=-0.0071558183254;   Eid[150]=13; Rid[150]=0; Zid[150]=-1;
  NLX_coes[0][151]=-0.0059157327150;   Eid[151]=13; Rid[151]=0; Zid[151]=+0;
  NLX_coes[0][152]=-0.0813780540382;   Eid[152]=13; Rid[152]=0; Zid[152]=+1;
  NLX_coes[0][153]=+0.0730297945709;   Eid[153]=13; Rid[153]=0; Zid[153]=+2;
  NLX_coes[0][154]=-0.0920548496227;   Eid[154]=13; Rid[154]=0; Zid[154]=+3;
  NLX_coes[0][155]=-0.0523946141631;   Eid[155]=13; Rid[155]=1; Zid[155]=-2;
  NLX_coes[0][156]=-0.0502826320070;   Eid[156]=13; Rid[156]=1; Zid[156]=-1;
  NLX_coes[0][157]=+0.0517116188209;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[0][158]=-0.0626058607265;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[0][159]=-0.0489268224086;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[0][160]=+0.0806848281484;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[0][161]=-0.0076919893249;   Eid[161]=14; Rid[161]=0; Zid[161]=-2;
  NLX_coes[0][162]=-0.0047461952846;   Eid[162]=14; Rid[162]=0; Zid[162]=-1;
  NLX_coes[0][163]=+0.0261325783682;   Eid[163]=14; Rid[163]=0; Zid[163]=+0;
  NLX_coes[0][164]=-0.0659658963650;   Eid[164]=14; Rid[164]=0; Zid[164]=+1;
  NLX_coes[0][165]=+0.0422656675406;   Eid[165]=14; Rid[165]=0; Zid[165]=+2;
  NLX_coes[0][166]=-0.0911276331346;   Eid[166]=14; Rid[166]=0; Zid[166]=+3;
  NLX_coes[0][167]=+0.3120468063594;   Eid[167]=14; Rid[167]=1; Zid[167]=-2;
  NLX_coes[0][168]=-0.1267173313144;   Eid[168]=14; Rid[168]=1; Zid[168]=-1;
  NLX_coes[0][169]=-0.0612323056332;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[0][170]=-0.1799725455907;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[0][171]=-0.0706609055313;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[0][172]=+0.1419666222465;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[0][173]=+0.0138020764065;   Eid[173]=15; Rid[173]=0; Zid[173]=-2;
  NLX_coes[0][174]=+0.0003452387236;   Eid[174]=15; Rid[174]=0; Zid[174]=-1;
  NLX_coes[0][175]=+0.0592676415217;   Eid[175]=15; Rid[175]=0; Zid[175]=+0;
  NLX_coes[0][176]=-0.0569990831112;   Eid[176]=15; Rid[176]=0; Zid[176]=+1;
  NLX_coes[0][177]=+0.0235543962494;   Eid[177]=15; Rid[177]=0; Zid[177]=+2;
  NLX_coes[0][178]=+0.0567115165684;   Eid[178]=15; Rid[178]=0; Zid[178]=+3;
  NLX_coes[0][179]=-0.1045631335096;   Eid[179]=15; Rid[179]=1; Zid[179]=-2;
  NLX_coes[0][180]=+0.0046740158160;   Eid[180]=15; Rid[180]=1; Zid[180]=-1;
  NLX_coes[0][181]=+0.1196483913222;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[0][182]=+0.0190414428256;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[0][183]=+0.0520705662009;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[0][184]=-0.0574037463756;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[0][185]=+0.9999999995048;   Eid[185]= 0; Rid[185]=0; Zid[185]=+0;
  NLX_coes[1][  0]=-0.0708754044490;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=-2;
  NLX_coes[1][  1]=+0.6296548270705;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=-1;
  NLX_coes[1][  2]=+0.1284776758034;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+1;
  NLX_coes[1][  3]=-0.5589870935956;   Eid[  3]= 0; Rid[  3]=0; Zid[  3]=+2;
  NLX_coes[1][  4]=+0.0428323460392;   Eid[  4]= 0; Rid[  4]=0; Zid[  4]=+3;
  NLX_coes[1][  5]=-0.2204678577107;   Eid[  5]= 1; Rid[  5]=0; Zid[  5]=-2;
  NLX_coes[1][  6]=+0.6299307426295;   Eid[  6]= 1; Rid[  6]=0; Zid[  6]=-1;
  NLX_coes[1][  7]=-0.4405578011215;   Eid[  7]= 1; Rid[  7]=0; Zid[  7]=+0;
  NLX_coes[1][  8]=+0.7733461587598;   Eid[  8]= 1; Rid[  8]=0; Zid[  8]=+1;
  NLX_coes[1][  9]=+0.0027640431616;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+2;
  NLX_coes[1][ 10]=-0.0036823584048;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+3;
  NLX_coes[1][ 11]=+0.2159191733688;   Eid[ 11]= 1; Rid[ 11]=1; Zid[ 11]=-2;
  NLX_coes[1][ 12]=+0.4270109280186;   Eid[ 12]= 1; Rid[ 12]=1; Zid[ 12]=-1;
  NLX_coes[1][ 13]=+0.5379479465402;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[1][ 14]=-1.4242517080446;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[1][ 15]=-0.2990312119789;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[1][ 16]=+0.0446080533265;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[1][ 17]=-0.2661387462784;   Eid[ 17]= 2; Rid[ 17]=0; Zid[ 17]=-2;
  NLX_coes[1][ 18]=+0.4693351418535;   Eid[ 18]= 2; Rid[ 18]=0; Zid[ 18]=-1;
  NLX_coes[1][ 19]=-0.2787113976374;   Eid[ 19]= 2; Rid[ 19]=0; Zid[ 19]=+0;
  NLX_coes[1][ 20]=+0.6293146631549;   Eid[ 20]= 2; Rid[ 20]=0; Zid[ 20]=+1;
  NLX_coes[1][ 21]=-0.1881518972808;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+2;
  NLX_coes[1][ 22]=-0.2010601742989;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+3;
  NLX_coes[1][ 23]=-0.1201474055424;   Eid[ 23]= 2; Rid[ 23]=1; Zid[ 23]=-2;
  NLX_coes[1][ 24]=-0.0418992871765;   Eid[ 24]= 2; Rid[ 24]=1; Zid[ 24]=-1;
  NLX_coes[1][ 25]=+0.6473520284534;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[1][ 26]=-0.1519921037714;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[1][ 27]=+1.0249463932524;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[1][ 28]=+0.9455258450198;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[1][ 29]=-0.2156023543751;   Eid[ 29]= 3; Rid[ 29]=0; Zid[ 29]=-2;
  NLX_coes[1][ 30]=+0.3310312097146;   Eid[ 30]= 3; Rid[ 30]=0; Zid[ 30]=-1;
  NLX_coes[1][ 31]=-0.2782982684209;   Eid[ 31]= 3; Rid[ 31]=0; Zid[ 31]=+0;
  NLX_coes[1][ 32]=+0.3794738870733;   Eid[ 32]= 3; Rid[ 32]=0; Zid[ 32]=+1;
  NLX_coes[1][ 33]=-0.3587308888890;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+2;
  NLX_coes[1][ 34]=-0.1277305854097;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+3;
  NLX_coes[1][ 35]=-0.1479421023790;   Eid[ 35]= 3; Rid[ 35]=1; Zid[ 35]=-2;
  NLX_coes[1][ 36]=-0.3246658558995;   Eid[ 36]= 3; Rid[ 36]=1; Zid[ 36]=-1;
  NLX_coes[1][ 37]=+0.4448777997258;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[1][ 38]=+0.0539777106641;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[1][ 39]=+0.8572866653428;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[1][ 40]=+0.3470942853381;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[1][ 41]=-0.1220186380977;   Eid[ 41]= 4; Rid[ 41]=0; Zid[ 41]=-2;
  NLX_coes[1][ 42]=+0.2441287201901;   Eid[ 42]= 4; Rid[ 42]=0; Zid[ 42]=-1;
  NLX_coes[1][ 43]=-0.2822788481084;   Eid[ 43]= 4; Rid[ 43]=0; Zid[ 43]=+0;
  NLX_coes[1][ 44]=+0.2242851841852;   Eid[ 44]= 4; Rid[ 44]=0; Zid[ 44]=+1;
  NLX_coes[1][ 45]=-0.3583899524381;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+2;
  NLX_coes[1][ 46]=+0.0681419849674;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+3;
  NLX_coes[1][ 47]=-0.0101415048924;   Eid[ 47]= 4; Rid[ 47]=1; Zid[ 47]=-2;
  NLX_coes[1][ 48]=-0.3922221141741;   Eid[ 48]= 4; Rid[ 48]=1; Zid[ 48]=-1;
  NLX_coes[1][ 49]=+0.2672918862211;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[1][ 50]=-0.0361941911524;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[1][ 51]=+0.3850834504136;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[1][ 52]=-0.3093689812441;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[1][ 53]=-0.0316017819406;   Eid[ 53]= 5; Rid[ 53]=0; Zid[ 53]=-2;
  NLX_coes[1][ 54]=+0.1876949595926;   Eid[ 54]= 5; Rid[ 54]=0; Zid[ 54]=-1;
  NLX_coes[1][ 55]=-0.2610232902095;   Eid[ 55]= 5; Rid[ 55]=0; Zid[ 55]=+0;
  NLX_coes[1][ 56]=+0.1748387495767;   Eid[ 56]= 5; Rid[ 56]=0; Zid[ 56]=+1;
  NLX_coes[1][ 57]=-0.2359798548041;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+2;
  NLX_coes[1][ 58]=+0.2463192666639;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+3;
  NLX_coes[1][ 59]=+0.1510864550866;   Eid[ 59]= 5; Rid[ 59]=1; Zid[ 59]=-2;
  NLX_coes[1][ 60]=-0.3151019393192;   Eid[ 60]= 5; Rid[ 60]=1; Zid[ 60]=-1;
  NLX_coes[1][ 61]=+0.2022972129601;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[1][ 62]=-0.1242291252455;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[1][ 63]=+0.0458790508577;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[1][ 64]=-0.5974927211213;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[1][ 65]=+0.0314295558950;   Eid[ 65]= 6; Rid[ 65]=0; Zid[ 65]=-2;
  NLX_coes[1][ 66]=+0.1360043775200;   Eid[ 66]= 6; Rid[ 66]=0; Zid[ 66]=-1;
  NLX_coes[1][ 67]=-0.2254316106138;   Eid[ 67]= 6; Rid[ 67]=0; Zid[ 67]=+0;
  NLX_coes[1][ 68]=+0.1839343678406;   Eid[ 68]= 6; Rid[ 68]=0; Zid[ 68]=+1;
  NLX_coes[1][ 69]=-0.0917165866357;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+2;
  NLX_coes[1][ 70]=+0.3026993000243;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+3;
  NLX_coes[1][ 71]=+0.2414719058329;   Eid[ 71]= 6; Rid[ 71]=1; Zid[ 71]=-2;
  NLX_coes[1][ 72]=-0.1878656614066;   Eid[ 72]= 6; Rid[ 72]=1; Zid[ 72]=-1;
  NLX_coes[1][ 73]=+0.2203347398579;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[1][ 74]=-0.1514815521792;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[1][ 75]=-0.0885946488375;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[1][ 76]=-0.5071525400833;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[1][ 77]=+0.0642638059058;   Eid[ 77]= 7; Rid[ 77]=0; Zid[ 77]=-2;
  NLX_coes[1][ 78]=+0.0753591777463;   Eid[ 78]= 7; Rid[ 78]=0; Zid[ 78]=-1;
  NLX_coes[1][ 79]=-0.1882786602456;   Eid[ 79]= 7; Rid[ 79]=0; Zid[ 79]=+0;
  NLX_coes[1][ 80]=+0.2077900344904;   Eid[ 80]= 7; Rid[ 80]=0; Zid[ 80]=+1;
  NLX_coes[1][ 81]=+0.0019892330256;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+2;
  NLX_coes[1][ 82]=+0.2078700754061;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+3;
  NLX_coes[1][ 83]=+0.2300386692838;   Eid[ 83]= 7; Rid[ 83]=1; Zid[ 83]=-2;
  NLX_coes[1][ 84]=-0.0825010252979;   Eid[ 84]= 7; Rid[ 84]=1; Zid[ 84]=-1;
  NLX_coes[1][ 85]=+0.2648807752046;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[1][ 86]=-0.1428957094343;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[1][ 87]=-0.0870045484284;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[1][ 88]=-0.2288116916453;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[1][ 89]=+0.0774576083034;   Eid[ 89]= 8; Rid[ 89]=0; Zid[ 89]=-2;
  NLX_coes[1][ 90]=+0.0051619878182;   Eid[ 90]= 8; Rid[ 90]=0; Zid[ 90]=-1;
  NLX_coes[1][ 91]=-0.1540758022820;   Eid[ 91]= 8; Rid[ 91]=0; Zid[ 91]=+0;
  NLX_coes[1][ 92]=+0.2230184613642;   Eid[ 92]= 8; Rid[ 92]=0; Zid[ 92]=+1;
  NLX_coes[1][ 93]=+0.0230584596923;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+2;
  NLX_coes[1][ 94]=+0.0202490827424;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+3;
  NLX_coes[1][ 95]=+0.1404608024180;   Eid[ 95]= 8; Rid[ 95]=1; Zid[ 95]=-2;
  NLX_coes[1][ 96]=-0.0332282372308;   Eid[ 96]= 8; Rid[ 96]=1; Zid[ 96]=-1;
  NLX_coes[1][ 97]=+0.2910253527578;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[1][ 98]=-0.1326309092385;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[1][ 99]=-0.0312912108729;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[1][100]=+0.0479952422642;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[1][101]=+0.0854668198243;   Eid[101]= 9; Rid[101]=0; Zid[101]=-2;
  NLX_coes[1][102]=-0.0666592713845;   Eid[102]= 9; Rid[102]=0; Zid[102]=-1;
  NLX_coes[1][103]=-0.1204461366783;   Eid[103]= 9; Rid[103]=0; Zid[103]=+0;
  NLX_coes[1][104]=+0.2234200529698;   Eid[104]= 9; Rid[104]=0; Zid[104]=+1;
  NLX_coes[1][105]=-0.0123621312885;   Eid[105]= 9; Rid[105]=0; Zid[105]=+2;
  NLX_coes[1][106]=-0.1601341615302;   Eid[106]= 9; Rid[106]=0; Zid[106]=+3;
  NLX_coes[1][107]=+0.0288572093923;   Eid[107]= 9; Rid[107]=1; Zid[107]=-2;
  NLX_coes[1][108]=-0.0381259770002;   Eid[108]= 9; Rid[108]=1; Zid[108]=-1;
  NLX_coes[1][109]=+0.2759659224355;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[1][110]=-0.1399163154112;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[1][111]=+0.0256931648182;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[1][112]=+0.2118876292122;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[1][113]=+0.0987103855448;   Eid[113]=10; Rid[113]=0; Zid[113]=-2;
  NLX_coes[1][114]=-0.1292086967351;   Eid[114]=10; Rid[114]=0; Zid[114]=-1;
  NLX_coes[1][115]=-0.0825905065556;   Eid[115]=10; Rid[115]=0; Zid[115]=+0;
  NLX_coes[1][116]=+0.2118161032648;   Eid[116]=10; Rid[116]=0; Zid[116]=+1;
  NLX_coes[1][117]=-0.0707218312202;   Eid[117]=10; Rid[117]=0; Zid[117]=+2;
  NLX_coes[1][118]=-0.2426978615450;   Eid[118]=10; Rid[118]=0; Zid[118]=+3;
  NLX_coes[1][119]=-0.0454062407128;   Eid[119]=10; Rid[119]=1; Zid[119]=-2;
  NLX_coes[1][120]=-0.0689029597082;   Eid[120]=10; Rid[120]=1; Zid[120]=-1;
  NLX_coes[1][121]=+0.2180696564910;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[1][122]=-0.1649684919826;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[1][123]=+0.0637500614528;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[1][124]=+0.2318764921014;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[1][125]=+0.1187172817656;   Eid[125]=11; Rid[125]=0; Zid[125]=-2;
  NLX_coes[1][126]=-0.1732174077203;   Eid[126]=11; Rid[126]=0; Zid[126]=-1;
  NLX_coes[1][127]=-0.0377937560203;   Eid[127]=11; Rid[127]=0; Zid[127]=+0;
  NLX_coes[1][128]=+0.1928394955251;   Eid[128]=11; Rid[128]=0; Zid[128]=+1;
  NLX_coes[1][129]=-0.1186221777195;   Eid[129]=11; Rid[129]=0; Zid[129]=+2;
  NLX_coes[1][130]=-0.1811722850714;   Eid[130]=11; Rid[130]=0; Zid[130]=+3;
  NLX_coes[1][131]=-0.0502007531930;   Eid[131]=11; Rid[131]=1; Zid[131]=-2;
  NLX_coes[1][132]=-0.0854468212402;   Eid[132]=11; Rid[132]=1; Zid[132]=-1;
  NLX_coes[1][133]=+0.1306315047620;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[1][134]=-0.1916843683829;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[1][135]=+0.0867521007631;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[1][136]=+0.1370660311686;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[1][137]=+0.1372891821206;   Eid[137]=12; Rid[137]=0; Zid[137]=-2;
  NLX_coes[1][138]=-0.1936039166802;   Eid[138]=12; Rid[138]=0; Zid[138]=-1;
  NLX_coes[1][139]=+0.0111813323688;   Eid[139]=12; Rid[139]=0; Zid[139]=+0;
  NLX_coes[1][140]=+0.1683875680145;   Eid[140]=12; Rid[140]=0; Zid[140]=+1;
  NLX_coes[1][141]=-0.1352175099588;   Eid[141]=12; Rid[141]=0; Zid[141]=+2;
  NLX_coes[1][142]=+0.0068516954967;   Eid[142]=12; Rid[142]=0; Zid[142]=+3;
  NLX_coes[1][143]=-0.0034492428634;   Eid[143]=12; Rid[143]=1; Zid[143]=-2;
  NLX_coes[1][144]=-0.0529988780983;   Eid[144]=12; Rid[144]=1; Zid[144]=-1;
  NLX_coes[1][145]=+0.0318867509996;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[1][146]=-0.1934824075438;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[1][147]=+0.1102302620096;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[1][148]=-0.0052210940503;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[1][149]=+0.1399551392892;   Eid[149]=13; Rid[149]=0; Zid[149]=-2;
  NLX_coes[1][150]=-0.1891136745963;   Eid[150]=13; Rid[150]=0; Zid[150]=-1;
  NLX_coes[1][151]=+0.0544860340711;   Eid[151]=13; Rid[151]=0; Zid[151]=+0;
  NLX_coes[1][152]=+0.1367398619448;   Eid[152]=13; Rid[152]=0; Zid[152]=+1;
  NLX_coes[1][153]=-0.1189554974110;   Eid[153]=13; Rid[153]=0; Zid[153]=+2;
  NLX_coes[1][154]=+0.2242742520997;   Eid[154]=13; Rid[154]=0; Zid[154]=+3;
  NLX_coes[1][155]=+0.0274314758753;   Eid[155]=13; Rid[155]=1; Zid[155]=-2;
  NLX_coes[1][156]=+0.0415011793928;   Eid[156]=13; Rid[156]=1; Zid[156]=-1;
  NLX_coes[1][157]=-0.0682062495493;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[1][158]=-0.1420802538532;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[1][159]=+0.1445755393124;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[1][160]=-0.1129735699999;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[1][161]=+0.1127616345510;   Eid[161]=14; Rid[161]=0; Zid[161]=-2;
  NLX_coes[1][162]=-0.1583598509534;   Eid[162]=14; Rid[162]=0; Zid[162]=-1;
  NLX_coes[1][163]=+0.0762413656178;   Eid[163]=14; Rid[163]=0; Zid[163]=+0;
  NLX_coes[1][164]=+0.0966536435406;   Eid[164]=14; Rid[164]=0; Zid[164]=+1;
  NLX_coes[1][165]=-0.0868508737922;   Eid[165]=14; Rid[165]=0; Zid[165]=+2;
  NLX_coes[1][166]=+0.2726098123171;   Eid[166]=14; Rid[166]=0; Zid[166]=+3;
  NLX_coes[1][167]=-0.0235938695361;   Eid[167]=14; Rid[167]=1; Zid[167]=-2;
  NLX_coes[1][168]=+0.1805030016216;   Eid[168]=14; Rid[168]=1; Zid[168]=-1;
  NLX_coes[1][169]=-0.1832963768734;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[1][170]=-0.0198068887110;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[1][171]=+0.1700218454723;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[1][172]=-0.1257719525643;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[1][173]=+0.0492038993612;   Eid[173]=15; Rid[173]=0; Zid[173]=-2;
  NLX_coes[1][174]=-0.0920207198537;   Eid[174]=15; Rid[174]=0; Zid[174]=-1;
  NLX_coes[1][175]=+0.0591038839844;   Eid[175]=15; Rid[175]=0; Zid[175]=+0;
  NLX_coes[1][176]=+0.0595266034672;   Eid[176]=15; Rid[176]=0; Zid[176]=+1;
  NLX_coes[1][177]=-0.0622688499860;   Eid[177]=15; Rid[177]=0; Zid[177]=+2;
  NLX_coes[1][178]=-0.1952598485645;   Eid[178]=15; Rid[178]=0; Zid[178]=+3;
  NLX_coes[1][179]=-0.0756485067648;   Eid[179]=15; Rid[179]=1; Zid[179]=-2;
  NLX_coes[1][180]=+0.3262113102543;   Eid[180]=15; Rid[180]=1; Zid[180]=-1;
  NLX_coes[1][181]=-0.3653411226383;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[1][182]=+0.1643573179401;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[1][183]=+0.0976063121016;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[1][184]=-0.0701988097874;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[1][185]=+0.0000000000468;   Eid[185]= 0; Rid[185]=0; Zid[185]=+0;
  NLX_coes[2][  0]=+0.0749466437444;   Eid[  0]= 0; Rid[  0]=0; Zid[  0]=-2;
  NLX_coes[2][  1]=+0.3016111837387;   Eid[  1]= 0; Rid[  1]=0; Zid[  1]=-1;
  NLX_coes[2][  2]=+0.1360464621422;   Eid[  2]= 0; Rid[  2]=0; Zid[  2]=+1;
  NLX_coes[2][  3]=-0.0441325322492;   Eid[  3]= 0; Rid[  3]=0; Zid[  3]=+2;
  NLX_coes[2][  4]=-0.0062998089455;   Eid[  4]= 0; Rid[  4]=0; Zid[  4]=+3;
  NLX_coes[2][  5]=-0.0095343830246;   Eid[  5]= 1; Rid[  5]=0; Zid[  5]=-2;
  NLX_coes[2][  6]=-0.0076808668439;   Eid[  6]= 1; Rid[  6]=0; Zid[  6]=-1;
  NLX_coes[2][  7]=-0.0124615955128;   Eid[  7]= 1; Rid[  7]=0; Zid[  7]=+0;
  NLX_coes[2][  8]=-0.0142206317442;   Eid[  8]= 1; Rid[  8]=0; Zid[  8]=+1;
  NLX_coes[2][  9]=-0.0129206352506;   Eid[  9]= 1; Rid[  9]=0; Zid[  9]=+2;
  NLX_coes[2][ 10]=-0.0380960102006;   Eid[ 10]= 1; Rid[ 10]=0; Zid[ 10]=+3;
  NLX_coes[2][ 11]=-0.0086212315135;   Eid[ 11]= 1; Rid[ 11]=1; Zid[ 11]=-2;
  NLX_coes[2][ 12]=-0.0479774238557;   Eid[ 12]= 1; Rid[ 12]=1; Zid[ 12]=-1;
  NLX_coes[2][ 13]=+0.3255365684273;   Eid[ 13]= 1; Rid[ 13]=1; Zid[ 13]=+0;
  NLX_coes[2][ 14]=+0.1113800393258;   Eid[ 14]= 1; Rid[ 14]=1; Zid[ 14]=+1;
  NLX_coes[2][ 15]=-0.1062445671550;   Eid[ 15]= 1; Rid[ 15]=1; Zid[ 15]=+2;
  NLX_coes[2][ 16]=+0.0341814636243;   Eid[ 16]= 1; Rid[ 16]=1; Zid[ 16]=+3;
  NLX_coes[2][ 17]=-0.0069781955125;   Eid[ 17]= 2; Rid[ 17]=0; Zid[ 17]=-2;
  NLX_coes[2][ 18]=-0.0422090078712;   Eid[ 18]= 2; Rid[ 18]=0; Zid[ 18]=-1;
  NLX_coes[2][ 19]=+0.0090776035299;   Eid[ 19]= 2; Rid[ 19]=0; Zid[ 19]=+0;
  NLX_coes[2][ 20]=+0.0507761765083;   Eid[ 20]= 2; Rid[ 20]=0; Zid[ 20]=+1;
  NLX_coes[2][ 21]=+0.1161482980252;   Eid[ 21]= 2; Rid[ 21]=0; Zid[ 21]=+2;
  NLX_coes[2][ 22]=+0.1673116785985;   Eid[ 22]= 2; Rid[ 22]=0; Zid[ 22]=+3;
  NLX_coes[2][ 23]=+0.0784356124011;   Eid[ 23]= 2; Rid[ 23]=1; Zid[ 23]=-2;
  NLX_coes[2][ 24]=+0.0121036427879;   Eid[ 24]= 2; Rid[ 24]=1; Zid[ 24]=-1;
  NLX_coes[2][ 25]=-0.0073454174685;   Eid[ 25]= 2; Rid[ 25]=1; Zid[ 25]=+0;
  NLX_coes[2][ 26]=-0.2009332596502;   Eid[ 26]= 2; Rid[ 26]=1; Zid[ 26]=+1;
  NLX_coes[2][ 27]=-0.1773714280262;   Eid[ 27]= 2; Rid[ 27]=1; Zid[ 27]=+2;
  NLX_coes[2][ 28]=-0.0676734942874;   Eid[ 28]= 2; Rid[ 28]=1; Zid[ 28]=+3;
  NLX_coes[2][ 29]=-0.0142690835852;   Eid[ 29]= 3; Rid[ 29]=0; Zid[ 29]=-2;
  NLX_coes[2][ 30]=-0.0281718453103;   Eid[ 30]= 3; Rid[ 30]=0; Zid[ 30]=-1;
  NLX_coes[2][ 31]=+0.0619403134891;   Eid[ 31]= 3; Rid[ 31]=0; Zid[ 31]=+0;
  NLX_coes[2][ 32]=+0.0769039259943;   Eid[ 32]= 3; Rid[ 32]=0; Zid[ 32]=+1;
  NLX_coes[2][ 33]=+0.0843948631682;   Eid[ 33]= 3; Rid[ 33]=0; Zid[ 33]=+2;
  NLX_coes[2][ 34]=-0.0842925723157;   Eid[ 34]= 3; Rid[ 34]=0; Zid[ 34]=+3;
  NLX_coes[2][ 35]=+0.0218997174139;   Eid[ 35]= 3; Rid[ 35]=1; Zid[ 35]=-2;
  NLX_coes[2][ 36]=+0.0469592765990;   Eid[ 36]= 3; Rid[ 36]=1; Zid[ 36]=-1;
  NLX_coes[2][ 37]=-0.0611536415808;   Eid[ 37]= 3; Rid[ 37]=1; Zid[ 37]=+0;
  NLX_coes[2][ 38]=-0.1724978380677;   Eid[ 38]= 3; Rid[ 38]=1; Zid[ 38]=+1;
  NLX_coes[2][ 39]=-0.0446890023248;   Eid[ 39]= 3; Rid[ 39]=1; Zid[ 39]=+2;
  NLX_coes[2][ 40]=+0.0883593498932;   Eid[ 40]= 3; Rid[ 40]=1; Zid[ 40]=+3;
  NLX_coes[2][ 41]=-0.0270967860707;   Eid[ 41]= 4; Rid[ 41]=0; Zid[ 41]=-2;
  NLX_coes[2][ 42]=-0.0217533209109;   Eid[ 42]= 4; Rid[ 42]=0; Zid[ 42]=-1;
  NLX_coes[2][ 43]=+0.0819373884932;   Eid[ 43]= 4; Rid[ 43]=0; Zid[ 43]=+0;
  NLX_coes[2][ 44]=+0.0672627253373;   Eid[ 44]= 4; Rid[ 44]=0; Zid[ 44]=+1;
  NLX_coes[2][ 45]=+0.0856410743412;   Eid[ 45]= 4; Rid[ 45]=0; Zid[ 45]=+2;
  NLX_coes[2][ 46]=-0.0547218146301;   Eid[ 46]= 4; Rid[ 46]=0; Zid[ 46]=+3;
  NLX_coes[2][ 47]=-0.0641705398936;   Eid[ 47]= 4; Rid[ 47]=1; Zid[ 47]=-2;
  NLX_coes[2][ 48]=+0.0435030688250;   Eid[ 48]= 4; Rid[ 48]=1; Zid[ 48]=-1;
  NLX_coes[2][ 49]=+0.0085327413446;   Eid[ 49]= 4; Rid[ 49]=1; Zid[ 49]=+0;
  NLX_coes[2][ 50]=+0.0120309376434;   Eid[ 50]= 4; Rid[ 50]=1; Zid[ 50]=+1;
  NLX_coes[2][ 51]=+0.1285492844826;   Eid[ 51]= 4; Rid[ 51]=1; Zid[ 51]=+2;
  NLX_coes[2][ 52]=-0.0239488260862;   Eid[ 52]= 4; Rid[ 52]=1; Zid[ 52]=+3;
  NLX_coes[2][ 53]=-0.0304940160997;   Eid[ 53]= 5; Rid[ 53]=0; Zid[ 53]=-2;
  NLX_coes[2][ 54]=-0.0268745747050;   Eid[ 54]= 5; Rid[ 54]=0; Zid[ 54]=-1;
  NLX_coes[2][ 55]=+0.0647047809617;   Eid[ 55]= 5; Rid[ 55]=0; Zid[ 55]=+0;
  NLX_coes[2][ 56]=+0.0196959898298;   Eid[ 56]= 5; Rid[ 56]=0; Zid[ 56]=+1;
  NLX_coes[2][ 57]=+0.0697353755440;   Eid[ 57]= 5; Rid[ 57]=0; Zid[ 57]=+2;
  NLX_coes[2][ 58]=+0.0375134243753;   Eid[ 58]= 5; Rid[ 58]=0; Zid[ 58]=+3;
  NLX_coes[2][ 59]=-0.1054152852394;   Eid[ 59]= 5; Rid[ 59]=1; Zid[ 59]=-2;
  NLX_coes[2][ 60]=-0.0229371388591;   Eid[ 60]= 5; Rid[ 60]=1; Zid[ 60]=-1;
  NLX_coes[2][ 61]=+0.0188912135959;   Eid[ 61]= 5; Rid[ 61]=1; Zid[ 61]=+0;
  NLX_coes[2][ 62]=+0.1012149519516;   Eid[ 62]= 5; Rid[ 62]=1; Zid[ 62]=+1;
  NLX_coes[2][ 63]=+0.2227165908130;   Eid[ 63]= 5; Rid[ 63]=1; Zid[ 63]=+2;
  NLX_coes[2][ 64]=+0.1591285152038;   Eid[ 64]= 5; Rid[ 64]=1; Zid[ 64]=+3;
  NLX_coes[2][ 65]=-0.0154918784397;   Eid[ 65]= 6; Rid[ 65]=0; Zid[ 65]=-2;
  NLX_coes[2][ 66]=-0.0265214156554;   Eid[ 66]= 6; Rid[ 66]=0; Zid[ 66]=-1;
  NLX_coes[2][ 67]=+0.0417052869710;   Eid[ 67]= 6; Rid[ 67]=0; Zid[ 67]=+0;
  NLX_coes[2][ 68]=-0.0289646652876;   Eid[ 68]= 6; Rid[ 68]=0; Zid[ 68]=+1;
  NLX_coes[2][ 69]=+0.0189185851233;   Eid[ 69]= 6; Rid[ 69]=0; Zid[ 69]=+2;
  NLX_coes[2][ 70]=-0.0324725702383;   Eid[ 70]= 6; Rid[ 70]=0; Zid[ 70]=+3;
  NLX_coes[2][ 71]=-0.0580256659867;   Eid[ 71]= 6; Rid[ 71]=1; Zid[ 71]=-2;
  NLX_coes[2][ 72]=-0.0914385367917;   Eid[ 72]= 6; Rid[ 72]=1; Zid[ 72]=-1;
  NLX_coes[2][ 73]=-0.0252866342034;   Eid[ 73]= 6; Rid[ 73]=1; Zid[ 73]=+0;
  NLX_coes[2][ 74]=+0.0695222980830;   Eid[ 74]= 6; Rid[ 74]=1; Zid[ 74]=+1;
  NLX_coes[2][ 75]=+0.1428174099034;   Eid[ 75]= 6; Rid[ 75]=1; Zid[ 75]=+2;
  NLX_coes[2][ 76]=+0.2064749295686;   Eid[ 76]= 6; Rid[ 76]=1; Zid[ 76]=+3;
  NLX_coes[2][ 77]=+0.0104592311638;   Eid[ 77]= 7; Rid[ 77]=0; Zid[ 77]=-2;
  NLX_coes[2][ 78]=-0.0151913561779;   Eid[ 78]= 7; Rid[ 78]=0; Zid[ 78]=-1;
  NLX_coes[2][ 79]=+0.0322599056958;   Eid[ 79]= 7; Rid[ 79]=0; Zid[ 79]=+0;
  NLX_coes[2][ 80]=-0.0457397348558;   Eid[ 80]= 7; Rid[ 80]=0; Zid[ 80]=+1;
  NLX_coes[2][ 81]=-0.0163414325020;   Eid[ 81]= 7; Rid[ 81]=0; Zid[ 81]=+2;
  NLX_coes[2][ 82]=-0.1425669183934;   Eid[ 82]= 7; Rid[ 82]=0; Zid[ 82]=+3;
  NLX_coes[2][ 83]=+0.0548762243579;   Eid[ 83]= 7; Rid[ 83]=1; Zid[ 83]=-2;
  NLX_coes[2][ 84]=-0.1045172347909;   Eid[ 84]= 7; Rid[ 84]=1; Zid[ 84]=-1;
  NLX_coes[2][ 85]=-0.0591831789712;   Eid[ 85]= 7; Rid[ 85]=1; Zid[ 85]=+0;
  NLX_coes[2][ 86]=+0.0089240551743;   Eid[ 86]= 7; Rid[ 86]=1; Zid[ 86]=+1;
  NLX_coes[2][ 87]=-0.0086434622286;   Eid[ 87]= 7; Rid[ 87]=1; Zid[ 87]=+2;
  NLX_coes[2][ 88]=+0.0557644609744;   Eid[ 88]= 7; Rid[ 88]=1; Zid[ 88]=+3;
  NLX_coes[2][ 89]=+0.0312084636007;   Eid[ 89]= 8; Rid[ 89]=0; Zid[ 89]=-2;
  NLX_coes[2][ 90]=-0.0009501670555;   Eid[ 90]= 8; Rid[ 90]=0; Zid[ 90]=-1;
  NLX_coes[2][ 91]=+0.0334609928009;   Eid[ 91]= 8; Rid[ 91]=0; Zid[ 91]=+0;
  NLX_coes[2][ 92]=-0.0291894167246;   Eid[ 92]= 8; Rid[ 92]=0; Zid[ 92]=+1;
  NLX_coes[2][ 93]=-0.0077821817470;   Eid[ 93]= 8; Rid[ 93]=0; Zid[ 93]=+2;
  NLX_coes[2][ 94]=-0.1363597408157;   Eid[ 94]= 8; Rid[ 94]=0; Zid[ 94]=+3;
  NLX_coes[2][ 95]=+0.1555471758503;   Eid[ 95]= 8; Rid[ 95]=1; Zid[ 95]=-2;
  NLX_coes[2][ 96]=-0.0524579099950;   Eid[ 96]= 8; Rid[ 96]=1; Zid[ 96]=-1;
  NLX_coes[2][ 97]=-0.0464500110259;   Eid[ 97]= 8; Rid[ 97]=1; Zid[ 97]=+0;
  NLX_coes[2][ 98]=-0.0084181964881;   Eid[ 98]= 8; Rid[ 98]=1; Zid[ 98]=+1;
  NLX_coes[2][ 99]=-0.1084151237869;   Eid[ 99]= 8; Rid[ 99]=1; Zid[ 99]=+2;
  NLX_coes[2][100]=-0.1113191755250;   Eid[100]= 8; Rid[100]=1; Zid[100]=+3;
  NLX_coes[2][101]=+0.0340901112917;   Eid[101]= 9; Rid[101]=0; Zid[101]=-2;
  NLX_coes[2][102]=+0.0048119557981;   Eid[102]= 9; Rid[102]=0; Zid[102]=-1;
  NLX_coes[2][103]=+0.0315829581932;   Eid[103]= 9; Rid[103]=0; Zid[103]=+0;
  NLX_coes[2][104]=-0.0000947578109;   Eid[104]= 9; Rid[104]=0; Zid[104]=+1;
  NLX_coes[2][105]=+0.0283557649708;   Eid[105]= 9; Rid[105]=0; Zid[105]=+2;
  NLX_coes[2][106]=+0.0025757342348;   Eid[106]= 9; Rid[106]=0; Zid[106]=+3;
  NLX_coes[2][107]=+0.1597407031878;   Eid[107]= 9; Rid[107]=1; Zid[107]=-2;
  NLX_coes[2][108]=+0.0300755960575;   Eid[108]= 9; Rid[108]=1; Zid[108]=-1;
  NLX_coes[2][109]=+0.0038570806298;   Eid[109]= 9; Rid[109]=1; Zid[109]=+0;
  NLX_coes[2][110]=+0.0312982009889;   Eid[110]= 9; Rid[110]=1; Zid[110]=+1;
  NLX_coes[2][111]=-0.1080882461294;   Eid[111]= 9; Rid[111]=1; Zid[111]=+2;
  NLX_coes[2][112]=-0.1518681217298;   Eid[112]= 9; Rid[112]=1; Zid[112]=+3;
  NLX_coes[2][113]=+0.0182142409572;   Eid[113]=10; Rid[113]=0; Zid[113]=-2;
  NLX_coes[2][114]=-0.0026480945501;   Eid[114]=10; Rid[114]=0; Zid[114]=-1;
  NLX_coes[2][115]=+0.0156994051986;   Eid[115]=10; Rid[115]=0; Zid[115]=+0;
  NLX_coes[2][116]=+0.0187048817614;   Eid[116]=10; Rid[116]=0; Zid[116]=+1;
  NLX_coes[2][117]=+0.0565366173684;   Eid[117]=10; Rid[117]=0; Zid[117]=+2;
  NLX_coes[2][118]=+0.1643646330683;   Eid[118]=10; Rid[118]=0; Zid[118]=+3;
  NLX_coes[2][119]=+0.0360300141936;   Eid[119]=10; Rid[119]=1; Zid[119]=-2;
  NLX_coes[2][120]=+0.0884956833952;   Eid[120]=10; Rid[120]=1; Zid[120]=-1;
  NLX_coes[2][121]=+0.0511578000561;   Eid[121]=10; Rid[121]=1; Zid[121]=+0;
  NLX_coes[2][122]=+0.0957183920977;   Eid[122]=10; Rid[122]=1; Zid[122]=+1;
  NLX_coes[2][123]=-0.0353207974269;   Eid[123]=10; Rid[123]=1; Zid[123]=+2;
  NLX_coes[2][124]=-0.0568711038732;   Eid[124]=10; Rid[124]=1; Zid[124]=+3;
  NLX_coes[2][125]=-0.0044931142925;   Eid[125]=11; Rid[125]=0; Zid[125]=-2;
  NLX_coes[2][126]=-0.0165251067277;   Eid[126]=11; Rid[126]=0; Zid[126]=-1;
  NLX_coes[2][127]=-0.0140029826502;   Eid[127]=11; Rid[127]=0; Zid[127]=+0;
  NLX_coes[2][128]=+0.0171953923441;   Eid[128]=11; Rid[128]=0; Zid[128]=+1;
  NLX_coes[2][129]=+0.0535534440909;   Eid[129]=11; Rid[129]=0; Zid[129]=+2;
  NLX_coes[2][130]=+0.2069168421441;   Eid[130]=11; Rid[130]=0; Zid[130]=+3;
  NLX_coes[2][131]=-0.1470822280188;   Eid[131]=11; Rid[131]=1; Zid[131]=-2;
  NLX_coes[2][132]=+0.0804365665226;   Eid[132]=11; Rid[132]=1; Zid[132]=-1;
  NLX_coes[2][133]=+0.0483620638924;   Eid[133]=11; Rid[133]=1; Zid[133]=+0;
  NLX_coes[2][134]=+0.1331991679923;   Eid[134]=11; Rid[134]=1; Zid[134]=+1;
  NLX_coes[2][135]=+0.0420203400081;   Eid[135]=11; Rid[135]=1; Zid[135]=+2;
  NLX_coes[2][136]=+0.0758662312243;   Eid[136]=11; Rid[136]=1; Zid[136]=+3;
  NLX_coes[2][137]=-0.0163591786659;   Eid[137]=12; Rid[137]=0; Zid[137]=-2;
  NLX_coes[2][138]=-0.0206099102330;   Eid[138]=12; Rid[138]=0; Zid[138]=-1;
  NLX_coes[2][139]=-0.0451134018551;   Eid[139]=12; Rid[139]=0; Zid[139]=+0;
  NLX_coes[2][140]=+0.0028290441519;   Eid[140]=12; Rid[140]=0; Zid[140]=+1;
  NLX_coes[2][141]=+0.0262953144515;   Eid[141]=12; Rid[141]=0; Zid[141]=+2;
  NLX_coes[2][142]=+0.0606733882982;   Eid[142]=12; Rid[142]=0; Zid[142]=+3;
  NLX_coes[2][143]=-0.2199789459980;   Eid[143]=12; Rid[143]=1; Zid[143]=-2;
  NLX_coes[2][144]=+0.0067864779915;   Eid[144]=12; Rid[144]=1; Zid[144]=-1;
  NLX_coes[2][145]=-0.0270904211964;   Eid[145]=12; Rid[145]=1; Zid[145]=+0;
  NLX_coes[2][146]=+0.1026107778080;   Eid[146]=12; Rid[146]=1; Zid[146]=+1;
  NLX_coes[2][147]=+0.0592776689676;   Eid[147]=12; Rid[147]=1; Zid[147]=+2;
  NLX_coes[2][148]=+0.1185562945915;   Eid[148]=12; Rid[148]=1; Zid[148]=+3;
  NLX_coes[2][149]=-0.0073647713234;   Eid[149]=13; Rid[149]=0; Zid[149]=-2;
  NLX_coes[2][150]=+0.0002441787444;   Eid[150]=13; Rid[150]=0; Zid[150]=-1;
  NLX_coes[2][151]=-0.0594229968592;   Eid[151]=13; Rid[151]=0; Zid[151]=+0;
  NLX_coes[2][152]=-0.0055836690738;   Eid[152]=13; Rid[152]=0; Zid[152]=+1;
  NLX_coes[2][153]=+0.0073567572310;   Eid[153]=13; Rid[153]=0; Zid[153]=+2;
  NLX_coes[2][154]=-0.1919583339713;   Eid[154]=13; Rid[154]=0; Zid[154]=+3;
  NLX_coes[2][155]=-0.0125623515553;   Eid[155]=13; Rid[155]=1; Zid[155]=-2;
  NLX_coes[2][156]=-0.0715074783304;   Eid[156]=13; Rid[156]=1; Zid[156]=-1;
  NLX_coes[2][157]=-0.1380507579452;   Eid[157]=13; Rid[157]=1; Zid[157]=+0;
  NLX_coes[2][158]=+0.0037379210153;   Eid[158]=13; Rid[158]=1; Zid[158]=+1;
  NLX_coes[2][159]=-0.0030414471009;   Eid[159]=13; Rid[159]=1; Zid[159]=+2;
  NLX_coes[2][160]=+0.0105360578200;   Eid[160]=13; Rid[160]=1; Zid[160]=+3;
  NLX_coes[2][161]=+0.0113132451524;   Eid[161]=14; Rid[161]=0; Zid[161]=-2;
  NLX_coes[2][162]=+0.0407428104900;   Eid[162]=14; Rid[162]=0; Zid[162]=-1;
  NLX_coes[2][163]=-0.0468141024954;   Eid[163]=14; Rid[163]=0; Zid[163]=+0;
  NLX_coes[2][164]=+0.0060873245744;   Eid[164]=14; Rid[164]=0; Zid[164]=+1;
  NLX_coes[2][165]=+0.0244068827712;   Eid[165]=14; Rid[165]=0; Zid[165]=+2;
  NLX_coes[2][166]=-0.2822914131503;   Eid[166]=14; Rid[166]=0; Zid[166]=+3;
  NLX_coes[2][167]=+0.3510265137077;   Eid[167]=14; Rid[167]=1; Zid[167]=-2;
  NLX_coes[2][168]=-0.0542707276016;   Eid[168]=14; Rid[168]=1; Zid[168]=-1;
  NLX_coes[2][169]=-0.1528669505903;   Eid[169]=14; Rid[169]=1; Zid[169]=+0;
  NLX_coes[2][170]=-0.0940995812279;   Eid[170]=14; Rid[170]=1; Zid[170]=+1;
  NLX_coes[2][171]=-0.0825753009323;   Eid[171]=14; Rid[171]=1; Zid[171]=+2;
  NLX_coes[2][172]=-0.1278375961805;   Eid[172]=14; Rid[172]=1; Zid[172]=+3;
  NLX_coes[2][173]=+0.0076919745293;   Eid[173]=15; Rid[173]=0; Zid[173]=-2;
  NLX_coes[2][174]=+0.0481808069897;   Eid[174]=15; Rid[174]=0; Zid[174]=-1;
  NLX_coes[2][175]=-0.0254164646694;   Eid[175]=15; Rid[175]=0; Zid[175]=+0;
  NLX_coes[2][176]=+0.0234227778210;   Eid[176]=15; Rid[176]=0; Zid[176]=+1;
  NLX_coes[2][177]=+0.0389498570512;   Eid[177]=15; Rid[177]=0; Zid[177]=+2;
  NLX_coes[2][178]=+0.2080848144165;   Eid[178]=15; Rid[178]=0; Zid[178]=+3;
  NLX_coes[2][179]=-0.1840887716509;   Eid[179]=15; Rid[179]=1; Zid[179]=-2;
  NLX_coes[2][180]=+0.0986467734769;   Eid[180]=15; Rid[180]=1; Zid[180]=-1;
  NLX_coes[2][181]=+0.1781858296246;   Eid[181]=15; Rid[181]=1; Zid[181]=+0;
  NLX_coes[2][182]=-0.0324185258200;   Eid[182]=15; Rid[182]=1; Zid[182]=+1;
  NLX_coes[2][183]=-0.0111714192111;   Eid[183]=15; Rid[183]=1; Zid[183]=+2;
  NLX_coes[2][184]=+0.1370917713475;   Eid[184]=15; Rid[184]=1; Zid[184]=+3;
  NLX_coes[2][185]=-0.0000000001926;   Eid[185]= 0; Rid[185]=0; Zid[185]=+0;

  /* convert rho and drho to r, Z, and e */

  x = rho;
  y = fabs(drho);

  xy = x/y;
  yx = y/x;
  ln = -log(8.0*PI) - 3.0*log(pow(x,4.0/3.0)/y);
  r = xy*ln;
  Z = 0.50*y/x;

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

  dr0 = (-4.0 + ln)/y;
  dr1 = (3.0 - ln)*xy/y;

  dZ0 = -0.5*yx/x;
  dZ1 = 0.5/x;

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

  for (n=0; n<(Nchi-1); n++){

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

  printf("VVV2 x=%15.12f y=%15.12f r=%15.12f Z=%15.12f e=%15.12f\n",x,y,r,Z,e);

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




