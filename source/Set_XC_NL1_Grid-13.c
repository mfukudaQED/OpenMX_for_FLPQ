#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

static int Num_TPF,Num_Grid1,Num_Grid2,Nchi,Nchi1,Nchi2;
static int basis1_flag,basis2_flag;
static int Eid[189],Rid[189],Zid[189];
static double NLX_coes1[3][21][21];
static double NLX_coes2[3][189];
static double min_rho16,max_rho16,min_R13,max_R13,al_R13,alpha_para;
static double Exponent_Exp,damp_para_B,damp_para_C;

static double mask_func_q(int m_flag, double q, double alpha_para);
static void gx_kernel(int mu, double rho, double drho, double gx[3]);
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
    double rho;
    double drho;
    double d;

    rho  = 0.3;
    drho = 0.1;

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
  {
    int i,k;
    double chi[3],t1,t2;
    double rho;
    double drho;
    double d;

    rho  = 0.3;
    drho = 0.1;

    gx_kernel(0, rho, drho, gx);

    printf("ABC1 %15.12f %15.12f %15.12f\n",gx[0],gx[1],gx[2]);

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

  /*
  {
    int N1,N2,j0;
    double d1,d2,x,y,tmp,R;

    N1 = Num_Grid1;
    N2 = Num_Grid2;
    d1 = (max_rho16-(min_rho16))/(double)N1;
    d2 = (max_R13-(min_R13))/(double)N2;

    printf("AAA1\n\n"); 

    for (i1=0; i1<=Num_Grid1; i1++){
      for (j1=0; j1<=Num_Grid2; j1++){

	x = min_rho16 + d1*(double)i1;
	y = min_R13 + d2*(double)j1;
        rho = pow(x,6.0);
        R = y*y*y + al_R13*y;
        tmp = 8.0*PI*rho*rho*rho*rho*exp(R);
        drho = pow(tmp,1.0/3.0);          
        gx_kernel(0, rho, drho, gx);
	printf("%18.15f %18.15f %18.15f\n",x,y,gx[0]);
      }
      printf("\n"); 
    }

  }

  MPI_Finalize();
  exit(0);
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
  int i,j,ii,jj,i1,j1,N1,N2,p,n;
  double tmp0,tmp1,s,s2,s3,s4,s5,t,t2,t3,t4,t5,al;
  double min_R,max_R,rho16,drho16_0,drho16_1;
  double dR0,dR1,R13,R,ln,x,y,d1,d2;
  double f[6][6],df0[6][6],df1[6][6];
  double threshold=1.0e-13;
  double chi,dchi0,dchi1,F,dF,a,b,c;
  double pZ,pe,pr,pe1,pr1,pZ1,id,jd,kd;
  double de0,de1,e,dZ0,dZ1,Z,dr0,dr1,w,r,tmp,xy,yx;

  x = fabs(rho);
  y = fabs(drho);

  if (x<threshold){

    gx[0] = 0.0;  
    gx[1] = 0.0;  
    gx[2] = 0.0;

    return;
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

  rho16 = pow(x,1.0/6.0);
  drho16_0 = 1.0/6.0*pow(x,-5.0/6.0);
  drho16_1 = 0.0;

  if (rho16<(min_rho16)){
    x = min_rho16*min_rho16*min_rho16*min_rho16*min_rho16*min_rho16;
    rho16 = min_rho16;
    drho16_0 = 0.0;
    drho16_1 = 0.0;
  }
  else if ((max_rho16)<rho16){
    x = max_rho16*max_rho16*max_rho16*max_rho16*max_rho16*max_rho16;
    rho16 = max_rho16;
    drho16_0 = 0.0;
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

  /* find indexes */

  i1 = (int)((rho16 - (min_rho16))/d1);
  if (i1<2)           i1 = 2;
  else if ((N1-3)<i1) i1 = N1-3;

  j1 = (int)((R13 - (min_R13))/d2);
  if (j1<2)           j1 = 2;
  else if ((N2-3)<j1) j1 = N2-3;

  /* convert rho16 and R13 to s and t */

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

  /* basis1 */

  if (basis1_flag==1){

    for (i=0; i<=5; i++){
      for (j=0; j<=5; j++){

	ii = i1+i-2;
	jj = j1+j-2;

	gx[0] += NLX_coes1[mu][ii][jj]*f[i][j]; 
	gx[1] += NLX_coes1[mu][ii][jj]*df0[i][j]; 
	gx[2] += NLX_coes1[mu][ii][jj]*df1[i][j]; 
      }
    }
  }

  /* basis2 */

  if (basis2_flag==1){

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

    a = Exponent_Exp;
    e = exp(-a*Z*r);
    de0 = -a*e*Z*dr0 - a*e*r*dZ0;
    de1 = -a*e*Z*dr1 - a*e*r*dZ1;

    b = damp_para_B;
    c = damp_para_C;
    F = 0.5*(erf(b*(r-c))+1.0);
    dF = b*exp(-b*b*(r-c)*(r-c))/sqrt(PI);

    for (n=0; n<Nchi2; n++){

      id = (double)Eid[n];
      jd = (double)Rid[n];
      kd = (double)Zid[n];

      pe1 = pow(e,id-1.0);
      pe = pe1*e;

      pr1 = pow(r,jd-1.0);
      pr = pr1*r;

      pZ1 = pow(Z,kd-1.0);
      pZ = pZ1*Z;

      chi = F*pe*pr*pZ;
      dchi0 = dF*dr0*pe*pr*pZ + F*(id*pe1*de0*pr*pZ + jd*pr1*dr0*pe*pZ + kd*pZ1*dZ0*pe*pr);
      dchi1 = dF*dr1*pe*pr*pZ + F*(id*pe1*de1*pr*pZ + jd*pr1*dr1*pe*pZ + kd*pZ1*dZ1*pe*pr);

      gx[0] += NLX_coes2[mu][n]*chi;
      gx[1] += NLX_coes2[mu][n]*dchi0;
      gx[2] += NLX_coes2[mu][n]*dchi1;
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
  basis1_flag = 1;
  basis2_flag = 1;
  Num_TPF = 3;
  Num_Grid1 = 20;
  Num_Grid2 = 20;
  Nchi1 = 441;
  Nchi2 = 189;
  Nchi = 630;
  min_rho16 = -0.400000000000000;
  max_rho16 =  8.000000000000000;
  min_R13 = -6.000000000000000;
  max_R13 =  4.000000000000000;
  al_R13 =  1.000000000000000;
  alpha_para =  6.000000000000000;
  Exponent_Exp =  0.250000000000000;
  damp_para_B =  1.000000000000000;
  damp_para_C = -2.000000000000000;

  NLX_coes1[0][  0][  0] =+2.646037937937557e-09;
  NLX_coes1[0][  0][  1] =+2.540469018179040e-01;
  NLX_coes1[0][  0][  2] =+4.860031725066692e-01;
  NLX_coes1[0][  0][  3] =+6.912880287524034e-01;
  NLX_coes1[0][  0][  4] =+8.648006501581298e-01;
  NLX_coes1[0][  0][  5] =+1.009141938733056e+00;
  NLX_coes1[0][  0][  6] =+1.133022269243363e+00;
  NLX_coes1[0][  0][  7] =+1.247033822319871e+00;
  NLX_coes1[0][  0][  8] =+1.349531663386570e+00;
  NLX_coes1[0][  0][  9] =+1.411779131189902e+00;
  NLX_coes1[0][  0][ 10] =+1.441855978309173e+00;
  NLX_coes1[0][  0][ 11] =+1.461309760547306e+00;
  NLX_coes1[0][  0][ 12] =+1.503509519034754e+00;
  NLX_coes1[0][  0][ 13] =+1.524401263005036e+00;
  NLX_coes1[0][  0][ 14] =+1.485898427626072e+00;
  NLX_coes1[0][  0][ 15] =+1.535449842486217e+00;
  NLX_coes1[0][  0][ 16] =+1.675814763950715e+00;
  NLX_coes1[0][  0][ 17] =+1.854929757166604e+00;
  NLX_coes1[0][  0][ 18] =+2.023643679703685e+00;
  NLX_coes1[0][  0][ 19] =+2.171462632582100e+00;
  NLX_coes1[0][  0][ 20] =+2.299090534137355e+00;
  NLX_coes1[0][  1][  0] =-2.954240794388458e-04;
  NLX_coes1[0][  1][  1] =+6.415901727812831e-02;
  NLX_coes1[0][  1][  2] =+1.589536362882147e-01;
  NLX_coes1[0][  1][  3] =+2.518243425704689e-01;
  NLX_coes1[0][  1][  4] =+3.354825421118532e-01;
  NLX_coes1[0][  1][  5] =+4.078791814569039e-01;
  NLX_coes1[0][  1][  6] =+4.702166513730794e-01;
  NLX_coes1[0][  1][  7] =+5.374861812107613e-01;
  NLX_coes1[0][  1][  8] =+6.232170115635348e-01;
  NLX_coes1[0][  1][  9] =+6.611251972305312e-01;
  NLX_coes1[0][  1][ 10] =+6.529308861590691e-01;
  NLX_coes1[0][  1][ 11] =+6.266913143493219e-01;
  NLX_coes1[0][  1][ 12] =+6.451844185463050e-01;
  NLX_coes1[0][  1][ 13] =+7.377113150919830e-01;
  NLX_coes1[0][  1][ 14] =+5.806499468312362e-01;
  NLX_coes1[0][  1][ 15] =+6.237495350015311e-01;
  NLX_coes1[0][  1][ 16] =+7.907759812156436e-01;
  NLX_coes1[0][  1][ 17] =+9.438130671629229e-01;
  NLX_coes1[0][  1][ 18] =+1.084898661550738e+00;
  NLX_coes1[0][  1][ 19] =+1.210089336151270e+00;
  NLX_coes1[0][  1][ 20] =+1.321024689046637e+00;
  NLX_coes1[0][  2][  0] =-1.429849782708176e-01;
  NLX_coes1[0][  2][  1] =-2.161902712142816e-01;
  NLX_coes1[0][  2][  2] =-2.587845990028512e-01;
  NLX_coes1[0][  2][  3] =-2.644120459939544e-01;
  NLX_coes1[0][  2][  4] =-2.500621945274398e-01;
  NLX_coes1[0][  2][  5] =-2.305601201087910e-01;
  NLX_coes1[0][  2][  6] =-2.268860048795262e-01;
  NLX_coes1[0][  2][  7] =-2.073979265753189e-01;
  NLX_coes1[0][  2][  8] =-1.795389504820577e-01;
  NLX_coes1[0][  2][  9] =-1.450811338681529e-02;
  NLX_coes1[0][  2][ 10] =-1.495352752711059e-01;
  NLX_coes1[0][  2][ 11] =-1.930070140612930e-01;
  NLX_coes1[0][  2][ 12] =-2.263263619474392e-01;
  NLX_coes1[0][  2][ 13] =-2.434992767912940e-01;
  NLX_coes1[0][  2][ 14] =-2.291775445768986e-01;
  NLX_coes1[0][  2][ 15] =-1.852956213970921e-01;
  NLX_coes1[0][  2][ 16] =-7.156125053764714e-02;
  NLX_coes1[0][  2][ 17] =+2.811619821001261e-02;
  NLX_coes1[0][  2][ 18] =+1.234621535536492e-01;
  NLX_coes1[0][  2][ 19] =+2.137082347288361e-01;
  NLX_coes1[0][  2][ 20] =+3.056306567816882e-01;
  NLX_coes1[0][  3][  0] =-5.462380640866054e-01;
  NLX_coes1[0][  3][  1] =-7.061898021412227e-01;
  NLX_coes1[0][  3][  2] =-8.454976901822714e-01;
  NLX_coes1[0][  3][  3] =-9.254289493119811e-01;
  NLX_coes1[0][  3][  4] =-9.285596291547880e-01;
  NLX_coes1[0][  3][  5] =-9.372071052484416e-01;
  NLX_coes1[0][  3][  6] =-9.304181501023223e-01;
  NLX_coes1[0][  3][  7] =-9.189181900312078e-01;
  NLX_coes1[0][  3][  8] =-8.964764244028061e-01;
  NLX_coes1[0][  3][  9] =-9.534343591288226e-01;
  NLX_coes1[0][  3][ 10] =-9.613815198239697e-01;
  NLX_coes1[0][  3][ 11] =-9.668994503013293e-01;
  NLX_coes1[0][  3][ 12] =-9.711409441689735e-01;
  NLX_coes1[0][  3][ 13] =-9.765542110562060e-01;
  NLX_coes1[0][  3][ 14] =-9.864245770781163e-01;
  NLX_coes1[0][  3][ 15] =-9.889185527145965e-01;
  NLX_coes1[0][  3][ 16] =-9.886987680399930e-01;
  NLX_coes1[0][  3][ 17] =-9.695132765068305e-01;
  NLX_coes1[0][  3][ 18] =-9.160942040520146e-01;
  NLX_coes1[0][  3][ 19] =-8.461895682509994e-01;
  NLX_coes1[0][  3][ 20] =-7.720919166911637e-01;
  NLX_coes1[0][  4][  0] =-1.210054683186260e+00;
  NLX_coes1[0][  4][  1] =-1.430272318310511e+00;
  NLX_coes1[0][  4][  2] =-1.640343182116092e+00;
  NLX_coes1[0][  4][  3] =-1.807005964360259e+00;
  NLX_coes1[0][  4][  4] =-1.840646850160079e+00;
  NLX_coes1[0][  4][  5] =-1.862953889638919e+00;
  NLX_coes1[0][  4][  6] =-1.862565690281906e+00;
  NLX_coes1[0][  4][  7] =-1.844500307579875e+00;
  NLX_coes1[0][  4][  8] =-1.809906682825423e+00;
  NLX_coes1[0][  4][  9] =-1.808506949667453e+00;
  NLX_coes1[0][  4][ 10] =-1.913563923420769e+00;
  NLX_coes1[0][  4][ 11] =-1.953740994711697e+00;
  NLX_coes1[0][  4][ 12] =-1.961003032937715e+00;
  NLX_coes1[0][  4][ 13] =-1.956782413421615e+00;
  NLX_coes1[0][  4][ 14] =-1.940309708943698e+00;
  NLX_coes1[0][  4][ 15] =-1.972661496185480e+00;
  NLX_coes1[0][  4][ 16] =-2.039471703294526e+00;
  NLX_coes1[0][  4][ 17] =-2.053961049243128e+00;
  NLX_coes1[0][  4][ 18] =-2.035958803731484e+00;
  NLX_coes1[0][  4][ 19] =-1.994364884013639e+00;
  NLX_coes1[0][  4][ 20] =-1.936966834047780e+00;
  NLX_coes1[0][  5][  0] =-2.134434835136507e+00;
  NLX_coes1[0][  5][  1] =-2.389665600659472e+00;
  NLX_coes1[0][  5][  2] =-2.632939557084268e+00;
  NLX_coes1[0][  5][  3] =-2.829651738555547e+00;
  NLX_coes1[0][  5][  4] =-2.942170622752781e+00;
  NLX_coes1[0][  5][  5] =-3.009060715403689e+00;
  NLX_coes1[0][  5][  6] =-3.042373112927881e+00;
  NLX_coes1[0][  5][  7] =-3.110768221111936e+00;
  NLX_coes1[0][  5][  8] =-3.287357718679653e+00;
  NLX_coes1[0][  5][  9] =-3.428001685939088e+00;
  NLX_coes1[0][  5][ 10] =-3.342179837303795e+00;
  NLX_coes1[0][  5][ 11] =-3.336724681177665e+00;
  NLX_coes1[0][  5][ 12] =-3.256803202162872e+00;
  NLX_coes1[0][  5][ 13] =-3.202521832307344e+00;
  NLX_coes1[0][  5][ 14] =-3.215249226975661e+00;
  NLX_coes1[0][  5][ 15] =-3.257371618148602e+00;
  NLX_coes1[0][  5][ 16] =-3.279265514817164e+00;
  NLX_coes1[0][  5][ 17] =-3.293662979756656e+00;
  NLX_coes1[0][  5][ 18] =-3.287120546120772e+00;
  NLX_coes1[0][  5][ 19] =-3.257307077970410e+00;
  NLX_coes1[0][  5][ 20] =-3.209551279755418e+00;
  NLX_coes1[0][  6][  0] =-3.319378519843776e+00;
  NLX_coes1[0][  6][  1] =-3.588634501291249e+00;
  NLX_coes1[0][  6][  2] =-3.845318316453770e+00;
  NLX_coes1[0][  6][  3] =-4.062603777246928e+00;
  NLX_coes1[0][  6][  4] =-4.227326360924941e+00;
  NLX_coes1[0][  6][  5] =-4.349714217301742e+00;
  NLX_coes1[0][  6][  6] =-4.457949286032943e+00;
  NLX_coes1[0][  6][  7] =-4.584994130815853e+00;
  NLX_coes1[0][  6][  8] =-4.729999603109829e+00;
  NLX_coes1[0][  6][  9] =-4.822021036939816e+00;
  NLX_coes1[0][  6][ 10] =-4.887375884882779e+00;
  NLX_coes1[0][  6][ 11] =-4.910140204262882e+00;
  NLX_coes1[0][  6][ 12] =-4.908723488794398e+00;
  NLX_coes1[0][  6][ 13] =-4.850931660930258e+00;
  NLX_coes1[0][  6][ 14] =-4.753082010796979e+00;
  NLX_coes1[0][  6][ 15] =-4.671154056414300e+00;
  NLX_coes1[0][  6][ 16] =-4.692686073643746e+00;
  NLX_coes1[0][  6][ 17] =-4.698672227974583e+00;
  NLX_coes1[0][  6][ 18] =-4.685188258053897e+00;
  NLX_coes1[0][  6][ 19] =-4.653270107014192e+00;
  NLX_coes1[0][  6][ 20] =-4.606231694569072e+00;
  NLX_coes1[0][  7][  0] =-4.764885737350786e+00;
  NLX_coes1[0][  7][  1] =-5.034988281179205e+00;
  NLX_coes1[0][  7][  2] =-5.294994723427987e+00;
  NLX_coes1[0][  7][  3] =-5.526922396244776e+00;
  NLX_coes1[0][  7][  4] =-5.723096441099943e+00;
  NLX_coes1[0][  7][  5] =-5.887655369119455e+00;
  NLX_coes1[0][  7][  6] =-6.036651628094435e+00;
  NLX_coes1[0][  7][  7] =-6.182802058701058e+00;
  NLX_coes1[0][  7][  8] =-6.317451251252122e+00;
  NLX_coes1[0][  7][  9] =-6.411344976469811e+00;
  NLX_coes1[0][  7][ 10] =-6.435666817217849e+00;
  NLX_coes1[0][  7][ 11] =-6.379335715698380e+00;
  NLX_coes1[0][  7][ 12] =-6.505842456922512e+00;
  NLX_coes1[0][  7][ 13] =-6.556716776096466e+00;
  NLX_coes1[0][  7][ 14] =-6.451522764918651e+00;
  NLX_coes1[0][  7][ 15] =-6.344529713742271e+00;
  NLX_coes1[0][  7][ 16] =-6.292733505223162e+00;
  NLX_coes1[0][  7][ 17] =-6.263421776084647e+00;
  NLX_coes1[0][  7][ 18] =-6.231225733066344e+00;
  NLX_coes1[0][  7][ 19] =-6.188156380761287e+00;
  NLX_coes1[0][  7][ 20] =-6.134545165478636e+00;
  NLX_coes1[0][  8][  0] =-6.470956487667912e+00;
  NLX_coes1[0][  8][  1] =-6.732452732854286e+00;
  NLX_coes1[0][  8][  2] =-6.987489925488733e+00;
  NLX_coes1[0][  8][  3] =-7.223165230313884e+00;
  NLX_coes1[0][  8][  4] =-7.432804292981329e+00;
  NLX_coes1[0][  8][  5] =-7.617008904891152e+00;
  NLX_coes1[0][  8][  6] =-7.781581241234848e+00;
  NLX_coes1[0][  8][  7] =-7.929479442618633e+00;
  NLX_coes1[0][  8][  8] =-8.054590010165899e+00;
  NLX_coes1[0][  8][  9] =-8.148668167045315e+00;
  NLX_coes1[0][  8][ 10] =-8.195915330848564e+00;
  NLX_coes1[0][  8][ 11] =-8.161061524481939e+00;
  NLX_coes1[0][  8][ 12] =-8.221906130635627e+00;
  NLX_coes1[0][  8][ 13] =-8.215729378287246e+00;
  NLX_coes1[0][  8][ 14] =-8.186106872837557e+00;
  NLX_coes1[0][  8][ 15] =-8.106671041904585e+00;
  NLX_coes1[0][  8][ 16] =-8.033477619738031e+00;
  NLX_coes1[0][  8][ 17] =-7.974868538022344e+00;
  NLX_coes1[0][  8][ 18] =-7.919436552539105e+00;
  NLX_coes1[0][  8][ 19] =-7.859886799676372e+00;
  NLX_coes1[0][  8][ 20] =-7.794614944258071e+00;
  NLX_coes1[0][  9][  0] =-8.437590770781718e+00;
  NLX_coes1[0][  9][  1] =-8.681679586190445e+00;
  NLX_coes1[0][  9][  2] =-8.922555940466898e+00;
  NLX_coes1[0][  9][  3] =-9.149704682963090e+00;
  NLX_coes1[0][  9][  4] =-9.356470792246478e+00;
  NLX_coes1[0][  9][  5] =-9.540563342728651e+00;
  NLX_coes1[0][  9][  6] =-9.702161170197263e+00;
  NLX_coes1[0][  9][  7] =-9.840237272655179e+00;
  NLX_coes1[0][  9][  8] =-9.950035732958305e+00;
  NLX_coes1[0][  9][  9] =-1.002371069490407e+01;
  NLX_coes1[0][  9][ 10] =-1.005458513825367e+01;
  NLX_coes1[0][  9][ 11] =-1.007429560462434e+01;
  NLX_coes1[0][  9][ 12] =-1.009668893534063e+01;
  NLX_coes1[0][  9][ 13] =-1.006598145707353e+01;
  NLX_coes1[0][  9][ 14] =-1.002794553267572e+01;
  NLX_coes1[0][  9][ 15] =-9.964433247393625e+00;
  NLX_coes1[0][  9][ 16] =-9.888646614017354e+00;
  NLX_coes1[0][  9][ 17] =-9.813163273587664e+00;
  NLX_coes1[0][  9][ 18] =-9.738352973741815e+00;
  NLX_coes1[0][  9][ 19] =-9.661579822858673e+00;
  NLX_coes1[0][  9][ 20] =-9.581945099435409e+00;
  NLX_coes1[0][ 10][  0] =-1.066478858668277e+01;
  NLX_coes1[0][ 10][  1] =-1.088215213639507e+01;
  NLX_coes1[0][ 10][  2] =-1.109871443529157e+01;
  NLX_coes1[0][ 10][  3] =-1.130501375748342e+01;
  NLX_coes1[0][ 10][  4] =-1.149414885532537e+01;
  NLX_coes1[0][ 10][  5] =-1.166198260446280e+01;
  NLX_coes1[0][ 10][  6] =-1.180603415989198e+01;
  NLX_coes1[0][ 10][  7] =-1.192378475121352e+01;
  NLX_coes1[0][ 10][  8] =-1.201177122836958e+01;
  NLX_coes1[0][ 10][  9] =-1.206634080386085e+01;
  NLX_coes1[0][ 10][ 10] =-1.208840047064317e+01;
  NLX_coes1[0][ 10][ 11] =-1.209585470345915e+01;
  NLX_coes1[0][ 10][ 12] =-1.210760847490486e+01;
  NLX_coes1[0][ 10][ 13] =-1.206788644070211e+01;
  NLX_coes1[0][ 10][ 14] =-1.201208277909970e+01;
  NLX_coes1[0][ 10][ 15] =-1.194092487006991e+01;
  NLX_coes1[0][ 10][ 16] =-1.185782696978222e+01;
  NLX_coes1[0][ 10][ 17] =-1.176925337836604e+01;
  NLX_coes1[0][ 10][ 18] =-1.167814693469927e+01;
  NLX_coes1[0][ 10][ 19] =-1.158487166331052e+01;
  NLX_coes1[0][ 10][ 20] =-1.148972289181857e+01;
  NLX_coes1[0][ 11][  0] =-1.315254993536761e+01;
  NLX_coes1[0][ 11][  1] =-1.333305855680984e+01;
  NLX_coes1[0][ 11][  2] =-1.351445687118442e+01;
  NLX_coes1[0][ 11][  3] =-1.368769126415385e+01;
  NLX_coes1[0][ 11][  4] =-1.384563105153165e+01;
  NLX_coes1[0][ 11][  5] =-1.398327301701787e+01;
  NLX_coes1[0][ 11][  6] =-1.409722750082156e+01;
  NLX_coes1[0][ 11][  7] =-1.418497905502781e+01;
  NLX_coes1[0][ 11][  8] =-1.424470284000702e+01;
  NLX_coes1[0][ 11][  9] =-1.427642356731311e+01;
  NLX_coes1[0][ 11][ 10] =-1.428514720439235e+01;
  NLX_coes1[0][ 11][ 11] =-1.427515455439146e+01;
  NLX_coes1[0][ 11][ 12] =-1.424649847605234e+01;
  NLX_coes1[0][ 11][ 13] =-1.419491842865267e+01;
  NLX_coes1[0][ 11][ 14] =-1.412321399138081e+01;
  NLX_coes1[0][ 11][ 15] =-1.403758044659660e+01;
  NLX_coes1[0][ 11][ 16] =-1.394140972707702e+01;
  NLX_coes1[0][ 11][ 17] =-1.383846533478680e+01;
  NLX_coes1[0][ 11][ 18] =-1.373150046586070e+01;
  NLX_coes1[0][ 11][ 19] =-1.362189314611252e+01;
  NLX_coes1[0][ 11][ 20] =-1.351055743167151e+01;
  NLX_coes1[0][ 12][  0] =-1.590087481683577e+01;
  NLX_coes1[0][ 12][  1] =-1.603351329210250e+01;
  NLX_coes1[0][ 12][  2] =-1.616823667505786e+01;
  NLX_coes1[0][ 12][  3] =-1.629602815091191e+01;
  NLX_coes1[0][ 12][  4] =-1.640965519498305e+01;
  NLX_coes1[0][ 12][  5] =-1.650397619679238e+01;
  NLX_coes1[0][ 12][  6] =-1.657570070997381e+01;
  NLX_coes1[0][ 12][  7] =-1.662298506643642e+01;
  NLX_coes1[0][ 12][  8] =-1.664531515749514e+01;
  NLX_coes1[0][ 12][  9] =-1.664383901587314e+01;
  NLX_coes1[0][ 12][ 10] =-1.662140943349829e+01;
  NLX_coes1[0][ 12][ 11] =-1.658064548249462e+01;
  NLX_coes1[0][ 12][ 12] =-1.652188552769301e+01;
  NLX_coes1[0][ 12][ 13] =-1.644574565794418e+01;
  NLX_coes1[0][ 12][ 14] =-1.635336193124308e+01;
  NLX_coes1[0][ 12][ 15] =-1.624883474755576e+01;
  NLX_coes1[0][ 12][ 16] =-1.613532506612353e+01;
  NLX_coes1[0][ 12][ 17] =-1.601555998009449e+01;
  NLX_coes1[0][ 12][ 18] =-1.589170966424519e+01;
  NLX_coes1[0][ 12][ 19] =-1.576519967666660e+01;
  NLX_coes1[0][ 12][ 20] =-1.563702383057407e+01;
  NLX_coes1[0][ 13][  0] =-1.890976323108804e+01;
  NLX_coes1[0][ 13][  1] =-1.898251324308923e+01;
  NLX_coes1[0][ 13][  2] =-1.905819354136218e+01;
  NLX_coes1[0][ 13][  3] =-1.912752670807881e+01;
  NLX_coes1[0][ 13][  4] =-1.918333507709211e+01;
  NLX_coes1[0][ 13][  5] =-1.922089421895326e+01;
  NLX_coes1[0][ 13][  6] =-1.923762935410707e+01;
  NLX_coes1[0][ 13][  7] =-1.923261766480404e+01;
  NLX_coes1[0][ 13][  8] =-1.920624396165583e+01;
  NLX_coes1[0][ 13][  9] =-1.916002218120489e+01;
  NLX_coes1[0][ 13][ 10] =-1.909609761022228e+01;
  NLX_coes1[0][ 13][ 11] =-1.901642030365235e+01;
  NLX_coes1[0][ 13][ 12] =-1.892221385357073e+01;
  NLX_coes1[0][ 13][ 13] =-1.881487771285818e+01;
  NLX_coes1[0][ 13][ 14] =-1.869594326458472e+01;
  NLX_coes1[0][ 13][ 15] =-1.856789962706966e+01;
  NLX_coes1[0][ 13][ 16] =-1.843322531600641e+01;
  NLX_coes1[0][ 13][ 17] =-1.829394423601360e+01;
  NLX_coes1[0][ 13][ 18] =-1.815159694581244e+01;
  NLX_coes1[0][ 13][ 19] =-1.800722085488004e+01;
  NLX_coes1[0][ 13][ 20] =-1.786151646965510e+01;
  NLX_coes1[0][ 14][  0] =-2.217921517812641e+01;
  NLX_coes1[0][ 14][  1] =-2.217881969896333e+01;
  NLX_coes1[0][ 14][  2] =-2.218185144562917e+01;
  NLX_coes1[0][ 14][  3] =-2.217845094836965e+01;
  NLX_coes1[0][ 14][  4] =-2.216174389559425e+01;
  NLX_coes1[0][ 14][  5] =-2.212802025325488e+01;
  NLX_coes1[0][ 14][  6] =-2.207595488767236e+01;
  NLX_coes1[0][ 14][  7] =-2.200577492299426e+01;
  NLX_coes1[0][ 14][  8] =-2.191868840411150e+01;
  NLX_coes1[0][ 14][  9] =-2.181650316716651e+01;
  NLX_coes1[0][ 14][ 10] =-2.170122300655227e+01;
  NLX_coes1[0][ 14][ 11] =-2.157467901829027e+01;
  NLX_coes1[0][ 14][ 12] =-2.143843049187800e+01;
  NLX_coes1[0][ 14][ 13] =-2.129394757653640e+01;
  NLX_coes1[0][ 14][ 14] =-2.114277457411245e+01;
  NLX_coes1[0][ 14][ 15] =-2.098661596580900e+01;
  NLX_coes1[0][ 14][ 16] =-2.082715321534867e+01;
  NLX_coes1[0][ 14][ 17] =-2.066570474941524e+01;
  NLX_coes1[0][ 14][ 18] =-2.050313107949719e+01;
  NLX_coes1[0][ 14][ 19] =-2.033985181675192e+01;
  NLX_coes1[0][ 14][ 20] =-2.017599522306268e+01;
  NLX_coes1[0][ 15][  0] =-2.570923065795447e+01;
  NLX_coes1[0][ 15][  1] =-2.562081757058206e+01;
  NLX_coes1[0][ 15][  2] =-2.553577417341786e+01;
  NLX_coes1[0][ 15][  3] =-2.544326649297537e+01;
  NLX_coes1[0][ 15][  4] =-2.533735640159702e+01;
  NLX_coes1[0][ 15][  5] =-2.521623570022685e+01;
  NLX_coes1[0][ 15][  6] =-2.508041221727735e+01;
  NLX_coes1[0][ 15][  7] =-2.493147095163970e+01;
  NLX_coes1[0][ 15][  8] =-2.477142115998655e+01;
  NLX_coes1[0][ 15][  9] =-2.460236856027859e+01;
  NLX_coes1[0][ 15][ 10] =-2.442630743571702e+01;
  NLX_coes1[0][ 15][ 11] =-2.424499407013948e+01;
  NLX_coes1[0][ 15][ 12] =-2.405994610408566e+01;
  NLX_coes1[0][ 15][ 13] =-2.387249958569033e+01;
  NLX_coes1[0][ 15][ 14] =-2.368390743277136e+01;
  NLX_coes1[0][ 15][ 15] =-2.349531943963706e+01;
  NLX_coes1[0][ 15][ 16] =-2.330769883753660e+01;
  NLX_coes1[0][ 15][ 17] =-2.312164033985891e+01;
  NLX_coes1[0][ 15][ 18] =-2.293726248505634e+01;
  NLX_coes1[0][ 15][ 19] =-2.275421014923627e+01;
  NLX_coes1[0][ 15][ 20] =-2.257179771135297e+01;
  NLX_coes1[0][ 16][  0] =-2.949980967057976e+01;
  NLX_coes1[0][ 16][  1] =-2.930634248774682e+01;
  NLX_coes1[0][ 16][  2] =-2.911499233501207e+01;
  NLX_coes1[0][ 16][  3] =-2.891362694390342e+01;
  NLX_coes1[0][ 16][  4] =-2.869901483955842e+01;
  NLX_coes1[0][ 16][  5] =-2.847253625707664e+01;
  NLX_coes1[0][ 16][  6] =-2.823702828350948e+01;
  NLX_coes1[0][ 16][  7] =-2.799542304015762e+01;
  NLX_coes1[0][ 16][  8] =-2.775033420586753e+01;
  NLX_coes1[0][ 16][  9] =-2.750398813202366e+01;
  NLX_coes1[0][ 16][ 10] =-2.725824039404524e+01;
  NLX_coes1[0][ 16][ 11] =-2.701461202788648e+01;
  NLX_coes1[0][ 16][ 12] =-2.677434351719480e+01;
  NLX_coes1[0][ 16][ 13] =-2.653844775173591e+01;
  NLX_coes1[0][ 16][ 14] =-2.630774897955595e+01;
  NLX_coes1[0][ 16][ 15] =-2.608285539099466e+01;
  NLX_coes1[0][ 16][ 16] =-2.586407644003510e+01;
  NLX_coes1[0][ 16][ 17] =-2.565129463968535e+01;
  NLX_coes1[0][ 16][ 18] =-2.544385073203210e+01;
  NLX_coes1[0][ 16][ 19] =-2.524050469220348e+01;
  NLX_coes1[0][ 16][ 20] =-2.503952561056813e+01;
  NLX_coes1[0][ 17][  0] =-3.355095221603408e+01;
  NLX_coes1[0][ 17][  1] =-3.323251726190086e+01;
  NLX_coes1[0][ 17][  2] =-3.291156858115502e+01;
  NLX_coes1[0][ 17][  3] =-3.257629842857570e+01;
  NLX_coes1[0][ 17][  4] =-3.223012805594617e+01;
  NLX_coes1[0][ 17][  5] =-3.187882648695366e+01;
  NLX_coes1[0][ 17][  6] =-3.152742092863067e+01;
  NLX_coes1[0][ 17][  7] =-3.117964240731381e+01;
  NLX_coes1[0][ 17][  8] =-3.083817844919351e+01;
  NLX_coes1[0][ 17][  9] =-3.050497884648345e+01;
  NLX_coes1[0][ 17][ 10] =-3.018147162603786e+01;
  NLX_coes1[0][ 17][ 11] =-2.986870393616621e+01;
  NLX_coes1[0][ 17][ 12] =-2.956743415438121e+01;
  NLX_coes1[0][ 17][ 13] =-2.927818668212371e+01;
  NLX_coes1[0][ 17][ 14] =-2.900126510883233e+01;
  NLX_coes1[0][ 17][ 15] =-2.873671381966760e+01;
  NLX_coes1[0][ 17][ 16] =-2.848422603653596e+01;
  NLX_coes1[0][ 17][ 17] =-2.824301123915169e+01;
  NLX_coes1[0][ 17][ 18] =-2.801164522104190e+01;
  NLX_coes1[0][ 17][ 19] =-2.778796410245490e+01;
  NLX_coes1[0][ 17][ 20] =-2.756898499768751e+01;
  NLX_coes1[0][ 18][  0] =-3.786265829459347e+01;
  NLX_coes1[0][ 18][  1] =-3.739569778865755e+01;
  NLX_coes1[0][ 18][  2] =-3.691016924287609e+01;
  NLX_coes1[0][ 18][  3] =-3.640907868903693e+01;
  NLX_coes1[0][ 18][  4] =-3.590606789775889e+01;
  NLX_coes1[0][ 18][  5] =-3.541062465129426e+01;
  NLX_coes1[0][ 18][  6] =-3.492827106947733e+01;
  NLX_coes1[0][ 18][  7] =-3.446227032822451e+01;
  NLX_coes1[0][ 18][  8] =-3.401457961668753e+01;
  NLX_coes1[0][ 18][  9] =-3.358635721718549e+01;
  NLX_coes1[0][ 18][ 10] =-3.317825526298730e+01;
  NLX_coes1[0][ 18][ 11] =-3.279059402734173e+01;
  NLX_coes1[0][ 18][ 12] =-3.242346122046980e+01;
  NLX_coes1[0][ 18][ 13] =-3.207675703557734e+01;
  NLX_coes1[0][ 18][ 14] =-3.175019014003206e+01;
  NLX_coes1[0][ 18][ 15] =-3.144322409507313e+01;
  NLX_coes1[0][ 18][ 16] =-3.115497098715918e+01;
  NLX_coes1[0][ 18][ 17] =-3.088403454984791e+01;
  NLX_coes1[0][ 18][ 18] =-3.062831367035527e+01;
  NLX_coes1[0][ 18][ 19] =-3.038476789355549e+01;
  NLX_coes1[0][ 18][ 20] =-3.014915178407302e+01;
  NLX_coes1[0][ 19][  0] =-4.243492790682240e+01;
  NLX_coes1[0][ 19][  1] =-4.178584887878506e+01;
  NLX_coes1[0][ 19][  2] =-4.107595892818600e+01;
  NLX_coes1[0][ 19][  3] =-4.037395428879682e+01;
  NLX_coes1[0][ 19][  4] =-3.969201926429103e+01;
  NLX_coes1[0][ 19][  5] =-3.903674045040726e+01;
  NLX_coes1[0][ 19][  6] =-3.841150292355005e+01;
  NLX_coes1[0][ 19][  7] =-3.781787422640885e+01;
  NLX_coes1[0][ 19][  8] =-3.725636434739444e+01;
  NLX_coes1[0][ 19][  9] =-3.672687936739394e+01;
  NLX_coes1[0][ 19][ 10] =-3.622898657508090e+01;
  NLX_coes1[0][ 19][ 11] =-3.576206405864860e+01;
  NLX_coes1[0][ 19][ 12] =-3.532537616834286e+01;
  NLX_coes1[0][ 19][ 13] =-3.491809566635794e+01;
  NLX_coes1[0][ 19][ 14] =-3.453928046506255e+01;
  NLX_coes1[0][ 19][ 15] =-3.418780578877044e+01;
  NLX_coes1[0][ 19][ 16] =-3.386224735843405e+01;
  NLX_coes1[0][ 19][ 17] =-3.356070374918912e+01;
  NLX_coes1[0][ 19][ 18] =-3.328052636088143e+01;
  NLX_coes1[0][ 19][ 19] =-3.301782294772157e+01;
  NLX_coes1[0][ 19][ 20] =-3.276820651532633e+01;
  NLX_coes1[0][ 20][  0] =-4.726776104396563e+01;
  NLX_coes1[0][ 20][  1] =-4.628743838249726e+01;
  NLX_coes1[0][ 20][  2] =-4.533705961651317e+01;
  NLX_coes1[0][ 20][  3] =-4.441882350874776e+01;
  NLX_coes1[0][ 20][  4] =-4.354554888015829e+01;
  NLX_coes1[0][ 20][  5] =-4.272085206960792e+01;
  NLX_coes1[0][ 20][  6] =-4.194532693484280e+01;
  NLX_coes1[0][ 20][  7] =-4.121822626119762e+01;
  NLX_coes1[0][ 20][  8] =-4.053818134427544e+01;
  NLX_coes1[0][ 20][  9] =-3.990355793623267e+01;
  NLX_coes1[0][ 20][ 10] =-3.931264236713842e+01;
  NLX_coes1[0][ 20][ 11] =-3.876373340294627e+01;
  NLX_coes1[0][ 20][ 12] =-3.825517530455843e+01;
  NLX_coes1[0][ 20][ 13] =-3.778534864896415e+01;
  NLX_coes1[0][ 20][ 14] =-3.735262520422621e+01;
  NLX_coes1[0][ 20][ 15] =-3.695528716922351e+01;
  NLX_coes1[0][ 20][ 16] =-3.659140693199076e+01;
  NLX_coes1[0][ 20][ 17] =-3.625867811857906e+01;
  NLX_coes1[0][ 20][ 18] =-3.595417855030386e+01;
  NLX_coes1[0][ 20][ 19] =-3.567409794879774e+01;
  NLX_coes1[0][ 20][ 20] =-3.541188047092612e+01;
  NLX_coes1[1][  0][  0] =+1.884114593404455e+01;
  NLX_coes1[1][  0][  1] =+1.886011319731888e+01;
  NLX_coes1[1][  0][  2] =+1.887646655715880e+01;
  NLX_coes1[1][  0][  3] =+1.888731039616864e+01;
  NLX_coes1[1][  0][  4] =+1.888860217990462e+01;
  NLX_coes1[1][  0][  5] =+1.887668098106202e+01;
  NLX_coes1[1][  0][  6] =+1.884898944894967e+01;
  NLX_coes1[1][  0][  7] =+1.880220660354068e+01;
  NLX_coes1[1][  0][  8] =+1.872725176592989e+01;
  NLX_coes1[1][  0][  9] =+1.861874511047867e+01;
  NLX_coes1[1][  0][ 10] =+1.849747652141621e+01;
  NLX_coes1[1][  0][ 11] =+1.839219430026501e+01;
  NLX_coes1[1][  0][ 12] =+1.832933014562602e+01;
  NLX_coes1[1][  0][ 13] =+1.829386878017340e+01;
  NLX_coes1[1][  0][ 14] =+1.826841949916327e+01;
  NLX_coes1[1][  0][ 15] =+1.824630926630307e+01;
  NLX_coes1[1][  0][ 16] =+1.824249997692328e+01;
  NLX_coes1[1][  0][ 17] =+1.825051811711851e+01;
  NLX_coes1[1][  0][ 18] =+1.825807201129585e+01;
  NLX_coes1[1][  0][ 19] =+1.826285583251509e+01;
  NLX_coes1[1][  0][ 20] =+1.826516013736141e+01;
  NLX_coes1[1][  1][  0] =+1.216721220803565e+01;
  NLX_coes1[1][  1][  1] =+1.218831048887721e+01;
  NLX_coes1[1][  1][  2] =+1.220906607076571e+01;
  NLX_coes1[1][  1][  3] =+1.222720303490713e+01;
  NLX_coes1[1][  1][  4] =+1.223767475563163e+01;
  NLX_coes1[1][  1][  5] =+1.223630143554452e+01;
  NLX_coes1[1][  1][  6] =+1.222131524321409e+01;
  NLX_coes1[1][  1][  7] =+1.219187457237885e+01;
  NLX_coes1[1][  1][  8] =+1.214229811415991e+01;
  NLX_coes1[1][  1][  9] =+1.203751944072572e+01;
  NLX_coes1[1][  1][ 10] =+1.191178490495084e+01;
  NLX_coes1[1][  1][ 11] =+1.179429842914382e+01;
  NLX_coes1[1][  1][ 12] =+1.173187455179880e+01;
  NLX_coes1[1][  1][ 13] =+1.171966457408049e+01;
  NLX_coes1[1][  1][ 14] =+1.167948668015401e+01;
  NLX_coes1[1][  1][ 15] =+1.164526531704778e+01;
  NLX_coes1[1][  1][ 16] =+1.165020383217088e+01;
  NLX_coes1[1][  1][ 17] =+1.165923397816736e+01;
  NLX_coes1[1][  1][ 18] =+1.166543295651767e+01;
  NLX_coes1[1][  1][ 19] =+1.166786255432413e+01;
  NLX_coes1[1][  1][ 20] =+1.166840245431215e+01;
  NLX_coes1[1][  2][  0] =+5.488668835267417e+00;
  NLX_coes1[1][  2][  1] =+5.510696940131462e+00;
  NLX_coes1[1][  2][  2] =+5.534877464840122e+00;
  NLX_coes1[1][  2][  3] =+5.560620621010654e+00;
  NLX_coes1[1][  2][  4] =+5.581622096671373e+00;
  NLX_coes1[1][  2][  5] =+5.589335377644639e+00;
  NLX_coes1[1][  2][  6] =+5.581576830446403e+00;
  NLX_coes1[1][  2][  7] =+5.571843016279022e+00;
  NLX_coes1[1][  2][  8] =+5.562177849451072e+00;
  NLX_coes1[1][  2][  9] =+5.498310355865534e+00;
  NLX_coes1[1][  2][ 10] =+5.338784885602960e+00;
  NLX_coes1[1][  2][ 11] =+5.198536195813341e+00;
  NLX_coes1[1][  2][ 12] =+5.119184163319123e+00;
  NLX_coes1[1][  2][ 13] =+5.184828090142431e+00;
  NLX_coes1[1][  2][ 14] =+5.073152313467173e+00;
  NLX_coes1[1][  2][ 15] =+5.020821075571489e+00;
  NLX_coes1[1][  2][ 16] =+5.070638231869875e+00;
  NLX_coes1[1][  2][ 17] =+5.072912347245198e+00;
  NLX_coes1[1][  2][ 18] =+5.070321523976943e+00;
  NLX_coes1[1][  2][ 19] =+5.069592232868152e+00;
  NLX_coes1[1][  2][ 20] =+5.068155609490282e+00;
  NLX_coes1[1][  3][  0] =-1.194946967500794e+00;
  NLX_coes1[1][  3][  1] =-1.173647953118437e+00;
  NLX_coes1[1][  3][  2] =-1.149672326727660e+00;
  NLX_coes1[1][  3][  3] =-1.119352084577412e+00;
  NLX_coes1[1][  3][  4] =-1.081438346175635e+00;
  NLX_coes1[1][  3][  5] =-1.066133622001246e+00;
  NLX_coes1[1][  3][  6] =-1.070328472026742e+00;
  NLX_coes1[1][  3][  7] =-1.067794339398233e+00;
  NLX_coes1[1][  3][  8] =-1.073754459949557e+00;
  NLX_coes1[1][  3][  9] =-9.557847708617216e-01;
  NLX_coes1[1][  3][ 10] =-1.195038597703773e+00;
  NLX_coes1[1][  3][ 11] =-1.337962016543333e+00;
  NLX_coes1[1][  3][ 12] =-1.403969447353127e+00;
  NLX_coes1[1][  3][ 13] =-1.441356531781619e+00;
  NLX_coes1[1][  3][ 14] =-1.507934166394218e+00;
  NLX_coes1[1][  3][ 15] =-1.517313555983188e+00;
  NLX_coes1[1][  3][ 16] =-1.519546787219761e+00;
  NLX_coes1[1][  3][ 17] =-1.524274551903172e+00;
  NLX_coes1[1][  3][ 18] =-1.528129088030312e+00;
  NLX_coes1[1][  3][ 19] =-1.533943060700504e+00;
  NLX_coes1[1][  3][ 20] =-1.538379453843240e+00;
  NLX_coes1[1][  4][  0] =-7.881266651283712e+00;
  NLX_coes1[1][  4][  1] =-7.861697680362284e+00;
  NLX_coes1[1][  4][  2] =-7.841960129686701e+00;
  NLX_coes1[1][  4][  3] =-7.820765871446441e+00;
  NLX_coes1[1][  4][  4] =-7.782250337378083e+00;
  NLX_coes1[1][  4][  5] =-7.759229061697900e+00;
  NLX_coes1[1][  4][  6] =-7.748126132865737e+00;
  NLX_coes1[1][  4][  7] =-7.753022631150074e+00;
  NLX_coes1[1][  4][  8] =-7.721535457422921e+00;
  NLX_coes1[1][  4][  9] =-7.632187787005062e+00;
  NLX_coes1[1][  4][ 10] =-7.724627829304174e+00;
  NLX_coes1[1][  4][ 11] =-7.772506706458556e+00;
  NLX_coes1[1][  4][ 12] =-7.875757318453171e+00;
  NLX_coes1[1][  4][ 13] =-7.997980906305225e+00;
  NLX_coes1[1][  4][ 14] =-8.118488805268873e+00;
  NLX_coes1[1][  4][ 15] =-8.117625355207617e+00;
  NLX_coes1[1][  4][ 16] =-8.123276211962613e+00;
  NLX_coes1[1][  4][ 17] =-8.132052085608116e+00;
  NLX_coes1[1][  4][ 18] =-8.140979398813874e+00;
  NLX_coes1[1][  4][ 19] =-8.148006975790079e+00;
  NLX_coes1[1][  4][ 20] =-8.153735901953574e+00;
  NLX_coes1[1][  5][  0] =-1.456739587123607e+01;
  NLX_coes1[1][  5][  1] =-1.454923522673234e+01;
  NLX_coes1[1][  5][  2] =-1.453096330168927e+01;
  NLX_coes1[1][  5][  3] =-1.451034761815917e+01;
  NLX_coes1[1][  5][  4] =-1.448320963342294e+01;
  NLX_coes1[1][  5][  5] =-1.446085486656574e+01;
  NLX_coes1[1][  5][  6] =-1.444322462220058e+01;
  NLX_coes1[1][  5][  7] =-1.444259608777203e+01;
  NLX_coes1[1][  5][  8] =-1.444649039081091e+01;
  NLX_coes1[1][  5][  9] =-1.443607070929912e+01;
  NLX_coes1[1][  5][ 10] =-1.446668357391376e+01;
  NLX_coes1[1][  5][ 11] =-1.445804228745245e+01;
  NLX_coes1[1][  5][ 12] =-1.451561072589537e+01;
  NLX_coes1[1][  5][ 13] =-1.460741726587360e+01;
  NLX_coes1[1][  5][ 14] =-1.468739077560396e+01;
  NLX_coes1[1][  5][ 15] =-1.473894106929978e+01;
  NLX_coes1[1][  5][ 16] =-1.474794139077394e+01;
  NLX_coes1[1][  5][ 17] =-1.475581538590154e+01;
  NLX_coes1[1][  5][ 18] =-1.476487570397165e+01;
  NLX_coes1[1][  5][ 19] =-1.477273720105548e+01;
  NLX_coes1[1][  5][ 20] =-1.477922012472442e+01;
  NLX_coes1[1][  6][  0] =-2.125111534041252e+01;
  NLX_coes1[1][  6][  1] =-2.123369912783043e+01;
  NLX_coes1[1][  6][  2] =-2.121588460403947e+01;
  NLX_coes1[1][  6][  3] =-2.119624566915969e+01;
  NLX_coes1[1][  6][  4] =-2.117519162291074e+01;
  NLX_coes1[1][  6][  5] =-2.115618398733229e+01;
  NLX_coes1[1][  6][  6] =-2.114376680310934e+01;
  NLX_coes1[1][  6][  7] =-2.113953646679250e+01;
  NLX_coes1[1][  6][  8] =-2.113649722307549e+01;
  NLX_coes1[1][  6][  9] =-2.113343117661713e+01;
  NLX_coes1[1][  6][ 10] =-2.116248303090035e+01;
  NLX_coes1[1][  6][ 11] =-2.120109004041277e+01;
  NLX_coes1[1][  6][ 12] =-2.128329761880709e+01;
  NLX_coes1[1][  6][ 13] =-2.134493197312815e+01;
  NLX_coes1[1][  6][ 14] =-2.135854885712978e+01;
  NLX_coes1[1][  6][ 15] =-2.135019019944447e+01;
  NLX_coes1[1][  6][ 16] =-2.137765365144578e+01;
  NLX_coes1[1][  6][ 17] =-2.139167947566316e+01;
  NLX_coes1[1][  6][ 18] =-2.140069503066221e+01;
  NLX_coes1[1][  6][ 19] =-2.140804256931357e+01;
  NLX_coes1[1][  6][ 20] =-2.141439878503926e+01;
  NLX_coes1[1][  7][  0] =-2.793196234962949e+01;
  NLX_coes1[1][  7][  1] =-2.791521719313192e+01;
  NLX_coes1[1][  7][  2] =-2.789810100101905e+01;
  NLX_coes1[1][  7][  3] =-2.788030170531619e+01;
  NLX_coes1[1][  7][  4] =-2.786273168909928e+01;
  NLX_coes1[1][  7][  5] =-2.784722183692664e+01;
  NLX_coes1[1][  7][  6] =-2.783646145767716e+01;
  NLX_coes1[1][  7][  7] =-2.783084206413123e+01;
  NLX_coes1[1][  7][  8] =-2.782892311249816e+01;
  NLX_coes1[1][  7][  9] =-2.782990299395482e+01;
  NLX_coes1[1][  7][ 10] =-2.782189173969780e+01;
  NLX_coes1[1][  7][ 11] =-2.780765934430438e+01;
  NLX_coes1[1][  7][ 12] =-2.794422747259099e+01;
  NLX_coes1[1][  7][ 13] =-2.804940609425171e+01;
  NLX_coes1[1][  7][ 14] =-2.805146404870719e+01;
  NLX_coes1[1][  7][ 15] =-2.803328948328102e+01;
  NLX_coes1[1][  7][ 16] =-2.803252906641605e+01;
  NLX_coes1[1][  7][ 17] =-2.803906299808994e+01;
  NLX_coes1[1][  7][ 18] =-2.804598338774829e+01;
  NLX_coes1[1][  7][ 19] =-2.805225767877736e+01;
  NLX_coes1[1][  7][ 20] =-2.805805559471333e+01;
  NLX_coes1[1][  8][  0] =-3.461033936629070e+01;
  NLX_coes1[1][  8][  1] =-3.459437411650464e+01;
  NLX_coes1[1][  8][  2] =-3.457825667428708e+01;
  NLX_coes1[1][  8][  3] =-3.456215906013681e+01;
  NLX_coes1[1][  8][  4] =-3.454683182766890e+01;
  NLX_coes1[1][  8][  5] =-3.453350694340268e+01;
  NLX_coes1[1][  8][  6] =-3.452353430106344e+01;
  NLX_coes1[1][  8][  7] =-3.451740287131334e+01;
  NLX_coes1[1][  8][  8] =-3.451552397667993e+01;
  NLX_coes1[1][  8][  9] =-3.452027608771527e+01;
  NLX_coes1[1][  8][ 10] =-3.452739112757099e+01;
  NLX_coes1[1][  8][ 11] =-3.451821600557443e+01;
  NLX_coes1[1][  8][ 12] =-3.459776115004585e+01;
  NLX_coes1[1][  8][ 13] =-3.465194519454307e+01;
  NLX_coes1[1][  8][ 14] =-3.469311955824750e+01;
  NLX_coes1[1][  8][ 15] =-3.469582846478175e+01;
  NLX_coes1[1][  8][ 16] =-3.469337347047391e+01;
  NLX_coes1[1][  8][ 17] =-3.469492809633363e+01;
  NLX_coes1[1][  8][ 18] =-3.469869372388521e+01;
  NLX_coes1[1][  8][ 19] =-3.470334571287787e+01;
  NLX_coes1[1][  8][ 20] =-3.470835436857932e+01;
  NLX_coes1[1][  9][  0] =-4.128676829228399e+01;
  NLX_coes1[1][  9][  1] =-4.127166076598556e+01;
  NLX_coes1[1][  9][  2] =-4.125660240488767e+01;
  NLX_coes1[1][  9][  3] =-4.124188483322270e+01;
  NLX_coes1[1][  9][  4] =-4.122809328610764e+01;
  NLX_coes1[1][  9][  5] =-4.121605111874786e+01;
  NLX_coes1[1][  9][  6] =-4.120654705934704e+01;
  NLX_coes1[1][  9][  7] =-4.120008853277781e+01;
  NLX_coes1[1][  9][  8] =-4.119683976213273e+01;
  NLX_coes1[1][  9][  9] =-4.119587661877392e+01;
  NLX_coes1[1][  9][ 10] =-4.119640181346742e+01;
  NLX_coes1[1][  9][ 11] =-4.121866085969167e+01;
  NLX_coes1[1][  9][ 12] =-4.126661614267024e+01;
  NLX_coes1[1][  9][ 13] =-4.129727749825310e+01;
  NLX_coes1[1][  9][ 14] =-4.132950252482315e+01;
  NLX_coes1[1][  9][ 15] =-4.134513578792760e+01;
  NLX_coes1[1][  9][ 16] =-4.134980408787109e+01;
  NLX_coes1[1][  9][ 17] =-4.135225887176932e+01;
  NLX_coes1[1][  9][ 18] =-4.135520923386378e+01;
  NLX_coes1[1][  9][ 19] =-4.135897050319871e+01;
  NLX_coes1[1][  9][ 20] =-4.136337983328539e+01;
  NLX_coes1[1][ 10][  0] =-4.796170466160416e+01;
  NLX_coes1[1][ 10][  1] =-4.794746898613608e+01;
  NLX_coes1[1][ 10][  2] =-4.793340405047687e+01;
  NLX_coes1[1][ 10][  3] =-4.791980526252210e+01;
  NLX_coes1[1][ 10][  4] =-4.790713249705070e+01;
  NLX_coes1[1][ 10][  5] =-4.789595123131998e+01;
  NLX_coes1[1][ 10][  6] =-4.788678252349746e+01;
  NLX_coes1[1][ 10][  7] =-4.787998606166156e+01;
  NLX_coes1[1][ 10][  8] =-4.787570438057347e+01;
  NLX_coes1[1][ 10][  9] =-4.787395710484421e+01;
  NLX_coes1[1][ 10][ 10] =-4.787678648462946e+01;
  NLX_coes1[1][ 10][ 11] =-4.789462686716472e+01;
  NLX_coes1[1][ 10][ 12] =-4.793569113479946e+01;
  NLX_coes1[1][ 10][ 13] =-4.795939769740680e+01;
  NLX_coes1[1][ 10][ 14] =-4.798131837235199e+01;
  NLX_coes1[1][ 10][ 15] =-4.799674145600012e+01;
  NLX_coes1[1][ 10][ 16] =-4.800503488215358e+01;
  NLX_coes1[1][ 10][ 17] =-4.800972126382784e+01;
  NLX_coes1[1][ 10][ 18] =-4.801341382711713e+01;
  NLX_coes1[1][ 10][ 19] =-4.801721673768419e+01;
  NLX_coes1[1][ 10][ 20] =-4.802144589819756e+01;
  NLX_coes1[1][ 11][  0] =-5.463552826331228e+01;
  NLX_coes1[1][ 11][  1] =-5.462213352190982e+01;
  NLX_coes1[1][ 11][  2] =-5.460896985620791e+01;
  NLX_coes1[1][ 11][  3] =-5.459631488759719e+01;
  NLX_coes1[1][ 11][  4] =-5.458454496259137e+01;
  NLX_coes1[1][ 11][  5] =-5.457409369681569e+01;
  NLX_coes1[1][ 11][  6] =-5.456539123286671e+01;
  NLX_coes1[1][ 11][  7] =-5.455884871538511e+01;
  NLX_coes1[1][ 11][  8] =-5.455498710317366e+01;
  NLX_coes1[1][ 11][  9] =-5.455504123484042e+01;
  NLX_coes1[1][ 11][ 10] =-5.456244575883656e+01;
  NLX_coes1[1][ 11][ 11] =-5.457866740218399e+01;
  NLX_coes1[1][ 11][ 12] =-5.460082932464611e+01;
  NLX_coes1[1][ 11][ 13] =-5.462202860468796e+01;
  NLX_coes1[1][ 11][ 14] =-5.463939554221504e+01;
  NLX_coes1[1][ 11][ 15] =-5.465286666728613e+01;
  NLX_coes1[1][ 11][ 16] =-5.466202651719895e+01;
  NLX_coes1[1][ 11][ 17] =-5.466813590803424e+01;
  NLX_coes1[1][ 11][ 18] =-5.467280169622065e+01;
  NLX_coes1[1][ 11][ 19] =-5.467707363090430e+01;
  NLX_coes1[1][ 11][ 20] =-5.468143303559151e+01;
  NLX_coes1[1][ 12][  0] =-6.130855409124563e+01;
  NLX_coes1[1][ 12][  1] =-6.129594556445993e+01;
  NLX_coes1[1][ 12][  2] =-6.128359770685739e+01;
  NLX_coes1[1][ 12][  3] =-6.127177455676889e+01;
  NLX_coes1[1][ 12][  4] =-6.126080874043453e+01;
  NLX_coes1[1][ 12][  5] =-6.125108470437023e+01;
  NLX_coes1[1][ 12][  6] =-6.124303052157222e+01;
  NLX_coes1[1][ 12][  7] =-6.123715730362002e+01;
  NLX_coes1[1][ 12][  8] =-6.123419163016442e+01;
  NLX_coes1[1][ 12][  9] =-6.123528790325406e+01;
  NLX_coes1[1][ 12][ 10] =-6.124185805631713e+01;
  NLX_coes1[1][ 12][ 11] =-6.125413882221461e+01;
  NLX_coes1[1][ 12][ 12] =-6.126988869350141e+01;
  NLX_coes1[1][ 12][ 13] =-6.128626480062612e+01;
  NLX_coes1[1][ 12][ 14] =-6.130054519929423e+01;
  NLX_coes1[1][ 12][ 15] =-6.131227170433154e+01;
  NLX_coes1[1][ 12][ 16] =-6.132125644969364e+01;
  NLX_coes1[1][ 12][ 17] =-6.132799442717715e+01;
  NLX_coes1[1][ 12][ 18] =-6.133336671186836e+01;
  NLX_coes1[1][ 12][ 19] =-6.133813508243038e+01;
  NLX_coes1[1][ 12][ 20] =-6.134275151400485e+01;
  NLX_coes1[1][ 13][  0] =-6.798103861118537e+01;
  NLX_coes1[1][ 13][  1] =-6.796914896876748e+01;
  NLX_coes1[1][ 13][  2] =-6.795753804686278e+01;
  NLX_coes1[1][ 13][  3] =-6.794646321311497e+01;
  NLX_coes1[1][ 13][  4] =-6.793623746263134e+01;
  NLX_coes1[1][ 13][  5] =-6.792722721528928e+01;
  NLX_coes1[1][ 13][  6] =-6.791985950362506e+01;
  NLX_coes1[1][ 13][  7] =-6.791465031394688e+01;
  NLX_coes1[1][ 13][  8] =-6.791224906269106e+01;
  NLX_coes1[1][ 13][  9] =-6.791342528069532e+01;
  NLX_coes1[1][ 13][ 10] =-6.791872257085475e+01;
  NLX_coes1[1][ 13][ 11] =-6.792788712367816e+01;
  NLX_coes1[1][ 13][ 12] =-6.793957666407982e+01;
  NLX_coes1[1][ 13][ 13] =-6.795212959063232e+01;
  NLX_coes1[1][ 13][ 14] =-6.796388498198095e+01;
  NLX_coes1[1][ 13][ 15] =-6.797407584699747e+01;
  NLX_coes1[1][ 13][ 16] =-6.798250551856918e+01;
  NLX_coes1[1][ 13][ 17] =-6.798936490624365e+01;
  NLX_coes1[1][ 13][ 18] =-6.799511524985560e+01;
  NLX_coes1[1][ 13][ 19] =-6.800024754282011e+01;
  NLX_coes1[1][ 13][ 20] =-6.800511921272171e+01;
  NLX_coes1[1][ 14][  0] =-7.465318414187983e+01;
  NLX_coes1[1][ 14][  1] =-7.464193714692469e+01;
  NLX_coes1[1][ 14][  2] =-7.463098227443352e+01;
  NLX_coes1[1][ 14][  3] =-7.462057297767991e+01;
  NLX_coes1[1][ 14][  4] =-7.461100985177006e+01;
  NLX_coes1[1][ 14][  5] =-7.460264144067001e+01;
  NLX_coes1[1][ 14][  6] =-7.459586760821864e+01;
  NLX_coes1[1][ 14][  7] =-7.459114363841157e+01;
  NLX_coes1[1][ 14][  8] =-7.458896592292245e+01;
  NLX_coes1[1][ 14][  9] =-7.458978380917414e+01;
  NLX_coes1[1][ 14][ 10] =-7.459377535973768e+01;
  NLX_coes1[1][ 14][ 11] =-7.460061653997184e+01;
  NLX_coes1[1][ 14][ 12] =-7.460946866105526e+01;
  NLX_coes1[1][ 14][ 13] =-7.461923301623361e+01;
  NLX_coes1[1][ 14][ 14] =-7.462887037597075e+01;
  NLX_coes1[1][ 14][ 15] =-7.463769289460738e+01;
  NLX_coes1[1][ 14][ 16] =-7.464543617530894e+01;
  NLX_coes1[1][ 14][ 17] =-7.465212901496874e+01;
  NLX_coes1[1][ 14][ 18] =-7.465799599630482e+01;
  NLX_coes1[1][ 14][ 19] =-7.466333102752090e+01;
  NLX_coes1[1][ 14][ 20] =-7.466839171666221e+01;
  NLX_coes1[1][ 15][  0] =-8.132514523806266e+01;
  NLX_coes1[1][ 15][  1] =-8.131445695915514e+01;
  NLX_coes1[1][ 15][  2] =-8.130406980642650e+01;
  NLX_coes1[1][ 15][  3] =-8.129423237817059e+01;
  NLX_coes1[1][ 15][  4] =-8.128523165259877e+01;
  NLX_coes1[1][ 15][  5] =-8.127739121785226e+01;
  NLX_coes1[1][ 15][  6] =-8.127106650678492e+01;
  NLX_coes1[1][ 15][  7] =-8.126663140102620e+01;
  NLX_coes1[1][ 15][  8] =-8.126444141815692e+01;
  NLX_coes1[1][ 15][  9] =-8.126475104855207e+01;
  NLX_coes1[1][ 15][ 10] =-8.126759191407638e+01;
  NLX_coes1[1][ 15][ 11] =-8.127268152595713e+01;
  NLX_coes1[1][ 15][ 12] =-8.127944781086684e+01;
  NLX_coes1[1][ 15][ 13] =-8.128715039483572e+01;
  NLX_coes1[1][ 15][ 14] =-8.129507934827348e+01;
  NLX_coes1[1][ 15][ 15] =-8.130269888780002e+01;
  NLX_coes1[1][ 15][ 16] =-8.130972819562994e+01;
  NLX_coes1[1][ 15][ 17] =-8.131610326458480e+01;
  NLX_coes1[1][ 15][ 18] =-8.132190999218265e+01;
  NLX_coes1[1][ 15][ 19] =-8.132730955328744e+01;
  NLX_coes1[1][ 15][ 20] =-8.133247377943881e+01;
  NLX_coes1[1][ 16][  0] =-8.799703727115656e+01;
  NLX_coes1[1][ 16][  1] =-8.798681773733919e+01;
  NLX_coes1[1][ 16][  2] =-8.797690269019671e+01;
  NLX_coes1[1][ 16][  3] =-8.796753370404572e+01;
  NLX_coes1[1][ 16][  4] =-8.795898140219192e+01;
  NLX_coes1[1][ 16][  5] =-8.795154009503494e+01;
  NLX_coes1[1][ 16][  6] =-8.794551682725522e+01;
  NLX_coes1[1][ 16][  7] =-8.794121045575409e+01;
  NLX_coes1[1][ 16][  8] =-8.793887352795940e+01;
  NLX_coes1[1][ 16][  9] =-8.793865327652794e+01;
  NLX_coes1[1][ 16][ 10] =-8.794052725601000e+01;
  NLX_coes1[1][ 16][ 11] =-8.794426793304997e+01;
  NLX_coes1[1][ 16][ 12] =-8.794946548724887e+01;
  NLX_coes1[1][ 16][ 13] =-8.795560399208118e+01;
  NLX_coes1[1][ 16][ 14] =-8.796217288097105e+01;
  NLX_coes1[1][ 16][ 15] =-8.796875927313900e+01;
  NLX_coes1[1][ 16][ 16] =-8.797510198776160e+01;
  NLX_coes1[1][ 16][ 17] =-8.798109042181511e+01;
  NLX_coes1[1][ 16][ 18] =-8.798672877940386e+01;
  NLX_coes1[1][ 16][ 19] =-8.799209077465444e+01;
  NLX_coes1[1][ 16][ 20] =-8.799727922349246e+01;
  NLX_coes1[1][ 17][  0] =-9.466894507496184e+01;
  NLX_coes1[1][ 17][  1] =-9.465910093847450e+01;
  NLX_coes1[1][ 17][  2] =-9.464955895192028e+01;
  NLX_coes1[1][ 17][  3] =-9.464055186994356e+01;
  NLX_coes1[1][ 17][  4] =-9.463233253449768e+01;
  NLX_coes1[1][ 17][  5] =-9.462516600404550e+01;
  NLX_coes1[1][ 17][  6] =-9.461931557519125e+01;
  NLX_coes1[1][ 17][  7] =-9.461502106941423e+01;
  NLX_coes1[1][ 17][  8] =-9.461246710352547e+01;
  NLX_coes1[1][ 17][  9] =-9.461174411019736e+01;
  NLX_coes1[1][ 17][ 10] =-9.461281382689398e+01;
  NLX_coes1[1][ 17][ 11] =-9.461549610442877e+01;
  NLX_coes1[1][ 17][ 12] =-9.461948808965622e+01;
  NLX_coes1[1][ 17][ 13] =-9.462441472867935e+01;
  NLX_coes1[1][ 17][ 14] =-9.462989564817809e+01;
  NLX_coes1[1][ 17][ 15] =-9.463560640147057e+01;
  NLX_coes1[1][ 17][ 16] =-9.464131634689407e+01;
  NLX_coes1[1][ 17][ 17] =-9.464689731990626e+01;
  NLX_coes1[1][ 17][ 18] =-9.465230741605166e+01;
  NLX_coes1[1][ 17][ 19] =-9.465756373389804e+01;
  NLX_coes1[1][ 17][ 20] =-9.466271485228127e+01;
  NLX_coes1[1][ 18][  0] =-1.013409292654121e+02;
  NLX_coes1[1][ 18][  1] =-1.013313668532916e+02;
  NLX_coes1[1][ 18][  2] =-1.013221011782351e+02;
  NLX_coes1[1][ 18][  3] =-1.013133542271910e+02;
  NLX_coes1[1][ 18][  4] =-1.013053613565801e+02;
  NLX_coes1[1][ 18][  5] =-1.012983608271554e+02;
  NLX_coes1[1][ 18][  6] =-1.012925799227845e+02;
  NLX_coes1[1][ 18][  7] =-1.012882155305087e+02;
  NLX_coes1[1][ 18][  8] =-1.012854096247666e+02;
  NLX_coes1[1][ 18][  9] =-1.012842235256576e+02;
  NLX_coes1[1][ 18][ 10] =-1.012846186544555e+02;
  NLX_coes1[1][ 18][ 11] =-1.012864524635004e+02;
  NLX_coes1[1][ 18][ 12] =-1.012894944313949e+02;
  NLX_coes1[1][ 18][ 13] =-1.012934606680882e+02;
  NLX_coes1[1][ 18][ 14] =-1.012980579292967e+02;
  NLX_coes1[1][ 18][ 15] =-1.013030252862590e+02;
  NLX_coes1[1][ 18][ 16] =-1.013081623269681e+02;
  NLX_coes1[1][ 18][ 17] =-1.013133394743340e+02;
  NLX_coes1[1][ 18][ 18] =-1.013184915456603e+02;
  NLX_coes1[1][ 18][ 19] =-1.013236006405320e+02;
  NLX_coes1[1][ 18][ 20] =-1.013286746718864e+02;
  NLX_coes1[1][ 19][  0] =-1.080130290825675e+02;
  NLX_coes1[1][ 19][  1] =-1.080036556548338e+02;
  NLX_coes1[1][ 19][  2] =-1.079945798103039e+02;
  NLX_coes1[1][ 19][  3] =-1.079860044679087e+02;
  NLX_coes1[1][ 19][  4] =-1.079781491367986e+02;
  NLX_coes1[1][ 19][  5] =-1.079712278071565e+02;
  NLX_coes1[1][ 19][  6] =-1.079654387760255e+02;
  NLX_coes1[1][ 19][  7] =-1.079609485822818e+02;
  NLX_coes1[1][ 19][  8] =-1.079578731003017e+02;
  NLX_coes1[1][ 19][  9] =-1.079562596790597e+02;
  NLX_coes1[1][ 19][ 10] =-1.079560755621053e+02;
  NLX_coes1[1][ 19][ 11] =-1.079572075056356e+02;
  NLX_coes1[1][ 19][ 12] =-1.079594749680246e+02;
  NLX_coes1[1][ 19][ 13] =-1.079626552310790e+02;
  NLX_coes1[1][ 19][ 14] =-1.079665147422069e+02;
  NLX_coes1[1][ 19][ 15] =-1.079708391579088e+02;
  NLX_coes1[1][ 19][ 16] =-1.079754551388558e+02;
  NLX_coes1[1][ 19][ 17] =-1.079802401397256e+02;
  NLX_coes1[1][ 19][ 18] =-1.079851199910519e+02;
  NLX_coes1[1][ 19][ 19] =-1.079900562633433e+02;
  NLX_coes1[1][ 19][ 20] =-1.079950383102428e+02;
  NLX_coes1[1][ 20][  0] =-1.146852476454943e+02;
  NLX_coes1[1][ 20][  1] =-1.146759948227045e+02;
  NLX_coes1[1][ 20][  2] =-1.146670312530021e+02;
  NLX_coes1[1][ 20][  3] =-1.146585630227820e+02;
  NLX_coes1[1][ 20][  4] =-1.146507834331255e+02;
  NLX_coes1[1][ 20][  5] =-1.146438821075359e+02;
  NLX_coes1[1][ 20][  6] =-1.146380331740781e+02;
  NLX_coes1[1][ 20][  7] =-1.146333812383867e+02;
  NLX_coes1[1][ 20][  8] =-1.146300261463836e+02;
  NLX_coes1[1][ 20][  9] =-1.146280095642241e+02;
  NLX_coes1[1][ 20][ 10] =-1.146273070762915e+02;
  NLX_coes1[1][ 20][ 11] =-1.146278289898198e+02;
  NLX_coes1[1][ 20][ 12] =-1.146294311631125e+02;
  NLX_coes1[1][ 20][ 13] =-1.146319343866989e+02;
  NLX_coes1[1][ 20][ 14] =-1.146351482991746e+02;
  NLX_coes1[1][ 20][ 15] =-1.146388944783418e+02;
  NLX_coes1[1][ 20][ 16] =-1.146430238325803e+02;
  NLX_coes1[1][ 20][ 17] =-1.146474253107063e+02;
  NLX_coes1[1][ 20][ 18] =-1.146520254548187e+02;
  NLX_coes1[1][ 20][ 19] =-1.146567807169139e+02;
  NLX_coes1[1][ 20][ 20] =-1.146616536946992e+02;
  NLX_coes1[2][  0][  0] =+1.453880527914586e+01;
  NLX_coes1[2][  0][  1] =+1.450370487730570e+01;
  NLX_coes1[2][  0][  2] =+1.447403250981582e+01;
  NLX_coes1[2][  0][  3] =+1.445440954083104e+01;
  NLX_coes1[2][  0][  4] =+1.444972985220448e+01;
  NLX_coes1[2][  0][  5] =+1.446140792620486e+01;
  NLX_coes1[2][  0][  6] =+1.448343109631579e+01;
  NLX_coes1[2][  0][  7] =+1.450185239203956e+01;
  NLX_coes1[2][  0][  8] =+1.450402008936817e+01;
  NLX_coes1[2][  0][  9] =+1.448897372453892e+01;
  NLX_coes1[2][  0][ 10] =+1.445761068952121e+01;
  NLX_coes1[2][  0][ 11] =+1.440440893761840e+01;
  NLX_coes1[2][  0][ 12] =+1.432995163982218e+01;
  NLX_coes1[2][  0][ 13] =+1.425050395204504e+01;
  NLX_coes1[2][  0][ 14] =+1.418682070266435e+01;
  NLX_coes1[2][  0][ 15] =+1.412506321311878e+01;
  NLX_coes1[2][  0][ 16] =+1.403940488076485e+01;
  NLX_coes1[2][  0][ 17] =+1.394131522072646e+01;
  NLX_coes1[2][  0][ 18] =+1.384809923702189e+01;
  NLX_coes1[2][  0][ 19] =+1.376073958334179e+01;
  NLX_coes1[2][  0][ 20] =+1.367728536932423e+01;
  NLX_coes1[2][  1][  0] =+1.092554871441412e+01;
  NLX_coes1[2][  1][  1] =+1.088667463315553e+01;
  NLX_coes1[2][  1][  2] =+1.085037255436631e+01;
  NLX_coes1[2][  1][  3] =+1.082105707202780e+01;
  NLX_coes1[2][  1][  4] =+1.080705464186760e+01;
  NLX_coes1[2][  1][  5] =+1.081320537951381e+01;
  NLX_coes1[2][  1][  6] =+1.083564959547884e+01;
  NLX_coes1[2][  1][  7] =+1.085712998604922e+01;
  NLX_coes1[2][  1][  8] =+1.085036473533053e+01;
  NLX_coes1[2][  1][  9] =+1.082899060275909e+01;
  NLX_coes1[2][  1][ 10] =+1.078293463028312e+01;
  NLX_coes1[2][  1][ 11] =+1.074064069772368e+01;
  NLX_coes1[2][  1][ 12] =+1.065807716737670e+01;
  NLX_coes1[2][  1][ 13] =+1.056643476802190e+01;
  NLX_coes1[2][  1][ 14] =+1.052130308761459e+01;
  NLX_coes1[2][  1][ 15] =+1.048286573136424e+01;
  NLX_coes1[2][  1][ 16] =+1.037964863352677e+01;
  NLX_coes1[2][  1][ 17] =+1.028718067306382e+01;
  NLX_coes1[2][  1][ 18] =+1.019702021142430e+01;
  NLX_coes1[2][  1][ 19] =+1.011522430380676e+01;
  NLX_coes1[2][  1][ 20] =+1.003482619387798e+01;
  NLX_coes1[2][  2][  0] =+7.319920145272881e+00;
  NLX_coes1[2][  2][  1] =+7.278057436466526e+00;
  NLX_coes1[2][  2][  2] =+7.234456644827074e+00;
  NLX_coes1[2][  2][  3] =+7.191056034935396e+00;
  NLX_coes1[2][  2][  4] =+7.159778203911475e+00;
  NLX_coes1[2][  2][  5] =+7.155483511178137e+00;
  NLX_coes1[2][  2][  6] =+7.183910361591981e+00;
  NLX_coes1[2][  2][  7] =+7.219400273074712e+00;
  NLX_coes1[2][  2][  8] =+7.223781742090873e+00;
  NLX_coes1[2][  2][  9] =+7.152461616243009e+00;
  NLX_coes1[2][  2][ 10] =+7.079199405747102e+00;
  NLX_coes1[2][  2][ 11] =+7.073173315952825e+00;
  NLX_coes1[2][  2][ 12] =+6.973952020667716e+00;
  NLX_coes1[2][  2][ 13] =+6.878456508791968e+00;
  NLX_coes1[2][  2][ 14] =+6.882750780030044e+00;
  NLX_coes1[2][  2][ 15] =+6.845543211725299e+00;
  NLX_coes1[2][  2][ 16] =+6.709712553314861e+00;
  NLX_coes1[2][  2][ 17] =+6.619608367693252e+00;
  NLX_coes1[2][  2][ 18] =+6.556460957792907e+00;
  NLX_coes1[2][  2][ 19] =+6.477207057069688e+00;
  NLX_coes1[2][  2][ 20] =+6.397911629891325e+00;
  NLX_coes1[2][  3][  0] =+3.723843894883980e+00;
  NLX_coes1[2][  3][  1] =+3.680772290825050e+00;
  NLX_coes1[2][  3][  2] =+3.633802905167869e+00;
  NLX_coes1[2][  3][  3] =+3.577382076469245e+00;
  NLX_coes1[2][  3][  4] =+3.511685187869564e+00;
  NLX_coes1[2][  3][  5] =+3.488543316949996e+00;
  NLX_coes1[2][  3][  6] =+3.509101797706618e+00;
  NLX_coes1[2][  3][  7] =+3.551016277957401e+00;
  NLX_coes1[2][  3][  8] =+3.629559808857544e+00;
  NLX_coes1[2][  3][  9] =+3.522213627825439e+00;
  NLX_coes1[2][  3][ 10] =+3.361992265677513e+00;
  NLX_coes1[2][  3][ 11] =+3.261496949097439e+00;
  NLX_coes1[2][  3][ 12] =+3.211888579709736e+00;
  NLX_coes1[2][  3][ 13] =+3.177505075978814e+00;
  NLX_coes1[2][  3][ 14] =+3.129994270562851e+00;
  NLX_coes1[2][  3][ 15] =+3.144275139014831e+00;
  NLX_coes1[2][  3][ 16] =+3.070525321806286e+00;
  NLX_coes1[2][  3][ 17] =+3.005077177878273e+00;
  NLX_coes1[2][  3][ 18] =+2.917372423199927e+00;
  NLX_coes1[2][  3][ 19] =+2.844292172056310e+00;
  NLX_coes1[2][  3][ 20] =+2.769358428743443e+00;
  NLX_coes1[2][  4][  0] =+1.356490352002231e-01;
  NLX_coes1[2][  4][  1] =+9.282357940006222e-02;
  NLX_coes1[2][  4][  2] =+4.896323608367291e-02;
  NLX_coes1[2][  4][  3] =+1.220536729802878e-03;
  NLX_coes1[2][  4][  4] =-8.076742980048592e-02;
  NLX_coes1[2][  4][  5] =-1.295992287658098e-01;
  NLX_coes1[2][  4][  6] =-1.272557889460667e-01;
  NLX_coes1[2][  4][  7] =-1.089916392535004e-01;
  NLX_coes1[2][  4][  8] =-6.957936225975817e-02;
  NLX_coes1[2][  4][  9] =-6.192178658817343e-02;
  NLX_coes1[2][  4][ 10] =-2.975326963659091e-01;
  NLX_coes1[2][  4][ 11] =-3.787170713185533e-01;
  NLX_coes1[2][  4][ 12] =-4.218745276749283e-01;
  NLX_coes1[2][  4][ 13] =-4.503740706534879e-01;
  NLX_coes1[2][  4][ 14] =-4.570814018373450e-01;
  NLX_coes1[2][  4][ 15] =-5.061495014504812e-01;
  NLX_coes1[2][  4][ 16] =-5.594751529224475e-01;
  NLX_coes1[2][  4][ 17] =-6.289742745942163e-01;
  NLX_coes1[2][  4][ 18] =-6.984434582196838e-01;
  NLX_coes1[2][  4][ 19] =-7.732344726175747e-01;
  NLX_coes1[2][  4][ 20] =-8.485530036155690e-01;
  NLX_coes1[2][  5][  0] =-3.446957782557269e+00;
  NLX_coes1[2][  5][  1] =-3.489632777659864e+00;
  NLX_coes1[2][  5][  2] =-3.534078510327791e+00;
  NLX_coes1[2][  5][  3] =-3.584384375304960e+00;
  NLX_coes1[2][  5][  4] =-3.646498908689458e+00;
  NLX_coes1[2][  5][  5] =-3.691208571835225e+00;
  NLX_coes1[2][  5][  6] =-3.705664039596559e+00;
  NLX_coes1[2][  5][  7] =-3.678355298837925e+00;
  NLX_coes1[2][  5][  8] =-3.598018440567287e+00;
  NLX_coes1[2][  5][  9] =-3.570999432285767e+00;
  NLX_coes1[2][  5][ 10] =-3.799305391769481e+00;
  NLX_coes1[2][  5][ 11] =-4.021803937040828e+00;
  NLX_coes1[2][  5][ 12] =-4.030585294935881e+00;
  NLX_coes1[2][  5][ 13] =-4.016496093995642e+00;
  NLX_coes1[2][  5][ 14] =-4.011280070598639e+00;
  NLX_coes1[2][  5][ 15] =-4.085673172087877e+00;
  NLX_coes1[2][  5][ 16] =-4.161680493010147e+00;
  NLX_coes1[2][  5][ 17] =-4.235815850153536e+00;
  NLX_coes1[2][  5][ 18] =-4.307776050545750e+00;
  NLX_coes1[2][  5][ 19] =-4.381484444862759e+00;
  NLX_coes1[2][  5][ 20] =-4.456005841876979e+00;
  NLX_coes1[2][  6][  0] =-7.025930877818570e+00;
  NLX_coes1[2][  6][  1] =-7.068555030267619e+00;
  NLX_coes1[2][  6][  2] =-7.112928311643614e+00;
  NLX_coes1[2][  6][  3] =-7.160689504163688e+00;
  NLX_coes1[2][  6][  4] =-7.207958360196121e+00;
  NLX_coes1[2][  6][  5] =-7.242370747409506e+00;
  NLX_coes1[2][  6][  6] =-7.251261942768630e+00;
  NLX_coes1[2][  6][  7] =-7.230651008608026e+00;
  NLX_coes1[2][  6][  8] =-7.202867347509536e+00;
  NLX_coes1[2][  6][  9] =-7.234990400172023e+00;
  NLX_coes1[2][  6][ 10] =-7.339827837924531e+00;
  NLX_coes1[2][  6][ 11] =-7.470880953287150e+00;
  NLX_coes1[2][  6][ 12] =-7.545166823013489e+00;
  NLX_coes1[2][  6][ 13] =-7.591837029476083e+00;
  NLX_coes1[2][  6][ 14] =-7.621515556666091e+00;
  NLX_coes1[2][  6][ 15] =-7.693204183024150e+00;
  NLX_coes1[2][  6][ 16] =-7.760182439174078e+00;
  NLX_coes1[2][  6][ 17] =-7.832786202912444e+00;
  NLX_coes1[2][  6][ 18] =-7.906549193670299e+00;
  NLX_coes1[2][  6][ 19] =-7.980621884964730e+00;
  NLX_coes1[2][  6][ 20] =-8.055065781632683e+00;
  NLX_coes1[2][  7][  0] =-1.060162257249160e+01;
  NLX_coes1[2][  7][  1] =-1.064349676960398e+01;
  NLX_coes1[2][  7][  2] =-1.068608987410414e+01;
  NLX_coes1[2][  7][  3] =-1.072868100683225e+01;
  NLX_coes1[2][  7][  4] =-1.076702482917941e+01;
  NLX_coes1[2][  7][  5] =-1.079467409004548e+01;
  NLX_coes1[2][  7][  6] =-1.080594358959403e+01;
  NLX_coes1[2][  7][  7] =-1.080369169896486e+01;
  NLX_coes1[2][  7][  8] =-1.080752721884146e+01;
  NLX_coes1[2][  7][  9] =-1.084639104285402e+01;
  NLX_coes1[2][  7][ 10] =-1.092404899931685e+01;
  NLX_coes1[2][  7][ 11] =-1.102098662839040e+01;
  NLX_coes1[2][  7][ 12] =-1.109658788905282e+01;
  NLX_coes1[2][  7][ 13] =-1.115443921290658e+01;
  NLX_coes1[2][  7][ 14] =-1.121416536047477e+01;
  NLX_coes1[2][  7][ 15] =-1.128103201850337e+01;
  NLX_coes1[2][  7][ 16] =-1.135190461422159e+01;
  NLX_coes1[2][  7][ 17] =-1.142458858224392e+01;
  NLX_coes1[2][  7][ 18] =-1.149847082927735e+01;
  NLX_coes1[2][  7][ 19] =-1.157275356110317e+01;
  NLX_coes1[2][  7][ 20] =-1.164717191386416e+01;
  NLX_coes1[2][  8][  0] =-1.417429954135985e+01;
  NLX_coes1[2][  8][  1] =-1.421497870467408e+01;
  NLX_coes1[2][  8][  2] =-1.425527549860733e+01;
  NLX_coes1[2][  8][  3] =-1.429381569774604e+01;
  NLX_coes1[2][  8][  4] =-1.432775530007225e+01;
  NLX_coes1[2][  8][  5] =-1.435391812727432e+01;
  NLX_coes1[2][  8][  6] =-1.437112511122109e+01;
  NLX_coes1[2][  8][  7] =-1.438411353149660e+01;
  NLX_coes1[2][  8][  8] =-1.440516214437826e+01;
  NLX_coes1[2][  8][  9] =-1.444639831763328e+01;
  NLX_coes1[2][  8][ 10] =-1.451005630866192e+01;
  NLX_coes1[2][  8][ 11] =-1.458842819185054e+01;
  NLX_coes1[2][  8][ 12] =-1.466182853372133e+01;
  NLX_coes1[2][  8][ 13] =-1.473180452796914e+01;
  NLX_coes1[2][  8][ 14] =-1.479689486042329e+01;
  NLX_coes1[2][  8][ 15] =-1.486587033104590e+01;
  NLX_coes1[2][  8][ 16] =-1.493796817476676e+01;
  NLX_coes1[2][  8][ 17] =-1.501125244224972e+01;
  NLX_coes1[2][  8][ 18] =-1.508531557956825e+01;
  NLX_coes1[2][  8][ 19] =-1.515965539612036e+01;
  NLX_coes1[2][  8][ 20] =-1.523399971650162e+01;
  NLX_coes1[2][  9][  0] =-1.774474945378686e+01;
  NLX_coes1[2][  9][  1] =-1.778428067879931e+01;
  NLX_coes1[2][  9][  2] =-1.782281280993912e+01;
  NLX_coes1[2][  9][  3] =-1.785917407702312e+01;
  NLX_coes1[2][  9][  4] =-1.789178488691493e+01;
  NLX_coes1[2][  9][  5] =-1.791948669462429e+01;
  NLX_coes1[2][  9][  6] =-1.794304282891316e+01;
  NLX_coes1[2][  9][  7] =-1.796662328664456e+01;
  NLX_coes1[2][  9][  8] =-1.799729484835836e+01;
  NLX_coes1[2][  9][  9] =-1.804109679858993e+01;
  NLX_coes1[2][  9][ 10] =-1.809938486047134e+01;
  NLX_coes1[2][  9][ 11] =-1.816652945945282e+01;
  NLX_coes1[2][  9][ 12] =-1.823622762907778e+01;
  NLX_coes1[2][  9][ 13] =-1.830847670051689e+01;
  NLX_coes1[2][  9][ 14] =-1.837806418760079e+01;
  NLX_coes1[2][  9][ 15] =-1.844848644017554e+01;
  NLX_coes1[2][  9][ 16] =-1.852086991680539e+01;
  NLX_coes1[2][  9][ 17] =-1.859440225649287e+01;
  NLX_coes1[2][  9][ 18] =-1.866848760759809e+01;
  NLX_coes1[2][  9][ 19] =-1.874274201276148e+01;
  NLX_coes1[2][  9][ 20] =-1.881694103393433e+01;
  NLX_coes1[2][ 10][  0] =-2.131396683995030e+01;
  NLX_coes1[2][ 10][  1] =-2.135270361557521e+01;
  NLX_coes1[2][ 10][  2] =-2.139031498337342e+01;
  NLX_coes1[2][ 10][  3] =-2.142601973967226e+01;
  NLX_coes1[2][ 10][  4] =-2.145911200943594e+01;
  NLX_coes1[2][ 10][  5] =-2.148951261856117e+01;
  NLX_coes1[2][ 10][  6] =-2.151853219387514e+01;
  NLX_coes1[2][ 10][  7] =-2.154926971968234e+01;
  NLX_coes1[2][ 10][  8] =-2.158586014727964e+01;
  NLX_coes1[2][ 10][  9] =-2.163162176308265e+01;
  NLX_coes1[2][ 10][ 10] =-2.168734098195274e+01;
  NLX_coes1[2][ 10][ 11] =-2.175032196872355e+01;
  NLX_coes1[2][ 10][ 12] =-2.181607669572342e+01;
  NLX_coes1[2][ 10][ 13] =-2.188665386735161e+01;
  NLX_coes1[2][ 10][ 14] =-2.195764135650247e+01;
  NLX_coes1[2][ 10][ 15] =-2.202918792210891e+01;
  NLX_coes1[2][ 10][ 16] =-2.210178864489227e+01;
  NLX_coes1[2][ 10][ 17] =-2.217524733258613e+01;
  NLX_coes1[2][ 10][ 18] =-2.224914871203744e+01;
  NLX_coes1[2][ 10][ 19] =-2.232317316725872e+01;
  NLX_coes1[2][ 10][ 20] =-2.239713120347472e+01;
  NLX_coes1[2][ 11][  0] =-2.488282866958803e+01;
  NLX_coes1[2][ 11][  1] =-2.492121269920580e+01;
  NLX_coes1[2][ 11][  2] =-2.495863820795750e+01;
  NLX_coes1[2][ 11][  3] =-2.499466569086361e+01;
  NLX_coes1[2][ 11][  4] =-2.502911600794957e+01;
  NLX_coes1[2][ 11][  5] =-2.506238460883899e+01;
  NLX_coes1[2][ 11][  6] =-2.509575937793379e+01;
  NLX_coes1[2][ 11][  7] =-2.513142653884293e+01;
  NLX_coes1[2][ 11][  8] =-2.517190599951357e+01;
  NLX_coes1[2][ 11][  9] =-2.521908091828544e+01;
  NLX_coes1[2][ 11][ 10] =-2.527328860851129e+01;
  NLX_coes1[2][ 11][ 11] =-2.533364109214609e+01;
  NLX_coes1[2][ 11][ 12] =-2.539863152003238e+01;
  NLX_coes1[2][ 11][ 13] =-2.546701616257178e+01;
  NLX_coes1[2][ 11][ 14] =-2.553743167430035e+01;
  NLX_coes1[2][ 11][ 15] =-2.560897976422970e+01;
  NLX_coes1[2][ 11][ 16] =-2.568144816780613e+01;
  NLX_coes1[2][ 11][ 17] =-2.575461596745375e+01;
  NLX_coes1[2][ 11][ 18] =-2.582817109747501e+01;
  NLX_coes1[2][ 11][ 19] =-2.590184456672604e+01;
  NLX_coes1[2][ 11][ 20] =-2.597547122686742e+01;
  NLX_coes1[2][ 12][  0] =-2.845196663515849e+01;
  NLX_coes1[2][ 12][  1] =-2.849039028418734e+01;
  NLX_coes1[2][ 12][  2] =-2.852813540123377e+01;
  NLX_coes1[2][ 12][  3] =-2.856501158811520e+01;
  NLX_coes1[2][ 12][  4] =-2.860112163131351e+01;
  NLX_coes1[2][ 12][  5] =-2.863701413892881e+01;
  NLX_coes1[2][ 12][  6] =-2.867378249266936e+01;
  NLX_coes1[2][ 12][  7] =-2.871296318997010e+01;
  NLX_coes1[2][ 12][  8] =-2.875616841599731e+01;
  NLX_coes1[2][ 12][  9] =-2.880458016161987e+01;
  NLX_coes1[2][ 12][ 10] =-2.885857706184137e+01;
  NLX_coes1[2][ 12][ 11] =-2.891776530991835e+01;
  NLX_coes1[2][ 12][ 12] =-2.898134127770464e+01;
  NLX_coes1[2][ 12][ 13] =-2.904826223750066e+01;
  NLX_coes1[2][ 12][ 14] =-2.911759684499003e+01;
  NLX_coes1[2][ 12][ 15] =-2.918849102261404e+01;
  NLX_coes1[2][ 12][ 16] =-2.926044806998894e+01;
  NLX_coes1[2][ 12][ 17] =-2.933311996920033e+01;
  NLX_coes1[2][ 12][ 18] =-2.940619731932432e+01;
  NLX_coes1[2][ 12][ 19] =-2.947942757474370e+01;
  NLX_coes1[2][ 12][ 20] =-2.955265090881835e+01;
  NLX_coes1[2][ 13][  0] =-3.202177028254039e+01;
  NLX_coes1[2][ 13][  1] =-3.206052058326556e+01;
  NLX_coes1[2][ 13][  2] =-3.209887747668363e+01;
  NLX_coes1[2][ 13][  3] =-3.213682004127907e+01;
  NLX_coes1[2][ 13][  4] =-3.217458558569862e+01;
  NLX_coes1[2][ 13][  5] =-3.221273179709441e+01;
  NLX_coes1[2][ 13][  6] =-3.225214383294445e+01;
  NLX_coes1[2][ 13][  7] =-3.229392597056340e+01;
  NLX_coes1[2][ 13][  8] =-3.233917156494744e+01;
  NLX_coes1[2][ 13][  9] =-3.238868730663686e+01;
  NLX_coes1[2][ 13][ 10] =-3.244280915767593e+01;
  NLX_coes1[2][ 13][ 11] =-3.250138660457348e+01;
  NLX_coes1[2][ 13][ 12] =-3.256391837462982e+01;
  NLX_coes1[2][ 13][ 13] =-3.262967219157792e+01;
  NLX_coes1[2][ 13][ 14] =-3.269790312637949e+01;
  NLX_coes1[2][ 13][ 15] =-3.276791016188556e+01;
  NLX_coes1[2][ 13][ 16] =-3.283914056930583e+01;
  NLX_coes1[2][ 13][ 17] =-3.291117709430240e+01;
  NLX_coes1[2][ 13][ 18] =-3.298368707964943e+01;
  NLX_coes1[2][ 13][ 19] =-3.305641269545895e+01;
  NLX_coes1[2][ 13][ 20] =-3.312918433825704e+01;
  NLX_coes1[2][ 14][  0] =-3.559244104104358e+01;
  NLX_coes1[2][ 14][  1] =-3.563169369643531e+01;
  NLX_coes1[2][ 14][  2] =-3.567079508634552e+01;
  NLX_coes1[2][ 14][  3] =-3.570983654363972e+01;
  NLX_coes1[2][ 14][  4] =-3.574911179124580e+01;
  NLX_coes1[2][ 14][  5] =-3.578913548942000e+01;
  NLX_coes1[2][ 14][  6] =-3.583061869969637e+01;
  NLX_coes1[2][ 14][  7] =-3.587437974505774e+01;
  NLX_coes1[2][ 14][  8] =-3.592119841264049e+01;
  NLX_coes1[2][ 14][  9] =-3.597165791853060e+01;
  NLX_coes1[2][ 14][ 10] =-3.602603718744759e+01;
  NLX_coes1[2][ 14][ 11] =-3.608428674038383e+01;
  NLX_coes1[2][ 14][ 12] =-3.614607830615638e+01;
  NLX_coes1[2][ 14][ 13] =-3.621089669537104e+01;
  NLX_coes1[2][ 14][ 14] =-3.627815082436506e+01;
  NLX_coes1[2][ 14][ 15] =-3.634725806845435e+01;
  NLX_coes1[2][ 14][ 16] =-3.641769936531530e+01;
  NLX_coes1[2][ 14][ 17] =-3.648904562055661e+01;
  NLX_coes1[2][ 14][ 18] =-3.656095038671185e+01;
  NLX_coes1[2][ 14][ 19] =-3.663314505336346e+01;
  NLX_coes1[2][ 14][ 20] =-3.670544346273731e+01;
  NLX_coes1[2][ 15][  0] =-3.916405135920090e+01;
  NLX_coes1[2][ 15][  1] =-3.920388714718462e+01;
  NLX_coes1[2][ 15][  2] =-3.924375844918571e+01;
  NLX_coes1[2][ 15][  3] =-3.928383066188211e+01;
  NLX_coes1[2][ 15][  4] =-3.932441673057166e+01;
  NLX_coes1[2][ 15][  5] =-3.936597723062037e+01;
  NLX_coes1[2][ 15][  6] =-3.940908945601751e+01;
  NLX_coes1[2][ 15][  7] =-3.945437979491221e+01;
  NLX_coes1[2][ 15][  8] =-3.950242878068180e+01;
  NLX_coes1[2][ 15][  9] =-3.955367421448938e+01;
  NLX_coes1[2][ 15][ 10] =-3.960834222380281e+01;
  NLX_coes1[2][ 15][ 11] =-3.966642420123253e+01;
  NLX_coes1[2][ 15][ 12] =-3.972769812655717e+01;
  NLX_coes1[2][ 15][ 13] =-3.979178578254991e+01;
  NLX_coes1[2][ 15][ 14] =-3.985822067431126e+01;
  NLX_coes1[2][ 15][ 15] =-3.992651440738043e+01;
  NLX_coes1[2][ 15][ 16] =-3.999620251291369e+01;
  NLX_coes1[2][ 15][ 17] =-4.006687433004421e+01;
  NLX_coes1[2][ 15][ 18] =-4.013818443058729e+01;
  NLX_coes1[2][ 15][ 19] =-4.020985654116059e+01;
  NLX_coes1[2][ 15][ 20] =-4.028168737924110e+01;
  NLX_coes1[2][ 16][  0] =-4.273659128630243e+01;
  NLX_coes1[2][ 16][  1] =-4.277701849701984e+01;
  NLX_coes1[2][ 16][  2] =-4.281761773277376e+01;
  NLX_coes1[2][ 16][  3] =-4.285860399242908e+01;
  NLX_coes1[2][ 16][  4] =-4.290029332511043e+01;
  NLX_coes1[2][ 16][  5] =-4.294309670157487e+01;
  NLX_coes1[2][ 16][  6] =-4.298749197250963e+01;
  NLX_coes1[2][ 16][  7] =-4.303397410931986e+01;
  NLX_coes1[2][ 16][  8] =-4.308299139359992e+01;
  NLX_coes1[2][ 16][  9] =-4.313488233866868e+01;
  NLX_coes1[2][ 16][ 10] =-4.318982932344289e+01;
  NLX_coes1[2][ 16][ 11] =-4.324783950219935e+01;
  NLX_coes1[2][ 16][ 12] =-4.330875483604788e+01;
  NLX_coes1[2][ 16][ 13] =-4.337228624883407e+01;
  NLX_coes1[2][ 16][ 14] =-4.343805938284639e+01;
  NLX_coes1[2][ 16][ 15] =-4.350566220590142e+01;
  NLX_coes1[2][ 16][ 16] =-4.357468404842110e+01;
  NLX_coes1[2][ 16][ 17] =-4.364474371865798e+01;
  NLX_coes1[2][ 16][ 18] =-4.371550662376483e+01;
  NLX_coes1[2][ 16][ 19] =-4.378669392698539e+01;
  NLX_coes1[2][ 16][ 20] =-4.385808756775638e+01;
  NLX_coes1[2][ 17][  0] =-4.630999914924624e+01;
  NLX_coes1[2][ 17][  1] =-4.635097429248985e+01;
  NLX_coes1[2][ 17][  2] =-4.639222033006410e+01;
  NLX_coes1[2][ 17][  3] =-4.643398639425389e+01;
  NLX_coes1[2][ 17][  4] =-4.647658478677525e+01;
  NLX_coes1[2][ 17][  5] =-4.652038516535107e+01;
  NLX_coes1[2][ 17][  6] =-4.656579192056011e+01;
  NLX_coes1[2][ 17][  7] =-4.661320809912746e+01;
  NLX_coes1[2][ 17][  8] =-4.666299119990246e+01;
  NLX_coes1[2][ 17][  9] =-4.671541006534282e+01;
  NLX_coes1[2][ 17][ 10] =-4.677061245315272e+01;
  NLX_coes1[2][ 17][ 11] =-4.682861004194919e+01;
  NLX_coes1[2][ 17][ 12] =-4.688928302342926e+01;
  NLX_coes1[2][ 17][ 13] =-4.695240142598226e+01;
  NLX_coes1[2][ 17][ 14] =-4.701765686162431e+01;
  NLX_coes1[2][ 17][ 15] =-4.708469742190664e+01;
  NLX_coes1[2][ 17][ 16] =-4.715315954520463e+01;
  NLX_coes1[2][ 17][ 17] =-4.722269314235921e+01;
  NLX_coes1[2][ 17][ 18] =-4.729297916167961e+01;
  NLX_coes1[2][ 17][ 19] =-4.736374065201358e+01;
  NLX_coes1[2][ 17][ 20] =-4.743474790140843e+01;
  NLX_coes1[2][ 18][  0] =-4.988417804789373e+01;
  NLX_coes1[2][ 18][  1] =-4.992562166292281e+01;
  NLX_coes1[2][ 18][  2] =-4.996741511559059e+01;
  NLX_coes1[2][ 18][  3] =-5.000982874962207e+01;
  NLX_coes1[2][ 18][  4] =-5.005316756070555e+01;
  NLX_coes1[2][ 18][  5] =-5.009776660582025e+01;
  NLX_coes1[2][ 18][  6] =-5.014397500714700e+01;
  NLX_coes1[2][ 18][  7] =-5.019212981719205e+01;
  NLX_coes1[2][ 18][  8] =-5.024252443024631e+01;
  NLX_coes1[2][ 18][  9] =-5.029537776402763e+01;
  NLX_coes1[2][ 18][ 10] =-5.035081062705206e+01;
  NLX_coes1[2][ 18][ 11] =-5.040883396603302e+01;
  NLX_coes1[2][ 18][ 12] =-5.046935081213113e+01;
  NLX_coes1[2][ 18][ 13] =-5.053217052413054e+01;
  NLX_coes1[2][ 18][ 14] =-5.059703164150793e+01;
  NLX_coes1[2][ 18][ 15] =-5.066362845375075e+01;
  NLX_coes1[2][ 18][ 16] =-5.073163683118617e+01;
  NLX_coes1[2][ 18][ 17] =-5.080073596935973e+01;
  NLX_coes1[2][ 18][ 18] =-5.087062453231611e+01;
  NLX_coes1[2][ 18][ 19] =-5.094103084867505e+01;
  NLX_coes1[2][ 18][ 20] =-5.101171836617672e+01;
  NLX_coes1[2][ 19][  0] =-5.345900220645368e+01;
  NLX_coes1[2][ 19][  1] =-5.350080475332673e+01;
  NLX_coes1[2][ 19][  2] =-5.354304913520033e+01;
  NLX_coes1[2][ 19][  3] =-5.358599560656637e+01;
  NLX_coes1[2][ 19][  4] =-5.362994069837811e+01;
  NLX_coes1[2][ 19][  5] =-5.367518857023671e+01;
  NLX_coes1[2][ 19][  6] =-5.372204445809906e+01;
  NLX_coes1[2][ 19][  7] =-5.377079610749035e+01;
  NLX_coes1[2][ 19][  8] =-5.382169014190129e+01;
  NLX_coes1[2][ 19][  9] =-5.387490877653871e+01;
  NLX_coes1[2][ 19][ 10] =-5.393055172017151e+01;
  NLX_coes1[2][ 19][ 11] =-5.398862675351383e+01;
  NLX_coes1[2][ 19][ 12] =-5.404905048093734e+01;
  NLX_coes1[2][ 19][ 13] =-5.411165858913191e+01;
  NLX_coes1[2][ 19][ 14] =-5.417622322070828e+01;
  NLX_coes1[2][ 19][ 15] =-5.424247408299966e+01;
  NLX_coes1[2][ 19][ 16] =-5.431011984702937e+01;
  NLX_coes1[2][ 19][ 17] =-5.437886688267216e+01;
  NLX_coes1[2][ 19][ 18] =-5.444843305839208e+01;
  NLX_coes1[2][ 19][ 19] =-5.451855364958646e+01;
  NLX_coes1[2][ 19][ 20] =-5.458900289381965e+01;
  NLX_coes1[2][ 20][  0] =-5.703427859545232e+01;
  NLX_coes1[2][ 20][  1] =-5.707636494427758e+01;
  NLX_coes1[2][ 20][  2] =-5.711896029478751e+01;
  NLX_coes1[2][ 20][  3] =-5.716235700697325e+01;
  NLX_coes1[2][ 20][  4] =-5.720681950701050e+01;
  NLX_coes1[2][ 20][  5] =-5.725261895377724e+01;
  NLX_coes1[2][ 20][  6] =-5.730002309928847e+01;
  NLX_coes1[2][ 20][  7] =-5.734928068635855e+01;
  NLX_coes1[2][ 20][  8] =-5.740060233656196e+01;
  NLX_coes1[2][ 20][  9] =-5.745414190196091e+01;
  NLX_coes1[2][ 20][ 10] =-5.750998193071627e+01;
  NLX_coes1[2][ 20][ 11] =-5.756812592997445e+01;
  NLX_coes1[2][ 20][ 12] =-5.762849865767991e+01;
  NLX_coes1[2][ 20][ 13] =-5.769095409392385e+01;
  NLX_coes1[2][ 20][ 14] =-5.775528940111767e+01;
  NLX_coes1[2][ 20][ 15] =-5.782126237372294e+01;
  NLX_coes1[2][ 20][ 16] =-5.788860967021990e+01;
  NLX_coes1[2][ 20][ 17] =-5.795706340803242e+01;
  NLX_coes1[2][ 20][ 18] =-5.802636420988920e+01;
  NLX_coes1[2][ 20][ 19] =-5.809627041579532e+01;
  NLX_coes1[2][ 20][ 20] =-5.816653851529762e+01;

  NLX_coes2[0][  0]=-1.215946866819101e+00;   Eid[  0]= 0; Rid[  0]= 0; Zid[  0]= 1;
  NLX_coes2[0][  1]=+5.652427195288375e-01;   Eid[  1]= 0; Rid[  1]= 0; Zid[  1]= 2;
  NLX_coes2[0][  2]=-1.208696812408274e-01;   Eid[  2]= 0; Rid[  2]= 0; Zid[  2]= 3;
  NLX_coes2[0][  3]=+3.988728275090771e-01;   Eid[  3]= 0; Rid[  3]= 1; Zid[  3]= 1;
  NLX_coes2[0][  4]=-2.929560459769579e-01;   Eid[  4]= 0; Rid[  4]= 1; Zid[  4]= 2;
  NLX_coes2[0][  5]=-3.243165776631379e-01;   Eid[  5]= 0; Rid[  5]= 1; Zid[  5]= 3;
  NLX_coes2[0][  6]=+2.842255242900868e-02;   Eid[  6]= 0; Rid[  6]= 2; Zid[  6]= 1;
  NLX_coes2[0][  7]=-6.105970311169855e-02;   Eid[  7]= 0; Rid[  7]= 2; Zid[  7]= 2;
  NLX_coes2[0][  8]=-8.506775265448904e-02;   Eid[  8]= 0; Rid[  8]= 2; Zid[  8]= 3;
  NLX_coes2[0][  9]=+5.469227413683363e-01;   Eid[  9]= 1; Rid[  9]= 0; Zid[  9]= 0;
  NLX_coes2[0][ 10]=-1.118822531814712e+00;   Eid[ 10]= 1; Rid[ 10]= 0; Zid[ 10]= 1;
  NLX_coes2[0][ 11]=+5.605273329949757e-01;   Eid[ 11]= 1; Rid[ 11]= 0; Zid[ 11]= 2;
  NLX_coes2[0][ 12]=+3.913335209036491e-01;   Eid[ 12]= 1; Rid[ 12]= 0; Zid[ 12]= 3;
  NLX_coes2[0][ 13]=-3.695015741677851e-02;   Eid[ 13]= 1; Rid[ 13]= 1; Zid[ 13]= 0;
  NLX_coes2[0][ 14]=+6.442828370619432e-01;   Eid[ 14]= 1; Rid[ 14]= 1; Zid[ 14]= 1;
  NLX_coes2[0][ 15]=-5.680791909605349e-01;   Eid[ 15]= 1; Rid[ 15]= 1; Zid[ 15]= 2;
  NLX_coes2[0][ 16]=-3.328131137257524e-02;   Eid[ 16]= 1; Rid[ 16]= 1; Zid[ 16]= 3;
  NLX_coes2[0][ 17]=+6.982310218065339e-02;   Eid[ 17]= 1; Rid[ 17]= 2; Zid[ 17]= 0;
  NLX_coes2[0][ 18]=-4.341672725275109e-01;   Eid[ 18]= 1; Rid[ 18]= 2; Zid[ 18]= 1;
  NLX_coes2[0][ 19]=-7.057276111410632e-01;   Eid[ 19]= 1; Rid[ 19]= 2; Zid[ 19]= 2;
  NLX_coes2[0][ 20]=+1.883062052928921e+00;   Eid[ 20]= 1; Rid[ 20]= 2; Zid[ 20]= 3;
  NLX_coes2[0][ 21]=+3.773251650346126e-01;   Eid[ 21]= 2; Rid[ 21]= 0; Zid[ 21]= 0;
  NLX_coes2[0][ 22]=-9.231317067636127e-01;   Eid[ 22]= 2; Rid[ 22]= 0; Zid[ 22]= 1;
  NLX_coes2[0][ 23]=+6.692081395810437e-01;   Eid[ 23]= 2; Rid[ 23]= 0; Zid[ 23]= 2;
  NLX_coes2[0][ 24]=-6.152859755048757e-02;   Eid[ 24]= 2; Rid[ 24]= 0; Zid[ 24]= 3;
  NLX_coes2[0][ 25]=+2.839764607331158e-01;   Eid[ 25]= 2; Rid[ 25]= 1; Zid[ 25]= 0;
  NLX_coes2[0][ 26]=+6.632451935000537e-01;   Eid[ 26]= 2; Rid[ 26]= 1; Zid[ 26]= 1;
  NLX_coes2[0][ 27]=-9.441585212718373e-01;   Eid[ 27]= 2; Rid[ 27]= 1; Zid[ 27]= 2;
  NLX_coes2[0][ 28]=-8.190137675419253e-01;   Eid[ 28]= 2; Rid[ 28]= 1; Zid[ 28]= 3;
  NLX_coes2[0][ 29]=+7.470831436182246e-01;   Eid[ 29]= 2; Rid[ 29]= 2; Zid[ 29]= 0;
  NLX_coes2[0][ 30]=-1.547077386268723e+00;   Eid[ 30]= 2; Rid[ 30]= 2; Zid[ 30]= 1;
  NLX_coes2[0][ 31]=+4.805196923256402e-01;   Eid[ 31]= 2; Rid[ 31]= 2; Zid[ 31]= 2;
  NLX_coes2[0][ 32]=+6.771953004999678e-01;   Eid[ 32]= 2; Rid[ 32]= 2; Zid[ 32]= 3;
  NLX_coes2[0][ 33]=+2.362345174627880e-01;   Eid[ 33]= 3; Rid[ 33]= 0; Zid[ 33]= 0;
  NLX_coes2[0][ 34]=-6.872332373691245e-01;   Eid[ 34]= 3; Rid[ 34]= 0; Zid[ 34]= 1;
  NLX_coes2[0][ 35]=+8.979835493846771e-01;   Eid[ 35]= 3; Rid[ 35]= 0; Zid[ 35]= 2;
  NLX_coes2[0][ 36]=-1.744941455875169e-01;   Eid[ 36]= 3; Rid[ 36]= 0; Zid[ 36]= 3;
  NLX_coes2[0][ 37]=+5.351674735524635e-01;   Eid[ 37]= 3; Rid[ 37]= 1; Zid[ 37]= 0;
  NLX_coes2[0][ 38]=+4.294636474332906e-01;   Eid[ 38]= 3; Rid[ 38]= 1; Zid[ 38]= 1;
  NLX_coes2[0][ 39]=-8.690122120840886e-01;   Eid[ 39]= 3; Rid[ 39]= 1; Zid[ 39]= 2;
  NLX_coes2[0][ 40]=-8.127747512530674e-01;   Eid[ 40]= 3; Rid[ 40]= 1; Zid[ 40]= 3;
  NLX_coes2[0][ 41]=-2.668513327924611e-01;   Eid[ 41]= 3; Rid[ 41]= 2; Zid[ 41]= 0;
  NLX_coes2[0][ 42]=-3.241465820357306e-01;   Eid[ 42]= 3; Rid[ 42]= 2; Zid[ 42]= 1;
  NLX_coes2[0][ 43]=+1.337172060621570e+00;   Eid[ 43]= 3; Rid[ 43]= 2; Zid[ 43]= 2;
  NLX_coes2[0][ 44]=-1.061633886544658e+00;   Eid[ 44]= 3; Rid[ 44]= 2; Zid[ 44]= 3;
  NLX_coes2[0][ 45]=+1.757658163008957e-01;   Eid[ 45]= 4; Rid[ 45]= 0; Zid[ 45]= 0;
  NLX_coes2[0][ 46]=-5.081673927419008e-01;   Eid[ 46]= 4; Rid[ 46]= 0; Zid[ 46]= 1;
  NLX_coes2[0][ 47]=+9.945919907918691e-01;   Eid[ 47]= 4; Rid[ 47]= 0; Zid[ 47]= 2;
  NLX_coes2[0][ 48]=-9.913116056256721e-02;   Eid[ 48]= 4; Rid[ 48]= 0; Zid[ 48]= 3;
  NLX_coes2[0][ 49]=+4.681146611943772e-01;   Eid[ 49]= 4; Rid[ 49]= 1; Zid[ 49]= 0;
  NLX_coes2[0][ 50]=+4.198029446840909e-02;   Eid[ 50]= 4; Rid[ 50]= 1; Zid[ 50]= 1;
  NLX_coes2[0][ 51]=-5.608343666399562e-01;   Eid[ 51]= 4; Rid[ 51]= 1; Zid[ 51]= 2;
  NLX_coes2[0][ 52]=+1.330119066668337e-01;   Eid[ 52]= 4; Rid[ 52]= 1; Zid[ 52]= 3;
  NLX_coes2[0][ 53]=-7.816010951600131e-01;   Eid[ 53]= 4; Rid[ 53]= 2; Zid[ 53]= 0;
  NLX_coes2[0][ 54]=+7.213392364739656e-01;   Eid[ 54]= 4; Rid[ 54]= 2; Zid[ 54]= 1;
  NLX_coes2[0][ 55]=+1.626839665930490e+00;   Eid[ 55]= 4; Rid[ 55]= 2; Zid[ 55]= 2;
  NLX_coes2[0][ 56]=-1.143435783623220e+00;   Eid[ 56]= 4; Rid[ 56]= 2; Zid[ 56]= 3;
  NLX_coes2[0][ 57]=+2.123662363101824e-01;   Eid[ 57]= 5; Rid[ 57]= 0; Zid[ 57]= 0;
  NLX_coes2[0][ 58]=-3.959796571751455e-01;   Eid[ 58]= 5; Rid[ 58]= 0; Zid[ 58]= 1;
  NLX_coes2[0][ 59]=+8.248793136768487e-01;   Eid[ 59]= 5; Rid[ 59]= 0; Zid[ 59]= 2;
  NLX_coes2[0][ 60]=-1.957712321193303e-01;   Eid[ 60]= 5; Rid[ 60]= 0; Zid[ 60]= 3;
  NLX_coes2[0][ 61]=+2.423154522291699e-01;   Eid[ 61]= 5; Rid[ 61]= 1; Zid[ 61]= 0;
  NLX_coes2[0][ 62]=-3.107609205300626e-01;   Eid[ 62]= 5; Rid[ 62]= 1; Zid[ 62]= 1;
  NLX_coes2[0][ 63]=-3.707311560637425e-01;   Eid[ 63]= 5; Rid[ 63]= 1; Zid[ 63]= 2;
  NLX_coes2[0][ 64]=+1.170261175926898e+00;   Eid[ 64]= 5; Rid[ 64]= 1; Zid[ 64]= 3;
  NLX_coes2[0][ 65]=-9.725691422476322e-01;   Eid[ 65]= 5; Rid[ 65]= 2; Zid[ 65]= 0;
  NLX_coes2[0][ 66]=+9.867838488636624e-01;   Eid[ 66]= 5; Rid[ 66]= 2; Zid[ 66]= 1;
  NLX_coes2[0][ 67]=+1.064243806318121e+00;   Eid[ 67]= 5; Rid[ 67]= 2; Zid[ 67]= 2;
  NLX_coes2[0][ 68]=-3.465221737398452e-01;   Eid[ 68]= 5; Rid[ 68]= 2; Zid[ 68]= 3;
  NLX_coes2[0][ 69]=+3.126357805127731e-01;   Eid[ 69]= 6; Rid[ 69]= 0; Zid[ 69]= 0;
  NLX_coes2[0][ 70]=-3.071136044523636e-01;   Eid[ 70]= 6; Rid[ 70]= 0; Zid[ 70]= 1;
  NLX_coes2[0][ 71]=+4.558637585327355e-01;   Eid[ 71]= 6; Rid[ 71]= 0; Zid[ 71]= 2;
  NLX_coes2[0][ 72]=-5.011653903998367e-01;   Eid[ 72]= 6; Rid[ 72]= 0; Zid[ 72]= 3;
  NLX_coes2[0][ 73]=-4.616540772765888e-04;   Eid[ 73]= 6; Rid[ 73]= 1; Zid[ 73]= 0;
  NLX_coes2[0][ 74]=-4.459229730488463e-01;   Eid[ 74]= 6; Rid[ 74]= 1; Zid[ 74]= 1;
  NLX_coes2[0][ 75]=-3.624801422884110e-01;   Eid[ 75]= 6; Rid[ 75]= 1; Zid[ 75]= 2;
  NLX_coes2[0][ 76]=+1.670458626829117e+00;   Eid[ 76]= 6; Rid[ 76]= 1; Zid[ 76]= 3;
  NLX_coes2[0][ 77]=-7.683759650108244e-01;   Eid[ 77]= 6; Rid[ 77]= 2; Zid[ 77]= 0;
  NLX_coes2[0][ 78]=+9.459183093469061e-01;   Eid[ 78]= 6; Rid[ 78]= 2; Zid[ 78]= 1;
  NLX_coes2[0][ 79]=-2.057173666624901e-02;   Eid[ 79]= 6; Rid[ 79]= 2; Zid[ 79]= 2;
  NLX_coes2[0][ 80]=+1.825195881807780e-01;   Eid[ 80]= 6; Rid[ 80]= 2; Zid[ 80]= 3;
  NLX_coes2[0][ 81]=+4.121369811595350e-01;   Eid[ 81]= 7; Rid[ 81]= 0; Zid[ 81]= 0;
  NLX_coes2[0][ 82]=-2.100355083927752e-01;   Eid[ 82]= 7; Rid[ 82]= 0; Zid[ 82]= 1;
  NLX_coes2[0][ 83]=+4.256365703724644e-02;   Eid[ 83]= 7; Rid[ 83]= 0; Zid[ 83]= 2;
  NLX_coes2[0][ 84]=-7.739795883362058e-01;   Eid[ 84]= 7; Rid[ 84]= 0; Zid[ 84]= 3;
  NLX_coes2[0][ 85]=-2.349037202061225e-01;   Eid[ 85]= 7; Rid[ 85]= 1; Zid[ 85]= 0;
  NLX_coes2[0][ 86]=-3.066393295763400e-01;   Eid[ 86]= 7; Rid[ 86]= 1; Zid[ 86]= 1;
  NLX_coes2[0][ 87]=-4.108837234208789e-01;   Eid[ 87]= 7; Rid[ 87]= 1; Zid[ 87]= 2;
  NLX_coes2[0][ 88]=+1.546335635859676e+00;   Eid[ 88]= 7; Rid[ 88]= 1; Zid[ 88]= 3;
  NLX_coes2[0][ 89]=-3.977620736259863e-01;   Eid[ 89]= 7; Rid[ 89]= 2; Zid[ 89]= 0;
  NLX_coes2[0][ 90]=+9.339523946423113e-01;   Eid[ 90]= 7; Rid[ 90]= 2; Zid[ 90]= 1;
  NLX_coes2[0][ 91]=-1.038055270785355e+00;   Eid[ 91]= 7; Rid[ 91]= 2; Zid[ 91]= 2;
  NLX_coes2[0][ 92]=+1.192532884560597e-01;   Eid[ 92]= 7; Rid[ 92]= 2; Zid[ 92]= 3;
  NLX_coes2[0][ 93]=+4.490619973473847e-01;   Eid[ 93]= 8; Rid[ 93]= 0; Zid[ 93]= 0;
  NLX_coes2[0][ 94]=-1.071064111106393e-01;   Eid[ 94]= 8; Rid[ 94]= 0; Zid[ 94]= 1;
  NLX_coes2[0][ 95]=-2.849042072034903e-01;   Eid[ 95]= 8; Rid[ 95]= 0; Zid[ 95]= 2;
  NLX_coes2[0][ 96]=-7.467627748935438e-01;   Eid[ 96]= 8; Rid[ 96]= 0; Zid[ 96]= 3;
  NLX_coes2[0][ 97]=-4.653050222918899e-01;   Eid[ 97]= 8; Rid[ 97]= 1; Zid[ 97]= 0;
  NLX_coes2[0][ 98]=+3.354629339837808e-02;   Eid[ 98]= 8; Rid[ 98]= 1; Zid[ 98]= 1;
  NLX_coes2[0][ 99]=-3.981832776167620e-01;   Eid[ 99]= 8; Rid[ 99]= 1; Zid[ 99]= 2;
  NLX_coes2[0][100]=+1.034349293161551e+00;   Eid[100]= 8; Rid[100]= 1; Zid[100]= 3;
  NLX_coes2[0][101]=-1.490380103576923e-01;   Eid[101]= 8; Rid[101]= 2; Zid[101]= 0;
  NLX_coes2[0][102]=+8.908052943298603e-01;   Eid[102]= 8; Rid[102]= 2; Zid[102]= 1;
  NLX_coes2[0][103]=-1.598377574215626e+00;   Eid[103]= 8; Rid[103]= 2; Zid[103]= 2;
  NLX_coes2[0][104]=-2.239055563387756e-01;   Eid[104]= 8; Rid[104]= 2; Zid[104]= 3;
  NLX_coes2[0][105]=+3.891506358476467e-01;   Eid[105]= 9; Rid[105]= 0; Zid[105]= 0;
  NLX_coes2[0][106]=-1.714113296791020e-02;   Eid[106]= 9; Rid[106]= 0; Zid[106]= 1;
  NLX_coes2[0][107]=-4.659958330705803e-01;   Eid[107]= 9; Rid[107]= 0; Zid[107]= 2;
  NLX_coes2[0][108]=-3.326056750603378e-01;   Eid[108]= 9; Rid[108]= 0; Zid[108]= 3;
  NLX_coes2[0][109]=-6.618242572158974e-01;   Eid[109]= 9; Rid[109]= 1; Zid[109]= 0;
  NLX_coes2[0][110]=+4.486067900101366e-01;   Eid[110]= 9; Rid[110]= 1; Zid[110]= 1;
  NLX_coes2[0][111]=-3.135996840582749e-01;   Eid[111]= 9; Rid[111]= 1; Zid[111]= 2;
  NLX_coes2[0][112]=+4.142080255634746e-01;   Eid[112]= 9; Rid[112]= 1; Zid[112]= 3;
  NLX_coes2[0][113]=-2.390910521337717e-02;   Eid[113]= 9; Rid[113]= 2; Zid[113]= 0;
  NLX_coes2[0][114]=+6.374081981185433e-01;   Eid[114]= 9; Rid[114]= 2; Zid[114]= 1;
  NLX_coes2[0][115]=-1.636874177076521e+00;   Eid[115]= 9; Rid[115]= 2; Zid[115]= 2;
  NLX_coes2[0][116]=-3.915092080442885e-01;   Eid[116]= 9; Rid[116]= 2; Zid[116]= 3;
  NLX_coes2[0][117]=+2.287998536846430e-01;   Eid[117]=10; Rid[117]= 0; Zid[117]= 0;
  NLX_coes2[0][118]=+5.058886809925901e-02;   Eid[118]=10; Rid[118]= 0; Zid[118]= 1;
  NLX_coes2[0][119]=-4.973903163367537e-01;   Eid[119]=10; Rid[119]= 0; Zid[119]= 2;
  NLX_coes2[0][120]=+3.135062460331319e-01;   Eid[120]=10; Rid[120]= 0; Zid[120]= 3;
  NLX_coes2[0][121]=-7.669018147887596e-01;   Eid[121]=10; Rid[121]= 1; Zid[121]= 0;
  NLX_coes2[0][122]=+8.211306709493673e-01;   Eid[122]=10; Rid[122]= 1; Zid[122]= 1;
  NLX_coes2[0][123]=-2.391062670133528e-01;   Eid[123]=10; Rid[123]= 1; Zid[123]= 2;
  NLX_coes2[0][124]=-1.462587901976049e-01;   Eid[124]=10; Rid[124]= 1; Zid[124]= 3;
  NLX_coes2[0][125]=+1.889792643588341e-01;   Eid[125]=10; Rid[125]= 2; Zid[125]= 0;
  NLX_coes2[0][126]=+1.840717068917624e-01;   Eid[126]=10; Rid[126]= 2; Zid[126]= 1;
  NLX_coes2[0][127]=-1.291414886115566e+00;   Eid[127]=10; Rid[127]= 2; Zid[127]= 2;
  NLX_coes2[0][128]=-1.301946208383997e-01;   Eid[128]=10; Rid[128]= 2; Zid[128]= 3;
  NLX_coes2[0][129]=-1.086452069158834e-02;   Eid[129]=11; Rid[129]= 0; Zid[129]= 0;
  NLX_coes2[0][130]=+1.126882432816693e-01;   Eid[130]=11; Rid[130]= 0; Zid[130]= 1;
  NLX_coes2[0][131]=-4.019598920457899e-01;   Eid[131]=11; Rid[131]= 0; Zid[131]= 2;
  NLX_coes2[0][132]=+8.647558169158809e-01;   Eid[132]=11; Rid[132]= 0; Zid[132]= 3;
  NLX_coes2[0][133]=-7.584287918533401e-01;   Eid[133]=11; Rid[133]= 1; Zid[133]= 0;
  NLX_coes2[0][134]=+1.071954016780580e+00;   Eid[134]=11; Rid[134]= 1; Zid[134]= 1;
  NLX_coes2[0][135]=-2.808316096729333e-01;   Eid[135]=11; Rid[135]= 1; Zid[135]= 2;
  NLX_coes2[0][136]=-5.941449204560439e-01;   Eid[136]=11; Rid[136]= 1; Zid[136]= 3;
  NLX_coes2[0][137]=+4.502000292152873e-01;   Eid[137]=11; Rid[137]= 2; Zid[137]= 0;
  NLX_coes2[0][138]=-2.304256463421097e-01;   Eid[138]=11; Rid[138]= 2; Zid[138]= 1;
  NLX_coes2[0][139]=-6.897694322623931e-01;   Eid[139]=11; Rid[139]= 2; Zid[139]= 2;
  NLX_coes2[0][140]=+4.917085600844646e-01;   Eid[140]=11; Rid[140]= 2; Zid[140]= 3;
  NLX_coes2[0][141]=-2.918124256372600e-01;   Eid[141]=12; Rid[141]= 0; Zid[141]= 0;
  NLX_coes2[0][142]=+2.056798771365357e-01;   Eid[142]=12; Rid[142]= 0; Zid[142]= 1;
  NLX_coes2[0][143]=-2.128591459233165e-01;   Eid[143]=12; Rid[143]= 0; Zid[143]= 2;
  NLX_coes2[0][144]=+9.730078558448406e-01;   Eid[144]=12; Rid[144]= 0; Zid[144]= 3;
  NLX_coes2[0][145]=-6.692690951119896e-01;   Eid[145]=12; Rid[145]= 1; Zid[145]= 0;
  NLX_coes2[0][146]=+1.145093057894647e+00;   Eid[146]=12; Rid[146]= 1; Zid[146]= 1;
  NLX_coes2[0][147]=-4.908890664439574e-01;   Eid[147]=12; Rid[147]= 1; Zid[147]= 2;
  NLX_coes2[0][148]=-8.845739592904539e-01;   Eid[148]=12; Rid[148]= 1; Zid[148]= 3;
  NLX_coes2[0][149]=+1.990212185640335e-01;   Eid[149]=12; Rid[149]= 2; Zid[149]= 0;
  NLX_coes2[0][150]=-4.026537793956244e-01;   Eid[150]=12; Rid[150]= 2; Zid[150]= 1;
  NLX_coes2[0][151]=+1.057961429612417e-01;   Eid[151]=12; Rid[151]= 2; Zid[151]= 2;
  NLX_coes2[0][152]=+1.162815826229039e+00;   Eid[152]=12; Rid[152]= 2; Zid[152]= 3;
  NLX_coes2[0][153]=-5.678130970247607e-01;   Eid[153]=13; Rid[153]= 0; Zid[153]= 0;
  NLX_coes2[0][154]=+3.672403094358571e-01;   Eid[154]=13; Rid[154]= 0; Zid[154]= 1;
  NLX_coes2[0][155]=+1.187824221718223e-02;   Eid[155]=13; Rid[155]= 0; Zid[155]= 2;
  NLX_coes2[0][156]=+4.958815361871505e-01;   Eid[156]=13; Rid[156]= 0; Zid[156]= 3;
  NLX_coes2[0][157]=-5.633351598306598e-01;   Eid[157]=13; Rid[157]= 1; Zid[157]= 0;
  NLX_coes2[0][158]=+1.020366099981994e+00;   Eid[158]=13; Rid[158]= 1; Zid[158]= 1;
  NLX_coes2[0][159]=-8.278952736661400e-01;   Eid[159]=13; Rid[159]= 1; Zid[159]= 2;
  NLX_coes2[0][160]=-8.210392217927663e-01;   Eid[160]=13; Rid[160]= 1; Zid[160]= 3;
  NLX_coes2[0][161]=-6.560347088434648e-01;   Eid[161]=13; Rid[161]= 2; Zid[161]= 0;
  NLX_coes2[0][162]=-5.731188089341833e-01;   Eid[162]=13; Rid[162]= 2; Zid[162]= 1;
  NLX_coes2[0][163]=+9.119174845224026e-01;   Eid[163]=13; Rid[163]= 2; Zid[163]= 2;
  NLX_coes2[0][164]=+1.447093340022419e+00;   Eid[164]=13; Rid[164]= 2; Zid[164]= 3;
  NLX_coes2[0][165]=-7.910951798464975e-01;   Eid[165]=14; Rid[165]= 0; Zid[165]= 0;
  NLX_coes2[0][166]=+6.215361603091556e-01;   Eid[166]=14; Rid[166]= 0; Zid[166]= 1;
  NLX_coes2[0][167]=+1.258468130947607e-01;   Eid[167]=14; Rid[167]= 0; Zid[167]= 2;
  NLX_coes2[0][168]=-2.597572042595845e-01;   Eid[168]=14; Rid[168]= 0; Zid[168]= 3;
  NLX_coes2[0][169]=-3.655745815769206e-01;   Eid[169]=14; Rid[169]= 1; Zid[169]= 0;
  NLX_coes2[0][170]=+7.509007816207515e-01;   Eid[170]=14; Rid[170]= 1; Zid[170]= 1;
  NLX_coes2[0][171]=-1.141402093307930e+00;   Eid[171]=14; Rid[171]= 1; Zid[171]= 2;
  NLX_coes2[0][172]=+1.157501067801496e-01;   Eid[172]=14; Rid[172]= 1; Zid[172]= 3;
  NLX_coes2[0][173]=-6.783443697685941e-02;   Eid[173]=14; Rid[173]= 2; Zid[173]= 0;
  NLX_coes2[0][174]=-8.130785571347355e-01;   Eid[174]=14; Rid[174]= 2; Zid[174]= 1;
  NLX_coes2[0][175]=+9.925705561277032e-01;   Eid[175]=14; Rid[175]= 2; Zid[175]= 2;
  NLX_coes2[0][176]=+8.973423800076171e-01;   Eid[176]=14; Rid[176]= 2; Zid[176]= 3;
  NLX_coes2[0][177]=-9.603645661710689e-01;   Eid[177]=15; Rid[177]= 0; Zid[177]= 0;
  NLX_coes2[0][178]=+9.165678585648646e-01;   Eid[178]=15; Rid[178]= 0; Zid[178]= 1;
  NLX_coes2[0][179]=-1.911488380768262e-01;   Eid[179]=15; Rid[179]= 0; Zid[179]= 2;
  NLX_coes2[0][180]=-3.234571033219261e-01;   Eid[180]=15; Rid[180]= 0; Zid[180]= 3;
  NLX_coes2[0][181]=-3.104899257225990e-01;   Eid[181]=15; Rid[181]= 1; Zid[181]= 0;
  NLX_coes2[0][182]=+6.947401222201857e-01;   Eid[182]=15; Rid[182]= 1; Zid[182]= 1;
  NLX_coes2[0][183]=-1.176725544536349e+00;   Eid[183]=15; Rid[183]= 1; Zid[183]= 2;
  NLX_coes2[0][184]=+2.821306767025967e+00;   Eid[184]=15; Rid[184]= 1; Zid[184]= 3;
  NLX_coes2[0][185]=+5.468283438201943e-02;   Eid[185]=15; Rid[185]= 2; Zid[185]= 0;
  NLX_coes2[0][186]=+1.034554980831810e+00;   Eid[186]=15; Rid[186]= 2; Zid[186]= 1;
  NLX_coes2[0][187]=-1.049006393636184e+00;   Eid[187]=15; Rid[187]= 2; Zid[187]= 2;
  NLX_coes2[0][188]=-7.732433001947824e-01;   Eid[188]=15; Rid[188]= 2; Zid[188]= 3;
  NLX_coes2[1][  0]=+2.941189770054177e+00;   Eid[  0]= 0; Rid[  0]= 0; Zid[  0]= 1;
  NLX_coes2[1][  1]=-7.186551372841505e-01;   Eid[  1]= 0; Rid[  1]= 0; Zid[  1]= 2;
  NLX_coes2[1][  2]=+2.074159426813577e-01;   Eid[  2]= 0; Rid[  2]= 0; Zid[  2]= 3;
  NLX_coes2[1][  3]=-7.195262224568086e-01;   Eid[  3]= 0; Rid[  3]= 1; Zid[  3]= 1;
  NLX_coes2[1][  4]=+2.622752091506641e-02;   Eid[  4]= 0; Rid[  4]= 1; Zid[  4]= 2;
  NLX_coes2[1][  5]=-7.240069694939630e-02;   Eid[  5]= 0; Rid[  5]= 1; Zid[  5]= 3;
  NLX_coes2[1][  6]=+1.088845125490687e+00;   Eid[  6]= 0; Rid[  6]= 2; Zid[  6]= 1;
  NLX_coes2[1][  7]=+2.721609546153359e+00;   Eid[  7]= 0; Rid[  7]= 2; Zid[  7]= 2;
  NLX_coes2[1][  8]=-5.582818117539251e-01;   Eid[  8]= 0; Rid[  8]= 2; Zid[  8]= 3;
  NLX_coes2[1][  9]=-2.956520413821047e+00;   Eid[  9]= 1; Rid[  9]= 0; Zid[  9]= 0;
  NLX_coes2[1][ 10]=+2.906739663493637e+00;   Eid[ 10]= 1; Rid[ 10]= 0; Zid[ 10]= 1;
  NLX_coes2[1][ 11]=-9.158513805348667e-01;   Eid[ 11]= 1; Rid[ 11]= 0; Zid[ 11]= 2;
  NLX_coes2[1][ 12]=-5.381657947274123e-01;   Eid[ 12]= 1; Rid[ 12]= 0; Zid[ 12]= 3;
  NLX_coes2[1][ 13]=-8.144397072455248e-01;   Eid[ 13]= 1; Rid[ 13]= 1; Zid[ 13]= 0;
  NLX_coes2[1][ 14]=-1.101707381826234e+00;   Eid[ 14]= 1; Rid[ 14]= 1; Zid[ 14]= 1;
  NLX_coes2[1][ 15]=+2.654550519776886e-01;   Eid[ 15]= 1; Rid[ 15]= 1; Zid[ 15]= 2;
  NLX_coes2[1][ 16]=+1.381492503951367e+00;   Eid[ 16]= 1; Rid[ 16]= 1; Zid[ 16]= 3;
  NLX_coes2[1][ 17]=-8.777387260131320e-01;   Eid[ 17]= 1; Rid[ 17]= 2; Zid[ 17]= 0;
  NLX_coes2[1][ 18]=-8.631301969245776e-01;   Eid[ 18]= 1; Rid[ 18]= 2; Zid[ 18]= 1;
  NLX_coes2[1][ 19]=+5.127221133571925e-01;   Eid[ 19]= 1; Rid[ 19]= 2; Zid[ 19]= 2;
  NLX_coes2[1][ 20]=-1.114394743221328e+00;   Eid[ 20]= 1; Rid[ 20]= 2; Zid[ 20]= 3;
  NLX_coes2[1][ 21]=-2.678281846378263e+00;   Eid[ 21]= 2; Rid[ 21]= 0; Zid[ 21]= 0;
  NLX_coes2[1][ 22]=+2.802093625526427e+00;   Eid[ 22]= 2; Rid[ 22]= 0; Zid[ 22]= 1;
  NLX_coes2[1][ 23]=-1.280275229085387e+00;   Eid[ 23]= 2; Rid[ 23]= 0; Zid[ 23]= 2;
  NLX_coes2[1][ 24]=-1.822047143331561e-01;   Eid[ 24]= 2; Rid[ 24]= 0; Zid[ 24]= 3;
  NLX_coes2[1][ 25]=-5.298482661052666e-01;   Eid[ 25]= 2; Rid[ 25]= 1; Zid[ 25]= 0;
  NLX_coes2[1][ 26]=-1.086913182810340e+00;   Eid[ 26]= 2; Rid[ 26]= 1; Zid[ 26]= 1;
  NLX_coes2[1][ 27]=+5.814287776402874e-01;   Eid[ 27]= 2; Rid[ 27]= 1; Zid[ 27]= 2;
  NLX_coes2[1][ 28]=+1.349210638441798e+00;   Eid[ 28]= 2; Rid[ 28]= 1; Zid[ 28]= 3;
  NLX_coes2[1][ 29]=-2.993172379477659e-01;   Eid[ 29]= 2; Rid[ 29]= 2; Zid[ 29]= 0;
  NLX_coes2[1][ 30]=-1.177687815053751e+00;   Eid[ 30]= 2; Rid[ 30]= 2; Zid[ 30]= 1;
  NLX_coes2[1][ 31]=-3.903381160714150e-01;   Eid[ 31]= 2; Rid[ 31]= 2; Zid[ 31]= 2;
  NLX_coes2[1][ 32]=-1.329504150558452e+00;   Eid[ 32]= 2; Rid[ 32]= 2; Zid[ 32]= 3;
  NLX_coes2[1][ 33]=-2.419184563560861e+00;   Eid[ 33]= 3; Rid[ 33]= 0; Zid[ 33]= 0;
  NLX_coes2[1][ 34]=+2.617904614968026e+00;   Eid[ 34]= 3; Rid[ 34]= 0; Zid[ 34]= 1;
  NLX_coes2[1][ 35]=-1.529896692102056e+00;   Eid[ 35]= 3; Rid[ 35]= 0; Zid[ 35]= 2;
  NLX_coes2[1][ 36]=+2.537087893960411e-01;   Eid[ 36]= 3; Rid[ 36]= 0; Zid[ 36]= 3;
  NLX_coes2[1][ 37]=-2.860702562203587e-01;   Eid[ 37]= 3; Rid[ 37]= 1; Zid[ 37]= 0;
  NLX_coes2[1][ 38]=-9.902381831208801e-01;   Eid[ 38]= 3; Rid[ 38]= 1; Zid[ 38]= 1;
  NLX_coes2[1][ 39]=+8.760436253969566e-01;   Eid[ 39]= 3; Rid[ 39]= 1; Zid[ 39]= 2;
  NLX_coes2[1][ 40]=+5.764833280354987e-01;   Eid[ 40]= 3; Rid[ 40]= 1; Zid[ 40]= 3;
  NLX_coes2[1][ 41]=+5.784775458495262e-01;   Eid[ 41]= 3; Rid[ 41]= 2; Zid[ 41]= 0;
  NLX_coes2[1][ 42]=-7.414111830892077e-01;   Eid[ 42]= 3; Rid[ 42]= 2; Zid[ 42]= 1;
  NLX_coes2[1][ 43]=-3.015068547281938e-01;   Eid[ 43]= 3; Rid[ 43]= 2; Zid[ 43]= 2;
  NLX_coes2[1][ 44]=-9.571915662982664e-01;   Eid[ 44]= 3; Rid[ 44]= 2; Zid[ 44]= 3;
  NLX_coes2[1][ 45]=-2.178293710328956e+00;   Eid[ 45]= 4; Rid[ 45]= 0; Zid[ 45]= 0;
  NLX_coes2[1][ 46]=+2.374177298426198e+00;   Eid[ 46]= 4; Rid[ 46]= 0; Zid[ 46]= 1;
  NLX_coes2[1][ 47]=-1.551767106146896e+00;   Eid[ 47]= 4; Rid[ 47]= 0; Zid[ 47]= 2;
  NLX_coes2[1][ 48]=+4.889915219340081e-01;   Eid[ 48]= 4; Rid[ 48]= 0; Zid[ 48]= 3;
  NLX_coes2[1][ 49]=-1.535958874813292e-01;   Eid[ 49]= 4; Rid[ 49]= 1; Zid[ 49]= 0;
  NLX_coes2[1][ 50]=-9.474160853164397e-01;   Eid[ 50]= 4; Rid[ 50]= 1; Zid[ 50]= 1;
  NLX_coes2[1][ 51]=+1.054724211986122e+00;   Eid[ 51]= 4; Rid[ 51]= 1; Zid[ 51]= 2;
  NLX_coes2[1][ 52]=-4.017091088194118e-01;   Eid[ 52]= 4; Rid[ 52]= 1; Zid[ 52]= 3;
  NLX_coes2[1][ 53]=+1.166004143471997e+00;   Eid[ 53]= 4; Rid[ 53]= 2; Zid[ 53]= 0;
  NLX_coes2[1][ 54]=-3.485822168862228e-01;   Eid[ 54]= 4; Rid[ 54]= 2; Zid[ 54]= 1;
  NLX_coes2[1][ 55]=-5.197924755917341e-02;   Eid[ 55]= 4; Rid[ 55]= 2; Zid[ 55]= 2;
  NLX_coes2[1][ 56]=-5.057793201883918e-01;   Eid[ 56]= 4; Rid[ 56]= 2; Zid[ 56]= 3;
  NLX_coes2[1][ 57]=-1.941180279617493e+00;   Eid[ 57]= 5; Rid[ 57]= 0; Zid[ 57]= 0;
  NLX_coes2[1][ 58]=+2.097349912099538e+00;   Eid[ 58]= 5; Rid[ 58]= 0; Zid[ 58]= 1;
  NLX_coes2[1][ 59]=-1.342601877060573e+00;   Eid[ 59]= 5; Rid[ 59]= 0; Zid[ 59]= 2;
  NLX_coes2[1][ 60]=+5.498660065781770e-01;   Eid[ 60]= 5; Rid[ 60]= 0; Zid[ 60]= 3;
  NLX_coes2[1][ 61]=-9.597907588771049e-02;   Eid[ 61]= 5; Rid[ 61]= 1; Zid[ 61]= 0;
  NLX_coes2[1][ 62]=-9.561985749808162e-01;   Eid[ 62]= 5; Rid[ 62]= 1; Zid[ 62]= 1;
  NLX_coes2[1][ 63]=+1.152039324153886e+00;   Eid[ 63]= 5; Rid[ 63]= 1; Zid[ 63]= 2;
  NLX_coes2[1][ 64]=-1.221918071428859e+00;   Eid[ 64]= 5; Rid[ 64]= 1; Zid[ 64]= 3;
  NLX_coes2[1][ 65]=+1.386520219885987e+00;   Eid[ 65]= 5; Rid[ 65]= 2; Zid[ 65]= 0;
  NLX_coes2[1][ 66]=-1.425274385534284e-01;   Eid[ 66]= 5; Rid[ 66]= 2; Zid[ 66]= 1;
  NLX_coes2[1][ 67]=+1.105882555500907e-01;   Eid[ 67]= 5; Rid[ 67]= 2; Zid[ 67]= 2;
  NLX_coes2[1][ 68]=-3.148496164766062e-01;   Eid[ 68]= 5; Rid[ 68]= 2; Zid[ 68]= 3;
  NLX_coes2[1][ 69]=-1.697258528926022e+00;   Eid[ 69]= 6; Rid[ 69]= 0; Zid[ 69]= 0;
  NLX_coes2[1][ 70]=+1.800083891074755e+00;   Eid[ 70]= 6; Rid[ 70]= 0; Zid[ 70]= 1;
  NLX_coes2[1][ 71]=-9.727110816955670e-01;   Eid[ 71]= 6; Rid[ 71]= 0; Zid[ 71]= 2;
  NLX_coes2[1][ 72]=+5.554854113763097e-01;   Eid[ 72]= 6; Rid[ 72]= 0; Zid[ 72]= 3;
  NLX_coes2[1][ 73]=-6.755793426191378e-02;   Eid[ 73]= 6; Rid[ 73]= 1; Zid[ 73]= 0;
  NLX_coes2[1][ 74]=-1.000899613133843e+00;   Eid[ 74]= 6; Rid[ 74]= 1; Zid[ 74]= 1;
  NLX_coes2[1][ 75]=+1.224284354443997e+00;   Eid[ 75]= 6; Rid[ 75]= 1; Zid[ 75]= 2;
  NLX_coes2[1][ 76]=-1.665229181506130e+00;   Eid[ 76]= 6; Rid[ 76]= 1; Zid[ 76]= 3;
  NLX_coes2[1][ 77]=+1.229466456192354e+00;   Eid[ 77]= 6; Rid[ 77]= 2; Zid[ 77]= 0;
  NLX_coes2[1][ 78]=-1.163889917524719e-01;   Eid[ 78]= 6; Rid[ 78]= 2; Zid[ 78]= 1;
  NLX_coes2[1][ 79]=+2.493970222140579e-01;   Eid[ 79]= 6; Rid[ 79]= 2; Zid[ 79]= 2;
  NLX_coes2[1][ 80]=-2.677676542362221e-01;   Eid[ 80]= 6; Rid[ 80]= 2; Zid[ 80]= 3;
  NLX_coes2[1][ 81]=-1.437636404305272e+00;   Eid[ 81]= 7; Rid[ 81]= 0; Zid[ 81]= 0;
  NLX_coes2[1][ 82]=+1.486444446231658e+00;   Eid[ 82]= 7; Rid[ 82]= 0; Zid[ 82]= 1;
  NLX_coes2[1][ 83]=-5.458952715756389e-01;   Eid[ 83]= 7; Rid[ 83]= 0; Zid[ 83]= 2;
  NLX_coes2[1][ 84]=+5.904421983052650e-01;   Eid[ 84]= 7; Rid[ 84]= 0; Zid[ 84]= 3;
  NLX_coes2[1][ 85]=-2.093822273644197e-02;   Eid[ 85]= 7; Rid[ 85]= 1; Zid[ 85]= 0;
  NLX_coes2[1][ 86]=-1.082119747637911e+00;   Eid[ 86]= 7; Rid[ 86]= 1; Zid[ 86]= 1;
  NLX_coes2[1][ 87]=+1.280302340341261e+00;   Eid[ 87]= 7; Rid[ 87]= 1; Zid[ 87]= 2;
  NLX_coes2[1][ 88]=-1.682629932844418e+00;   Eid[ 88]= 7; Rid[ 88]= 1; Zid[ 88]= 3;
  NLX_coes2[1][ 89]=+7.338339952103687e-01;   Eid[ 89]= 7; Rid[ 89]= 2; Zid[ 89]= 0;
  NLX_coes2[1][ 90]=-2.928262728130513e-01;   Eid[ 90]= 7; Rid[ 90]= 2; Zid[ 90]= 1;
  NLX_coes2[1][ 91]=+3.984815136439068e-01;   Eid[ 91]= 7; Rid[ 91]= 2; Zid[ 91]= 2;
  NLX_coes2[1][ 92]=-1.584147604853995e-01;   Eid[ 92]= 7; Rid[ 92]= 2; Zid[ 92]= 3;
  NLX_coes2[1][ 93]=-1.153470462756030e+00;   Eid[ 93]= 8; Rid[ 93]= 0; Zid[ 93]= 0;
  NLX_coes2[1][ 94]=+1.163628400683444e+00;   Eid[ 94]= 8; Rid[ 94]= 0; Zid[ 94]= 1;
  NLX_coes2[1][ 95]=-1.597811054247513e-01;   Eid[ 95]= 8; Rid[ 95]= 0; Zid[ 95]= 2;
  NLX_coes2[1][ 96]=+6.476213160171486e-01;   Eid[ 96]= 8; Rid[ 96]= 0; Zid[ 96]= 3;
  NLX_coes2[1][ 97]=+9.595058780883549e-02;   Eid[ 97]= 8; Rid[ 97]= 1; Zid[ 97]= 0;
  NLX_coes2[1][ 98]=-1.194977945845323e+00;   Eid[ 98]= 8; Rid[ 98]= 1; Zid[ 98]= 1;
  NLX_coes2[1][ 99]=+1.292039257098852e+00;   Eid[ 99]= 8; Rid[ 99]= 1; Zid[ 99]= 2;
  NLX_coes2[1][100]=-1.362499117662880e+00;   Eid[100]= 8; Rid[100]= 1; Zid[100]= 3;
  NLX_coes2[1][101]=+6.032039824117547e-02;   Eid[101]= 8; Rid[101]= 2; Zid[101]= 0;
  NLX_coes2[1][102]=-6.703399875837432e-01;   Eid[102]= 8; Rid[102]= 2; Zid[102]= 1;
  NLX_coes2[1][103]=+4.841929373933773e-01;   Eid[103]= 8; Rid[103]= 2; Zid[103]= 2;
  NLX_coes2[1][104]=+8.196347932711884e-02;   Eid[104]= 8; Rid[104]= 2; Zid[104]= 3;
  NLX_coes2[1][105]=-8.399409333393208e-01;   Eid[105]= 9; Rid[105]= 0; Zid[105]= 0;
  NLX_coes2[1][106]=+8.461274540326537e-01;   Eid[106]= 9; Rid[106]= 0; Zid[106]= 1;
  NLX_coes2[1][107]=+1.203255522885407e-01;   Eid[107]= 9; Rid[107]= 0; Zid[107]= 2;
  NLX_coes2[1][108]=+6.249372069411808e-01;   Eid[108]= 9; Rid[108]= 0; Zid[108]= 3;
  NLX_coes2[1][109]=+3.212268890634631e-01;   Eid[109]= 9; Rid[109]= 1; Zid[109]= 0;
  NLX_coes2[1][110]=-1.309447514280182e+00;   Eid[110]= 9; Rid[110]= 1; Zid[110]= 1;
  NLX_coes2[1][111]=+1.236662980163482e+00;   Eid[111]= 9; Rid[111]= 1; Zid[111]= 2;
  NLX_coes2[1][112]=-8.615578243912887e-01;   Eid[112]= 9; Rid[112]= 1; Zid[112]= 3;
  NLX_coes2[1][113]=-5.172173129157170e-01;   Eid[113]= 9; Rid[113]= 2; Zid[113]= 0;
  NLX_coes2[1][114]=-1.121432990343670e+00;   Eid[114]= 9; Rid[114]= 2; Zid[114]= 1;
  NLX_coes2[1][115]=+3.942571725555594e-01;   Eid[115]= 9; Rid[115]= 2; Zid[115]= 2;
  NLX_coes2[1][116]=+3.506229547494966e-01;   Eid[116]= 9; Rid[116]= 2; Zid[116]= 3;
  NLX_coes2[1][117]=-5.028811506522359e-01;   Eid[117]=10; Rid[117]= 0; Zid[117]= 0;
  NLX_coes2[1][118]=+5.496409666743479e-01;   Eid[118]=10; Rid[118]= 0; Zid[118]= 1;
  NLX_coes2[1][119]=+2.692609732012125e-01;   Eid[119]=10; Rid[119]= 0; Zid[119]= 2;
  NLX_coes2[1][120]=+3.669785390381644e-01;   Eid[120]=10; Rid[120]= 0; Zid[120]= 3;
  NLX_coes2[1][121]=+6.392454374708199e-01;   Eid[121]=10; Rid[121]= 1; Zid[121]= 0;
  NLX_coes2[1][122]=-1.372986237429766e+00;   Eid[122]=10; Rid[122]= 1; Zid[122]= 1;
  NLX_coes2[1][123]=+1.130960322528294e+00;   Eid[123]=10; Rid[123]= 1; Zid[123]= 2;
  NLX_coes2[1][124]=-3.387553496275770e-01;   Eid[124]=10; Rid[124]= 1; Zid[124]= 3;
  NLX_coes2[1][125]=-7.339113599252879e-01;   Eid[125]=10; Rid[125]= 2; Zid[125]= 0;
  NLX_coes2[1][126]=-1.355785891272155e+00;   Eid[126]=10; Rid[126]= 2; Zid[126]= 1;
  NLX_coes2[1][127]=+8.347314451677000e-02;   Eid[127]=10; Rid[127]= 2; Zid[127]= 2;
  NLX_coes2[1][128]=+4.489941515372540e-01;   Eid[128]=10; Rid[128]= 2; Zid[128]= 3;
  NLX_coes2[1][129]=-1.614198060539191e-01;   Eid[129]=11; Rid[129]= 0; Zid[129]= 0;
  NLX_coes2[1][130]=+2.795008659556490e-01;   Eid[130]=11; Rid[130]= 0; Zid[130]= 1;
  NLX_coes2[1][131]=+2.980723523183119e-01;   Eid[131]=11; Rid[131]= 0; Zid[131]= 2;
  NLX_coes2[1][132]=-2.427396624664517e-01;   Eid[132]=11; Rid[132]= 0; Zid[132]= 3;
  NLX_coes2[1][133]=+9.493616300581120e-01;   Eid[133]=11; Rid[133]= 1; Zid[133]= 0;
  NLX_coes2[1][134]=-1.341911658455337e+00;   Eid[134]=11; Rid[134]= 1; Zid[134]= 1;
  NLX_coes2[1][135]=+1.041113407465734e+00;   Eid[135]=11; Rid[135]= 1; Zid[135]= 2;
  NLX_coes2[1][136]=+8.609114488496251e-02;   Eid[136]=11; Rid[136]= 1; Zid[136]= 3;
  NLX_coes2[1][137]=-5.207271439024820e-01;   Eid[137]=11; Rid[137]= 2; Zid[137]= 0;
  NLX_coes2[1][138]=-9.979404239792452e-01;   Eid[138]=11; Rid[138]= 2; Zid[138]= 1;
  NLX_coes2[1][139]=-3.347555387492185e-01;   Eid[139]=11; Rid[139]= 2; Zid[139]= 2;
  NLX_coes2[1][140]=+2.073448361566314e-01;   Eid[140]=11; Rid[140]= 2; Zid[140]= 3;
  NLX_coes2[1][141]=+1.602534229062051e-01;   Eid[141]=12; Rid[141]= 0; Zid[141]= 0;
  NLX_coes2[1][142]=+2.031941379020431e-02;   Eid[142]=12; Rid[142]= 0; Zid[142]= 1;
  NLX_coes2[1][143]=+2.409829294273549e-01;   Eid[143]=12; Rid[143]= 0; Zid[143]= 2;
  NLX_coes2[1][144]=-1.143818190105408e+00;   Eid[144]=12; Rid[144]= 0; Zid[144]= 3;
  NLX_coes2[1][145]=+1.068212418067456e+00;   Eid[145]=12; Rid[145]= 1; Zid[145]= 0;
  NLX_coes2[1][146]=-1.219913539256835e+00;   Eid[146]=12; Rid[146]= 1; Zid[146]= 1;
  NLX_coes2[1][147]=+1.055146611182007e+00;   Eid[147]=12; Rid[147]= 1; Zid[147]= 2;
  NLX_coes2[1][148]=+3.344949060013613e-01;   Eid[148]=12; Rid[148]= 1; Zid[148]= 3;
  NLX_coes2[1][149]=-8.201283175675643e-02;   Eid[149]=12; Rid[149]= 2; Zid[149]= 0;
  NLX_coes2[1][150]=+1.726832153401746e-01;   Eid[150]=12; Rid[150]= 2; Zid[150]= 1;
  NLX_coes2[1][151]=-5.902515188513442e-01;   Eid[151]=12; Rid[151]= 2; Zid[151]= 2;
  NLX_coes2[1][152]=-3.592722213882858e-01;   Eid[152]=12; Rid[152]= 2; Zid[152]= 3;
  NLX_coes2[1][153]=+4.472689870405144e-01;   Eid[153]=13; Rid[153]= 0; Zid[153]= 0;
  NLX_coes2[1][154]=-2.613902182676004e-01;   Eid[154]=13; Rid[154]= 0; Zid[154]= 1;
  NLX_coes2[1][155]=+1.528553503644678e-01;   Eid[155]=13; Rid[155]= 0; Zid[155]= 2;
  NLX_coes2[1][156]=-1.917114117125269e+00;   Eid[156]=13; Rid[156]= 0; Zid[156]= 3;
  NLX_coes2[1][157]=+8.238421970808605e-01;   Eid[157]=13; Rid[157]= 1; Zid[157]= 0;
  NLX_coes2[1][158]=-1.083947161419454e+00;   Eid[158]=13; Rid[158]= 1; Zid[158]= 1;
  NLX_coes2[1][159]=+1.220822461871707e+00;   Eid[159]=13; Rid[159]= 1; Zid[159]= 2;
  NLX_coes2[1][160]=+3.143174606379166e-01;   Eid[160]=13; Rid[160]= 1; Zid[160]= 3;
  NLX_coes2[1][161]=+3.368792809887850e-01;   Eid[161]=13; Rid[161]= 2; Zid[161]= 0;
  NLX_coes2[1][162]=+1.790381645887034e+00;   Eid[162]=13; Rid[162]= 2; Zid[162]= 1;
  NLX_coes2[1][163]=-4.140413795036330e-01;   Eid[163]=13; Rid[163]= 2; Zid[163]= 2;
  NLX_coes2[1][164]=-9.099905450086047e-01;   Eid[164]=13; Rid[164]= 2; Zid[164]= 3;
  NLX_coes2[1][165]=+7.098619563319648e-01;   Eid[165]=14; Rid[165]= 0; Zid[165]= 0;
  NLX_coes2[1][166]=-5.958907176841691e-01;   Eid[166]=14; Rid[166]= 0; Zid[166]= 1;
  NLX_coes2[1][167]=+1.250248627006114e-01;   Eid[167]=14; Rid[167]= 0; Zid[167]= 2;
  NLX_coes2[1][168]=-1.509284399524985e+00;   Eid[168]=14; Rid[168]= 0; Zid[168]= 3;
  NLX_coes2[1][169]=+2.581372892846718e-01;   Eid[169]=14; Rid[169]= 1; Zid[169]= 0;
  NLX_coes2[1][170]=-1.039509745814442e+00;   Eid[170]=14; Rid[170]= 1; Zid[170]= 1;
  NLX_coes2[1][171]=+1.446264920779734e+00;   Eid[171]=14; Rid[171]= 1; Zid[171]= 2;
  NLX_coes2[1][172]=-2.063813662391170e-01;   Eid[172]=14; Rid[172]= 1; Zid[172]= 3;
  NLX_coes2[1][173]=+7.776067203273109e-01;   Eid[173]=14; Rid[173]= 2; Zid[173]= 0;
  NLX_coes2[1][174]=+2.321775708624348e+00;   Eid[174]=14; Rid[174]= 2; Zid[174]= 1;
  NLX_coes2[1][175]=+5.086189535430304e-02;   Eid[175]=14; Rid[175]= 2; Zid[175]= 2;
  NLX_coes2[1][176]=-6.941293987495369e-01;   Eid[176]=14; Rid[176]= 2; Zid[176]= 3;
  NLX_coes2[1][177]=+9.716651009405570e-01;   Eid[177]=15; Rid[177]= 0; Zid[177]= 0;
  NLX_coes2[1][178]=-9.637521448891643e-01;   Eid[178]=15; Rid[178]= 0; Zid[178]= 1;
  NLX_coes2[1][179]=+3.464663296340451e-01;   Eid[179]=15; Rid[179]= 0; Zid[179]= 2;
  NLX_coes2[1][180]=+2.116156483562697e+00;   Eid[180]=15; Rid[180]= 0; Zid[180]= 3;
  NLX_coes2[1][181]=+4.013897672174895e-02;   Eid[181]=15; Rid[181]= 1; Zid[181]= 0;
  NLX_coes2[1][182]=-1.026035556295001e+00;   Eid[182]=15; Rid[182]= 1; Zid[182]= 1;
  NLX_coes2[1][183]=+1.417099977861381e+00;   Eid[183]=15; Rid[183]= 1; Zid[183]= 2;
  NLX_coes2[1][184]=-1.840666781275283e+00;   Eid[184]=15; Rid[184]= 1; Zid[184]= 3;
  NLX_coes2[1][185]=-3.359681498762609e-01;   Eid[185]=15; Rid[185]= 2; Zid[185]= 0;
  NLX_coes2[1][186]=-1.844470059985287e+00;   Eid[186]=15; Rid[186]= 2; Zid[186]= 1;
  NLX_coes2[1][187]=-5.922978169888971e-01;   Eid[187]=15; Rid[187]= 2; Zid[187]= 2;
  NLX_coes2[1][188]=+1.326077935137412e+00;   Eid[188]=15; Rid[188]= 2; Zid[188]= 3;
  NLX_coes2[2][  0]=-1.130078090786957e-01;   Eid[  0]= 0; Rid[  0]= 0; Zid[  0]= 1;
  NLX_coes2[2][  1]=+4.387608565445420e-01;   Eid[  1]= 0; Rid[  1]= 0; Zid[  1]= 2;
  NLX_coes2[2][  2]=-2.805876830277769e-01;   Eid[  2]= 0; Rid[  2]= 0; Zid[  2]= 3;
  NLX_coes2[2][  3]=-3.655186570847531e-01;   Eid[  3]= 0; Rid[  3]= 1; Zid[  3]= 1;
  NLX_coes2[2][  4]=+2.111029241656577e-01;   Eid[  4]= 0; Rid[  4]= 1; Zid[  4]= 2;
  NLX_coes2[2][  5]=-5.874760524278093e-02;   Eid[  5]= 0; Rid[  5]= 1; Zid[  5]= 3;
  NLX_coes2[2][  6]=+1.594803907311823e-02;   Eid[  6]= 0; Rid[  6]= 2; Zid[  6]= 1;
  NLX_coes2[2][  7]=+7.234838764382279e-01;   Eid[  7]= 0; Rid[  7]= 2; Zid[  7]= 2;
  NLX_coes2[2][  8]=-1.646910252533943e-01;   Eid[  8]= 0; Rid[  8]= 2; Zid[  8]= 3;
  NLX_coes2[2][  9]=-5.418730915382892e-01;   Eid[  9]= 1; Rid[  9]= 0; Zid[  9]= 0;
  NLX_coes2[2][ 10]=-1.028209419868797e-01;   Eid[ 10]= 1; Rid[ 10]= 0; Zid[ 10]= 1;
  NLX_coes2[2][ 11]=+4.256851403635951e-01;   Eid[ 11]= 1; Rid[ 11]= 0; Zid[ 11]= 2;
  NLX_coes2[2][ 12]=+5.421203243297911e-01;   Eid[ 12]= 1; Rid[ 12]= 0; Zid[ 12]= 3;
  NLX_coes2[2][ 13]=-8.429335450136947e-01;   Eid[ 13]= 1; Rid[ 13]= 1; Zid[ 13]= 0;
  NLX_coes2[2][ 14]=-2.811508534665771e-01;   Eid[ 14]= 1; Rid[ 14]= 1; Zid[ 14]= 1;
  NLX_coes2[2][ 15]=-2.508649307073565e-01;   Eid[ 15]= 1; Rid[ 15]= 1; Zid[ 15]= 2;
  NLX_coes2[2][ 16]=-5.533490599540759e-01;   Eid[ 16]= 1; Rid[ 16]= 1; Zid[ 16]= 3;
  NLX_coes2[2][ 17]=+1.042027475845244e+00;   Eid[ 17]= 1; Rid[ 17]= 2; Zid[ 17]= 0;
  NLX_coes2[2][ 18]=-9.233312564029121e-01;   Eid[ 18]= 1; Rid[ 18]= 2; Zid[ 18]= 1;
  NLX_coes2[2][ 19]=-7.940365524840868e-01;   Eid[ 19]= 1; Rid[ 19]= 2; Zid[ 19]= 2;
  NLX_coes2[2][ 20]=+5.335326312270336e-01;   Eid[ 20]= 1; Rid[ 20]= 2; Zid[ 20]= 3;
  NLX_coes2[2][ 21]=-4.858113984667253e-01;   Eid[ 21]= 2; Rid[ 21]= 0; Zid[ 21]= 0;
  NLX_coes2[2][ 22]=-4.228910390166192e-02;   Eid[ 22]= 2; Rid[ 22]= 0; Zid[ 22]= 1;
  NLX_coes2[2][ 23]=+6.160432955735408e-01;   Eid[ 23]= 2; Rid[ 23]= 0; Zid[ 23]= 2;
  NLX_coes2[2][ 24]=+4.328379140484951e-01;   Eid[ 24]= 2; Rid[ 24]= 0; Zid[ 24]= 3;
  NLX_coes2[2][ 25]=-7.917060571113821e-01;   Eid[ 25]= 2; Rid[ 25]= 1; Zid[ 25]= 0;
  NLX_coes2[2][ 26]=-2.126425192857616e-01;   Eid[ 26]= 2; Rid[ 26]= 1; Zid[ 26]= 1;
  NLX_coes2[2][ 27]=-2.024602278535682e-01;   Eid[ 27]= 2; Rid[ 27]= 1; Zid[ 27]= 2;
  NLX_coes2[2][ 28]=-6.879976661973244e-01;   Eid[ 28]= 2; Rid[ 28]= 1; Zid[ 28]= 3;
  NLX_coes2[2][ 29]=+5.872094235342328e-01;   Eid[ 29]= 2; Rid[ 29]= 2; Zid[ 29]= 0;
  NLX_coes2[2][ 30]=+2.232884674567104e-01;   Eid[ 30]= 2; Rid[ 30]= 2; Zid[ 30]= 1;
  NLX_coes2[2][ 31]=+2.160135366473183e-01;   Eid[ 31]= 2; Rid[ 31]= 2; Zid[ 31]= 2;
  NLX_coes2[2][ 32]=-3.219042525965385e-01;   Eid[ 32]= 2; Rid[ 32]= 2; Zid[ 32]= 3;
  NLX_coes2[2][ 33]=-4.231075088193653e-01;   Eid[ 33]= 3; Rid[ 33]= 0; Zid[ 33]= 0;
  NLX_coes2[2][ 34]=+1.581569207944854e-03;   Eid[ 34]= 3; Rid[ 34]= 0; Zid[ 34]= 1;
  NLX_coes2[2][ 35]=+7.042153936252367e-01;   Eid[ 35]= 3; Rid[ 35]= 0; Zid[ 35]= 2;
  NLX_coes2[2][ 36]=-8.491930753061364e-02;   Eid[ 36]= 3; Rid[ 36]= 0; Zid[ 36]= 3;
  NLX_coes2[2][ 37]=-8.363414749577266e-01;   Eid[ 37]= 3; Rid[ 37]= 1; Zid[ 37]= 0;
  NLX_coes2[2][ 38]=-2.874733061257939e-01;   Eid[ 38]= 3; Rid[ 38]= 1; Zid[ 38]= 1;
  NLX_coes2[2][ 39]=-1.553675369872954e-01;   Eid[ 39]= 3; Rid[ 39]= 1; Zid[ 39]= 2;
  NLX_coes2[2][ 40]=+6.769276470817099e-03;   Eid[ 40]= 3; Rid[ 40]= 1; Zid[ 40]= 3;
  NLX_coes2[2][ 41]=+1.345583743988416e-01;   Eid[ 41]= 3; Rid[ 41]= 2; Zid[ 41]= 0;
  NLX_coes2[2][ 42]=+2.664577837986711e-02;   Eid[ 42]= 3; Rid[ 42]= 2; Zid[ 42]= 1;
  NLX_coes2[2][ 43]=+2.014662779851294e-01;   Eid[ 43]= 3; Rid[ 43]= 2; Zid[ 43]= 2;
  NLX_coes2[2][ 44]=-1.729478509584230e-01;   Eid[ 44]= 3; Rid[ 44]= 2; Zid[ 44]= 3;
  NLX_coes2[2][ 45]=-3.499795344429396e-01;   Eid[ 45]= 4; Rid[ 45]= 0; Zid[ 45]= 0;
  NLX_coes2[2][ 46]=+3.121205379885602e-02;   Eid[ 46]= 4; Rid[ 46]= 0; Zid[ 46]= 1;
  NLX_coes2[2][ 47]=+6.203889901017554e-01;   Eid[ 47]= 4; Rid[ 47]= 0; Zid[ 47]= 2;
  NLX_coes2[2][ 48]=-7.412684333162457e-01;   Eid[ 48]= 4; Rid[ 48]= 0; Zid[ 48]= 3;
  NLX_coes2[2][ 49]=-7.975533927320807e-01;   Eid[ 49]= 4; Rid[ 49]= 1; Zid[ 49]= 0;
  NLX_coes2[2][ 50]=-2.777913395884709e-01;   Eid[ 50]= 4; Rid[ 50]= 1; Zid[ 50]= 1;
  NLX_coes2[2][ 51]=-6.370061388458570e-02;   Eid[ 51]= 4; Rid[ 51]= 1; Zid[ 51]= 2;
  NLX_coes2[2][ 52]=+6.315041154347898e-01;   Eid[ 52]= 4; Rid[ 52]= 1; Zid[ 52]= 3;
  NLX_coes2[2][ 53]=+2.217153367777467e-01;   Eid[ 53]= 4; Rid[ 53]= 2; Zid[ 53]= 0;
  NLX_coes2[2][ 54]=-3.063502412077540e-01;   Eid[ 54]= 4; Rid[ 54]= 2; Zid[ 54]= 1;
  NLX_coes2[2][ 55]=-2.591587591499578e-01;   Eid[ 55]= 4; Rid[ 55]= 2; Zid[ 55]= 2;
  NLX_coes2[2][ 56]=-5.852281330192209e-01;   Eid[ 56]= 4; Rid[ 56]= 2; Zid[ 56]= 3;
  NLX_coes2[2][ 57]=-2.911786360242192e-01;   Eid[ 57]= 5; Rid[ 57]= 0; Zid[ 57]= 0;
  NLX_coes2[2][ 58]=+2.600717486812524e-02;   Eid[ 58]= 5; Rid[ 58]= 0; Zid[ 58]= 1;
  NLX_coes2[2][ 59]=+4.080466394391348e-01;   Eid[ 59]= 5; Rid[ 59]= 0; Zid[ 59]= 2;
  NLX_coes2[2][ 60]=-1.099525609058777e+00;   Eid[ 60]= 5; Rid[ 60]= 0; Zid[ 60]= 3;
  NLX_coes2[2][ 61]=-6.978885253538289e-01;   Eid[ 61]= 5; Rid[ 61]= 1; Zid[ 61]= 0;
  NLX_coes2[2][ 62]=-1.866677655725303e-01;   Eid[ 62]= 5; Rid[ 62]= 1; Zid[ 62]= 1;
  NLX_coes2[2][ 63]=+1.108149357469648e-01;   Eid[ 63]= 5; Rid[ 63]= 1; Zid[ 63]= 2;
  NLX_coes2[2][ 64]=+1.020706838904227e+00;   Eid[ 64]= 5; Rid[ 64]= 1; Zid[ 64]= 3;
  NLX_coes2[2][ 65]=+5.640274618423941e-01;   Eid[ 65]= 5; Rid[ 65]= 2; Zid[ 65]= 0;
  NLX_coes2[2][ 66]=-4.752479694382898e-01;   Eid[ 66]= 5; Rid[ 66]= 2; Zid[ 66]= 1;
  NLX_coes2[2][ 67]=-4.051314281012746e-01;   Eid[ 67]= 5; Rid[ 67]= 2; Zid[ 67]= 2;
  NLX_coes2[2][ 68]=-7.109188784898198e-01;   Eid[ 68]= 5; Rid[ 68]= 2; Zid[ 68]= 3;
  NLX_coes2[2][ 69]=-2.562555621658564e-01;   Eid[ 69]= 6; Rid[ 69]= 0; Zid[ 69]= 0;
  NLX_coes2[2][ 70]=-2.043916964004305e-02;   Eid[ 70]= 6; Rid[ 70]= 0; Zid[ 70]= 1;
  NLX_coes2[2][ 71]=+1.375725622792718e-01;   Eid[ 71]= 6; Rid[ 71]= 0; Zid[ 71]= 2;
  NLX_coes2[2][ 72]=-9.331391931973814e-01;   Eid[ 72]= 6; Rid[ 72]= 0; Zid[ 72]= 3;
  NLX_coes2[2][ 73]=-5.601276725875884e-01;   Eid[ 73]= 6; Rid[ 73]= 1; Zid[ 73]= 0;
  NLX_coes2[2][ 74]=-9.810028641985620e-02;   Eid[ 74]= 6; Rid[ 74]= 1; Zid[ 74]= 1;
  NLX_coes2[2][ 75]=+2.413954971823173e-01;   Eid[ 75]= 6; Rid[ 75]= 1; Zid[ 75]= 2;
  NLX_coes2[2][ 76]=+1.080762431625584e+00;   Eid[ 76]= 6; Rid[ 76]= 1; Zid[ 76]= 3;
  NLX_coes2[2][ 77]=+8.917006193534561e-01;   Eid[ 77]= 6; Rid[ 77]= 2; Zid[ 77]= 0;
  NLX_coes2[2][ 78]=-6.354706032426022e-01;   Eid[ 78]= 6; Rid[ 78]= 2; Zid[ 78]= 1;
  NLX_coes2[2][ 79]=-2.822802416888984e-01;   Eid[ 79]= 6; Rid[ 79]= 2; Zid[ 79]= 2;
  NLX_coes2[2][ 80]=-2.548129886305527e-01;   Eid[ 80]= 6; Rid[ 80]= 2; Zid[ 80]= 3;
  NLX_coes2[2][ 81]=-2.390340915607522e-01;   Eid[ 81]= 7; Rid[ 81]= 0; Zid[ 81]= 0;
  NLX_coes2[2][ 82]=-8.155909557183229e-02;   Eid[ 82]= 7; Rid[ 82]= 0; Zid[ 82]= 1;
  NLX_coes2[2][ 83]=-1.001385735190436e-01;   Eid[ 83]= 7; Rid[ 83]= 0; Zid[ 83]= 2;
  NLX_coes2[2][ 84]=-3.013805465397326e-01;   Eid[ 84]= 7; Rid[ 84]= 0; Zid[ 84]= 3;
  NLX_coes2[2][ 85]=-3.756879575808825e-01;   Eid[ 85]= 7; Rid[ 85]= 1; Zid[ 85]= 0;
  NLX_coes2[2][ 86]=-4.478361052987782e-02;   Eid[ 86]= 7; Rid[ 86]= 1; Zid[ 86]= 1;
  NLX_coes2[2][ 87]=+2.183340806212675e-01;   Eid[ 87]= 7; Rid[ 87]= 1; Zid[ 87]= 2;
  NLX_coes2[2][ 88]=+7.595306140303320e-01;   Eid[ 88]= 7; Rid[ 88]= 1; Zid[ 88]= 3;
  NLX_coes2[2][ 89]=+1.104797449241467e+00;   Eid[ 89]= 7; Rid[ 89]= 2; Zid[ 89]= 0;
  NLX_coes2[2][ 90]=-8.436674565267243e-01;   Eid[ 90]= 7; Rid[ 90]= 2; Zid[ 90]= 1;
  NLX_coes2[2][ 91]=-1.626214050679242e-01;   Eid[ 91]= 7; Rid[ 91]= 2; Zid[ 91]= 2;
  NLX_coes2[2][ 92]=+4.375044423876598e-01;   Eid[ 92]= 7; Rid[ 92]= 2; Zid[ 92]= 3;
  NLX_coes2[2][ 93]=-2.327110685976232e-01;   Eid[ 93]= 8; Rid[ 93]= 0; Zid[ 93]= 0;
  NLX_coes2[2][ 94]=-1.164816040858924e-01;   Eid[ 94]= 8; Rid[ 94]= 0; Zid[ 94]= 1;
  NLX_coes2[2][ 95]=-2.171241920962431e-01;   Eid[ 95]= 8; Rid[ 95]= 0; Zid[ 95]= 2;
  NLX_coes2[2][ 96]=+5.163385401947863e-01;   Eid[ 96]= 8; Rid[ 96]= 0; Zid[ 96]= 3;
  NLX_coes2[2][ 97]=-1.417760859814906e-01;   Eid[ 97]= 8; Rid[ 97]= 1; Zid[ 97]= 0;
  NLX_coes2[2][ 98]=-4.725835925942639e-03;   Eid[ 98]= 8; Rid[ 98]= 1; Zid[ 98]= 1;
  NLX_coes2[2][ 99]=+4.118435581943292e-02;   Eid[ 99]= 8; Rid[ 99]= 1; Zid[ 99]= 2;
  NLX_coes2[2][100]=+1.425594581238701e-01;   Eid[100]= 8; Rid[100]= 1; Zid[100]= 3;
  NLX_coes2[2][101]=+1.106553319400778e+00;   Eid[101]= 8; Rid[101]= 2; Zid[101]= 0;
  NLX_coes2[2][102]=-1.010036165801236e+00;   Eid[102]= 8; Rid[102]= 2; Zid[102]= 1;
  NLX_coes2[2][103]=-1.810541147645853e-01;   Eid[103]= 8; Rid[103]= 2; Zid[103]= 2;
  NLX_coes2[2][104]=+9.154052257118382e-01;   Eid[104]= 8; Rid[104]= 2; Zid[104]= 3;
  NLX_coes2[2][105]=-2.382597618259922e-01;   Eid[105]= 9; Rid[105]= 0; Zid[105]= 0;
  NLX_coes2[2][106]=-9.665772071816001e-02;   Eid[106]= 9; Rid[106]= 0; Zid[106]= 1;
  NLX_coes2[2][107]=-1.655975928883445e-01;   Eid[107]= 9; Rid[107]= 0; Zid[107]= 2;
  NLX_coes2[2][108]=+1.129260354599469e+00;   Eid[108]= 9; Rid[108]= 0; Zid[108]= 3;
  NLX_coes2[2][109]=+1.171684961764939e-01;   Eid[109]= 9; Rid[109]= 1; Zid[109]= 0;
  NLX_coes2[2][110]=+5.325221410383649e-02;   Eid[110]= 9; Rid[110]= 1; Zid[110]= 1;
  NLX_coes2[2][111]=-2.012477390564129e-01;   Eid[111]= 9; Rid[111]= 1; Zid[111]= 2;
  NLX_coes2[2][112]=-5.417229725894274e-01;   Eid[112]= 9; Rid[112]= 1; Zid[112]= 3;
  NLX_coes2[2][113]=+7.567175544265266e-01;   Eid[113]= 9; Rid[113]= 2; Zid[113]= 0;
  NLX_coes2[2][114]=-1.025895997110420e+00;   Eid[114]= 9; Rid[114]= 2; Zid[114]= 1;
  NLX_coes2[2][115]=-2.860312643711020e-01;   Eid[115]= 9; Rid[115]= 2; Zid[115]= 2;
  NLX_coes2[2][116]=+9.458726648946861e-01;   Eid[116]= 9; Rid[116]= 2; Zid[116]= 3;
  NLX_coes2[2][117]=-2.612837286422319e-01;   Eid[117]=10; Rid[117]= 0; Zid[117]= 0;
  NLX_coes2[2][118]=-1.971214199897186e-02;   Eid[118]=10; Rid[118]= 0; Zid[118]= 1;
  NLX_coes2[2][119]=+3.459076166060691e-02;   Eid[119]=10; Rid[119]= 0; Zid[119]= 2;
  NLX_coes2[2][120]=+1.170439815725324e+00;   Eid[120]=10; Rid[120]= 0; Zid[120]= 3;
  NLX_coes2[2][121]=+3.542072627808060e-01;   Eid[121]=10; Rid[121]= 1; Zid[121]= 0;
  NLX_coes2[2][122]=+1.325611716431354e-01;   Eid[122]=10; Rid[122]= 1; Zid[122]= 1;
  NLX_coes2[2][123]=-4.000249131930538e-01;   Eid[123]=10; Rid[123]= 1; Zid[123]= 2;
  NLX_coes2[2][124]=-9.976909982657655e-01;   Eid[124]=10; Rid[124]= 1; Zid[124]= 3;
  NLX_coes2[2][125]=+1.004109030568835e-02;   Eid[125]=10; Rid[125]= 2; Zid[125]= 0;
  NLX_coes2[2][126]=-8.445905407234567e-01;   Eid[126]=10; Rid[126]= 2; Zid[126]= 1;
  NLX_coes2[2][127]=-3.240758010032184e-01;   Eid[127]=10; Rid[127]= 2; Zid[127]= 2;
  NLX_coes2[2][128]=+6.024288234362819e-01;   Eid[128]=10; Rid[128]= 2; Zid[128]= 3;
  NLX_coes2[2][129]=-3.039771991477991e-01;   Eid[129]=11; Rid[129]= 0; Zid[129]= 0;
  NLX_coes2[2][130]=+9.451018443234874e-02;   Eid[130]=11; Rid[130]= 0; Zid[130]= 1;
  NLX_coes2[2][131]=+2.926976125611916e-01;   Eid[131]=11; Rid[131]= 0; Zid[131]= 2;
  NLX_coes2[2][132]=+4.532903367419840e-01;   Eid[132]=11; Rid[132]= 0; Zid[132]= 3;
  NLX_coes2[2][133]=+5.223237560633900e-01;   Eid[133]=11; Rid[133]= 1; Zid[133]= 0;
  NLX_coes2[2][134]=+1.994558821778276e-01;   Eid[134]=11; Rid[134]= 1; Zid[134]= 1;
  NLX_coes2[2][135]=-4.971562906852185e-01;   Eid[135]=11; Rid[135]= 1; Zid[135]= 2;
  NLX_coes2[2][136]=-9.716612100792956e-01;   Eid[136]=11; Rid[136]= 1; Zid[136]= 3;
  NLX_coes2[2][137]=-8.191931235968012e-01;   Eid[137]=11; Rid[137]= 2; Zid[137]= 0;
  NLX_coes2[2][138]=-4.524698319282858e-01;   Eid[138]=11; Rid[138]= 2; Zid[138]= 1;
  NLX_coes2[2][139]=-1.695736161268264e-01;   Eid[139]=11; Rid[139]= 2; Zid[139]= 2;
  NLX_coes2[2][140]=+1.990295570719026e-01;   Eid[140]=11; Rid[140]= 2; Zid[140]= 3;
  NLX_coes2[2][141]=-3.555130363089876e-01;   Eid[141]=12; Rid[141]= 0; Zid[141]= 0;
  NLX_coes2[2][142]=+2.226754189057073e-01;   Eid[142]=12; Rid[142]= 0; Zid[142]= 1;
  NLX_coes2[2][143]=+4.709968350432456e-01;   Eid[143]=12; Rid[143]= 0; Zid[143]= 2;
  NLX_coes2[2][144]=-8.515555973557473e-01;   Eid[144]=12; Rid[144]= 0; Zid[144]= 3;
  NLX_coes2[2][145]=+5.565701103581210e-01;   Eid[145]=12; Rid[145]= 1; Zid[145]= 0;
  NLX_coes2[2][146]=+2.009671971727418e-01;   Eid[146]=12; Rid[146]= 1; Zid[146]= 1;
  NLX_coes2[2][147]=-5.286737794061089e-01;   Eid[147]=12; Rid[147]= 1; Zid[147]= 2;
  NLX_coes2[2][148]=-3.598995446442988e-01;   Eid[148]=12; Rid[148]= 1; Zid[148]= 3;
  NLX_coes2[2][149]=-1.080269957361538e+00;   Eid[149]=12; Rid[149]= 2; Zid[149]= 0;
  NLX_coes2[2][150]=+2.213948888855180e-01;   Eid[150]=12; Rid[150]= 2; Zid[150]= 1;
  NLX_coes2[2][151]=+1.678571365077307e-01;   Eid[151]=12; Rid[151]= 2; Zid[151]= 2;
  NLX_coes2[2][152]=+1.425453630342010e-01;   Eid[152]=12; Rid[152]= 2; Zid[152]= 3;
  NLX_coes2[2][153]=-3.968810479840358e-01;   Eid[153]=13; Rid[153]= 0; Zid[153]= 0;
  NLX_coes2[2][154]=+3.636795534120457e-01;   Eid[154]=13; Rid[154]= 0; Zid[154]= 1;
  NLX_coes2[2][155]=+4.341866056709694e-01;   Eid[155]=13; Rid[155]= 0; Zid[155]= 2;
  NLX_coes2[2][156]=-2.027911199053020e+00;   Eid[156]=13; Rid[156]= 0; Zid[156]= 3;
  NLX_coes2[2][157]=+3.758886550033978e-01;   Eid[157]=13; Rid[157]= 1; Zid[157]= 0;
  NLX_coes2[2][158]=+1.204900593622739e-01;   Eid[158]=13; Rid[158]= 1; Zid[158]= 1;
  NLX_coes2[2][159]=-6.168142996457036e-01;   Eid[159]=13; Rid[159]= 1; Zid[159]= 2;
  NLX_coes2[2][160]=+7.093039627238933e-01;   Eid[160]=13; Rid[160]= 1; Zid[160]= 3;
  NLX_coes2[2][161]=-4.128295639204045e-01;   Eid[161]=13; Rid[161]= 2; Zid[161]= 0;
  NLX_coes2[2][162]=+1.276353283590633e+00;   Eid[162]=13; Rid[162]= 2; Zid[162]= 1;
  NLX_coes2[2][163]=+4.348798352314966e-01;   Eid[163]=13; Rid[163]= 2; Zid[163]= 2;
  NLX_coes2[2][164]=+6.611040489076465e-01;   Eid[164]=13; Rid[164]= 2; Zid[164]= 3;
  NLX_coes2[2][165]=-4.151288362375942e-01;   Eid[165]=14; Rid[165]= 0; Zid[165]= 0;
  NLX_coes2[2][166]=+5.468042931043423e-01;   Eid[166]=14; Rid[166]= 0; Zid[166]= 1;
  NLX_coes2[2][167]=+1.086995349477703e-01;   Eid[167]=14; Rid[167]= 0; Zid[167]= 2;
  NLX_coes2[2][168]=-1.616130379081053e+00;   Eid[168]=14; Rid[168]= 0; Zid[168]= 3;
  NLX_coes2[2][169]=-8.002968000597770e-02;   Eid[169]=14; Rid[169]= 1; Zid[169]= 0;
  NLX_coes2[2][170]=+5.307114167115638e-02;   Eid[170]=14; Rid[170]= 1; Zid[170]= 1;
  NLX_coes2[2][171]=-8.809067255721175e-01;   Eid[171]=14; Rid[171]= 1; Zid[171]= 2;
  NLX_coes2[2][172]=+1.866601852708186e+00;   Eid[172]=14; Rid[172]= 1; Zid[172]= 3;
  NLX_coes2[2][173]=+7.702498628541261e-02;   Eid[173]=14; Rid[173]= 2; Zid[173]= 0;
  NLX_coes2[2][174]=+2.239639772838604e+00;   Eid[174]=14; Rid[174]= 2; Zid[174]= 1;
  NLX_coes2[2][175]=-1.407433566362572e-01;   Eid[175]=14; Rid[175]= 2; Zid[175]= 2;
  NLX_coes2[2][176]=+1.414513218193092e+00;   Eid[176]=14; Rid[176]= 2; Zid[176]= 3;
  NLX_coes2[2][177]=-4.555994108597190e-01;   Eid[177]=15; Rid[177]= 0; Zid[177]= 0;
  NLX_coes2[2][178]=+8.110334790911797e-01;   Eid[178]=15; Rid[178]= 0; Zid[178]= 1;
  NLX_coes2[2][179]=-4.650272304862146e-01;   Eid[179]=15; Rid[179]= 0; Zid[179]= 2;
  NLX_coes2[2][180]=+2.777836250757918e+00;   Eid[180]=15; Rid[180]= 0; Zid[180]= 3;
  NLX_coes2[2][181]=-5.241463996199409e-01;   Eid[181]=15; Rid[181]= 1; Zid[181]= 0;
  NLX_coes2[2][182]=+4.279169810991766e-01;   Eid[182]=15; Rid[182]= 1; Zid[182]= 1;
  NLX_coes2[2][183]=-1.218936535824251e+00;   Eid[183]=15; Rid[183]= 1; Zid[183]= 2;
  NLX_coes2[2][184]=+2.611325687273314e+00;   Eid[184]=15; Rid[184]= 1; Zid[184]= 3;
  NLX_coes2[2][185]=-1.046518190563420e-02;   Eid[185]=15; Rid[185]= 2; Zid[185]= 0;
  NLX_coes2[2][186]=+4.548069329069218e-01;   Eid[186]=15; Rid[186]= 2; Zid[186]= 1;
  NLX_coes2[2][187]=-3.572696002678306e+00;   Eid[187]=15; Rid[187]= 2; Zid[187]= 2;
  NLX_coes2[2][188]=+8.437962119357831e-01;   Eid[188]=15; Rid[188]= 2; Zid[188]= 3;
}

