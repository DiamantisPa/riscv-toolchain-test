#ifndef __DCNN_CONV2D__
#define __DCNN_CONV2D__

#include "defines.h"
#include "ac_math/ac_sigmoid_pwl.h"
// #include "mc_scverify.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype_I, typename dtype_O, 
            int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
            int Pt, int Pb, int Pl, int Pr,int str_r,int str_c,int groups, Activation ACT=linear, bool quant=false>
    class Conv2D {
      private:

        typedef weightT       wtype;
        typedef biasT         btype;

        typedef compactDataT<dtype_I, Tn> unrolled_dti;
        typedef compactDataT<dtype_O, Tm> unrolled_dto;

        typedef ndmatrix::Mat3d<dtype_I, R, C, N>  chanI; // Input feature channel datatype
        typedef ndmatrix::Mat3d<dtype_O, int(float((R-K+1+ Pt+Pb-1)/str_r)+1), int(float((C-L+1 +Pl + Pr-1)/str_c)+1), M>  chanO; // Output feature channel datatype
        typedef ndmatrix::Mat2d<wtype, M, K*K*N/groups>   chanW; // Load weights channel datatype
        typedef ndmatrix::Mat1d<btype, M>          chanB; // Load bias channel datatype
                
        dtype_O activation(dtype_I &x) {
          dtype_O res;
          switch (ACT) {
            case relu: { 
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
            case relu6: {
              res = (x > 0 && x < 6) ? (dtype_O)x : (x >= 6) ? (dtype_O)6 : (dtype_O)0;
              break;  
            }
            case linear: { 
              res = (dtype_O)x;
              break;
            }
            case sigmoid: {
              #ifdef __USE_FIXED__
              const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in = (x < 0) ? (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)(-x) : (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)x;
              ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp;
              // dtype_O tmp;
              ac_math::ac_sigmoid_pwl(in, tmp);
              //res = ac_math::ac_sigmoid_pwl<dtype_O, AC_TRN, dtype_I>(x);
              const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in1 = 0;
              ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp1;
              ac_math::ac_sigmoid_pwl(in1, tmp1);
              res = (x < 0) ? (dtype_O)(tmp1 - tmp) : (dtype_O)tmp;
              #else
              res = 1/(1+exp(-x));
              #endif
              break;
            }
            default: {
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
          } // end switch

          return res;
        };

      public:
        Conv2D() {};
        ~Conv2D() {};

       void run(chanW &w, chanB &b, chanI &din, chanO &dout) {

          const int height_col=int(float((R-K+ Pt+Pb)/str_r)+1);
          const int width_col=int(float((C-L +Pl + Pr)/str_c)+1);
          ndmatrix::Mat2d<dtype_O, K*L*N ,height_col*width_col > im2col;
          ndmatrix::Mat2d<dtype_O, M ,height_col*width_col > temp_array;
          int channels_col = N * K * L;
          for (int c = 0; c < channels_col; ++c) {
            int w_offset = c % L;
            int h_offset = (c / L) % K;
            int c_im = c / L / K;
            for (int h = 0; h < height_col; ++h) {
              for (int wi = 0; wi < width_col; ++wi) {
                  int im_row = h_offset + h * str_r-Pt;
                  int im_col = w_offset + wi * str_c-Pl;
                  if (im_row < 0 || im_col < 0 || im_row >= R || im_col >= C){
                    im2col[c][wi+h*width_col]=0;
                  } 
                  else{
                    im2col[c][wi+h*width_col]=din[im_row][im_col][c_im];
                  }
              }
            }
          }

          //GEMM
          for(int g = 0; g < groups; g++){
            for (int i=0;i<M/groups;i++){
              for (int j=0;j<height_col*width_col;j++){
                temp_array[g*M/groups+i][j] = b[g*M/groups+i]; 
                for (int k = 0; k < K*L*N/groups; k++) { 
                  temp_array[g*M/groups+i][j] += w[g*M/groups+i][k] * im2col[k+K*L*N/groups*g][j]; 
                }
              }
            }
          }
          //
          for(int r=0;r<height_col;r++){
            for(int c=0;c<width_col;c++){
              for(int m=0;m<M;m++){
                dout[r][c][m]=activation(temp_array[m][c+r*width_col]);
              }
            }
          }
        }
    };

    template<typename dtype_I, typename dtype_O, 
            int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
            int Pt, int Pb, int Pl, int Pr,int str_r,int str_c,int groups, Activation ACT>
    class Conv2D<dtype_I,dtype_O,R,C,N,M,K,L,Tm,Tn,Tr,Tc,Pt,Pb,Pl,Pr,str_r,str_c,groups,ACT,true> {
      private:

        typedef weightT       wtype;
        typedef biasT         btype;
        typedef scaleT         stype;

        typedef compactDataT<dtype_I, Tn> unrolled_dti;
        typedef compactDataT<dtype_O, Tm> unrolled_dto;

        typedef ndmatrix::Mat3d<dtype_I, R, C, N>  chanI; // Input feature channel datatype
        typedef ndmatrix::Mat3d<dtype_O, int(float((R-K+1+ Pt+Pb-1)/str_r)+1), int(float((C-L+1 +Pl + Pr-1)/str_c)+1), M>  chanO; // Output feature channel datatype
        typedef ndmatrix::Mat2d<wtype, M, N*K*L/groups> chanW; // Load weights channel datatype
        typedef ndmatrix::Mat1d<btype, M>          chanB; // Load bias channel datatype
        typedef ndmatrix::Mat1d<stype, M>          chanWS; // Load weight scale channel datatype
        typedef ndmatrix::Mat1d<wtype, M>          chanWZP; // Load weight zero point channel datatype
        typedef ndmatrix::Mat1d<stype, 1>          chanIS; // Load input scale channel datatype
        typedef ndmatrix::Mat1d<wtype, 1>          chanIZP; // Load input zero point channel datatype
        typedef ndmatrix::Mat1d<stype, 1>          chanIS_next; // Load next input scale channel datatype
        typedef ndmatrix::Mat1d<wtype, 1>          chanIZP_next; // Load next input scale channel datatype
        
        
                
      dtype_O activation(dtype_O &x) {
          dtype_O res;
          switch (ACT) {
            case relu: { 
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
            case linear: { 
              res = (dtype_O)x;
              break;
            }
            case sigmoid: {
              #ifdef __USE_FIXED__
              const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in = (x < 0) ? (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)(-x) : (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)x;
              ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp;
              // dtype_O tmp;
              ac_math::ac_sigmoid_pwl(in, tmp);
              //res = ac_math::ac_sigmoid_pwl<dtype_O, AC_TRN, dtype_I>(x);
              const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in1 = 0;
              ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp1;
              ac_math::ac_sigmoid_pwl(in1, tmp1);
              res = (x < 0) ? (dtype_O)(tmp1 - tmp) : (dtype_O)tmp;
              #else
              res = 1/(1+exp(-x));
              #endif
              break;
            }
            default: {
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
          } // end switch

          return res;
        };

      public:
        Conv2D() {};
        ~Conv2D() {};

       void run(chanW &w, chanB &b, chanWS &ws, chanWZP &wzp, chanIS &is, chanIZP &izp, chanIS_next &is_next, chanIZP_next &izp_next, chanI &din, chanO &dout) {

          const int height_col=int(float((R-K+ Pt+Pb)/str_r)+1);
          const int width_col=int(float((C-L +Pl + Pr)/str_c)+1);
  
          ndmatrix::Mat2d<dtype_O, M, height_col*width_col> out_buffer;
          ndmatrix::Mat2d<dtype_I, K*L*N ,height_col*width_col > im2col;
          ndmatrix::Mat2d<dtype_O, M ,height_col*width_col > temp_array;
          
          int channels_col = N * K * L;
          for (int c = 0; c < channels_col; ++c) {
            int w_offset = c % L;
            int h_offset = (c / L) % K;
            int c_im = c / L / K;
            for (int h = 0; h < height_col; ++h) {
              for (int wi = 0; wi < width_col; ++wi) {
                  int im_row = h_offset + h * str_r-Pt;
                  int im_col = w_offset + wi * str_c-Pl;
                  if (im_row < 0 || im_col < 0 || im_row >= R || im_col >= C){
                    im2col[c][wi+h*width_col]=0;
                  } 
                  else{
                    im2col[c][wi+h*width_col]=din[im_row][im_col][c_im];
                  }
              }
            }
          }

          //GEMM
          for(int g = 0; g < groups; g++){
            for (int i=0;i<M/groups;i++){
              for (int j=0;j<height_col*width_col;j++){
                temp_array[g*M/groups+i][j] = float(b[g*M/groups+i]/(ws[g*M/groups+i]*is[0])); 
                dtype_O temp_2_sum = 0;
                dtype_O temp_3_sum = 0; 
                for (int k = 0; k < K*L*N/groups; k++) { 
                  temp_array[g*M/groups+i][j] += w[g*M/groups+i][k] * im2col[k+K*L*N/groups*g][j];
                  temp_2_sum+=w[g*M/groups+i][k]*izp[0]+wzp[g*M/groups+i]*izp[0];
                  temp_3_sum+=im2col[k+K*L*N/groups*g][j]*wzp[g*M/groups+i]; 
                }
                temp_array[g*M/groups+i][j]=round((temp_array[g*M/groups+i][j]-temp_2_sum-temp_3_sum)*(ws[g*M/groups+i]*is[0])/(is_next[0])+izp_next[0]);
              }
            }
          }
          //
          for(int r=0;r<height_col;r++){
            for(int c=0;c<width_col;c++){
              for(int m=0;m<M;m++){
                dout[r][c][m]=activation(temp_array[m][c+r*width_col]);
              }
            }
          }
        }
    };
  }; // (namespace) cpp
}; // (namespace) dcnn

#endif
