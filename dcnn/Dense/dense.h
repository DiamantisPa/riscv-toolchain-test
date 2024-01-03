#ifndef __DCNN_DENSE__
#define __DCNN_DENSE__

#include "defines.h"
#include "math.h"
#include <algorithm>
// #include "mc_scverify.h"

namespace dcnn {

  namespace cpp {

    // Accuracy takes values 1, 2, 4
    template<typename dtype_I, typename dtype_O,
            int N, int M, int Tm, int Tn, Activation ACT=linear, bool quant=false>
    class Dense {
    private:

      typedef weightT       wtype;
      typedef biasT         btype;

      typedef dtype_O softmax_sumT;

      typedef ndmatrix::Mat1d<dtype_I, N>  chanI;
      typedef ndmatrix::Mat1d<dtype_O, M>  chanO;
      typedef ndmatrix::Mat2d<wtype, N, M> chanW;
      typedef ndmatrix::Mat1d<btype, M>    chanB;

      dtype_O sfm_max;
      softmax_sumT sfm_sum;

      void softmax_max_sum(ndmatrix::Mat1d<signedDataT, M> &vec) {
        sfm_max = vec[0];
        for (int i = 0; i < M; i++) {
          sfm_max = (sfm_max > vec[i]) ? sfm_max : (dtype_O)vec[i];
        }
        sfm_sum = 0;
        for (int i = 0; i < M; i++) {
          sfm_sum += std::exp(vec[i]-sfm_max);
        }
      }


      dtype_O activation(signedDataT &x) {
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
          case softmax: { 
            // res = (dtype_O)x;
            res = (dtype_O)(expf(x-sfm_max)/sfm_sum); // TODO: write softmax
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
      Dense() {};
      ~Dense() {};

      void run(chanW &w, chanB &b, chanI &din, chanO &dout) {

        ndmatrix::Mat1d<signedDataT, M> acc;
        
        IMAP: for (int n = 0; n < N; n+=Tn) {
          OMAP: for (int m = 0; m < M; m+=Tm) {
            _TIM: for (int tn = 0; tn < Tn; tn++) {
              _TOM: for (int tm = 0; tm < Tm; tm++) {

                if (n+tn == 0) acc[m+tm] = b[m+tm];
                
                acc[m+tm] += din[n+tn] * w[n+tn][m+tm];
                
              } // (loop) _TOM
            } // (loop) _TIM
          } // (loop) OMAP      
        } // (loop) IMAP

        if (ACT == softmax) {
          softmax_max_sum(acc);
        }

        for (int m = 0; m < M; m+=Tm) {
          for (int tm = 0; tm < Tm; tm++) {
            dout[m+tm] = activation(acc[m+tm]);
          }
        }
      };
    };

     // Accuracy takes values 1, 2, 4
    template<typename dtype_I, typename dtype_O,
            int N, int M, int Tm, int Tn, Activation ACT>
    class Dense<dtype_I,dtype_O,N,M,Tm,Tn,ACT,true> {
    private:

      typedef weightT       wtype;
      typedef biasT         btype;
      typedef scaleT        stype;

      typedef dtype_O softmax_sumT;

      typedef ndmatrix::Mat1d<dtype_I, N>  chanI;
      typedef ndmatrix::Mat1d<dtype_O, M>  chanO;
      typedef ndmatrix::Mat2d<wtype, N, M> chanW;
      typedef ndmatrix::Mat1d<btype, M>    chanB;
      typedef ndmatrix::Mat1d<stype, M>    chanWS; // Load bias channel datatype
      typedef ndmatrix::Mat1d<wtype, M>    chanWZP; // Load bias channel datatype
      typedef ndmatrix::Mat1d<stype, 1>    chanIS; // Load input scale channel datatype
      typedef ndmatrix::Mat1d<wtype, 1>    chanIZP; // Load input zero point channel datatype
      typedef ndmatrix::Mat1d<stype, 1>    chanIS_next; // Load next input scale channel datatype
      typedef ndmatrix::Mat1d<wtype, 1>    chanIZP_next; // Load next input scale channel datatype

      dtype_O sfm_max;
      softmax_sumT sfm_sum;

      void softmax_max_sum(ndmatrix::Mat1d<signedDataT, M> &vec) {
        sfm_max = vec[0];
        for (int i = 0; i < M; i++) {
          sfm_max = (sfm_max > vec[i]) ? sfm_max : (dtype_O)vec[i];
        }
        sfm_sum = 0;
        for (int i = 0; i < M; i++) {
          sfm_sum += std::exp(vec[i]-sfm_max);
        }
      }


      dtype_O activation(signedDataT &x) {
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
          case softmax: { 
            // res = (dtype_O)x;
            res = (dtype_O)(expf(x-sfm_max)/sfm_sum); // TODO: write softmax
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
      Dense() {};
      ~Dense() {};

      void run(chanW &w, chanB &b, chanWS &ws, chanWZP &wzp, chanIS &is, chanIZP &izp, chanIS_next &is_next, chanIZP_next &izp_next, chanI &din, chanO &dout) {

        ndmatrix::Mat1d<dtype_O, M> acc;
        ndmatrix::Mat1d<dtype_O, M> temp;
        ndmatrix::Mat1d<dtype_O, M> temp_sum_2;
        ndmatrix::Mat1d<dtype_O, M> temp_sum_3;     

        
        IMAP: for (int n = 0; n < N; n+=Tn) {
          _TIM: for (int tn = 0; tn < Tn; tn++) {
            OMAP: for (int m = 0; m < M; m+=Tm) {
              _TOM: for (int tm = 0; tm < Tm; tm++) {
              if(n+tn==0){
                temp[m+tm] = float(b[m+tm]/(ws[m+tm]*is[0]));
                temp_sum_2[m+tm] = 0;
                temp_sum_3[m+tm] = 0;
              }
                temp[m+tm] += (din[n+tn]) * w[n+tn][m+tm];
                temp_sum_2[m+tm] += w[n+tn][m+tm]*izp[0]+wzp[m+tm]*izp[0];
                temp_sum_3[m+tm] += din[n+tn]*wzp[m+tm];
              } // (loop) _TOM
            } // (loop) _TIM
          } // (loop) OMAP     
        } // (loop) IMAP

        for (int m = 0; m < M; m++){
          acc[m]=round((temp[m]-temp_sum_2[m]-temp_sum_3[m])*(ws[m]*is[0])/is_next[0]+izp_next[0]);
        }
        if (ACT == softmax) {
          softmax_max_sum(acc);
        }

        for (int m = 0; m < M; m+=Tm) {
          for (int tm = 0; tm < Tm; tm++) {
            dout[m+tm] = activation(acc[m+tm]);
          }
        }
      };
    };

  }; // (namespace) cpp
}; // (namespace) dcnn

#endif
