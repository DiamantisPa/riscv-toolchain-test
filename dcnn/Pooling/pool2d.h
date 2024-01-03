#ifndef __ADNNET_POOL2D__
#define __ADNNET_POOL2D__

#include "defines.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype, int R, int C, int M, int K, int L, int Tm, int Tr=1, int Tc=1, Pooling P=Max>
    class Pool2D {
      private:

        
        typedef ndmatrix::Mat3d<dtype, R, C, M>     chanI; 
        typedef ndmatrix::Mat3d<dtype, R/K, C/L, M> chanO; 

        typedef ndmatrix::Mat2d<dtype, K, L>     kernelT;
        typedef ndmatrix::Mat3d<dtype, K, C, Tm>  bufferT;
        typedef ndmatrix::Mat3d<dtype, Tm, K, L> windowT;

        windowT window;
        bufferT buffer;

        dtype max(dtype window[K][L]) {
          dtype max = window[0][0];
          for (int k = 0; k < K; k++)
            for (int l = 0; l < L; l++)
              max = (max > window[k][l]) ? max : window[k][l];
          return max;
        };

        dtype avg(dtype window[K][L]) {
          dtype sum_2=0;
          for(int k=0;k<K;k++){
            dtype sum_1=0;
            for(int l=0;l<L;l++){
              sum_1+=window[k][l];
            }
            sum_2+=float(float(sum_1/L)/K);
          }
          return sum_2;
        };

        dtype f_pool(dtype window[K][L]) {
          dtype out;
          switch (P) {
            case Max: out = max(window); break;
            case Avg: out = avg(window); break;
            default:  out = max(window); break;
          }

          return out;
        }

      public:

        Pool2D() {};
        ~Pool2D() {};

        void run(chanI &inp, chanO &out) {
          
          ROWS: for (int r = 0; r < R; r+=Tr*K) {  // foreach row of the input feature maps
            COLS: for (int c = 0; c < C; c+=Tc*L) {  // foreach column of the input feature maps
              _TRS: for (int tr = 0; tr < Tr; tr+=K) { // foreach row of the tile
                _TCS: for (int tc = 0; tc < Tc; tc+=L) { // foreach column of the tile

                  MAPS: for (int m = 0; m < M; m+=Tm) {
                    for (int tm = 0; tm < Tm; tm++) {
                      
                      dtype window[K][L];
                      // update window
                      for (int k = 0; k < K; k++) {
                        for (int l = 0; l < L; l++) {
                        
                          window[k][l] = inp[r+tr+k][c+tc+l][m+tm];

                        }
                      }
                      out[(r+tr)/K][(c+tc)/L][m+tm] = f_pool(window);
                    }

                  } // (loop) MAPS
                } // _TCS
              } // _TRS
            } // COLS
          } // ROWS

        }; // (function) compute_layer
    };

  }; // (namespace) cpp
}; // (namespace) dcnn

#endif 
