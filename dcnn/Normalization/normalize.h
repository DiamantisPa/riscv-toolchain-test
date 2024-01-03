#ifndef __ADNNET_NORMALIZE__
#define __ADNNET_NORMALIZE__

#include "defines.h"
// #include "mc_scverify.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype, typename otype, int R, int C, int M, int Tm, int Tr=1, int Tc=1>
    class BatchNormalization {
      private:

        typedef scaleT        stype;
        typedef biasT         btype;

        typedef ndmatrix::Mat3d<dtype, R, C, M>  chanI;
        typedef ndmatrix::Mat3d<dtype, R, C, M>  chanO;
        typedef ndmatrix::Mat1d<btype, M>        chanW;
        typedef ndmatrix::Mat1d<btype, M>        chanB;

      public:
        BatchNormalization() {};
        ~BatchNormalization() {};

        void run(chanW &s, chanB &b, chanI &inp, chanO &out){

          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the feature maps
              _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile
                  OMAP: for ( int m = 0; m < M; m+=Tm) {  // foreach feature map
                    _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                      out[r+tr][c+tc][m+tm] = (inp[r+tr][c+tc][m+tm] * s[m+tm]) + b[m+tm];
                    }
                  }// OMAP
                } // _TCS
              } // _TRS
            } // COLS
          } // ROWS
        }; // run
    };
  }
} // (namespace) dcnn

#endif
