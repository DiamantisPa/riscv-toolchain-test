#ifndef __ADNNET_RESIDUAL__
#define __ADNNET_RESIDUAL__

#include "defines.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype_I, typename dtype_O,
            int R, int C, int M, int Tm=1, int Tr=1, int Tc=1>
    class Residual{
      private:
      
        typedef ndmatrix::Mat3d<dtype_I, R, C, M> chanI;
        typedef ndmatrix::Mat3d<dtype_I, R, C, M> chanI1;
        typedef ndmatrix::Mat3d<dtype_O, R, C, M> chanO;
        
      public:
        Residual() {};
        ~Residual() {};
        
        void run(chanI &inp, chanI1 &inp1, chanO &out){

          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the feature maps
              OMAP: for (int m = 0; m < M; m+=Tm) {  // foreach feature map
                _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                  _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile
                    _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                        
                      out[r+tr][c+tc][m+tm] = inp[r+tr][c+tc][m+tm]+inp1[r+tr][c+tc][m+tm];
                      
                    } // _TOM
                  } // _TCS
                } // _TRS
              }// OMAP
            } // COLS
          } // ROWS
        };
    };

  }; // namespace cpp

}; // (namespace) dcnn

#endif
