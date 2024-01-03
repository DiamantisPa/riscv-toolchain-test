#ifndef __ADNNET_RELU__
#define __ADNNET_RELU__

#include "defines.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype_I, typename dtype_O,
            int R, int C, int M, int Tm=1, int Tr=1, int Tc=1>
    class Relu {
      private:
      
        typedef ndmatrix::Mat3d<dtype_I, R, C, M> chanI;
        typedef ndmatrix::Mat3d<dtype_I, R, C, M> chanO;
        
      public:
        Relu() {};
        ~Relu() {};
        
        void run(chanI &inp, chanO &out){

          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the feature maps
              OMAP: for (int m = 0; m < M; m+=Tm) {  // foreach feature map
                _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                  _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile
                    _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                        
                      dtype_I preAct = inp[r+tr][c+tc][m+tm];
                      dtype_O postAct = (preAct > 0) ? (dtype_O)preAct : (dtype_O)0;
                      out[r+tr][c+tc][m+tm] = postAct;
                      
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
