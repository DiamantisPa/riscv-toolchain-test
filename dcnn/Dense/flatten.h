#ifndef __DCNN_FLATTEN__
#define __DCNN_FLATTEN__

#include "defines.h"
// #include "mc_scverify.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype, int R, int C, int N, int Tn, int Tm, int Tr, int Tc>
    class Flatten {
    private:

      typedef ndmatrix::Mat3d<dtype, R, C, N> chanI;
      typedef ndmatrix::Mat1d<dtype, R*C*N>   chanO; 

    public:
      Flatten() {};
      ~Flatten() {};

      void run(chanI &din, chanO &dout) {

        IMAP: for (int n = 0; n < N; n+=Tn) {
          COLS: for (int c = 0; c < C; c+=Tc) {  
            ROWS: for (int r = 0; r < R; r+=Tr) {  
              _TIM: for (int tn = 0; tn < Tn; tn++) {
                _TCS: for (int tc = 0; tc < Tc; tc++) { 
                  _TRS: for (int tr = 0; tr < Tr; tr++) { 
                    dout[(r+tr)*C+(c+tc)+(R*C*(n+tn))] = din[r+tr][c+tc][n+tn];

                  } // _TIM
                } // IMAP
              } // _TCS
            } // _TRS
          } // COLS
        } // ROWS
      }; // (function) run


    }; // (class) Flatten

  }; // (namespace) cpp
}; // (namespace) dcnn

#endif
