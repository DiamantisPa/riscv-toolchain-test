#ifndef __DCNN_MODEL__
#define __DCNN_MODEL__

#include "defines.h"
#include "conv2d.h"
#include "normalize.h"
#include "pool2d.h"
#include "dense.h"
#include "flatten.h"
#include "relu6.h"
#include "residual.h"

namespace dcnn {
namespace cpp {

#pragma hls_design
class lenet {
private:

  typedef unsignedDataT     dtype_I;
  typedef signedDataT       dtype_O;
  typedef weightT           wtype;
  typedef biasT             btype;
  typedef scaleT            stype;

  typedef ndmatrix::Mat3d<dtype_I, 28, 28, 1> chanI;
  typedef ndmatrix::Mat1d<dtype_O, 10> chanO;
  typedef ndmatrix::Mat2d<wtype,6,25> chanW_0;
  typedef ndmatrix::Mat1d<btype, 6> chanB_0;
  typedef ndmatrix::Mat1d<stype, 6> chanWS_0;
  typedef ndmatrix::Mat1d<wtype, 6> chanWZP_0;
  typedef ndmatrix::Mat1d<stype, 1> chanIS_0;
  typedef ndmatrix::Mat1d<wtype, 1> chanIZP_0;
  typedef ndmatrix::Mat2d<wtype,16,150> chanW_1;
  typedef ndmatrix::Mat1d<btype, 16> chanB_1;
  typedef ndmatrix::Mat1d<stype, 16> chanWS_1;
  typedef ndmatrix::Mat1d<wtype, 16> chanWZP_1;
  typedef ndmatrix::Mat1d<stype, 1> chanIS_1;
  typedef ndmatrix::Mat1d<wtype, 1> chanIZP_1;
  typedef ndmatrix::Mat2d<wtype,400,120> chanW_2;
  typedef ndmatrix::Mat1d<btype, 120> chanB_2;
  typedef ndmatrix::Mat1d<stype, 120> chanWS_2;
  typedef ndmatrix::Mat1d<wtype, 120> chanWZP_2;
  typedef ndmatrix::Mat1d<stype, 1> chanIS_2;
  typedef ndmatrix::Mat1d<wtype, 1> chanIZP_2;
  typedef ndmatrix::Mat2d<wtype,120,84> chanW_3;
  typedef ndmatrix::Mat1d<btype, 84> chanB_3;
  typedef ndmatrix::Mat1d<stype, 84> chanWS_3;
  typedef ndmatrix::Mat1d<wtype, 84> chanWZP_3;
  typedef ndmatrix::Mat1d<stype, 1> chanIS_3;
  typedef ndmatrix::Mat1d<wtype, 1> chanIZP_3;
  typedef ndmatrix::Mat2d<wtype,84,10> chanW_4;
  typedef ndmatrix::Mat1d<btype, 10> chanB_4;
  typedef ndmatrix::Mat1d<stype, 10> chanWS_4;
  typedef ndmatrix::Mat1d<wtype, 10> chanWZP_4;
  typedef ndmatrix::Mat1d<stype, 1> chanIS_4;
  typedef ndmatrix::Mat1d<wtype, 1> chanIZP_4;
  typedef ndmatrix::Mat1d<stype, 1> chanIS_5;
  typedef ndmatrix::Mat1d<wtype, 1> chanIZP_5;

  Conv2D<unsignedDataT, signedDataT, 28, 28, 1, 6, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, linear, true>  conv_0;
  Pool2D<signedDataT, 28, 28, 6, 2, 2, 1, 1, 1, Max> max_0;
  Conv2D<signedDataT, signedDataT, 14, 14, 6, 16, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, linear, true>  conv_1;
  Pool2D<signedDataT, 10, 10, 16, 2, 2, 1, 1, 1, Max> max_1;
  Flatten<signedDataT, 5, 5, 16, 1, 1, 1, 1>  flat_0;
  Dense<signedDataT, signedDataT, 400, 120, 1, 1, linear, true>  fc_0;
  Dense<signedDataT, signedDataT, 120, 84, 1, 1, linear, true>  fc_1;
  Dense<signedDataT, signedDataT, 84, 10, 1, 1, linear, true>  fc_2;

  ndmatrix::Mat3d<signedDataT,28,28,6> conv_0_o;
  ndmatrix::Mat3d<signedDataT,14,14,6> max_0_o;
  ndmatrix::Mat3d<signedDataT,10,10,16> conv_1_o;
  ndmatrix::Mat3d<signedDataT,5,5,16> max_1_o;
  ndmatrix::Mat1d<signedDataT,400> flat_0_o;
  ndmatrix::Mat1d<signedDataT,120> fc_0_o;
  ndmatrix::Mat1d<signedDataT,84> fc_1_o;

public:
  lenet() {};
  ~lenet() {};

  #pragma hls_design interface
  void predict(chanW_0&w0, chanB_0&b0, chanWS_0&ws0, chanWZP_0&wzp0, chanIS_0&is0, chanIZP_0&izp0, 
  chanW_1&w1, chanB_1&b1, chanWS_1&ws1, chanWZP_1&wzp1, chanIS_1&is1, chanIZP_1&izp1, 
  chanW_2&w2, chanB_2&b2, chanWS_2&ws2, chanWZP_2&wzp2, chanIS_2&is2, chanIZP_2&izp2, 
  chanW_3&w3, chanB_3&b3, chanWS_3&ws3, chanWZP_3&wzp3, chanIS_3&is3, chanIZP_3&izp3, 
  chanW_4&w4, chanB_4&b4, chanWS_4&ws4, chanWZP_4&wzp4, chanIS_4&is4, chanIZP_4&izp4, 
  chanIS_5&is5, chanIZP_5&izp5, 
  chanI &inp, chanO &out) {

    conv_0.run(w0, b0, ws0, wzp0, is0, izp0, is1, izp1, inp, conv_0_o);
    max_0.run(conv_0_o, max_0_o);
    conv_1.run(w1, b1, ws1, wzp1, is1, izp1, is2, izp2, max_0_o, conv_1_o);
    max_1.run(conv_1_o, max_1_o);
    flat_0.run(max_1_o, flat_0_o);
    fc_0.run(w2, b2, ws2, wzp2, is2, izp2, is3, izp3, flat_0_o, fc_0_o);
    fc_1.run(w3, b3, ws3, wzp3, is3, izp3, is4, izp4, fc_0_o, fc_1_o);
    fc_2.run(w4, b4, ws4, wzp4, is4, izp4, is5, izp5, fc_1_o, out);

  }; // (function) predict
}; // (class) lenet


}; // (namespace) cpp
}; // (namespace) dcnn

#endif

