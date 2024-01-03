#include "model/dcnn_lenet.h"
#include "helpers.h"

#ifdef TARGET_LIBGCC_SDATA_SECTION
  extern void *_dso_handle __attribute_ ((_section_ (TARGET_LIBGCC_SDATA_SECTION)));
#endif
#ifdef HAVE_GAS_HIDDEN
 extern void *_dso_handle __attribute_ ((_visibility_ ("hidden")));
#endif
#ifdef CRTSTUFFS_O
 void *__dso_handle = &__dso_handle;
#else
 void *__dso_handle = 0;
#endif

int main(int argc, char* argv[]) {
//void notmain() {
  ndmatrix::Mat3d<dcnn::unsignedDataT,28, 28, 1> inp;
  read_3d_ndarray_from_txt<dcnn::unsignedDataT,28, 28, 1>(inp, "images/new_img_160.txt");

  ndmatrix::Mat1d<dcnn::scaleT, 1> is0;
  read_1d_ndarray_from_txt<dcnn::scaleT, 1>(is0, "parameters/start/start_is.txt");
  ndmatrix::Mat1d<dcnn::weightT, 1> izp0;
  read_1d_ndarray_from_txt<dcnn::weightT, 1>(izp0, "parameters/start/start_izp.txt");

  ndmatrix::Mat2d<dcnn::weightT, 6, 25> w0;
  read_2d_ndarray_from_txt<dcnn::weightT, 6, 25>(w0, "parameters/weights/conv_0_w.txt");
  ndmatrix::Mat1d<dcnn::biasT, 6> b0;
  read_1d_ndarray_from_txt<dcnn::biasT, 6>(b0, "parameters/biases/conv_0_b.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 6> ws0;
  read_1d_ndarray_from_txt<dcnn::scaleT, 6>(ws0, "parameters/weight_scale/conv_0_ws.txt");
  ndmatrix::Mat1d<dcnn::weightT, 6> wzp0;
  read_1d_ndarray_from_txt<dcnn::weightT, 6>(wzp0, "parameters/weight_zero_point/conv_0_wzp.txt");
  ndmatrix::Mat1d<dcnn::weightT, 1> izp1;
  read_1d_ndarray_from_txt<dcnn::weightT, 1>(izp1, "parameters/input_zero_point/conv_0_izp.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 1> is1;
  read_1d_ndarray_from_txt<dcnn::scaleT, 1>(is1, "parameters/input_scale/conv_0_is.txt");


  ndmatrix::Mat2d<dcnn::weightT, 16, 150> w1;
  read_2d_ndarray_from_txt<dcnn::weightT, 16, 150>(w1, "parameters/weights/conv_1_w.txt");
  ndmatrix::Mat1d<dcnn::biasT, 16> b1;
  read_1d_ndarray_from_txt<dcnn::biasT, 16>(b1, "parameters/biases/conv_1_b.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 16> ws1;
  read_1d_ndarray_from_txt<dcnn::scaleT, 16>(ws1, "parameters/weight_scale/conv_1_ws.txt");
  ndmatrix::Mat1d<dcnn::weightT, 16> wzp1;
  read_1d_ndarray_from_txt<dcnn::weightT, 16>(wzp1, "parameters/weight_zero_point/conv_1_wzp.txt");
  ndmatrix::Mat1d<dcnn::weightT, 1> izp2;
  read_1d_ndarray_from_txt<dcnn::weightT, 1>(izp2, "parameters/input_zero_point/conv_1_izp.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 1> is2;
  read_1d_ndarray_from_txt<dcnn::scaleT, 1>(is2, "parameters/input_scale/conv_1_is.txt");



  ndmatrix::Mat2d<dcnn::weightT, 400, 120> w2;
  read_2d_ndarray_from_txt<dcnn::weightT, 400, 120>(w2, "parameters/weights/fc_0_w.txt");
  ndmatrix::Mat1d<dcnn::biasT, 120> b2;
  read_1d_ndarray_from_txt<dcnn::biasT, 120>(b2, "parameters/biases/fc_0_b.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 120> ws2;
  read_1d_ndarray_from_txt<dcnn::scaleT, 120>(ws2, "parameters/weight_scale/fc_0_ws.txt");
  ndmatrix::Mat1d<dcnn::weightT, 120> wzp2;
  read_1d_ndarray_from_txt<dcnn::weightT, 120>(wzp2, "parameters/weight_zero_point/fc_0_wzp.txt");
  ndmatrix::Mat1d<dcnn::weightT, 1> izp3;
  read_1d_ndarray_from_txt<dcnn::weightT, 1>(izp3, "parameters/input_zero_point/fc_0_izp.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 1> is3;
  read_1d_ndarray_from_txt<dcnn::scaleT, 1>(is3, "parameters/input_scale/fc_0_is.txt");

  ndmatrix::Mat2d<dcnn::weightT, 120, 84> w3;
  read_2d_ndarray_from_txt<dcnn::weightT, 120, 84>(w3, "parameters/weights/fc_1_w.txt");
  ndmatrix::Mat1d<dcnn::biasT, 84> b3;
  read_1d_ndarray_from_txt<dcnn::biasT, 84>(b3, "parameters/biases/fc_1_b.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 84> ws3;
  read_1d_ndarray_from_txt<dcnn::scaleT, 84>(ws3, "parameters/weight_scale/fc_1_ws.txt");
  ndmatrix::Mat1d<dcnn::weightT, 84> wzp3;
  read_1d_ndarray_from_txt<dcnn::weightT, 84>(wzp3, "parameters/weight_zero_point/fc_1_wzp.txt");
  ndmatrix::Mat1d<dcnn::weightT, 1> izp4;
  read_1d_ndarray_from_txt<dcnn::weightT, 1>(izp4, "parameters/input_zero_point/fc_1_izp.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 1> is4;
  read_1d_ndarray_from_txt<dcnn::scaleT, 1>(is4, "parameters/input_scale/fc_1_is.txt");

  ndmatrix::Mat2d<dcnn::weightT, 84, 10> w4;
  read_2d_ndarray_from_txt<dcnn::weightT, 84, 10>(w4, "parameters/weights/fc_2_w.txt");
  ndmatrix::Mat1d<dcnn::biasT, 10> b4;
  read_1d_ndarray_from_txt<dcnn::biasT, 10>(b4, "parameters/biases/fc_2_b.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 10> ws4;
  read_1d_ndarray_from_txt<dcnn::scaleT, 10>(ws4, "parameters/weight_scale/fc_2_ws.txt");
  ndmatrix::Mat1d<dcnn::weightT, 10> wzp4;
  read_1d_ndarray_from_txt<dcnn::weightT, 10>(wzp4, "parameters/weight_zero_point/fc_2_wzp.txt");
  ndmatrix::Mat1d<dcnn::weightT, 1> izp5;
  read_1d_ndarray_from_txt<dcnn::weightT, 1>(izp5, "parameters/input_zero_point/fc_2_izp.txt");
  ndmatrix::Mat1d<dcnn::scaleT, 1> is5;
  read_1d_ndarray_from_txt<dcnn::scaleT, 1>(is5, "parameters/input_scale/fc_2_is.txt");

  ndmatrix::Mat1d<dcnn::signedDataT, 10> out;

  dcnn::cpp::lenet model;

  model.predict( 
    w0, b0, ws0, wzp0, is0, izp0, 
    w1, b1, ws1, wzp1, is1, izp1, 
    w2, b2, ws2, wzp2, is2, izp2, 
    w3, b3, ws3, wzp3, is3, izp3, 
    w4, b4, ws4, wzp4, is4, izp4, 
    is5, izp5, 
    inp, out);

  write_1d_ndarray_to_txt<dcnn::weightT, 10>(out, "output.txt");

  // Extra code here


  return 0;
}
