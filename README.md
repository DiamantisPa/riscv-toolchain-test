## LENET

The Lenet model is trained with quantization aware training through pytorch, in the MNIST dataset, and is placed in the `Lenet` file, that is part of the `examples` file. The MNIST dataset has 10 classes for every number between 0 and 9 and the Lenet model performs classification on the provided images.

 **The images dimensions should be 28 x 28**
 
 To compile for bare-metal inside the `examples/lenet/` folder run `make riscv`

