DIR1=$(shell pwd)/../../dcnn/
DIR2=$(shell pwd)/../../dcnn/Convolutions/
DIR3=$(shell pwd)/../../dcnn/Activations/
DIR4=$(shell pwd)/../../dcnn/Dense/
DIR5=$(shell pwd)/../../dcnn/Normalization/
DIR6=$(shell pwd)/../../dcnn/Pooling/
DIR7=$(shell pwd)/../../libs_include/
DIR8=$(shell pwd)/../../libs_include/include/
DIR9=$(shell pwd)/../../dcnn/Residual/

INC=$(DIR1) $(DIR2) $(DIR3) $(DIR4) $(DIR5) $(DIR6) $(DIR7) $(DIR8) $(DIR9)
INC_PARAMS=$(foreach d, $(INC),-I $d)
make:
	g++ dcnn_lenet_tb.cpp -o lenet $(INC_PARAMS) -O3
	./lenet

riscv:
	riscv32-unknown-elf-g++ -march=rv32im -mabi=ilp32 -T lscript bootstrap.s -nostartfiles -fno-common -static -fno-builtin-printf -specs=nosys.specs $(INC_PARAMS) dcnn_lenet_tb.cpp -o dcnn_lenet_tb.o
