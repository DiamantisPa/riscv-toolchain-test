.globl _start
_start:
li a0,1000000
mv sp,a0
jal main
hang: j hang
