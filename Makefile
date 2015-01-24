CFLAGS=--gpu-architecture sm_13 -lcudart -lcublas -g -O3

all:
	nvcc mainA.cu -o mainA $(CFLAGS)
	nvcc mainB.cu -o mainB $(CFLAGS)
	nvcc mainC.cu -o mainC $(CFLAGS)

partA:
	nvcc mainA.cu -o mainA $(CFLAGS)

partB:
	nvcc mainB.cu -o mainB $(CFLAGS)

partC:
	nvcc mainC.cu -o mainC $(CFLAGS)

clean:
	rm -f mainA mainB mainC