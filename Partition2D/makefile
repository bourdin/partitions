ALL: Partition2D

include ${SLEPC_DIR}/bmake/slepc_common

Partition2D: Partition2D.o  chkopts
	-${CLINKER} -o Partition2D Partition2D.o ${SLEPC_LIB}
