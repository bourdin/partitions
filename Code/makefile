ALL: PartitionSimplified 
include ${SLEPC_DIR}/bmake/slepc_common

PartitionMultiLevel: PartitionMultiLevel.o  chkopts
	-${CLINKER} -o PartitionMultiLevel PartitionMultiLevel.o ${SLEPC_LIB}

PartitionSimplified: PartitionSimplified.o  chkopts
	-${CLINKER} -o PartitionSimplified PartitionSimplified.o ${SLEPC_LIB}

Partition2DSeq: Partition2DSeq.o  chkopts
	-${CLINKER} -o Partition2DSeq Partition2DSeq.o ${SLEPC_LIB}

Partition3DSeq: Partition3DSeq.o  chkopts
	-${CLINKER} -o Partition3DSeq Partition3DSeq.o ${SLEPC_LIB}

Partition2DSeqAvg: Partition2DSeqAvg.o  chkopts
	-${CLINKER} -o Partition2DSeqAvg Partition2DSeqAvg.o ${SLEPC_LIB}

Partition2D: Partition2D.o  chkopts
	-${CLINKER} -o Partition2D Partition2D.o ${SLEPC_LIB}

Partition2DPer: Partition2DPer.o  chkopts
	-${CLINKER} -o Partition2DPer Partition2DPer.o ${SLEPC_LIB}

Partition2DMR: Partition2DMR.o  chkopts
	-${CLINKER} -o Partition2DMR Partition2DMR.o ${SLEPC_LIB}
