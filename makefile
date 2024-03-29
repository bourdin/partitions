ALL: PartitionSimplified 
include ${SLEPC_DIR}/conf/slepc_common

PartitionMultiLevel: PartitionMultiLevel.o  chkopts
	-${CLINKER} -o PartitionMultiLevel PartitionMultiLevel.o ${SLEPC_LIB} -lpng

PartitionExactProj: PartitionExactProj.o  chkopts
	-${CLINKER} -o PartitionExactProj PartitionExactProj.o ${SLEPC_LIB}

MakeComposite: MakeComposite.o chkopts
	-${CLINKER} -o MakeComposite MakeComposite.o ${PETSC_LIB}
    
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

txt2PNG: txt2PNG.o  chkopts
	-${CLINKER} -o txt2PNG txt2PNG.o ${PETSC_LIB} -lpng
