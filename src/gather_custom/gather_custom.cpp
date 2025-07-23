#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue

template<class T, class TINDEX>
class Kernel {
public:
    __aicore__ inline Kernel() {}
    __aicore__ inline void Init(GM_ADDR input, int64_t dim, GM_ADDR index, GM_ADDR output, uint32_t num_elements)
    {
        this->totalElements = num_elements;
        this->blockIdx = AscendC::GetBlockIdx();
        this->blockNum = AscendC::GetBlockNum();
        
        this->blockLength = (num_elements + blockNum - 1) / blockNum;// calculate elements per block, rounding up
        
        // adjust for last block
        uint32_t remaining = num_elements - blockIdx * blockLength;
        if (remaining < blockLength) {
            blockLength = remaining;
        }
        
        this->tileNum = 1;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        
        if (tileLength == 0) tileLength = 1;// Safety check
        
        InputGm.SetGlobalBuffer((__gm__ T *)input + blockIdx * blockLength, blockLength);
        IndexGm.SetGlobalBuffer((__gm__ TINDEX *)index + blockIdx * blockLength, blockLength);
        OutputGm.SetGlobalBuffer((__gm__ T *)output + blockIdx * blockLength, blockLength); 
        
        pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueIndex, BUFFER_NUM, this->tileLength * sizeof(TINDEX));
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = (blockLength + tileLength - 1) / tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        AscendC::LocalTensor<TINDEX> indexLocal = inQueueIndex.AllocTensor<TINDEX>();

        uint32_t copyLength = tileLength;
        if (progress * tileLength + tileLength > blockLength) {
            copyLength = blockLength - progress * tileLength;
        }

        AscendC::DataCopy(inputLocal, InputGm[progress * tileLength], copyLength);
        AscendC::DataCopy(indexLocal, IndexGm[progress * tileLength], copyLength);

        inQueueInput.EnQue(inputLocal);
        inQueueIndex.EnQue(indexLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
        AscendC::LocalTensor<TINDEX> indexLocal = inQueueIndex.DeQue<TINDEX>();
        AscendC::LocalTensor<T> outLocal = outQueueOutput.AllocTensor<T>();

        uint32_t computeLength = tileLength;
        if (progress * tileLength + tileLength > blockLength) {
            computeLength = blockLength - progress * tileLength;
        }

        for(uint32_t i = 0; i < computeLength; ++i) {
            TINDEX idx = indexLocal.GetValue(i);
            if (idx >= 0 && idx < totalElements) {
                outLocal.SetValue(i, inputLocal.GetValue(idx));
            } else { // handle out-of-bounds index
                outLocal.SetValue(i, 0);
            }
        }
        
        outQueueOutput.EnQue<T>(outLocal);
        inQueueInput.FreeTensor(inputLocal);
        inQueueIndex.FreeTensor(indexLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> outLocal = outQueueOutput.DeQue<T>();
        uint32_t copyLength = tileLength;
        if (progress * tileLength + tileLength > blockLength) {
            copyLength = blockLength - progress * tileLength;
        }
        AscendC::DataCopy(OutputGm[progress * tileLength], outLocal, copyLength);
        outQueueOutput.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueInput, inQueueIndex;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
    AscendC::GlobalTensor<T> InputGm;
    AscendC::GlobalTensor<TINDEX> IndexGm;
    AscendC::GlobalTensor<T> OutputGm;
    uint32_t totalElements;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gather_custom(GM_ADDR input, int64_t dim, GM_ADDR index, GM_ADDR output, uint32_t num_elements)
{
    Kernel<float, int64_t> op;
    op.Init(input, dim, index, output, num_elements);
    op.Process();
}

void launch_gather(
    uint32_t blockDim,
    void* stream,
    uint8_t* input,
    int64_t dim,
    uint8_t* index,
    uint8_t* output,
    uint32_t num_elements
) {
    gather_custom<<<blockDim, stream>>>(
        input,
        dim,
        index,
        output,
        num_elements
    );
}