#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue

// https://bbs.huaweicloud.com/blogs/439155

template<class T, class TINDEX>
class Kernel {
public:
    __aicore__ inline Kernel() {}
    __aicore__ inline void Init(GM_ADDR input, int64_t dim, GM_ADDR index, GM_ADDR output, uint32_t totalLength)
    {
        AscendC::printf("Init:: Dim is %ld\n", dim);
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = 1;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        
        AscendC::printf("Init:: Dim is %ld   GetBlockNum()=%ld\n", dim , AscendC::GetBlockNum());

        InputGm.SetGlobalBuffer((__gm__ T *)input + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        IndexGm.SetGlobalBuffer((__gm__ TINDEX *)index + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        OutputGm.SetGlobalBuffer((__gm__ T *)output + this->blockLength * AscendC::GetBlockIdx(), this->blockLength); 
        pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueIndex, BUFFER_NUM, this->tileLength * sizeof(TINDEX));
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, this->tileLength * sizeof(T));
        // yGm.SetGlobalBuffer((__gm__ T *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // zGm.SetGlobalBuffer((__gm__ T *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        // pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        // pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        // int32_t loopCount = this->tileNum * BUFFER_NUM;
        // for (int32_t i = 0; i < loopCount; i++) {
        //     CopyIn(i);
        //     Compute(i);
        //     CopyOut(i);
        // }

        int32_t loopCount = 1;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }

    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // AscendC::printf("CopyIN this->tileLength %ld\n", this->tileLength);
        AscendC::LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        AscendC::LocalTensor<TINDEX> indexLocal = inQueueIndex.AllocTensor<TINDEX>();

        AscendC::DataCopy(inputLocal, InputGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(indexLocal, IndexGm[progress * this->tileLength], this->tileLength);

        inQueueInput.EnQue(inputLocal);
        inQueueIndex.EnQue(indexLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {

        AscendC::LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
        AscendC::LocalTensor<TINDEX> indexLocal = inQueueIndex.DeQue<TINDEX>();
        AscendC::LocalTensor<T> outLocal = outQueueOutput.AllocTensor<T>();

        // AscendC::printf(" GETVALUE [%ld]\n" , indexLocal.GetValue(1));
        // AscendC::printf(" GET [%ld]\n" , indexLocal.GetValue(1));

        for(uint32_t i=0;i<this->tileLength;++i)
        { 
            outLocal.SetValue(i,inputLocal.GetValue(indexLocal.GetValue(i)));
        }
           
        // AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        // AscendC::DumpTensor(inputLocal,0,this->tileLength);
        // AscendC::DumpTensor(indexLocal,0,this->tileLength);
        
        outQueueOutput.EnQue<T>(outLocal);
        inQueueInput.FreeTensor(inputLocal);
        inQueueIndex.FreeTensor(indexLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> outLocal = outQueueOutput.DeQue<T>();
        AscendC::DataCopy(OutputGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueueOutput.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueInput, inQueueIndex;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
    AscendC::GlobalTensor<T> InputGm;
    AscendC::GlobalTensor<TINDEX> IndexGm;
    AscendC::GlobalTensor<T> OutputGm;
    // AscendC::TPipe pipe;
    // AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    // AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    // AscendC::GlobalTensor<T> xGm;
    // AscendC::GlobalTensor<T> yGm;
    // AscendC::GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gather_custom(GM_ADDR input, int64_t dim, GM_ADDR index, GM_ADDR output, uint32_t totalLength)
{
    Kernel<float,int64_t> op;
    op.Init(input, dim, index, output, totalLength);
    op.Process();
}
// extern "C" uint32_t aclrtlaunch_gather_custom(uint32_t blockDim, aclrtStream stream, void* input, int64_t dim, void* index, void* output, uint32_t totalLength);
// void launch_gather(
//     uint32_t block_dim, void *l2ctrl, void *stream,
//     uint8_t *input, uint8_t *dim, int32_t index, output, totalLength)
// {
//     gather_custom<<<block_dim, l2ctrl, stream>>>(input, dim, index, totalLength);
// }
void launch_gather(
    uint32_t blockDim,
    // void *l2ctrl,
    void *stream,
    uint8_t* input,
    int64_t dim,
    uint8_t* index,
    uint8_t* output,
    uint32_t totalLength
) {
    // GM_ADDR input_gm = (GM_ADDR)input;
    // GM_ADDR index_gm = (GM_ADDR)index;
    // GM_ADDR output_gm = (GM_ADDR)output;
    
    // gather_custom<<<blockDim, l2ctrl, stream>>>(
    gather_custom<<<blockDim, stream>>>(
        input, //input_gm,
        dim,
        index, //index_gm,
        output, //output_gm,
        totalLength
    );
}
