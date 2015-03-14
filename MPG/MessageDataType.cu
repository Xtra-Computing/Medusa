#include "../Algorithm/MessageDataType.h"
#include "../MedusaRT/Utilities.h"


//---------------------------------------------------------------------------------------------------------//

MessageArray::MessageArray()
{
	val = NULL;
	size = 0;
}

void MessageArray::resize(int new_size)
{
	if(size)
	{
		free(val);
	}
	size = new_size;
	if(size > 0)
		CPUMalloc((void**)&val, sizeof(int)*new_size);

}

//---------------------------------------------------------------------------------------------------------//


void D_MessageArray::Fill(MessageArray ma)
{
	if(size)
	{
		CUDA_SAFE_CALL(cudaFree(d_val));
	}
	size = ma.size;
	GPUMalloc((void**)&d_val, sizeof(MVT)*size);
	cudaMemcpy(d_val, ma.val,sizeof(MVT)*size, cudaMemcpyHostToDevice);
}

void D_MessageArray::resize(int new_size)
{
	if(size)
	{
		CUDA_SAFE_CALL(cudaFree(d_val));
	}
	size = new_size;
	GPUMalloc((void**)&d_val, sizeof(MVT)*size);
}

