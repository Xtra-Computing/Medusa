/*
 * Compatability.h
 *
 *  Created on: Jul 4, 2016
 *      Author: cheyulin
 */

#ifndef COMPATABILITY_H_
#define COMPATABILITY_H_

#ifndef Replace_CUDA_SAFE_CALL
#define Replace_CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(x) checkCudaErrors(x)
#define cutilSafeCall(x) checkCudaErrors(x)
#endif

#ifndef Replace_cutilCheckMsg
#define Replace_cutilCheckMsg
#define cutilCheckMsg(x) getLastCudaError(x)
#endif

#ifndef Replace_cutilDeviceSynchronize
#define Replace_cutilDeviceSynchronize
#define cutilDeviceSynchronize() cudaDeviceSynchronize()
#endif

#ifndef Replace_cutCreateTimer
#define Replace_cutCreateTimer
#define cutCreateTimer(x) sdkCreateTimer(x)
#endif

#ifndef Replace_cutResetTimer
#define Replace_cutResetTimer
#define cutResetTimer(x) sdkResetTimer(x)
#endif

#ifndef Replace_cutStartTimer
#define Replace_cutStartTimer
#define cutStartTimer(x) sdkStartTimer(x)
#endif

#ifndef Replace_cutStopTimer
#define Replace_cutStopTimer
#define cutStopTimer(x) sdkStopTimer(x)
#endif

#ifndef Replace_cutGetTimerValue
#define Replace_cutGetTimerValue
#define cutGetTimerValue(x) sdkGetTimerValue(x)
#endif


#endif /* COMPATABILITY_H_ */
