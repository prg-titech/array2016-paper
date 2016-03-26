#define REORDER 0

#define GOOD_WEATHER 0
#define BAD_WEATHER 1

#define TAG_Car 0
#define TAG_Pedestrian 1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <random>
//#include <array> 
#include <algorithm> 

#define NUM_CARS 4096
#define NUM_PEDS 16384
#define NUM_STREETS 500
#define MAX_CONNECTIONS 10
#define MAX_LEN 25

using namespace std;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

__device__ float *d_Actor_progress;
__device__ int *d_Actor_street;
__device__ float *d_Car_max_velocity;
__device__ float *d_Street_length;
__device__ float *d_Street_max_velocity;
__device__ int *d_Street_neighbors;
__device__ int *d_Array_Street_size;
__device__ int *d_Array_Street_offset;
__device__ int *d_Array_Street_arrays;
__device__ int *d_input_actor_tag;
__device__ int *d_input_actor_id;
__device__ int *d_jobs;
__device__ int *d_randomn;

__device__ void method_Car_move(int actor_id, int weather)
{
	float weather_multiplier;
	if (weather == GOOD_WEATHER) 
	{
		weather_multiplier = 1.0;
	}
	else if (weather == BAD_WEATHER)
	{
		weather_multiplier = 0.75;
	}

	float speed = min(d_Car_max_velocity[actor_id], d_Street_max_velocity[d_Actor_street[actor_id]]) * weather_multiplier;
	d_Actor_progress[actor_id] = d_Actor_progress[actor_id] + (speed / 60.0); /* 1 tick = 1 minute */

	if (d_Actor_progress[actor_id] >= d_Street_length[d_Actor_street[actor_id]])
	{
		// move to different street
		int array_id = d_Street_neighbors[d_Actor_street[actor_id]];
		int neighbor_index = d_randomn[d_Actor_street[actor_id]] % d_Array_Street_size[array_id];
		d_Actor_street[actor_id] = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
		d_Actor_progress[actor_id] = 0.0f;
	}
}

__device__ void method_Pedestrian_move(int actor_id, int weather)
{
	float speed = d_randomn[((int) (d_Actor_progress[actor_id]*d_Actor_progress[actor_id])) % NUM_STREETS] % 7 - 2;
	d_Actor_progress[actor_id] = d_Actor_progress[actor_id] + (speed / 60.0);

	if (d_Actor_progress[actor_id] >= d_Street_length[d_Actor_street[actor_id]])
	{
		// move to different street
		int array_id = d_Street_neighbors[d_Actor_street[actor_id]];
		int neighbor_index = d_randomn[d_Actor_street[actor_id]] % d_Array_Street_size[array_id];
		d_Actor_street[actor_id] = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
		d_Actor_progress[actor_id] = 0.0f;
	}
}

__device__ void block(int actor_tag, int actor_id, int weather, int ticks)
{
	for (int i = 0; i < ticks; i++)
	{
		switch (actor_tag)
		{
			case TAG_Car:
				method_Car_move(actor_id, weather);
				break;
			case TAG_Pedestrian:
				method_Pedestrian_move(actor_id, weather);
				break;
		}
	}
}

__global__ void kernel(int weather,	int ticks,
	float *v_d_Actor_progress, int *v_d_Actor_street, float *v_d_Car_max_velocity,
	float *v_d_Street_length, float *v_d_Street_max_velocity, int *v_d_Street_neighbors,
	int *v_d_Array_Street_size, int *v_d_Array_Street_offset, int *v_d_Array_Street_arrays,
	int *v_d_input_actor_tag, int *v_d_input_actor_id, int *v_d_jobs, int *v_d_randomn)
{
	d_Actor_progress = v_d_Actor_progress;
	d_Actor_street = v_d_Actor_street;
	d_Car_max_velocity = v_d_Car_max_velocity;
	d_Street_length = v_d_Street_length;
	d_Street_max_velocity = v_d_Street_max_velocity;
	d_Street_neighbors = v_d_Street_neighbors;
	d_Array_Street_size = v_d_Array_Street_size;
	d_Array_Street_offset = v_d_Array_Street_offset;
	d_Array_Street_arrays = v_d_Array_Street_arrays;
	d_input_actor_tag = v_d_input_actor_tag;
	d_input_actor_id = v_d_input_actor_id;
	d_jobs = v_d_jobs;
	d_randomn = v_d_randomn;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__syncthreads();

#if (REORDER)
	block(d_input_actor_tag[d_jobs[tid]], d_input_actor_id[d_jobs[tid]], weather, ticks);
#else
	block(d_input_actor_tag[tid], d_input_actor_id[tid], weather, ticks);
#endif
}

int main()
{
	printf("Setting up scenario...\n");
	srand(42);

	// streets
	float *Street_length = new float[NUM_STREETS];
	float *Street_max_velocity = new float[NUM_STREETS];
	int *Street_neighbors = new int[NUM_STREETS];

	for (int i = 0; i < NUM_STREETS; i++)
	{
		Street_length[i] = rand() % MAX_LEN + 1;
		Street_max_velocity[i] = rand() % 40 + 45;	/* speed between 45 and 105 */
		Street_neighbors[i] = i;
	}

	// neighbors
	int *Array_Street_offset = new int[NUM_STREETS];
	int *Array_Street_size = new int[NUM_STREETS];
	int num_connections = 0;

	for (int i = 0; i < NUM_STREETS; i++)
	{
		Array_Street_offset[i] = num_connections;
		int connections = rand() % MAX_CONNECTIONS + 1;
		Array_Street_size[i] = connections;
		num_connections += connections;
	}

	int *Array_Street_arrays = new int[num_connections];
	for (int i = 0; i < num_connections; i++)
	{
		Array_Street_arrays[i] = rand() % NUM_STREETS;
	}

	// actors
	int *Actor_street = new int[NUM_PEDS + NUM_CARS];
	float *Actor_progress = new float[NUM_PEDS + NUM_CARS];
	float *Car_max_velocity = new float[NUM_CARS + NUM_PEDS];
	int *Actor_tag = new int[NUM_PEDS + NUM_CARS];
	int *Actor_id = new int[NUM_PEDS + NUM_CARS];

	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_street[i] = rand() % NUM_STREETS;
		Actor_progress[i] = rand() % 10;
		Car_max_velocity[i] = rand() % 20 + 65;
	}

	for (int i = 0; i < NUM_PEDS; i++)
	{
		Actor_tag[i] = TAG_Pedestrian;
		Actor_id[i] = i;
	}

	for (int i = NUM_PEDS; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_tag[i] = TAG_Car;
		Actor_id[i] = i;
	}

	std::srand(42);
#if !(REORDER)
	random_shuffle(Actor_tag, Actor_tag + NUM_CARS + NUM_PEDS);
#endif

	// jobs (dummy)
	int *jobs = new int[NUM_PEDS + NUM_CARS];

	for (int i = 0; i < NUM_CARS + NUM_PEDS; i++)
	{
		jobs[i] = i;
	}

	// random numbers
	int *randomn = new int[NUM_STREETS];
	for (int i = 0; i < NUM_STREETS; i++)
	{
		randomn[i] = i; 
	}

	printf("Scenario set up.\n");

	printf("Copying data to GPU...\n");
	float *v_d_Actor_progress;
	int *v_d_Actor_street;
	float *v_d_Car_max_velocity;
	float *v_d_Street_length;
	float *v_d_Street_max_velocity;
	int *v_d_Street_neighbors;
	int *v_d_Array_Street_size;
	int *v_d_Array_Street_offset;
	int *v_d_Array_Street_arrays;
	int *v_d_input_actor_tag;
	int *v_d_input_actor_id;
	int *v_d_jobs;
	int *v_d_randomn;

	CudaSafeCall(cudaMalloc((void**) &v_d_Actor_progress, sizeof(float) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_Actor_street, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_Car_max_velocity, sizeof(float) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_Street_length, sizeof(float) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Street_max_velocity, sizeof(float) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Street_neighbors, sizeof(int) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_size, sizeof(int) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_offset, sizeof(int) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_arrays, sizeof(int) * num_connections));
	CudaSafeCall(cudaMalloc((void**) &v_d_input_actor_tag, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_input_actor_id, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_jobs, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_randomn, sizeof(int) * NUM_STREETS));

	CudaSafeCall(cudaMemcpy(v_d_Actor_progress, &Actor_progress[0], sizeof(float) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Actor_street, &Actor_street[0], sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Car_max_velocity, &Car_max_velocity[0], sizeof(float) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Street_length, &Street_length[0], sizeof(float) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Street_max_velocity, &Street_max_velocity[0], sizeof(float) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Street_neighbors, &Street_neighbors[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_size, &Array_Street_size[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_offset, &Array_Street_offset[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_arrays, &Array_Street_arrays[0], sizeof(int) * num_connections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_input_actor_tag, &Actor_tag[0], sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_input_actor_id, &Actor_id[0], sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_jobs, &jobs[0], sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_randomn, &randomn[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));

	printf("Finished copying data.\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("Launching kernel...\n");
	cudaEventRecord(start);
	kernel<<<dim3(32), dim3((NUM_PEDS + NUM_CARS) / 32)>>>(GOOD_WEATHER, 1000000, 
		v_d_Actor_progress, v_d_Actor_street, v_d_Car_max_velocity, v_d_Street_length, v_d_Street_max_velocity,
		v_d_Street_neighbors, v_d_Array_Street_size, v_d_Array_Street_offset, v_d_Array_Street_arrays, v_d_input_actor_tag,
		v_d_input_actor_id, v_d_jobs, v_d_randomn);
	CudaCheckError();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	CudaCheckError();
	printf("Kernel finished.\n");

	cudaMemcpy(Actor_progress, v_d_Actor_progress, sizeof(float) * (NUM_PEDS + NUM_CARS), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
//		printf(" %f ", Actor_progress[i]);
	}

	cudaMemcpy(Actor_street, v_d_Actor_street, sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
//		printf(" %i ", Actor_street[i]);
	}

	printf("\n\n\nElapsed time millis: %f\n", milliseconds);
}
