
#define GOOD_WEATHER 0
#define BAD_WEATHER 1

#define TAG_Car 0
#define TAG_Pedestrian 1

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

__global__ void kernel(float *Actor_progress, 
	int *Actor_street, 
	float *Car_max_velocity,
	float *Street_length,
	float *Street_max_velocity,
	int *Street_neighbors,
	int *Array_Street_size,
	int *Array_Street_offset,
	int *Array_Street_arrays,
	int *input_actor_tag,
	int *input_actor_id,
	int weather,
	int ticks)
{

}