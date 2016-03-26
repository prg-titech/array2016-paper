NUM_CARS = 1000
NUM_PEDS = 4000
NUM_STREETS = 500
MAX_CONNECTIONS = 10

class Car
	attr_accessor :street

	def initialize(street)
		@street = street
		@max_velocity = 0.0
		@progress = 0.0
	end
end

class Pedestrian
	attr_accessor :street

	def initialize(street)
		@street = street
		@progress = 0.0
	end
end

class Street
	attr_accessor :neighbors

	def initialize
		@max_velocity = 0.0
		@length = 0.0
		@neighbors = []
	end

	def add_neighbor(s)
		@neighbors.push(s)
	end
end

INST_VARS = {Car: [:@street, :@max_velocity, :@progress],
	Pedestrian: [:@street, :@progress],
	Street: [:@neighbors, :@length, :@max_velocity]}

# Generate street network
streets = Array.new(NUM_STREETS) do
	Street.new
end

streets.each do |street|
	for i in 1..rand(MAX_CONNECTIONS)
		street.add_neighbor(streets[rand(NUM_STREETS)])
	end
end

# Generate actors
actors = Array.new(NUM_PEDS + NUM_CARS)
for i in 0..(NUM_PEDS - 1)
	actors[i] = Pedestrian.new(streets[rand(NUM_STREETS)])
end
for i in NUM_PEDS..(NUM_PEDS + NUM_CARS - 1)
	actors[i] = Car.new(streets[rand(NUM_STREETS)])
end
actors.shuffle!


# Start tracing
roots = actors
obj_ids = {Car: {}, Pedestrian: {}, Street: {}}
next_ids = {Car: 0, Pedestrian: 0, Street: 0}

def trace(obj)
	if !obj_ids[obj.class].has_key?(obj)
		# object not yet traced
		id = next_ids[obj.class]
		next_ids[obj.class] += 1

		obj_ids[obj.class][obj] = id

		INST_VARS[obj.class].each do |iv_name|
			iv_value = obj.instance_variable_get(iv_name)

			if iv_value.class != Fixnum and iv_value.class != Float and iv_value.class != Bool
				trace(iv_value)
			end
		end
	end
end

roots.each do |obj|
	trace(obj)
end

# Write columns
col_Actors_street = Array.new(next_ids[Car] + next_ids[Pedestrian])
col_Actors_max_velocity = Array.new(next_ids[Car] + next_ids[Pedestrian])
col_Actors_progress = Array.new(next_ids[Car] + next_ids[Pedestrian])
col_Street_progress = Array.new(next_ids[Street])