require 'benchmark'

NUM_CARS = ARGV[0].to_i
NUM_PEDS = ARGV[1].to_i
NUM_STREETS = ARGV[2].to_i
MAX_CONNECTIONS = ARGV[3].to_i

class Actor

end

class Car < Actor
	attr_accessor :street, :max_velocity, :progress

	def initialize(street)
		@street = street
		@max_velocity = 0.0
		@progress = 0.0
	end
end

class Pedestrian < Actor
	attr_accessor :street, :progress

	def initialize(street)
		@street = street
		@progress = 0.0
	end
end

class Street
	attr_accessor :neighbors, :length, :max_velocity

	def initialize
		@max_velocity = 0.0
		@length = 0.0
		@neighbors = []
	end

	def add_neighbor(s)
		@neighbors.push(s)
	end
end

INST_VARS = {Car => [:@street, :@max_velocity, :@progress],
	Pedestrian => [:@street, :@progress],
	Street => [:@neighbors, :@length, :@max_velocity],
	Array => []}

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

puts "NUM_CARS: #{NUM_CARS}, NUM_PEDS: #{NUM_PEDS}, NUM_STREETS: #{NUM_STREETS}, MAX_CONNECTIONS: #{MAX_CONNECTIONS}"
puts Benchmark.measure {

# Start tracing
roots = actors
OBJ_IDS = {Car => {}, Pedestrian => {}, Street => {}, Array => {}}
NEXT_IDS = {Actor => 0, Street => 0, Array => 0}
array_counter = 0

def trace(obj)
	if !OBJ_IDS[obj.class].has_key?(obj)
		# object not yet traced

		# superclass check for polymorphic
		counter_class = obj.class
		if counter_class == Pedestrian or counter_class == Car
			counter_class = Actor
		end

		id = NEXT_IDS[counter_class]
		NEXT_IDS[counter_class] += 1

		OBJ_IDS[obj.class][obj] = id

		INST_VARS[obj.class].each do |iv_name|
			iv_value = obj.instance_variable_get(iv_name)

			if iv_value.class != Fixnum and iv_value.class != Float and iv_value.class != TrueClass and iv_value.class != FalseClass and iv_value.class != Array
				trace(iv_value)
			end
		end

		if obj.class == Array
			array_counter += obj.size

			obj.each do |el|
				if el.class != Fixnum and el.class != Float and el.class != Bool and el.class != Array
					trace(el)
				end
			end
		end
	end
end

roots.each do |obj|
	trace(obj)
end

# Write columns
col_Actors_street = Array.new(NEXT_IDS[Actor])
col_Car_max_velocity = Array.new(NEXT_IDS[Actor])
col_Actors_progress = Array.new(NEXT_IDS[Actor])
col_Actors_tag = Array.new(NEXT_IDS[Actor])
col_Street_max_velocity = Array.new(NEXT_IDS[Street])
col_Street_length = Array.new(NEXT_IDS[Street])
col_Street_neighbors = Array.new(NEXT_IDS[Street])
col_Array_offset = Array.new(NEXT_IDS[Array])
col_Array_size = Array.new(NEXT_IDS[Array])
col_Array_arrays = Array.new(array_counter)

OBJ_IDS[Car].each_pair do |key, value|
	col_Actors_street[value] = OBJ_IDS[Street][key.street]
	col_Actors_tag[value] = 1
	col_Actors_progress[value] = key.progress
	col_Car_max_velocity[value] = key.max_velocity
end

OBJ_IDS[Pedestrian].each_pair do |key, value|
	col_Actors_street[value] = OBJ_IDS[Street][key.street]
	col_Actors_tag[value] = 1
	col_Actors_progress[value] = key.progress
end

OBJ_IDS[Street].each_pair do |key, value|
	col_Street_length[value] = key.length
	col_Street_max_velocity[value] = key.max_velocity
	col_Street_neighbors[value] = OBJ_IDS[Array][key.neighbors]
end

arr_offset = 0
OBJ_IDS[Array].each_pair do |key, value|
	col_Array_size[value] = key.size
	col_Array_offset[value] = arr_offset

	for i in 0..(key.size - 1)
		col_Array_arrays[arr_offset] = key[i]
		arr_offset += 1
	end
end
}

#puts col_Actors_street