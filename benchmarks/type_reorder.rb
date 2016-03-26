require 'benchmark'

NUM_CLS = [1, 100, 1000]
NUM_ITEMS = [100000, 500000, 1000000, 5000000, 10000000, 50000000]
WARP_SIZE = 32

def sort(base, w)
	types = Hash.new

	base.each_with_index do |obj, idx|
		if !types.has_key?(obj.class)
			types[obj.class] = []
		end
		types[obj.class].push(idx)
	end

	arr_size = 0
	types.each_value do |arr|
		arr_size += ((arr.size.to_f / w).ceil * w).to_i
	end
	result = Array.new(arr_size)

	next_idx = 0
	types.each_value do |arr|
		arr.each do |idx|
			result[next_idx] = idx
			next_idx += 1
		end

		next_idx = ((next_idx.to_f / w).ceil * w).to_i
	end

	return result
end


puts "WARP SIZE: #{WARP_SIZE}\n\n"

for num_cls in NUM_CLS
	for num_items in NUM_ITEMS

		classes = Array.new(num_cls) do
			Class.new
		end

		base = Array.new(num_items) do
			classes[rand(num_cls)].new
		end

		puts "CLS: #{num_cls}, ITEMS: #{num_items}"
		puts Benchmark.measure {
			sort(base, WARP_SIZE)
		}

		puts "\n\n"
	end
end