#!/usr/bin/env python
# make a horizontal bar chart

from pylab import *
val = [(0.180000 + 0.190000)/2, (0.850000+0.860000)/2, (3.800000+3.840000)/2, (22.030000+21.580000)/2]    # the bar lengths
pos = arange(4)+.5    # the bar centers on the y axis

figure(figsize=(10,5))
barh(pos,val, align='center', color='w', hatch=' ')
yticks(pos, ('500 streets, 4096 cars,\n 16384 pedestrians', '2000 streets, 16384 cars,\n 65536 pedestrians', '8000 streets, 65536 cars,\n 262144 pedestrians', '32000 streets, 262144 cars,\n 1048576 pedestrians'))
xlabel('Runtime (seconds)')
grid(False)

show()