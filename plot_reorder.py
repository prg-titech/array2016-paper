import matplotlib.pyplot as plt

x_axis = [100000, 500000, 1000000, 5000000, 10000000]
p1, = plt.plot(x_axis, [0.130000,0.730000,1.450000,7.310000, 15.020000], 'k:')
p2, = plt.plot(x_axis, [0.150000,0.720000,1.460000,8.250000, 15.620000], 'k--')
p3, = plt.plot(x_axis, [0.150000,0.780000,1.550000,9.130000, 16.960000], 'k')

plt.ylabel('Runtime (seconds)')
plt.xlabel('Number of elements in base array')

labels = x_axis

# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x_axis, labels, rotation='vertical')
plt.legend([p1, p2, p3], ['#types: 1', '#types: 100', '#types: 1000'], loc=2)

plt.show()