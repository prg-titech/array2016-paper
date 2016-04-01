import numpy as np
import matplotlib.pyplot as plt

N = 3


ind = np.arange(N) # the x locations for the groups
width = 0.4       # the width of the bars

fig, ax = plt.subplots(figsize=(10,5))

ax.set_ylim(0,32) # outliers only
#ax2.set_ylim(0,35) # most of the data

#ax.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
#ax.tick_params(labeltop='off') # don't put tick labels at the top
ax.xaxis.tick_bottom()
fig.subplots_adjust(hspace=0.1)

# call-site-specific
noneV = (11.63, 16.01, 26.71)
rectsNone = ax.bar(ind, noneV, width, color='w', hatch=' ')
#ax2.bar(ind, noneV, width, color='w')

# call-target-specific uncached
classCached = (7.329, 11.50, 20.88)
rectsClassCached = ax.bar(ind+width, classCached, width, color='w', hatch='o')
#ax2.bar(ind+width, classCached, width, color='w', hatch='/')

# call-target-specific cached
#classUncached = (2.634, 3.358, 5.583, 6.838)
#rectsClassUncached = ax.bar(ind+2*width, classUncached, width, color='w', hatch='o')
#ax2.bar(ind+2*width, classUncached, width, color='w', hatch='o')



# add some text for labels, title and axes ticks
#ax2.set_ylabel('Runtime (ms)')
#ax.set_title('Average rendering runtime per frame')
ax.set_ylabel('Kernel runtime (seconds)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('(a) SoA layout', '(b) AoS layout \n (obj. size 32B)', '(c) AoS layout \n (obj. size 72B)') )
#ax2.set_yticks(ax2.get_yticks()[:-1])
ax.set_yticks(ax.get_yticks()[1:])

ax.legend( (rectsNone[0], rectsClassCached[0]), ('without reordering', 'with reordering') , loc=2)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            ax.text(rect.get_x()+rect.get_width()/2., height+2, 'n/a',
                ha='center', va='bottom', rotation='vertical')       
        else:
            ax.text(rect.get_x()+rect.get_width()/2., height+1.5, '%.2f'%float(height),
                    ha='center', va='bottom', rotation='vertical')

autolabel(rectsNone)
autolabel(rectsClassCached)




plt.show()