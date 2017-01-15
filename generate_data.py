import numpy as np
from random import randint

'''
Determines whether or not there is overlap between 2 rooms.
This function is used inside of the generate_grids function below.
Input: room1 = [xmin_1, x_max_1, ymin_1, ymax_1]
	   room2 = [xmin_2, xmax_2, ymin_2, ymax_2]
'''
def is_overlap(room1, room2):
	[x_min1, x_max1, y_min1, y_max1] = room1
	[x_min2, x_max2, y_min2, y_max2] = room2

	intersection = max(0, min(x_max1, x_max2) - max(x_min1, x_min2)) * max(0, min(y_max1, y_max2) - max(y_min1, y_min2))
	union = (x_max1 - x_min1) * (y_max1 - y_min1) + (x_max2 - x_min2) * (y_max2 - y_min2) - intersection

	overlap = float(intersection) / float(union)
    
	if (overlap > 0):
		return True
	else:
		return False

'''
Generates grids containing 4 rooms, each with a different color.
Colors are Red (20), Orange (30), Yellow (40), and Green (50).
Everything other than these rooms in the grid is Gray (60).
Room sizes are 1x2, 2x1, or 2x2.
Each grid has size 28x28.

Returns vector of size [num_grids x 784]
'''
def generate_grids(num_grids=100, grid_size=28, num_rooms=4):
	colors = {'red': 20, 'orange': 30, 'yellow': 40, 'green': 50, 'gray': 60}
	colors_arr = [20, 30, 40, 50]

	grid_count = 0 #Number of grids made so far
	returned_grids = np.zeros((num_grids, grid_size*grid_size))

	#while grid_count < num_grids:
	#	for height in xrange(1,3):
	#		for width in xrange(1,3):

	#Array containing all pairs of allowed height,width combinations for the rooms.
	allowed_dimensions = [(height, width) for height in xrange(1,3) for width in xrange(1,3)]

	#Generate all the grids.
	for i in xrange(num_grids):
		#Initialize the grid and initialize the color of each square in the grid to be gray.
		grid = colors['gray'] * np.ones((grid_size, grid_size))

		#Randomly select the dimensions of the four rooms.
		room_dimensions = [allowed_dimensions[randint(0, len(allowed_dimensions) - 1)] for j in xrange(num_rooms)]

		#Randomly select the locations of the four rooms given their selected dimensions.
		room_locations = []
		for j in xrange(num_rooms):

			overlap = True
			x1 = 0
			y1 = 0
			height1, width1 = room_dimensions[j]

			#Make sure the newly generated location doesn't overlap with other room locations.
			#If it does, then pick a new location for the new room.
			while overlap:

				overlap = False

				x1 = randint(0, grid_size - 1 - width)
				y1 = randint(0, grid_size - 1 - height)
				room1 = [x1, x1+width1, y1, y1+height1]

				if (len(room_locations)) > 0:
					for m in xrange(len(room_locations)):

						room2 = room_locations[m]
						if is_overlap(room1, room2):
							overlap = True

			room_locations.append([x1, x1+width1, y1, y1+height1])

			#Color the room the appropriate color in the grid
			#The color of the room corresponds to which room it is
			for w in xrange(x1, x1+width):
				for h in xrange(y1, y1+height):
					grid[h][w] = colors_arr[j]

		grid = np.reshape(grid, [1, grid_size*grid_size])
		returned_grids[i] = grid

	return returned_grids


a = generate_grids()
print np.reshape(a[0], (28,28))
