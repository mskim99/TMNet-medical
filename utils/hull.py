#! /usr/bin/python

import math
import sys
import numpy as np
import torch

def set_correct_normal(possible_internal_points,plane): #Make the orientation of Normal correct
	for point in possible_internal_points:
		dist = dotProduct(plane.normal,point - plane.pointA)
		if(dist != 0) :
			if(dist > 10**-10):
				plane.normal[0, 0] = -1*plane.normal[0, 0]
				plane.normal[0, 1] = -1*plane.normal[0, 1]
				plane.normal[0, 2] = -1*plane.normal[0, 2]
				return         

def printV(vec): # Print points
	print(vec.x, vec.y, vec.z)

def cross(pointA, pointB): # Cross product
	x = (pointA[0, 1] * pointB[0, 2]) - (pointA[0, 2] * pointB[0, 1])
	y = (pointA[0, 2] * pointB[0, 0]) - (pointA[0, 0] * pointB[0, 2])
	z = (pointA[0, 0] * pointB[0, 1]) - (pointA[0, 1] * pointB[0, 0])
	return Point(x, y, z)

def dotProduct(pointA, pointB): # Dot product
	return (pointA[0, 0] * pointB[0, 0] + pointA[0, 1] * pointB[0, 1] + pointA[0, 2] * pointB[0, 2])

def checker_plane(a, b): #Check if two planes are equal or not

	print(a.pointC[0])

	if ((a.pointA[0, 0].item() == b.pointA[0, 0].item()) and (a.pointA[0, 1].item() == b.pointA[0, 1].item()) and (a.pointA[0, 2].item() == b.pointA[0, 2].item())):
		if ((a.pointB[0, 0].item() == b.pointB[0, 0].item()) and (a.pointB[0, 1].item() == b.pointB[0, 1].item()) and (a.pointB[0, 2].item() == b.pointB[0, 2].item())):
			if ((a.pointC[0, 0].item() == b.pointC[0, 0].item()) and (a.pointC[1, 0].item() == b.pointC[0, 1].item()) and (a.pointC[2, 0].item() == b.pointC[2, 0].item())):
				return True

		elif ((a.pointB[0, 0].item() == b.pointC[0].item()) and (a.pointB[0, 1].item() == b.pointC[1].item()) and (a.pointB[0, 2].item() == b.pointC[2].item())):
			if ((a.pointC[0].item() == b.pointB[0, 0].item()) and (a.pointC[1].item() == b.pointB[0, 1].item()) and (a.pointC[2].item() == b.pointB[0, 2].item())):
				return True
				
	if ((a.pointA[0, 0].item() == b.pointB[0, 0].item()) and (a.pointA[0, 1].item() == b.pointB[0, 1].item()) and (a.pointA[0, 2].item() == b.pointB[0, 2].item())):
		if ((a.pointB[0, 0].item() == b.pointA[0, 0].item()) and (a.pointB[0, 1].item() == b.pointA[0, 1].item()) and (a.pointB[0, 2].item() == b.pointA[0, 2].item())):
			if ((a.pointC[0].item() == b.pointC[0].item()) and (a.pointC[1].item() == b.pointC[1].item()) and (a.pointC[2].item() == b.pointC[2].item())):
				return True

		elif ((a.pointB[0, 0].item() == b.pointC[0].item()) and (a.pointB[0, 1].item() == b.pointC[1].item()) and (a.pointB[0, 2].item() == b.pointC[2].item())):
			if ((a.pointC[0].item() == b.pointA[0, 0].item()) and (a.pointC[1].item() == b.pointA[0, 1].item()) and (a.pointC[2].item() == b.pointA[0, 2].item())):
				return True

	if ((a.pointA[0, 0].item() == b.pointC[0].item()) and (a.pointA[0, 1].item() == b.pointC[1].item()) and (a.pointA[0, 2].item() == b.pointC[2].item())):
		if ((a.pointB[0, 0].item() == b.pointA[0, 0].item()) and (a.pointB[0, 1].item() == b.pointA[0, 1].item()) and (a.pointB[0, 2].item() == b.pointA[0, 2].item())):
			if ((a.pointC[0].item() == b.pointB[0, 0].item()) and (a.pointC[1].item() == b.pointB[0, 1].item()) and (a.pointC[2].item() == b.pointB[0, 2].item())):
				return True

		elif ((a.pointB[0, 0].item() == b.pointC[0].item()) and (a.pointB[0, 1].item() == b.pointC[1].item()) and (a.pointB[0, 2].item() == b.pointC[2].item())):
			if ((a.pointC[0].item() == b.pointB[0, 0].item()) and (a.pointC[1].item() == b.pointB[0, 1].item()) and (a.pointC[2].item() == b.pointB[0, 2].item())):
				return True			
			
	return False

def checker_edge(a, b): # Check if 2 edges have same 2 vertices

	if ((a.pointA == b.pointA)and(a.pointB == b.pointB)) or ((a.pointB == b.pointA)and(a.pointA == b.pointB)):
		return True

	return False
		
class Edge: # Make a object of type Edge which have two points denoting the vertices of the edges
	def __init__(self,pointA,pointB):
		self.pointA = pointA
		self.pointB = pointB

	def __str__(self):
		string = "Edge"
		string += "\n\tA: "+ str(self.pointA.x)+","+str(self.pointA.y)+","+str(self.pointA.z)
		string += "\n\tB: "+ str(self.pointB.x)+","+str(self.pointB.y)+","+str(self.pointB.z)
		return string

	def __hash__(self):
		return hash((self.pointA,self.pointB))

	def __eq__(self,other):
		# print "comparing Edges"
		return checker_edge(self,other)

class Point: #Point class denoting the points in the space
	def __init__(self, x=None, y=None, z=None):
		self.x = x
		self.y = y
		self.z = z

	def __sub__(self, pointX):
		return	Point(self.x - pointX.x, self.y - pointX.y, self.z - pointX.z)

	def __add__(self, pointX):
		return	Point(self.x + pointX.x, self.y + pointX.y, self.z + pointX.z)		

	def length(self):
		return math.sqrt(self.x**2 + self.y**2 + self.z**2) 

	def __str__(self):
		return str(self.x)+","+str(self.y)+","+str(self.z)

	def __hash__(self):
		return hash((self.x,self.y,self.z))

	def __eq__(self,other):
		# print "Checking equality of Point"
		return (self.x==other.x) and(self.y==other.y) and(self.z==other.z) 	

class Plane: # Plane class having 3 points for a triangle
	def __init__(self, pointA, pointB, pointC):
		self.pointA = pointA
		self.pointB = pointB
		self.pointC = pointC
		self.normal = None
		self.distance = None
		self.calcNorm()
		self.to_do = set()	
		self.edge1 = Edge(pointA, pointB)
		self.edge2 = Edge(pointB, pointC)
		self.edge3 = Edge(pointC, pointA)

	def calcNorm(self):
		point1 = self.pointA - self.pointB
		point2 = self.pointB - self.pointC
		normVector = torch.cross(point1,point2)
		length = torch.linalg.norm(normVector)
		normVector[0, 0] = normVector[0, 0] / length
		normVector[0, 1] = normVector[0, 1] / length
		normVector[0, 2] = normVector[0, 2] / length
		self.normal = normVector
		self.distance = dotProduct(self.normal,self.pointA)

	def dist(self, pointX):
		return (dotProduct(self.normal,pointX - self.pointA))

	def get_edges(self):
		return [self.edge1, self.edge2, self.edge3]

	def calculate_to_do(self, points, temp=None):
		if (temp != None):
			for p in temp:
				dist = self.dist(p)
				if dist > 10**(-10):
					self.to_do.add(p)

		else:
			for i in range(0, points.size(2)):
				p = points[0, :, i]
				dist = self.dist(p)
				if dist > 10**(-10):
					self.to_do.add(p)

	def __eq__(self,other):
		# print 'Checking Plane Equality'
		return checker_plane(self,other)

	def __str__(self):
		string =  "Plane : "
		string += "\n\tX: "+str(self.pointA.x)+","+str(self.pointA.y)+","+str(self.pointA.z)
		string += "\n\tY: "+str(self.pointB.x)+","+str(self.pointB.y)+","+str(self.pointB.z)
		string += "\n\tZ: "+str(self.pointC.x)+","+str(self.pointC.y)+","+str(self.pointC.z)
		string += "\n\tNormal: "+str(self.normal.x)+","+str(self.normal.y)+","+str(self.normal.z)
		return string

	def __hash__(self):
		return hash((self.pointA,self.pointB,self.pointC))

def calc_horizon(list_of_planes, visited_planes,plane,eye_point,edge_list): # Calculating the horizon for an eye to make new faces
	if (plane.dist(eye_point) > 10**-10):
		visited_planes.append(plane)
		edges = plane.get_edges()
		for edge in edges:
			neighbour = adjacent_plane(list_of_planes, plane,edge)
			if (neighbour not in visited_planes):
				result = calc_horizon(visited_planes,neighbour,eye_point,edge_list)
				if(result == 0):
					edge_list.add(edge)

		return 1
	
	else:
		return 0
				
def adjacent_plane(list_of_planes, main_plane,edge): # Finding adjacent planes to an edge
	for plane in list_of_planes:
		edges = plane.get_edges()
		if (plane != main_plane) and (edge in edges):
			return plane


def distLine(pointA, pointB, pointX): #Calculate the distance of a point from a line
	vec1 = pointX - pointA
	vec2 = pointX - pointB
	vec3 = pointB - pointA
	vec4 = cross(vec1, vec2)
	if torch.linalg.norm(vec3) == 0:
		return None

	else:
		return vec4.length() / torch.linalg.norm(vec3)

def max_dist_line_point(points, pointA, pointB): #Calculate the maximum distant point from a line for initial simplex
	maxDist = 0;
	maxDistPoint = torch.zeros([1, 3])
	maxDistPoint = maxDistPoint.cuda()

	for i in range(0, points.size(2)):
		point = points[0, :, i]
		if not torch.all(pointA.eq(point)) and torch.all(pointB.eq(point)):
			dist = abs(distLine(pointA,pointB,point))
			if dist>maxDist:
				maxDistPoint = point
				maxDist = dist

	return maxDistPoint

def max_dist_plane_point(points, plane): # Calculate the maximum distance from the plane
	maxDist = 0
	maxDistPoint = torch.zeros([1, 3])
	maxDistPoint = maxDistPoint.cuda()
	
	for i in range(0, points.size(2)):
		point = points[0, :, i]
		dist = abs(plane.dist(point))
		if (dist > maxDist):
			maxDist = dist
			maxDistPoint = point

	return maxDistPoint

def find_eye_point(plane, to_do_list): # Calculate the maximum distance from the plane
	maxDist = 0
	for point in to_do_list:
		dist = plane.dist(point)
		if (dist > maxDist):
			maxDist = dist
			maxDistPoint = point

	return maxDistPoint    

def initial_dis(p, q): # Gives the Euclidean distance
	return math.sqrt((p[0, 0]-q[0, 0])**2+(p[0, 1]-q[0, 1])**2+(p[0, 2]-q[0, 2])**2)

def initial_max(now): # From the extreme points calculate the 2 most distant points
	maxi = -1
	found = [[], []]
	for i in range(0, 6):
		for j in range(i+1, 6):
			dist = initial_dis(now[i], now[j])
			if dist > maxi:
				found = [now[i], now[j]]

	return found	

def initial(points, num): # To calculate the extreme points to make the initial simplex

	x_min_temp = 10**9
	x_max_temp = -10**9
	y_min_temp = 10**9
	y_max_temp = -10**9
	z_min_temp = 10**9
	z_max_temp = -10**9
	for i in range(0, num):
		if points[:, 0, i] > x_max_temp: # i, x
			x_max_temp = points[:, 0, i]
			x_max = points[:, :, i]

		if points[:, 0, i] < x_min_temp:
			x_min_temp = points[:, 0, i]
			x_min = points[:, :, i]

		if points[:, 1, i] > y_max_temp:  # i, y
			y_max_temp = points[:, 1, i]
			y_max = points[:, :, i]

		if points[:, 1, i] < y_min_temp:
			y_min_temp = points[:, 1, i]
			y_min = points[:, :, i]

		if points[:, 2, i] > z_max_temp:
			z_max_temp = points[:, 2, i]
			z_max = points[:, :, i]

		if points[:, 2, i] < z_min_temp:
			z_min_temp = points[:, 2, i]
			z_min = points[:, :, i]

	return (x_max, x_min, y_max, y_min, z_max, z_min)						

def convex_hull(points):

	extremes = initial(points, points.size(2)) # calculate the extreme points for every axis.
	initial_line = initial_max(extremes) # Make the initial line by joining farthest 2 points
	third_point = max_dist_line_point(points, initial_line[0], initial_line[1]) # Calculate the 3rd point to make a plane
	first_plane = Plane(initial_line[0], initial_line[1], third_point) # Make the initial plane by joining 3rd point to the line

	fourth_point = max_dist_plane_point(points, first_plane) # Make the fourth plane to make a tetrahedron

	possible_internal_points = [initial_line[0],initial_line[1],third_point,fourth_point] # List that helps in calculating orientation of point

	second_plane = Plane(initial_line[0], initial_line[1], fourth_point) # The other planes of the tetrahedron
	third_plane = Plane(initial_line[0], fourth_point, third_point)
	fourth_plane = Plane(initial_line[1], third_point, fourth_point)

	set_correct_normal(possible_internal_points,first_plane) # Setting the orientation of normal correct
	set_correct_normal(possible_internal_points,second_plane)
	set_correct_normal(possible_internal_points,third_plane)
	set_correct_normal(possible_internal_points,fourth_plane)

	first_plane.calculate_to_do(points) # Calculating the to_do list which stores the point for which  eye_point have to be found
	second_plane.calculate_to_do(points)
	third_plane.calculate_to_do(points)
	fourth_plane.calculate_to_do(points)

	list_of_planes = [] # List containing all the planes
	list_of_planes.append(first_plane)
	list_of_planes.append(second_plane)
	list_of_planes.append(third_plane)
	list_of_planes.append(fourth_plane)

	any_left = True # Checking if planes with to do list is over

	while any_left:
		any_left = False
		for working_plane in list_of_planes:
			if len(working_plane.to_do) > 0:
				any_left = True
				eye_point = find_eye_point(working_plane, working_plane.to_do) # Calculate the eye point of the face

				edge_list = set()
				visited_planes = []

				calc_horizon(list_of_planes, visited_planes, working_plane, eye_point, edge_list)  # Calculate the horizon

				for internal_plane in visited_planes: # Remove the internal planes
					list_of_planes.remove(internal_plane)

				for edge in edge_list:	# Make new planes
					new_plane = Plane(edge.pointA, edge.pointB, eye_point)
					set_correct_normal(possible_internal_points,new_plane)

					temp_to_do = set()
					for internal_plane in visited_planes:
						temp_to_do = temp_to_do.union(internal_plane.to_do)

					new_plane.calculate_to_do(temp_to_do)

					list_of_planes.append(new_plane)

	final_vertices = set()
	for plane in list_of_planes:
		final_vertices.add(plane.pointA)
		final_vertices.add(plane.pointB)
		final_vertices.add(plane.pointC)

	return final_vertices