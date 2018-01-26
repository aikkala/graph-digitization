# graph-digitization
A Matlab function to digitize graphs (only the area defined by positive x-axis and positive y-axis) from images. Tested with Matlab R2017a.

## How to use
The function expects one parameter, a path indicating where the image is located. The image should be 

- in a common image format (jpg, png, gif, etc.)
- oriented such that positive x-axis is the lowest line, and positive y-axis is the leftmost line (as you'd normally view a graph)

## How the function works
Shortly, the function

1. does a Hough transform
1. finds the longest lines
1. assumes the line with smallest y-coordinate (of center of mass) is the positive x-axis, and 
1. the line with smallest x-coordinate (and nearly at right angle against positive x-axis) is the positive y-axis
1. finds the origo (intersection of abovementioned lines)
1. finds the points inside the square defined by positive x- and y-axes
1. transforms them from image's coordinate frame into the graph's coordinate frame
1. does spline interpolation

## Example

The input image

![alt text](https://github.com/aikkala/graph-digitization/blob/master/example/example_input.jpg "Example input")

The outputs

![alt text](https://github.com/aikkala/graph-digitization/blob/master/example/example_output1.png "Example output1")

![alt text](https://github.com/aikkala/graph-digitization/blob/master/example/example_output2.png "Example output2")


## Notes
The algorithm is not particularly robust: it is based on some heuristics, and is sensitive to parameter tuning. 
