% NOTE: Image should be in some image format (jpg, png, gif), and in
% correct orientation
function digitize_graph(path_to_image)

image_size = [640, 860];

% Open the image as grayscale image
gray_img = flipud(rgb2gray(imread(path_to_image)));
figure;

% Preprocess the image
[B, B_nonsmooth] = preprocess(gray_img, image_size);

% Find positive x-axis ('line1') and positive y-axis ('line2')
[line1, line2] = find_axes(B);

% Find origo
eq1 = get_line_equation(line1);
eq2 = get_line_equation(line2);
origo = find_intersection(eq1, eq2);
plot(origo(1), origo(2), 'rs', 'markerfacecolor', 'r', 'markersize', 10);

% We need to update vec2 so that it begins from origo
line2.point1 = origo';
vec2 = get_vector(line2);

% Find those pixels that are inside the area defined by found axes
point3 = [line1.point2(1) + vec2(1), line1.point2(2) + vec2(2)];
% Plot these lines; makes it easier to see when things go wrong
plot([line1.point2(1), point3(1)], [line1.point2(2), point3(2)], 'g');
plot([line2.point2(1), point3(1)], [line2.point2(2), point3(2)], 'g');

% Calculate the equations for these new lines
line3 = struct();
line3.point1 = [line1.point2(1), line1.point2(2)];
line3.point2 = [point3(1), point3(2)];
eq3 = get_line_equation(line3);

line4 = struct();
line4.point1 = [line2.point2(1), line2.point2(2)];
line4.point2 = [point3(1), point3(2)];
eq4 = get_line_equation(line4);

% Calculate the skeleton line
skel = bwmorph(B_nonsmooth, 'skel', inf);

% Go through each pixel in skel, calculate whether the pixel is inside the
% defined square
[i, j] = find(skel>0);

img_valid_points = false(size(B));

for idx = 1:size(i,1)
     coords = [j(idx), i(idx)];
%      plot(coords(1), coords(2),'rx')
     
     % Calculate distance to each line
     % lines along x-axis
     dist1 = point_distance_from_line(coords, eq1);
     dist4 = point_distance_from_line(coords, eq4);
     
     % lines along y-axis
     dist2 = point_distance_from_line(coords, eq2);
     dist3 = point_distance_from_line(coords, eq3);
     
     % Check the distances; points must be far enough from the axes
     if abs(dist1) < 5 || abs(dist2) < 5 || abs(dist3) < 5 || abs(dist4) < 5
         continue;
     % The points must also be inside the defined lines/equations
     elseif dist1 > 0 || dist4 < 0
         continue;
     end
     
     % Note: eq2 and eq3 must be treated differently when A is positive or
     % negative
     if strcmp(eq2.type, 'y') && eq2.A < 0 && dist2 > 0
         continue;
     elseif (strcmp(eq2.type, 'x') || eq2.A >= 0) && dist2 < 0
         continue;
     end

     if strcmp(eq3.type, 'y') && eq3.A < 0 && dist3 < 0
         continue;
     elseif (strcmp(eq3.type, 'x') || eq3.A >= 0) && dist3 > 0
         continue;
     end

     img_valid_points(i(idx),j(idx)) = 1;
end

% Find the biggest connected line (gets rid of any remaining noise pixels)
cc = bwareafilt(img_valid_points,1);

% Get the coordinates of the remaining line
[i, j] = find(cc);

% Plot the valid points
plot(j,i, 'r.', 'markersize', 0.5);

% We need to calculate transformation from image's coordinates to graph's
% coordinates
vec1 = get_vector(line1);
rot_matrix = estimate_rotation_matrix(vec1);

% Transform to new coordinates
valid_points_transformed = transform_coordinates(j, i, rot_matrix, origo, vec2);
    
% Fit a smoothing spline to the data
curve = fit(valid_points_transformed(1,:)', valid_points_transformed(2,:)', 'smoothingspline', 'smoothingparam', 1-1e-6);
x_interp = 0:0.001:1;
y_interp = feval(curve,x_interp);
figure, plot(x_interp, y_interp);
axis([0,1,0,1]);

end

% Transform coordinates from image's frame to graph's frame
function valid_points_transformed = transform_coordinates(j, i, rot_matrix, origo, vec2)
valid_points_transformed = zeros(2,length(j));
for point_idx = 1:length(j)
    coords = [j(point_idx); i(point_idx)];
    % Rotate and translate
    valid_points_transformed(:,point_idx) = (rot_matrix*(-origo))+coords;
    % Scale y-coordinate, maximum is the length of vec2 (positive y-axis)
    valid_points_transformed(2,point_idx) = valid_points_transformed(2,point_idx)/sqrt(sum(vec2.^2));
end

% There's no x scale -- make the points start from zero and end at one
valid_points_transformed(1,:) = valid_points_transformed(1,:) - min(valid_points_transformed(1,:),[],2);
valid_points_transformed(1,:) = valid_points_transformed(1,:)./max(valid_points_transformed(1,:),[],2);
end

% Estimate rotation matrix from positive x-axis
function rot_matrix = estimate_rotation_matrix(vec1)
if (vec1(2) < 0)
    rot_angle = -asin(-vec1(2)/vec1(1));
else
    rot_angle = asin(vec1(2)/vec1(1));
end
rot_matrix = [cos(rot_angle), -sin(rot_angle); sin(rot_angle), cos(rot_angle)];
end

% Calculate a point's distance from given line
function dist = point_distance_from_line(point, eq)
if strcmp(eq.type, 'x')
    dist = point(1) - eq.A;
else
    dist = (eq.A*point(1) + -1*point(2) + eq.B)/sqrt(eq.A^2 + 1);
end
end

% Find the intersection of two lines
function coords = find_intersection(line1, line2)
% Make sure lines aren't parallel
if strcmp(line1.type, 'x') && strcmp(line2.type, 'x')
    error 'Lines are parallel'
elseif strcmp(line1.type, 'y') && line1.A == 0 ...
    && strcmp(line2.type, 'y') && line2.A == 0
    error 'Lines are parallel'
end

% Calculate the intersection
if strcmp(line1.type, 'x')
    x = line1.A;
    y = line2.A * x + line2.B;
elseif strcmp(line2.type, 'x')
    x = line2.A;
    y = line1.A * x + line1.B;
else
    x = (line2.B - line1.B)/(line1.A - line2.A);
    y = line1.A * x + line1.B;
end

coords = [x; y];

end

% Preprocess image
function [B, B_nonsmooth] = preprocess(gray_img, image_size)
% Remove edges (there may be some unwanted noise)
pad = round(0.05*min(size(gray_img,1), size(gray_img,2)));
I = gray_img(pad:end-pad, pad:end-pad);

% Resize image
I = imresize(I, image_size);

% Plot the image before smoothing, and flip the axis so plotting is easier
imshow(I); set(gca,'Ydir','Normal'); hold on;

% Smooth
I_smooth = imgaussfilt(I,1.5);

% Convert to binary image, remove small blobs
B = I_smooth<240;
B = bwareaopen(B, 1000);

% Another, nonsmoothed and more sensitive image
B_nonsmooth = I<252;
B_nonsmooth = bwareaopen(B_nonsmooth, 1000);
end

% Find positive x- and y-axes from the binary image B
function [pos_x, pos_y] = find_axes(B)

% Create Hough transform
[H,T,R] = hough(B);

% Find peaks from the transform
P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));

% Find lines, measure their length and position
lines = houghlines(B,T,R,P,'FillGap',50,'MinLength',400);

lengths = zeros(1, length(lines));
masspoints = zeros(2, length(lines));
for k = 1:length(lines)
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
    
    % Calculate length
    len = norm(lines(k).point1 - lines(k).point2);
    lengths(k) = len;
   
    % Also calculate the "masspoint" so we can determinte which will be
    % positive x-axis (and positive y-axis)
    masspoints(:,k) = mean([lines(k).point1; lines(k).point2])';
end

% Now the crucial point; we assume that the line with smallest y-coord
% value is positive x-axis (unless its angle with image's positive x-axis
% is >45 degrees)
x_axis_found = false;
[~, sorted_idxs] = sort(masspoints(2,:));
for idx = 1:numel(sorted_idxs)
    pos_x = lines(sorted_idxs(idx));
    vec = get_vector(pos_x);
    alpha = rad2deg(asin(vec(2)/vec(1)));
    if abs(alpha) < 45
        x_axis_found = true;
        break
    end
end

if ~x_axis_found
    error('Couldn''t find positive x-axis!');
end

% Transform the line into a vector form
vec1 = get_vector(pos_x);

% Another crucial assumption; positive y-axis is the line with smallest
% x-coord (and its angle with positive x-axis should be near 90 degrees)
y_axis_found = false;
[~, sorted_idxs] = sort(masspoints(1,:));
for idx = 1:numel(sorted_idxs)
    pos_y = lines(sorted_idxs(idx));
    vec2 = get_vector(pos_y);

    % Normalise vectors
    vec1n = vec1./sqrt(sum(vec1.^2));
    vec2n = vec2./sqrt(sum(vec2.^2));
    
    % Calculate dot product (zero when right angle)
    p = vec1n'*vec2n;
    
    if p < 0.1
        y_axis_found = true;
        break;
    end
end

if ~y_axis_found
    error('Couldn''t find positive y-axis!');
end

% Plot the selected lines
xy = [pos_x.point1; pos_x.point2];
plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
xy = [pos_y.point1; pos_y.point2];
plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

end

% Get equation from a line
function eq = get_line_equation(line)
eq = struct();
if line.point1(1) == line.point2(1)
    eq.type = 'x';
    eq.A = line.point1(1);
else
    eq = struct();
    eq.type = 'y';
    eq.A = (line.point2(2) - line.point1(2))/(line.point2(1) - line.point1(1));
    eq.B = -(line.point2(2) - line.point1(2))*line.point1(1)/(line.point2(1) - line.point1(1)) + line.point1(2);
end
end

% Get vector from a line
function vec = get_vector(line)
% Calculate the vector
vec = [line.point2(1)-line.point1(1); line.point2(2)-line.point1(2)];
if sign(max(vec)) < 0
    vec = vec.*-1;
end
end