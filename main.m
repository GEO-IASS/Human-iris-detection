function main
clear all;close all;clc;

%% Getting image
img = imread('10.jpg');

% Setting parameters
angW = 45;  % Predifined viewing field

%% Detecting human face
faceDetector = vision.CascadeObjectDetector;
bboxes = step(faceDetector, img);
IFaces = insertObjectAnnotation(img, 'rectangle', bboxes, 'Face');
figure, imshow(IFaces), title('Detected faces');

%% Extracting human eyes
figure(2);imshow(img);hold on
[m,n,s] = size(img);    % Size of image
cn = 0;                 % The sequence number of eyes
for i=1:size(bboxes,1)
    % The current box for human face
    bb = bboxes(i,:);
    % Generating mask
    bw = false(m,n);
    bw(bb(2):bb(2)+bb(4), bb(1):bb(1)+bb(3)) = true;
    % Enlarge some parts of image
    se = strel('square',10);
    bw = imdilate(bw,se);
    % Extracting the current human face
    img2 = img;    
    if size(img,3) == 3
        img2 = rgb2gray(img);
    end
    img2(~bw) = 0;
    % Extracting the coordinates of human face characteristics
    data = detectEye(img2);
    % Displaying labels
    if(~isempty(data))
        % Refreshing sequence number
        cn = cn+1;
        
        % Left eyes
        eyeL = int32(data(37:42,:));
        wE = max(eyeL(:,1)) - min(eyeL(:,1));  % The width of eyes
        plot(mean(eyeL(:,1)),mean(eyeL(:,2)),'g+');
        figure(2);plot(eyeL(:,1),eyeL(:,2),'g*');
        % Preserve the central position of eyes
        xL(cn) = mean(eyeL(:,1));
        yL(cn) = mean(eyeL(:,2));
        % Generating eyes mask
        bw = false(m,n);
        ind = sub2ind([m,n],eyeL(:,2),eyeL(:,1));
        bw(ind) = true;
        % Extracting eye balls
        [r,c] = extractEyeball(bw,img2);
        xL2(cn) = c;
        yL2(cn) = r;        
        zL2(cn) = double(( xL2(cn)-xL(cn) ) * 180/wE);
        
        
        % Right eyes
        eyeR = int32(data(43:48,:));
        wE = max(eyeR(:,1)) - min(eyeR(:,1));   % The width of eyes
        figure(2);plot(eyeR(:,1),eyeR(:,2),'r*');
        plot(mean(eyeR(:,1)),mean(eyeR(:,2)),'r+');
        % Preserve the central position
        xR(cn) = mean(eyeR(:,1));
        yR(cn) = mean(eyeR(:,2));
        % Generating mask
        bw = false(m,n);
        ind = sub2ind([m,n],eyeR(:,2),eyeR(:,1));
        bw(ind) = true;
        % Extracting eye balls
        [r,c] = extractEyeball(bw,img2);
        xR2(cn) = c;
        yR2(cn) = r;        
        zR2(cn) = double(( xR2(cn)-xR(cn) ) * 180/wE);
       
      

%% Creating the central position in image
figure;imshow(img);
hold on
for i=1:cn
    % Extracting the coordinate of left eye
    x = xL2(i);
    y = yL2(i);
    z = zL2(i);
    plot(x,y,'g+');
    plot(mean(eyeL(:,1)),mean(eyeL(:,2)),'b+');
    text(x-100,y-100,sprintf('L(%d)(%0.0f, %0.0f, %0.0f)',i,x,y,z),'BackgroundColor',[.7 .9 .7]);
    legend('yellow and blue mean the central position of eyes, red and green mean the position of iris');
    
    % Extracting the coordiante of right eye
    x = xR2(i);
    y = yR2(i);
    z = zR2(i);
    plot(x,y,'r+');
    plot(mean(eyeR(:,1)),mean(eyeR(:,2)),'y+');
    text(x-100,y-100,sprintf('R(%d)(%0.0f, %0.0f, %0.0f)',i,x,y,z),'BackgroundColor',[.7 .9 .7]);
end
hold off;
drawnow;
    end
end
%% Drawing 3D coordinate
figure;
hold on
for i=1:cn
    % Extracting the left eye coordinate
    x = xL2(i);
    y = yL2(i);
    z = zL2(i);
    plot3(x,y,z,'g+');
    text(x-5,y-5,z,sprintf('L(%d)(%0.0f, %0.0f, %0.0f)',i,x,y,z),'BackgroundColor',[.7 .9 .7]);
    
    % Extracting the right eye coordinate
    x = xR2(i);
    y = yR2(i);
    z = zR2(i);
    plot3(x,y,z,'r+');
    text(x-5,y-5,z,sprintf('R(%d)(%0.0f, %0.0f, %0.0f)',i,x,y,z),'BackgroundColor',[.7 .9 .7]);
end
hold off;
grid on
drawnow;


end

%% Extracting the central position of eyes
function [r,c] = extractEyeball(bw,img2)

r = [];
c = [];

bw = bwconvhull(bw);
[rows,cols] = find(bw);
r1 = min(rows);
bw(1:r1+5,:) = false;
img3 = img2;
img3(~bw) = 255;
bw = img3<20;

[label,num] = bwlabel(bw,8);
sts = regionprops(label,'Area','Centroid');
[y,idx] = sort(cat(1,sts.Area),'descend');
xy = sts(idx).Centroid;

r = xy(2);
c = xy(1);

figure;imshow(bw);drawnow;

end

%% detecting human eyes
function [data] = detectEye(img)

data = [];

% Loading ASM model
addpath('.\model');
S = load('model_param.mat');
Model = S.Model;
pc_version = computer();
if(strcmp(pc_version,'PCWIN')) % currently the code just supports windows OS
    addpath('.\face_detect_64');
    addpath('.\mex_64');
elseif(strcmp(pc_version, 'PCWIN64'))
    addpath('.\face_detect_64');
    addpath('.\mex_64');
end
Model.frontalL = @(X) Select(X, Model.frontal_landmark);
Model.leftL = @(X) Select(X, Model.left_landmark);
Model.rightL = @(X) Select(X, Model.right_landmark);

% Detecting the labels of human face
[shape, pglobal, visible] = faceAlign(img, Model, [], 1);
if(~isempty(shape))
    % Changing into xy coordinates
    data = reshape(shape,Model.nPts,2);
end

end















