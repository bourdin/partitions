function[U,Phi] = PlotPartition(k)


Phi_filename=strcat('Partition_Phi_Best-000.txt');
tmp=load(Phi_filename);
[nx,ny]=size(tmp);

U=zeros(nx,ny,k);
Phi=zeros(nx,ny,k);
for i=1:k,
    U_filename=strcat('Partition_U_Best-', num2str(i-1, '%.3d'),'.txt');
    Phi_filename=strcat('Partition_Phi_Best-', num2str(i-1, '%.3d'),'.txt');
    U(:,:,i) = load(U_filename);
    Phi(:,:,i) = load(Phi_filename);
end

U_all = zeros(nx,ny);
Phi_all = zeros(nx, ny);
for i=1:k,
    U_all = U_all + U(:,:,i);
    Phi_all = Phi_all + i * Phi(:,:,i);
end

linecolor          = 0.55*ones(1,3);
%----------------------------------------------------------
% les U
try
close(1);
end
figure(1);
hold off;
n                  = 3*nx;
x                  = linspace(-1,2,n);
y                  = linspace(-1,2,n);
[X,Y]              = meshgrid(x,y);
hold on;
surf(X,Y,[[U_all,U_all,U_all];[U_all,U_all,U_all];[U_all,U_all,U_all]]/max(abs(U_all(:))));
line([0  0],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([1  1],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([-1 2],       [0  0],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([-1 2],       [1  1],     [1 1],  'LineWidth', 3, 'Color', linecolor);
title('U');
axis equal;axis off
shading interp;
colorbar;

 
%----------------------------------------------------------
% Les Phi
try
close(2);
end
figure(2)

surf(X,Y,[[Phi_all, Phi_all, Phi_all]; [Phi_all, Phi_all, Phi_all] ; [Phi_all, Phi_all, Phi_all]]/max(abs(Phi_all(:))))
view(2)
hold on
line([0  0],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([1  1],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([-1 2],       [0  0],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([-1 2],       [1  1],     [1 1],  'LineWidth', 3, 'Color', linecolor);
title('Phi');
axis equal;axis off
shading interp;
colorbar;
daspect([ 1 1  1]);

