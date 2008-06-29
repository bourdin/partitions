function[U,Phi] = PlotPartitionSimplifiedPer()


%Phi_filename=strcat('Partition_Phi-000-',num2str(iter,'%.5d'),'.txt');
Phi_filename=strcat('Partition_Phi_all.txt');
Phi_all=load(Phi_filename);
U_filename=strcat('Partition_U_all.txt');
U_all=load(U_filename);
[nx,ny]=size(Phi_all);


linecolor          = 0.55*ones(1,3);
%----------------------------------------------------------
% les U
try
close(1);
end
pfigure(1);
hold off;
n                  = 3*nx;
x                  = linspace(-1,2,n);
y                  = linspace(-1,2,n);
[X,Y]              = meshgrid(x,y);
hold on;
surf(X,Y,[[U_all,U_all,U_all];[U_all,U_all,U_all];[U_all,U_all,U_all]]/max(abs(U_all(:))));
%line([0  0],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
%line([1  1],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
%line([-1 2],       [0  0],     [1 1],  'LineWidth', 3, 'Color', linecolor);
%line([-1 2],       [1  1],     [1 1],  'LineWidth', 3, 'Color', linecolor);
%title(strcat('U: iteration ',num2str(iter)));
title('U');
axis equal;axis off
shading interp;
colorbar;

 
%----------------------------------------------------------
% Les Phi
try
close(2);
end
pfigure(2)

surf(X,Y,[[Phi_all, Phi_all, Phi_all]; [Phi_all, Phi_all, Phi_all] ; [Phi_all, Phi_all, Phi_all]]/max(abs(Phi_all(:))))
view(2)
hold on
line([0  0],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([1  1],       [-1 2],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([-1 2],       [0  0],     [1 1],  'LineWidth', 3, 'Color', linecolor);
line([-1 2],       [1  1],     [1 1],  'LineWidth', 3, 'Color', linecolor);
%title(strcat('Phi: iteration ',num2str(iter)));
title('Phi');
axis equal;axis off
shading interp;
colorbar;
daspect([ 1 1  1]);


