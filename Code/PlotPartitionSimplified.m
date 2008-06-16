function[U,Phi] = PlotPartitionSimplified()


%Phi_filename=strcat('Partition_Phi-000-',num2str(iter,'%.5d'),'.txt');
Phi_filename=strcat('Partition_Phi_all.txt');
Phi_all=load(Phi_filename);
U_filename=strcat('Partition_Phi_all.txt');
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
x                  = linspace(0,1,nx);
y                  = linspace(0,1,nx);
[X,Y]              = meshgrid(x,y);
hold on;
surf(X,Y,U_all/max(abs(U_all(:))));
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

surf(X,Y,Phi_all/max(abs(Phi_all(:))))
view(2)
hold on
%title(strcat('Phi: iteration ',num2str(iter)));
title('Phi');
axis equal;axis off
shading interp;
colorbar;
daspect([ 1 1  1]);

