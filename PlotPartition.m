function[U,Phi] = PlotPartition(k)


%Phi_filename=strcat('Partition_Phi-000-',num2str(iter,'%.5d'),'.txt');
Phi_filename=strcat('Partition_Phi-000.txt');
tmp=load(Phi_filename);
[nx,ny]=size(tmp);

U=zeros(nx,ny,k);
Phi=zeros(nx,ny,k);
for i=1:k,
%    U_filename=strcat('Partition_U-', num2str(i-1, '%.3d'), '-', num2str(iter,'%.5d'),'.txt');
%    Phi_filename=strcat('Partition_Phi-', num2str(i-1, '%.3d'), '-', num2str(iter,'%.5d'),'.txt');
    U_filename=strcat('Partition_U-', num2str(i-1,'%.3d'),'.txt');
    Phi_filename=strcat('Partition_Phi-', num2str(i-1,'%.3d'),'.txt');
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

