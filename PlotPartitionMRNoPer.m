function[U,Phi] = PlotPartitionMRNoPer(k,numrun)


Phi_filename=strcat('Partition_Phi-000-000.txt');
tmp=load(Phi_filename);
[nx,ny]=size(tmp);

U=zeros(nx,ny,k);
Phi=zeros(nx,ny,k);
for i=1:k,
    U_filename=strcat('Partition_U-', num2str(i-1, '%.3d-'), num2str(numrun,'%.3d'),'.txt');
    Phi_filename=strcat('Partition_Phi-', num2str(i-1, '%.3d-'), num2str(numrun,'%.3d'), '.txt');
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
n                  = 3*nx;
x                  = linspace(0,1,n);
y                  = linspace(0,1,n);
[X,Y]              = meshgrid(x,y);
hold on;
%surf(U_all/max(abs(U_all(:))));
imagesc(U_all);
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

%surf(Phi_all/max(abs(Phi_all(:))))
imagesc(Phi_all);
view(2)
hold on
title('Phi');
axis equal;axis off
shading interp;
colorbar;
daspect([ 1 1  1]);

