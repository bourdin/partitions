function[U,Phi] = PlotPartition(k,iter)


Phi_filename=strcat('Partition_Phi-000-',num2str(iter,'%.5d'),'.txt');
tmp=load(Phi_filename);
[nx,ny]=size(tmp);

U=zeros(nx,ny,k);
Phi=zeros(nx,ny,k);
for i=1:k,
    U_filename=strcat('Partition_U-', num2str(i-1, '%.3d'), '-', num2str(iter,'%.5d'),'.txt');
    Phi_filename=strcat('Partition_Phi-', num2str(i-1, '%.3d'), '-', num2str(iter,'%.5d'),'.txt');
    U(:,:,i) = load(U_filename);
    Phi(:,:,i) = load(Phi_filename);
end

U_all = zeros(nx,ny);
Phi_all = zeros(nx, ny);
for i=1:k,
    U_all = U_all + U(:,:,i);
    Phi_all = Phi_all + i * Phi(:,:,i);
end

close(1);
pfigure(1);
hold off;
line([1 3*nx],        [ny+1, ny+1],     [1,1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
hold on;
line([1 3*nx],        [2*ny+1, 2*ny+1], [1,1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
line([nx+1 nx+1],     [1, 2*ny],        [1,1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
line([2*nx+1 2*nx+1], [1, 3*ny],        [1,1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
surface([[U_all,U_all,U_all];[U_all,U_all,U_all];[U_all,U_all,U_all]]);
title(strcat('U: iteration ',num2str(iter)));
axis equal;
shading interp;
colorbar;

close(2);
pfigure(2);
hold off;
line([1 3*nx],        [ny+1, ny+1],     [k+1,k+1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
hold on;
line([1 3*nx],        [2*ny+1, 2*ny+1], [k+1,k+1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
line([nx+1 nx+1],     [1, 3*ny],        [k+1,k+1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
line([2*nx+1 2*nx+1], [1, 3*ny],        [k+1,k+1],  'LineWidth', 1, 'Color', [.75 .75 .75]);
surface([[Phi_all, Phi_all, Phi_all]; [Phi_all, Phi_all, Phi_all] ; [Phi_all, Phi_all, Phi_all]]);
title(strcat('Phi: iteration ',num2str(iter)));
axis equal;
shading interp;
colorbar;
