function []=MakeCompositeTXT(level, numprocs)
% Generate composite file out of a individual U and Phi

for i=1:numprocs,
    U_filename   = strcat('Partition_U-',   num2str(i-1, '%.3d'),'-level',num2str(level, '%1d'), '.txt');
    Phi_filename = strcat('Partition_Phi-', num2str(i-1, '%.3d'),'-level',num2str(level, '%1d'), '.txt');
    U   = load(U_filename);
    Phi = load(Phi_filename);
    if i==1,
        psi = Phi;
        Uall = U;
    else
        psi = psi + Phi * (1+1);
        Uall = Uall + U;
    end
end

U_filename   = strcat('Partition_U_all-level',num2str(level, '%1d'), '.txt');
Phi_filename = strcat('Partition_Phi_all-level',num2str(level, '%1d'), '.txt');

save(U_filename, 'Uall', '-ASCII');
save(Phi_filename, 'psi', '-ASCII');
