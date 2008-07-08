function [valt,x_0s,std0] = fdm_assymp_eigenvalues(numeig,nbvertex,mu,nphi,periodic,dimension,reload,nstep,numex)

%
%
%
%  fdm_assymp_eigenvalues(numeig,nbvertex,mu,nphi,periodic,dimension,reload,nstep)
%  fdm_assymp_eigenvalues(1,5e1,1e3,8,0,2,0,1) sans reload
%  fdm_assymp_eigenvalues(1,6e1,1e3,8,0,2,1,1) avec reload
%
%

global cost_hist phis_hist comptco save_phihist


clear cost_hist phis_hist comptco save_phihist

%====================================
% Variables par defaut et generateur aleatoire
%rand('seed',2);
save_phihist             = 0;
%comptco                  = 0;

% Parametres globaux
if(exist('nbvertex')==0)
 numeig                 = 1;
end

if(exist('numex')==0)
 numex                  = 1;
end

if(exist('nbvertex')==0)
 nbvertex               = 50;
end

if(exist('nphi')==0)
 nphi                   = 8;                % Nombre de cellules de la partition
end

if(exist('dimension')==0)
 dimension              = 2;                % Nombre de cellules de la partition
end

if(exist('mu')==0)
 mu                     = 1e3;
end

if(exist('reload')==0)
 reload                 = 0;
end

if(dimension==2)
 dimm                   = '';
else
 dimm                   = '3D';
end

if(periodic)
 strperiodic            = '';
else
 strperiodic            = 'notperiodic';
end

rand('seed',numex);
disp(' ')
disp('*** Attention : le nb de points de la grille doit etre divisible par nphi... ***')
nbvertex                             = round(nbvertex/nphi)*nphi;

%====================================
% Grille et matrice du laplacien
adresse1                 = pwd;
cd /home/oudet/Matlab/Toolbox/Fdm/;
disp(' ')
disp('Debut d assemblage de la matrice du laplacien')

if(dimension==2)
 [M,Xl,Xp,Xm,Yp,Ym,dx,dy]           = laplace2d_periodic_sparse(nbvertex);
elseif(dimension==3)
 disp('!!! Travail en dimension 3 !!!')
 [M,Xl,Xp,Xm,Yp,Ym,Zp,Zm,dx,dy,dz]  = laplace3d_periodic_sparse(nbvertex);
end

[M,g]                                = build_and_test_laplacian_matrices(nbvertex,0,periodic); % Attention, par periodicite le nb de pt de la grille n est pas nbvertex^2
Xg                       = g.xs{1};
Yg                       = g.xs{2};
Xl                       = [Xg(:),Yg(:)];
eval(strcat(' cd  /','  ',adresse1));
disp('Fin d assemblage de la matrice du laplacien')
disp(' ')

n                        = nphi*size(Xl,1);   % Nombre de parametres
npt                      = size(Xl,1);



%====================================
% Recherche d'un point damissible
if(exist('Mx_0')==0)
 % Reloading eventuel
 if(reload)
   disp(' ')
   disp('*** Reloading ***')
   disp(' ')
   nbvertexold          = 100;
   nbvertexold          = round(nbvertexold/nphi)*nphi;
   muold                = 1e3;
   Mx_0                 = reloadproject(numeig,strperiodic,nbvertexold,Xl,nphi,muold,dimension,numex);
 else
   Mx_0                 = 2*rand(npt,nphi)-1;
 end
 gradphis               = zeros(npt,nphi);
end

nnn                      = n;
phit                     = zeros(1,n/nphi); % Utilise pour la variance cible

ccol                     = ones(size(Mx_0,1),1);
cligne                   = (npt/nphi)*ones(1,size(Mx_0,2));
Mx_0                     = project_affine_direct(Mx_0,cligne,ccol);


%====================================
% Definition de la structure d optimisation
Prob.INFOS_PERSO.pexp    = 1; % exposant etant utilise pour approcher le max 
if(Prob.INFOS_PERSO.pexp~=1)
 strperiodic            = strcat(strperiodic,'_pexp_',num2str(Prob.INFOS_PERSO.pexp));
end


Prob.INFOS_PERSO.numeig  = numeig;
Prob.INFOS_PERSO.M       = M;
Prob.INFOS_PERSO.nphi    = nphi;
Prob.INFOS_PERSO.npt     = npt;
Prob.INFOS_PERSO.mu      = mu;
Prob.INFOS_PERSO.dim     = dimension;
Prob.INFOS_PERSO.mu      = Prob.INFOS_PERSO.mu*ones(1,nphi);

Prob.INFOS_PERSO.Xm      = abs(Xm);
Prob.INFOS_PERSO.Ym      = abs(Ym);
Prob.INFOS_PERSO.Xp      = abs(Xp);
Prob.INFOS_PERSO.Yp      = abs(Yp);
Prob.INFOS_PERSO.dx      = dx;
Prob.INFOS_PERSO.dy      = dy;
if(dimension==3)
 Prob.INFOS_PERSO.Zm    = abs(Zm);
 Prob.INFOS_PERSO.Zp    = abs(Zp);
 Prob.INFOS_PERSO.dz    = dz;
end
Prob.INFOS_PERSO.cw      = sqrt(2)/6;
Prob.INFOS_PERSO.mus   = Prob.INFOS_PERSO.mu;

maxitp                   = 1e3;
compt                    = 1;
if(exist('nstep')==0)
 nstep                  = max(round(mu/1e3),1);
end
if(nstep>1)
 muliste                = linspace(1e3,mu,nstep);
else
 muliste                = mu;
end
muliste_save             = muliste;
Prob.INFOS_PERSO.mu      = Prob.INFOS_PERSO.mu(1);

%====================================
% Test de la derivation
x                        = rand(size(Mx_0));
val                      = cost_f(x,Prob);
gradphis                 = cost_g(x,Prob);


% Direction aleatoire
dr                       = 2*rand(size(Mx_0))-1;

% Approximation par differences finies
epst                     = 1e-6;
val1                     = cost_f(x+epst*dr,Prob);
val2                     = cost_f(x-epst*dr,Prob);
diffv                    = (val1-val2)/(2*epst);

disp(' ')
disp('Test du calcul de derivee')
disp(num2str(sum(sum(gradphis.*dr)),'%4.18f'))
disp(num2str(diffv,'%4.18f'))
disp(' ')


%====================================
% Debut de l optimisation


kconv                    = 0;
minstd0                  = 0;
fa0                      = 1e4;
Mx_0s                    = Mx_0;
comptvp                  = 1;
cost_hist                = zeros(1,maxitp);
nbvar                    = length(Mx_0(:));
tic
%while(or(kconv<=nstep,or(abs(max(min(Mx_0,[],1)))>1e-3,abs(min(max(Mx_0,[],1))-1)>1e-3)))
while(kconv<nstep)

 kconv                  = kconv + 1;
 if(kconv==1)
   muliste              = muliste_save;
 end


 if(nstep>1)  
   if(kconv<=nstep)
     Prob.INFOS_PERSO.mu                 = muliste_save(max(kconv,1));%muliste(max(kconv,1));
   else
     %Prob.INFOS_PERSO.mu(:)               = Prob.INFOS_PERSO.mu(:)/2;
   end
   disp(num2str(Prob.INFOS_PERSO.mu(1)))
 end


 factaway               = fa0*(nstep-kconv);

 caway                  = (nphi*Prob.INFOS_PERSO.mu(1))/10;

 ngradv                 = 1e3;
 valt                   = cost_f(Mx_0,Prob);
 valtold                = 1e14;
 alpha                  = 1e-2;
 stdold                 = 0;
 comptee                = 0;
 gradaway               = 0*Mx_0;
 debut                  = 1;

 % Critere d arret adaptatif
 if(kconv>=nstep)
   critstop             = 1e-8;
 else %if(kconv<=nstep/2)
   critstop             = 1e-6;
 end


 while(and(ngradv>critstop,alpha>1e-4))   

   [valt,struct_eig]    = cost_f(Mx_0,Prob);
   gradphis             = cost_g(Mx_0,Prob);

   % Projection des gradients
   %%
   gradphis             = project_affine_direct(gradphis,0*cligne,0*ccol);
   %ngradv               = norm(gradphis(:));
   try
     ngradv             = sum(abs(Mx_0(:)-Mx_0old(:)))/nbvar;
   catch
     ngradv             = 1e3;
   end

   %  Calcul du pas maximal admissible
   Mx_0old              = Mx_0;
   phist                = Mx_0(:) - alpha*gradphis(:);
   Mx_0(:)              = project_simplex_wrong(phist,nphi);

   alpha                = min(1.1*alpha,1e-2);
   valtold              = valt;


   if(compt<=5e1)
     phis_hist{comptvp} = Mx_0(:);
     comptvp            = comptvp + 1;
   end

   eigsphi1t              = struct_eig.eigs{1}; 
   cost_hist(:,compt)     = valt;    
   eigsphi1_hist(:,compt) = eigsphi1t(:);

   disp(['It '  pnum2str(compt,'%d',6) ' | Nconv ' pnum2str(kconv,'%4.4d',4) ' | Step ' pnum2str(alpha,'%4.6f',10) ...
	    ' | Ngradv ' pnum2str(ngradv,'%4.8f',12)...
	    ' | Cout ' pnum2str(valtold,'%4.4f',10) ...
	    ' | mu ' pnum2str(min(Prob.INFOS_PERSO.mu),'%4.4f',8)...
	    ' | maxmin ' pnum2str(max(min(Mx_0,[],1)),'%4.8f',10)...
	    ' | minmax ' pnum2str(min(max(Mx_0,[],1)),'%4.8f',10)])
   % Affichage
   if(or(mod(compt,1e2)==0,compt==1))
     disp(['It '  pnum2str(compt,'%d',6) ' | Nconv ' pnum2str(kconv,'%4.4d',4) ' | Step ' pnum2str(alpha,'%4.6f',10) ...
	    ' | Ngradv ' pnum2str(ngradv,'%4.8f',12)...
	    ' | Cout ' pnum2str(valt,'%4.4f',10) ...
	    ' | mu ' pnum2str(min(Prob.INFOS_PERSO.mu),'%4.4f',8)...
	    ' | maxmin ' pnum2str(max(min(Mx_0,[],1)),'%4.8f',10)...
	    ' | minmax ' pnum2str(min(max(Mx_0,[],1)),'%4.8f',10)])



     % Sauvegarde
     Result.Prob        = Prob;
     Result.x_k         = Mx_0(:);
     phis_hist{comptvp} = Result.x_k;
     comptvp            = comptvp + 1;
     eval(['save Runs/res' strperiodic '_numeig_' num2str(numeig) '_Nbun_' num2str(numex) '_nbptg_' ...
	  num2str(nbvertex) '_ncells_' num2str(nphi) '_mu_' num2str(mu) ...
	  '.mat Result phis_hist  eigsphi1_hist cost_hist;']);

   end

   compt                = compt +1;
   debut                = 0;
 end


 disp(['It '  pnum2str(compt,'%d',6) ' | Nconv ' pnum2str(kconv,'%4.4d',4) ' | Step ' pnum2str(alpha,'%4.6f',10) ...
	    ' | Ngradv ' pnum2str(ngradv,'%4.8f',12)...
	    ' | Cout ' pnum2str(valt,'%4.4f',10) ...
	    ' | mu ' pnum2str(min(Prob.INFOS_PERSO.mu),'%4.4f',8)...
	    ' | maxmin ' pnum2str(max(min(Mx_0,[],1)),'%4.8f',10)...
	    ' | minmax ' pnum2str(min(max(Mx_0,[],1)),'%4.8f',10)])



 % Fin de boucle kconv
 compt                  = compt +1;


end


% Sauvegarde
Result.Prob         = Prob;
Result.x_k          = Mx_0(:);

disp(['save Runs/res' strperiodic '_numeig_' num2str(numeig) '_Nbun_' num2str(numex) '_nbptg_' ...
	  num2str(nbvertex) '_ncells_' num2str(nphi) '_mu_' num2str(mu) ...
	  '.mat Result phis_hist  eigsphi1_hist cost_hist;']);

eval(['save Runs/res' strperiodic '_numeig_' num2str(numeig) '_Nbun_' num2str(numex) '_nbptg_' ...
	  num2str(nbvertex) '_ncells_' num2str(nphi) '_mu_' num2str(mu) ...
	  '.mat Result phis_hist  eigsphi1_hist cost_hist;']);




toc

%====================================
function phisp = project_simplex(phis,nphi)



dim                      = nphi;
n                        = length(phis)/dim;
X                        = reshape(phis,n,dim);
I                        = zeros(n,dim); % Vaut 1 si X(I) = 0 et 0 sinon

for k = 1:dim
 %%%%%
 % On projette X sur VI courant
 X                      = X.*(1-I);  % xij = 0 si Iij = 1
 sumXI                  = sum(X,2);  % Attention ordre important
 nI                     = max(dim - sum(I,2),1e-6);
 %nI                     = dim - sum(I,2);
 %X(nI>0,:)              = X(nI>0,:) - (1-I(nI>0,:)).*(((sumXI(nI>0)-1)./nI(nI>0))*ones(1,dim));
 X                      = X         - (1-I).*(((sumXI-1)./nI)*ones(1,dim));
 % Seconde etape
 I                      = min(I + (X<0),1);
 X                      = X - I.*(X<0).*X;
end


phisp                    = X(:);




%====================================
function phisp = project_simplex_wrong(phis,nphi)



dim                      = nphi;
n                        = length(phis)/dim;
X                        = reshape(phis,n,dim);
X                        = max(X,0);
sX                       = sum(X,2);
I                        = find(sX>1);
X(I,:)                   = X(I,:)./(sX(I)*ones(1,dim));
phisp                    = zeros(size(phis));
phisp(:)                 = X(:);


%====================================
function Mx_0 = project_affine_direct(Mx_0,cligne,ccol)

if(size(ccol,1)==1)
 ccol                   = ccol';
end
if(size(cligne,2)==1)
 cligne                 = cligne';
end

npt                      = size(Mx_0,1);
nphi                     = size(Mx_0,2);


fi                       = 2*(sum(Mx_0,2)-ccol);
ej                       = 2*(sum(Mx_0,1)-cligne);


% On calcul sligne par la resolution d un petit systeme
A                        = npt*eye(nphi) - npt*ones(nphi)/nphi;
B                        = ej - sum(fi)/nphi;

% On enleve une des contraintes de volume inutile
lambdaj                  = [(A(1:(end-1),1:(end-1))\B(1:end-1)')',0];


sligne                   = sum(lambdaj);
lambdai                  = (fi - sligne)/nphi;
Mx_0                     = Mx_0 - (lambdai*ones(1,nphi) + ones(npt,1)*lambdaj)/2;





%====================================
function Mx_0n = reloadproject(numeig,strperiodic,nbvertexold,Xl,nphi,mu,dimension,numex)


% Reloading et projection
eval(['load Runs/res' strperiodic '_numeig_' num2str(numeig) '_Nbun_' num2str(numex) '_nbptg_' ...
     num2str(nbvertexold) '_ncells_' num2str(nphi) '_mu_' num2str(mu) ...
     '.mat;']);

phis                     = Result.x_k;
nt                       = length(phis)/nphi;
Mx_0                     = reshape(phis,nt,nphi);
n                        = 3*sqrt(nt);

x                        = linspace(-2,1,n);
y                        = linspace(-2,1,n);
z                        = linspace(-2,1,n);
if(dimension==2)
 [Xo,Yo]                = meshgrid(x,y);
elseif(dimension==3)
 [Xo,Yo,Zo]             = meshgrid(x,y,z);
end

Mx_0n                    = zeros(size(Xl,1),nphi);
for k = 1:nphi

 %matrixt                = smooth(matrixt,30);
 if(dimension==2)
   F                    = reshape(Mx_0(:,k),n/3,n/3)';
   Mx_0n(:,k)           = interp2(Xo,Yo,[[F, F, F]; [F, F, F] ; [F, F, F]],Xl(:,1),Xl(:,2)); 
 elseif(dimension==3)
   Mx_0n(:,k)           = interp3(Xo,Yo,Zo,matrixt,Xl(:,1),Xl(:,2),Xl(:,3));
 end

end

function [val,struct_eig] = cost_f(phis,Prob)

global gradphi  phiok  cost_hist phis_hist comptco save_phihist


%phis                                            = project_simplex_wrong(phis,Prob.INFOS_PERSO.nphi);

gradphir                                        = zeros(size(phis));
phis                                            = phis(:);
if(size(phis,1)==1)
 phis                                          = phis';
end

if(isfield(Prob.INFOS_PERSO,'pexp'))   % exposant etant utilise pour approcher le max 
 pexp                                          = Prob.INFOS_PERSO.pexp;
else
 pexp                                          = 1;
end

if(isfield(Prob.INFOS_PERSO,'numcomp'))
 numcomp                                       = Prob.INFOS_PERSO.numcomp;
else
 numcomp                                       = Prob.INFOS_PERSO.numeig;
end

numeig                                          = Prob.INFOS_PERSO.numeig;
nphi                                            = Prob.INFOS_PERSO.nphi;
val                                             = 0;
gradphi                                         = zeros(size(phis));
compt                                           = 1;
M                                               = Prob.INFOS_PERSO.M;
options.tol                                     = 1e-18;
options.maxit                                   = 1e8;
options.disp                                    = 0;
options.issym                                   = 1;
options.isreal                                  = 1;
phiok                                           = phis;


for k = 1:nphi
 compte                                        = (compt + length(phis)/nphi-1);
 Mk                                            = - M;

 for j = 1:size(M,1)
   Mk(j,j)                                     = Mk(j,j) + Prob.INFOS_PERSO.mu*(1-phis(compt-1+j));
 end

 [V,D]                                         = eigs(Mk,[],numcomp,'sm',options);

 %format long;diag(D)
 %optt.DISP                                     = 0; 
 %optt.MAXIT                                    = 1e2;
 %optt.INNERIT
 %[D1,V]                                         = eigifp(Mk,numcomp,optt); 


 [D,Is]                                        = sort(diag(D));
 V                                             = V(:,Is);
 if(nargout>=2)
   struct_eig.vectp{k}                         = zeros(length(phis)/nphi,numcomp);
   struct_eig.eigs{k}                          = D(:);
   for t = 1:numcomp
     vp                                        = V(:,t);
     struct_eig.vectp{k}(:,t)                  = -(vp.^2)/(sum(vp.^2))*(pexp*D(numeig)^(pexp-1));
   end
   gradphi(compt:compte,1)                     = struct_eig.vectp{k}(:,numeig);
 else
   vp                                          = V(:,numeig);
   gradphi(compt:compte,1)                     = -(vp.^2)/(sum(vp.^2))*(pexp*D(numeig)^(pexp-1));
 end

 compt                                         = compte + 1;
 val                                           = val + D(numeig)^pexp;
end


if(length(comptco)>0)
 comptco                                       = comptco + 1;
 if(or(comptco<51,mod(comptco,1)==0))
   cost_hist(comptco)                          = val;
   if(save_phihist)
     phis_hist{comptco}                        = phis(:);
   end
 end
end

gradphir(:)                                     = gradphi(:);
gradphi                                         = gradphir;
gradphi                                         = Prob.INFOS_PERSO.mu(1)*gradphi;

%====================================
function phisp = project_simplex_wrong(phis,nphi)



dim                      = nphi;
n                        = length(phis)/dim;
X                        = reshape(phis,n,dim);
X                        = max(X,0);
sX                       = sum(X,2);
I                        = find(sX>1);
X(I,:)                   = X(I,:)./(sX(I)*ones(1,dim));
phisp                    = zeros(size(phis));
phisp(:)                 = X(:);

%====================================
function phisp = project_simplex(phis,nphi)



dim                      = nphi;
n                        = length(phis)/dim;
X                        = reshape(phis,n,dim);
I                        = zeros(n,dim); % Vaut 1 si X(I) = 0 et 0 sinon

for k = 1:dim
 %%%%%
 % On projette X sur VI courant
 X                      = X.*(1-I);  % xij = 0 si Iij = 1
 sumXI                  = sum(X,2);  % Attention ordre important
 nI                     = max(dim - sum(I,2),1e-6);
 %nI                     = dim - sum(I,2);
 %X(nI>0,:)              = X(nI>0,:) - (1-I(nI>0,:)).*(((sumXI(nI>0)-1)./nI(nI>0))*ones(1,dim));
 X                      = X         - (1-I).*(((sumXI-1)./nI)*ones(1,dim));
 % Seconde etape
 I                      = min(I + (X<0),1);
 X                      = X - I.*(X<0).*X;
end

phisp                    = X(:);

function gradv = cost_g(phis,Prob)


global gradphi  phiok


if(size(phis,1)==1)
 phis                                          = phis';
end

if(exist('phiok')==0)
 phiok                                         = zeros(size(phis));
end
if(isempty(phiok))
 phiok                                         = zeros(size(phis));
end



if(sum(abs(phiok(:)-phis(:)))<1e-8)
 gradv                                         = gradphi;
 return
else
 val                                           = cost_f(phis,Prob);
 gradv                                         = gradphi;
 return 
end