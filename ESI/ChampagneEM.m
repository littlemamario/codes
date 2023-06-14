function [X,VarVoxel,gamma] = Champagne(Y,L)
epsilon = 0.0001;
Y = Y/max(max(abs(Y)));
[M,T] = size(Y);
N = length(L(1,:));
Cy = Y*Y.'/T;
%initialization
lamda_init = 0.001*norm(Cy,'fro')/(M^2);%noise covariance
X0 = L.'*pinv(L*L.'+1e-5*eye(M))*Y;
gamma_init = diag(X0*X0.'/T);
%iteration
lamda = lamda_init;
gamma = gamma_init;
Gamma = diag(gamma);
Sigmay = lamda*eye(M)+L*Gamma*L.';
Sigmayinv = pinv(Sigmay);
% modelevidence = trace(Cy*Sigmayinv);
modelevidence = log(abs(det(Sigmay))+10e-8) + trace(Cy*Sigmayinv);%model evidence by initialization
modelevidence_old = -1000;
NumChampIter = 0;
while (abs(modelevidence-modelevidence_old)/abs(modelevidence_old) >= epsilon) && (NumChampIter <= 500)
    NumChampIter = NumChampIter + 1;
    modelevidence_old = modelevidence;
%     gamma_old = gamma;
    X = Gamma*L.'*Sigmayinv*Y;%mean matrix
    Sigmax = Gamma-Gamma*L.'*pinv(lamda*eye(M)+L*Gamma*L.')*L*Gamma;
    for n = 1:N
        zn = L(:,n).'*Sigmayinv*L(:,n);
        gamma(n) = sqrt(X(n,:)*X(n,:).'/(T*zn));%update variances of sources
    end
    lamda = sum(diag((Y-L*X).'*(Y-L*X)))/(T*M)+trace(L.'*L*Sigmax)/M;%update noise variance
    %Calculate model evidence
    Gamma = diag(gamma);
    Sigmay = lamda*eye(M)+L*Gamma*L.';
    Sigmayinv = pinv(Sigmay);
%     modelevidence = trace(Cy*Sigmayinv);
    modelevidence = log(abs(det(Sigmay))+10e-8) + trace(Cy*Sigmayinv);
end
%Final source estimation
X = Gamma*L.'*Sigmayinv*Y;
VarVoxel = diag(X*X.'/T);
VarVoxel = VarVoxel/max(VarVoxel);%Variance of source per voxel