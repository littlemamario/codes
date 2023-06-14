clc
clear all
close all
%-----------读取导程场以及网格文件：文件需要放在当前目录下的ParamMat文件夹中----------%
%Gain_constrained：导程场矩阵，维度为 电极通道数×顶点数
%Faces：矩阵形式，每行表示一个三角网格的三个顶点的（在导程场矩阵中对应列的）索引
%GridLoc：矩阵形式，表示为导程场矩阵每列对应顶点的坐标值，维度为 顶点数×3
%Region与Vertices1：均为cell形式，两者的各单元依次一一对应，分别表示 脑区标记（缩写的字符串形式） 和 该脑区所包含的所有顶点的（在导程场矩阵中对应列的）索引集合（行向量形式）
ParamMat_name_data = dir([pwd,'\ParamMat\*.mat']);
ParamMat_dir_data = [pwd,'\ParamMat\'];
for kk = 1:size(ParamMat_name_data,1)
    ParamMatName = ParamMat_name_data(kk,1).name;
    filename = [ParamMat_dir_data ParamMatName];
    load(filename);
end
%运动皮层上的顶点索引值
IdxMotVox = [217,234,247,248,256,258,263,266,267,273,275,281,282,283,289,293,302,306,312,318,323,326,328,334,335,336,341,342,353,359,360,361,370,377,378,385,392,404,411,412,415,420,421,426,437,438,455,458,465,471,476,480,486,489,518,528,551,926,956,960,961,962,983,984,994,997,1003,1006,1007,1018,1023,1033,1048,1049,1051,1055,1056,1082,1085,1093,1095,1098,1105,1120,1128,1133,1134,1167,1172,1173,1174,1177,1192,1197,1198,1208,1226,1232,1235,1245,1258]; 
NumVertMot = length(IdxMotVox);%运动皮层上的顶点个数
%%%%--------------产生仿真数据Y：M×T维----------------------------------
SNR = 0;%信噪比
T = 500;%所用数据中的采样点个数
[M,N] = size(Gain_constrained);%电极数为M，皮层上的voxel顶点数为N
LocMot = [217,234,247,289,353,486,926,1098];
NumLocMot = length(LocMot);
LocOut = [23,55,89,116,160,668,690,772,863,1358,1400,1422];
NumLocOut = length(LocOut);
% LocGroundTruth = LocMot;%预设存在源的位置
% LocGroundTruth = LocOut;%预设存在源的位置
LocGroundTruth = [LocOut LocMot];%预设存在源的位置
NumSocLoc = length(LocGroundTruth);%预设存在源的个数
Ynf = Gain_constrained(:,LocGroundTruth(:))*randn(NumSocLoc,T);%仿真无噪信号
Y = Ynf + (10^(-SNR/20))*randn(M,T);%加噪得最终的仿真信号
%%%%%%%%%%%%%%    Y 可以被真实信号替换掉！   %%%%%%%%%%%%%%%%%%%%%%%%%%%
procflag = 1;
%%%   procflag = 1:对数据和导程场矩阵不做任何处理，直接做ESI
%%%   procflag = 2:先对数据和导程场矩阵做正交投影，然后做ESI：只在运动皮层上成像
%%%   procflag = 3:先对数据和导程场矩阵做斜投影，然后做ESI：只在运动皮层上成像
if procflag == 1
    L = Gain_constrained;
    [X,VarVoxel,gamma] = Champagne(Y,L);
elseif procflag == 2
    L = Gain_constrained(:,IdxMotVox(:));
    G = Gain_constrained;
    G(:,IdxMotVox(:)) = [];
    [U,S,V] = svd(G,'econ');
    Kg = floor((2/3)*M);
    Ug = U(:,1:Kg);
    Pjg = eye(M)-Ug*pinv(Ug.'*Ug)*Ug.';
    Ypj = Pjg*Y;
    Lpj = Pjg*L;
    [Xmot,VarVoxelMot,gammaMot] = Champagne(Ypj,Lpj);
    X = zeros(N,T);
    X(IdxMotVox(:),:) = Xmot;
    VarVoxel = zeros(N,1);
    VarVoxel(IdxMotVox(:)) = VarVoxelMot;
    gamma = zeros(N,1);
    gamma(IdxMotVox(:)) = gammaMot;
elseif procflag == 3
    L = Gain_constrained(:,IdxMotVox(:));
    G = Gain_constrained;
    G(:,IdxMotVox(:)) = [];
    [U,S,V] = svd(G,'econ');
    Kg = floor((2/3)*M);
    Ug = U(:,1:Kg);
    [U,S,V] = svd(L,'econ');
    Kl = M - Kg;
    Ul = U(:,1:Kl);
    Pjg = eye(M)-Ug*pinv(Ug.'*Ug)*Ug.';
    E = Ul*pinv(Ul.'*Pjg*Ul)*Ul.'*Pjg;
    Ye = E*Y;
    Le = E*L;
    [Xmot,VarVoxelMot,gammaMot] = Champagne(Ye,Le);
    X = zeros(N,T);
    X(IdxMotVox(:),:) = Xmot;
    VarVoxel = zeros(N,1);
    VarVoxel(IdxMotVox(:)) = VarVoxelMot;
    gamma = zeros(N,1);
    gamma(IdxMotVox(:)) = gammaMot;
else
    disp('Error! procflag should be 1, 2, or 3.')
end
%画图
figure(1)
plot(1:N,VarVoxel)
for k = 1:NumLocMot
    hold on
    xv = LocMot(k)*ones(1,100);
    yv = 0.01:0.01:1;
    plot(xv,yv,'r:');
end
for k = 1:NumLocOut
    hold on
    xv = LocOut(k)*ones(1,100);
    yv = 0.01:0.01:1;
    plot(xv,yv,'k:');
end
hold off