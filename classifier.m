%%多特征分别识别选择最小残差类
close all;
clear;
clc;  
addpath(genpath(pwd));
S=load('Indian_pines_corrected.mat');
Hsi=S.indian_pines_corrected;
S=load('Indian_pines_gt.mat');
GroundTure=S.indian_pines_gt;


TotalLabel = length(unique(GroundTure));
Lab=GroundTure;
[M,N]=size(Lab);
Hsi=double(Hsi)./repmat(sqrt(sum(Hsi.^2,3)),[1,1,size(Hsi,3)]);
Hsi0=mybffilter( Hsi ,10);

hsi0=reshape(Hsi0,[M*N,size(Hsi0,3)]);
lab=reshape(Lab,[M*N,1]);
%% 

new11=myvecdctj( hsi0,9,3 );%new12=myvecdctj( hsi0,5,2 );
hsi10=hsi0;
Hsi10=reshape(hsi10,[M,N,size(hsi10,2)]);
hsi20=new11;
Hsi20=reshape(hsi20,[M,N,size(hsi20,2)]);
Hsi0=Hsi;
hsi0=reshape(Hsi0,[M*N,size(Hsi0,3)]);

K = 1;                              %OMP算法中信号的稀疏度
loopCnt =50;                       %为减少随机性而计算10次，结果取平均值
trnnum=5;                        %每一类选取训练样本的个数

clsCnt = TotalLabel-1;                        %分类的数量
clsNum = zeros(1, clsCnt);          %每个类别的数据的总数量
trnNum = zeros(1, clsCnt);          %每个类别的数据选作训练数据的数量
tstNum = zeros(1, clsCnt);          %每个类别的数据选作测试数据的数量
conMat = zeros(clsCnt,clsCnt);      %混淆矩阵，保存测试数据的预测类别与实际类别的关系
conMat1 = zeros(clsCnt,clsCnt);
conMat2 = zeros(clsCnt,clsCnt);
 for i = 1 : clsCnt
   index = find(lab == i);                 %找到标记为i的数据的下标
   clsNum(i) = size(index,1);                   %标记为i的数据的总数量
   %trnNum(i) = ceil(clsNum(i) * trnPer);        %选取作为训练样本的数量
   trnNum(i) = trnnum; 
   tstNum(i) = clsNum(i) - trnNum(i);           %剩余的为测试样本的数量
 end


%开始计算
acc=0;acc1=0;acc2=0;OA=0;OA1=0;OA2=0;
for loop = 1 : loopCnt              %重复10次
    lab0=zeros(M*N,1);lab1=zeros(M*N,1);lab2=zeros(M*N,1);
    trnFet0 = [];                        %保存训练数据
    trnFet10 = [];                        %保存训练数据
    trnFet20 = [];                        %保存训练数据
    trnLab0 = [];                        %保存训练数据对应的类型
    
    trnFet = [];                        %保存特征提取后的训练数据
    trnFet1 = [];                        %保存特征提取后的训练数据
    trnFet2 = [];                        %保存特征提取后的训练数据
    trnLab = [];                        %保存训练数据对应的类型
    tstFet = [];                        %保存特征提取后的测试数据
    tstFet1 = [];                        %保存特征提取后的测试数据
    tstFet2 = [];                        %保存特征提取后的测试数据
    tstLab = [];                        %保存测试数据对应的类型
    %每种类别随机选取**作为样本数据
    index1=[];
    for i = 1 : clsCnt
       index = find(lab == i);                  %找到标记为i的数据的下标
       random_index = index(randperm(length(index)));%结果为打乱顺序后的下标序列

       index = random_index(1:trnNum(i));            %在乱序中取前**作为训练样本，index保存它们的下标
       index1(:,i)=index(:);
       trnFet0 = [trnFet0 hsi0(index,:)'];         %将训练样本的数据依次填充进trnFet数组
       %%%%%%%%%%%%%%%%%%%%%%%%
       trnFet10 = [trnFet10 hsi10(index,:)'];         %将训练样本的数据依次填充进trnFet数组
       trnFet20 = [trnFet20 hsi20(index,:)'];         %将训练样本的数据依次填充进trnFet数组
       %%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       trnLab0 = [trnLab0 ones(1,length(index))*i];    %将训练样本的标记依次填充进trnLab数组

    end
    %omp之前依据不同分类先特征提取
    le1=30;le2=45;
    newhsi=NWFE_Hsi(Hsi0,trnFet0,trnLab0,le1);
    hsi=reshape(newhsi,[M*N,size(newhsi,3)]);
    newhsi1=NWFE_Hsi(Hsi10,trnFet10,trnLab0,le1);
    hsi1=reshape(newhsi1,[M*N,size(newhsi1,3)]);
    newhsi2=NWFE_Hsi(Hsi20,trnFet20,trnLab0,le2);
    hsi2=reshape(newhsi2,[M*N,size(newhsi2,3)]);
 
    %
    for i = 1 : clsCnt
        
       index(:) = index1(:,i);
       trnFet = [trnFet hsi(index,:)'];         %将训练样本的数据依次填充进trnFet数组
       trnFet1 = [trnFet1 hsi1(index,:)'];         %将训练样本的数据依次填充进trnFet数组
       trnFet2 = [trnFet2 hsi2(index,:)'];         %将训练样本的数据依次填充进trnFet数组
       trnLab = [trnLab ones(1,length(index))*i];    %将训练样本的标记依次填充进trnLab数组
    end 
    %%综合残差汇总及其分类汇总
   %     %用OMP算法预测训练数据的分类

    for i = 1 : size(hsi,1)              %对每一个测试样本i做预测
       if ( lab(i)>0)
       x = hsi(i,:);                     %提取出某个测试样本的200维数据x
       x=x';
       sparse = OMP(trnFet,x,K);            %用OMP算法求出其稀疏系数矩阵
       residual = zeros(1,clsCnt);          %该组样本对每个类别的重构冗余
       for j = 1:1:clsCnt                   %将该测试样本重构成每一种类别
           index = find(trnLab == j);       %选取字典中类别为j的数据
           
           D_c = trnFet(:,index);  %字典中类别为j的列
           s_c = sparse(index);             %稀疏矩阵中本应为j类的位置的系数
          
           temp = x - D_c*s_c;              %
           residual(j) = norm(temp,2);      %计算重构冗余
       end
       %residual
       preLab = find(residual == min(residual));     %找出重构冗余最小的类别，即为预测的类别
       conMat(preLab(1),lab(i)) = conMat(preLab(1),lab(i)) + 1; %根据实际类别与预测结果，更新混淆矩阵
       lab0(i)=preLab(1);
       end
    end
    oa=trace(conMat)/sum(conMat(:))
    for i = 1 : clsCnt
    conMat(:,i) = conMat(:,i) ./ (clsNum(i));
    end
   ave_acc = sum(diag(conMat))/clsCnt
   acc=acc+ave_acc;
   OA=OA+oa;
   
   %%%%%%%%%%%
       %用OMP算法预测训练数据的分类1

    for i = 1 : size(hsi1,1)              %对每一个测试样本i做预测
       if ( lab(i)>0)
       x = hsi1(i,:);                     %提取出某个测试样本的200维数据x
       x=x';
       sparse = OMP(trnFet1,x,K);            %用OMP算法求出其稀疏系数矩阵
       residual = zeros(1,clsCnt);          %该组样本对每个类别的重构冗余
       for j = 1:1:clsCnt                   %将该测试样本重构成每一种类别
           index = find(trnLab == j);       %选取字典中类别为j的数据
           
           D_c = trnFet1(:,index);  %字典中类别为j的列
           s_c = sparse(index);             %稀疏矩阵中本应为j类的位置的系数
          
           temp = x - D_c*s_c;              %
           residual(j) = norm(temp,2);      %计算重构冗余
       end
       %residual
       preLab = find(residual == min(residual));     %找出重构冗余最小的类别，即为预测的类别
       conMat1(preLab(1),lab(i)) = conMat1(preLab(1),lab(i)) + 1; %根据实际类别与预测结果，更新混淆矩阵
       lab1(i)=preLab(1);
       end
    end
    oa1=trace(conMat1)/sum(conMat1(:))
    for i = 1 : clsCnt
    conMat1(:,i) = conMat1(:,i) ./ (clsNum(i));
    end
   ave_acc1 = sum(diag(conMat1))/clsCnt
   acc1=acc1+ave_acc1;
   OA1=OA1+oa1;
   %%%%%%%%%%%
       %用OMP算法预测训练数据的分类2

    for i = 1 : size(hsi2,1)              %对每一个测试样本i做预测
       if ( lab(i)>0)
       x = hsi2(i,:);                     %提取出某个测试样本的200维数据x
       x=x';
       sparse = OMP(trnFet2,x,K);            %用OMP算法求出其稀疏系数矩阵
       residual = zeros(1,clsCnt);          %该组样本对每个类别的重构冗余
       for j = 1:1:clsCnt                   %将该测试样本重构成每一种类别
           index = find(trnLab == j);       %选取字典中类别为j的数据
           
           D_c = trnFet2(:,index);  %字典中类别为j的列
           s_c = sparse(index);             %稀疏矩阵中本应为j类的位置的系数
          
           temp = x - D_c*s_c;              %
           residual(j) = norm(temp,2);      %计算重构冗余
       end
       %residual
       preLab = find(residual == min(residual));     %找出重构冗余最小的类别，即为预测的类别
       conMat2(preLab(1),lab(i)) = conMat2(preLab(1),lab(i)) + 1; %根据实际类别与预测结果，更新混淆矩阵
       lab2(i)=preLab(1);
       end
    end
    oa2=trace(conMat2)/sum(conMat2(:))
    for i = 1 : clsCnt
    conMat2(:,i) = conMat2(:,i) ./ (clsNum(i));
    end
   
   ave_acc2 = sum(diag(conMat2))/clsCnt
   acc2=acc2+ave_acc2;
   OA2=OA2+oa2;
   loop
Lab0=reshape(lab0,[M,N,size(lab0,2)]);
Lab1=reshape(lab1,[M,N,size(lab1,2)]);
Lab2=reshape(lab2,[M,N,size(lab2,2)]);
%colorMap = rand(TotalLabel,3)
colorMap = [0.0034    0.7655    0.5616
    0.3167    0.7986    0.7129
    0.6999    0.6363    0.9708
    0.2553    0.2556    0.9160
    0.3135    0.0015    0.3834
    0.2940    0.8495    0.2898
    0.5776    0.6748    0.2754
    0.4261    0.0205    0.7786
    0.5287    0.3347    0.3878
    0.9195    0.7659    0.8520
    0.0380    0.5160    0.8518
    0.4288    0.0194    0.2332
    0.1106    0.3200    0.2896
    0.2265    0.4587    0.1553
    0.1646    0.5545    0.9827
    0.4627    0.5080    0.8313
    0.3460    0.6672    0.5999
];
colorMap(1,:)=0;
img = zeros(M,N);
img(:) = GroundTure;
figure;
subplot 221;
imshow(img,colorMap);
title('original image');
img(:)= Lab0;
subplot 222;
imshow(img,colorMap);
title('recognized image0');
img(:)= Lab1;
subplot 223;
imshow(img,colorMap);
title('recognized image1');
img(:)= Lab2;
subplot 224;
imshow(img,colorMap);
title('recognized image2');

end

acc=acc/loopCnt 
acc1=acc1/loopCnt 
acc2=acc2/loopCnt 
OA=OA/loopCnt
OA1=OA1/loopCnt
OA2=OA2/loopCnt
%printConMat(conMat);
