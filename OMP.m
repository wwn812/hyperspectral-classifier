function [y] = OMP(D,x,K)
% 输入参数：
%       D - 字典（必须是过完备字典）
%       x - 观测信号
%       K - 信号的稀疏度
% 输出参数：
%       y - 稀疏稀疏矩阵.

%初始化
y = zeros(size(D,2),size(x,2));             %
index = zeros(1,K);                         %支撑索引集初始为空
residual = x;                               %残差初始为x

for i = 1 : K
    pro = D' * residual; %字典各列与残差的乘积
    pos = find(abs(pro)==max(abs(pro)));    %找到内积绝对值最大的列，即与残差最相关的列
   
    index(i) = pos(1);                      %将寻找到的列作为支撑索引，加入信号支撑集
    temp = pinv(D(:,index(1:i))) * x;     %对选取的原子集合施密特正交化，之后与观测信号运算得到最小二乘解
    residual = x - D(:,index(1:i)) * temp;  %更新残差
    %temp = pinv(D(:,index(i))) * x;     %对选取的原子集合施密特正交化，之后与观测信号运算得到最小二乘解
    %residual = x - D(:,index(i)) * temp;  %更新残差
    %residual1 = norm(residual,2);
end
if (~isempty(index))
    y(index,:) = temp;
end
