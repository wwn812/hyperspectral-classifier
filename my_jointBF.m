function newHsi=  my_jointBF(Hsi,group,pcaHsi)
M=size(Hsi,1);
N=size(Hsi,2);
K=size(Hsi,3);
k=size(pcaHsi,3);
newHsi=zeros(M,N,K);
w=group;
xy=-(w-1)/2:(w-1)/2; 
xy2=xy.^2;
dist=exp(-(repmat(xy2,[w,1])+repmat(xy2',[1,w]))/(2*((w-1)/4)^2));
T  =  maxFilter(pcaHsi(:,:,1), w) ;
for i=1:M
    for j=1:N
         r1=max(i-(w-1)/2,1);r2=min(i+(w-1)/2,M);
         c1=max(j-(w-1)/2,1);c2=min(j+(w-1)/2,N);
         HH=Hsi(r1:r2,c1:c2,:);
         pcaHH=pcaHsi(r1:r2,c1:c2,:);
         HH=reshape(HH,[(r2-r1+1)*(c2-c1+1),K]);
         pcaHH=reshape(pcaHH,[(r2-r1+1)*(c2-c1+1),k]);
         pcaHH_center=pcaHsi(i,j,:);
         D= EuDist2(pcaHH,pcaHH_center(:)',0);
         Ws=dist(r1-i+(w+1)/2:r2-i+(w+1)/2,c1-j+(w+1)/2:c2-j+(w+1)/2);
         Ww=exp(-D/((T/10))^2); 
         W=Ww.*(Ws(:));
         W=W/(sum(W)+10^-6);
         HH=W'*HH;
         newHsi(i,j,:)=HH(:);
    end
end