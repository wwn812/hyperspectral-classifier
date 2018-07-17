function eigvector =NWFE(Data,label,len)
C=unique(label);
%Sb
Sb=0;
for c1=1:length(C)
    Celse=C(find(C~=C(c1)));
    index1=find(label==C(c1));
    for c2=1:length(Celse)
        index=find(label==Celse(c2)); 
        sumlambda=0;
        subSb=0;
    for i=1:length(index1)
        dist1=sqrt(sum((Data(:,index)-repmat(Data(:,index1(i)),[1,length(index)])).^2))';
        w=(1./dist1)./sum(1./dist1);
        Mij=Data(:,index)*w;
        Xdiff=Data(:,index1(i))-Mij;
        dist2=sqrt(sum(Xdiff.^2));
        lambda=1/dist2;
        subSb=subSb+(lambda*Xdiff)*Xdiff';
        sumlambda=sumlambda+lambda;
    end
        subSb=subSb/sumlambda;
        Sb=Sb+subSb;
    end
end

%Sw
Sw=0;
for c1=1:length(C)
    index=find(label==C(c1));
    sumlambda=0;
    subSw=0;
    for i=1:length(index)
        dist1=0.01+sqrt(sum((Data(:,index)-repmat(Data(:,index(i)),[1,length(index)])).^2))';
        w=(1./dist1)./sum(1./dist1);
        Mij=Data(:,index)*w;
        Xdiff=Data(:,index(i))-Mij;
        dist2=sqrt(sum(Xdiff.^2));
        lambda=1/dist2;
        subSw=subSw+(lambda*Xdiff)*Xdiff';
        sumlambda=sumlambda+lambda;
    end
        subSw=subSw/sumlambda;
        Sw=Sw+subSw;
end
Sw=Sw+0.5*diag(diag(Sw));
[eigvec,eigval_matrix]=eig(Sb,Sw);

eigval=diag(eigval_matrix);
[sort_eigval,sort_eigval_index]=sort(abs(eigval));
eigvector=eigvec(:,sort_eigval_index(end:-1:(end-len+1)));
eigvalue=sort_eigval;




        