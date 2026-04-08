function [W,index,lossFun]=DKAGL(X,c,k,Q,alpha,beta,eta,maxIter)
%% Unsupervised feature selection via discriminative k-means and adaptive graph learning
%% input:
% X:dxn,data matrix which satisfying X1=0
% c:clusters number
% alpha,beta>0: regularization parameter
% k: the number of neighbors
% Q: dependency score matrix of features
%% output:
% W: feature weight matrix
% index: the score of features
% lossFun: the value of objective function
%% Initialize relevant matrix
X=normalize(X);% 对每一列进行归一化处理，使其均值为0，标准差为1
X=X';%centered on features
[d,n]=size(X);
lossFun=zeros(1,maxIter);
W=rand(d,c);
temp_D=0.5*(sqrt(sum(W.^2,2)+eps)).^(-1);
D1=diag(temp_D);
I_n=ones(n,1);
% I_c=ones(c,1);
Y=full(sparse(1:n,randi([1,c],1,n),1,n,c));
G=Y*(Y'*Y)^(-0.5);
t1=2;
S=similarity_t1(X,k,t1);
S=(S+S')/2;
% L=Lap(S);
D=diag(sum(S,2));
L=D-S;
%L = diag(sum(S, 1)) - S;
%X=X';
% X=normalize(X);
% X=X';
C=X*I_n;
%% update relevant varibles
for i=1:maxIter
    fprintf('第%d次迭代\n',i);
    %% update W and b
    V=updateU(W);
    temp_W=X*X'+alpha*V+beta*X*L*X'+eta*D1*Q;
    W=real(pinv(temp_W)*X*G);
    temp_D=0.5*(sqrt(sum(W.^2,2)+eps)).^(-1);
    D1=diag(temp_D);
    S=similarity_t1(W'*X,k,t1);
    S=(S+S')/2;
    D=diag(sum(S,2));
    L=D-S;
    b=G'*I_n/n;
    %% compute M
    M=X'*W+I_n*b';
    %% update Y
    for j=1:n
        for r=1:c
            y=Y(:,r);
            m=M(:,r);
            if Y(j,r)==0
                l1=(y'*m+M(j,r))/sqrt(y'*y+1);
                l2=(y'*m)/sqrt(y'*y);
                t(j,r)=l1-l2;
            else
            l1=(y'*m)/sqrt(y'*y);
            l2=(y'*m-M(j,r))/sqrt(y'*y-1);
            t(j,r)=l1-l2;
            end
            clear l1 l2;
            %t(j,r)=(Y(r,:)'*M(r,:)+(1-Y(j,r))*M(j,r))/sqrt(Y(r,:)'*Y(r,:)+(1-Y(j,r)))-(Y(r,:)'*M(r,:)-Y(j,r)*M(j,r))/sqrt(Y(r,:)'*Y(r,:)-Y(j,r));
        end
    end
    for j=1:n
        for r=1:c
            if t(j,r)==max(t(j,:))
                Y(j,r)=1;
            else
                Y(j,r)=0;
            end
        end
    end
    G=Y*pinv((Y'*Y)^(0.5));
    tem1=trace(W'*updateU(W)*W);
    tem2=trace(W'*D1*Q*W);
    tem3=norm(X'*W+I_n*b'-G,'fro')^2;
    tem4=trace(W'*X*L*X'*W);
    lossFun(i)=tem3+alpha*tem1+beta*tem4+eta*tem2;
    clear tem1 tem2 tem3 tem4;
end
score=sum((W.*W),2);
[~,index]=sort(score,'descend');
%newFea=X(index(1:m),:);
end
