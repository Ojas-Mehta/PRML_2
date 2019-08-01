%x12 is 2nd column of 1st class.
[x11_train,x12_train]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_train.txt','%f %f');
[x11_test,x12_test]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_test.txt','%f %f');
[x11_val,x12_val]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_val.txt','%f %f');

[x21_train,x22_train]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_train.txt','%f %f');
[x21_test,x22_test]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_test.txt','%f %f');
[x21_val,x22_val]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_val.txt','%f %f');

[Ntrain1,nq]=size(x11_train);
[Ntest1,nq]=size(x11_test);
[Nval1,nq]=size(x11_val);

[Ntrain2,nq]=size(x21_train);
[Ntest2,nq]=size(x21_test);
[Nval2,nq]=size(x21_val);

xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];

Ntest=length(xy);

N=Ntrain1+Ntrain2;

dist=zeros(N,2);
freq=zeros(2,1);
predicted=zeros(Ntest,1);

q=6; %Hyper-parameter.

k=1;
for i=1:Ntest
    
    freq=zeros(4,1);
    for j=1:Ntrain1
      dist(k,1)=(x11_train(j)-xy(i,1))^2 + (x12_train(j)-xy(i,2))^2;
      dist(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist(k,1)=(x21_train(j)-xy(i,1))^2 + (x22_train(j)-xy(i,2))^2;
      dist(k,2)=2;
      k=k+1;
    end
    
    dist=sortrows(dist,1);
    
    %q-nearest neighbours.
    for j=1:q
        freq(dist(j,2))=freq(dist(j,2))+1;
    end
    
    max=0;max1=0;
    for j=1:2
        if(freq(j)>max1)
            max1=freq(j);
            max=j; 
        end
    end
    
    predicted(i)=max;  
    k=1;
end

decisionmap = reshape(predicted, image_size);

figure;
imagesc(xrange,yrange,decisionmap);

hold on

set(gca,'ydir','normal');
cmap = [0.7 1 0.7;0.9 0.9 1];
colormap(cmap);

scatter(x11_train,x12_train,'g');
scatter(x21_train,x22_train,'b');
hold off
