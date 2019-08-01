%x12 is 2nd column of 1st class.
[x11_train,x12_train]=textread('..\data_assign2_group5\group5\overlapping\class1_train.txt','%f %f');
[x11_test,x12_test]=textread('..\data_assign2_group5\group5\overlapping\class1_test.txt','%f %f');
[x11_val,x12_val]=textread('..\data_assign2_group5\group5\overlapping\class1_val.txt','%f %f');

[x21_train,x22_train]=textread('..\data_assign2_group5\group5\overlapping\class2_train.txt','%f %f');
[x21_test,x22_test]=textread('..\data_assign2_group5\group5\overlapping\class2_test.txt','%f %f');
[x21_val,x22_val]=textread('..\data_assign2_group5\group5\overlapping\class2_val.txt','%f %f');

[x31_train,x32_train]=textread('..\data_assign2_group5\group5\overlapping\class3_train.txt','%f %f');
[x31_test,x32_test]=textread('..\data_assign2_group5\group5\overlapping\class3_test.txt','%f %f');
[x31_val,x32_val]=textread('..\data_assign2_group5\group5\overlapping\class3_val.txt','%f %f');

[x41_train,x42_train]=textread('..\data_assign2_group5\group5\overlapping\class4_train.txt','%f %f');
[x41_test,x42_test]=textread('..\data_assign2_group5\group5\overlapping\class4_test.txt','%f %f');
[x41_val,x42_val]=textread('..\data_assign2_group5\group5\overlapping\class4_val.txt','%f %f');

[Ntrain1,nq]=size(x11_train);
[Ntest1,nq]=size(x11_test);
[Nval1,nq]=size(x11_val);

[Ntrain2,nq]=size(x21_train);
[Ntest2,nq]=size(x21_test);
[Nval2,nq]=size(x21_val);

[Ntrain3,nq]=size(x31_train);
[Ntest3,nq]=size(x31_test);
[Nval3,nq]=size(x31_val);

[Ntrain4,nq]=size(x41_train);
[Ntest4,nq]=size(x41_test);
[Nval4,nq]=size(x41_val);


xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];

Ntest=length(xy);

N=Ntrain1+Ntrain2+Ntrain3+Ntrain4;

dist=zeros(N,2);
freq=zeros(4,1);
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
    
    for j=1:Ntrain3
      dist(k,1)=(x31_train(j)-xy(i,1))^2 + (x32_train(j)-xy(i,2))^2;
      dist(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist(k,1)=(x41_train(j)-xy(i,1))^2 + (x42_train(j)-xy(i,2))^2;
      dist(k,2)=4;
      k=k+1;
    end
    
    dist=sortrows(dist,1);
    
    %q-nearest neighbours.
    for j=1:q
        freq(dist(j,2))=freq(dist(j,2))+1;
    end
    
    max=0;
    for j=1:4
        if(freq(j)>max) 
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
cmap = [0.7 1 0.7;0.9 0.9 1;1 0.7 1;1 0.8 0.8];
colormap(cmap);

scatter(x11_train,x12_train,'g');
scatter(x21_train,x22_train,'b');
scatter(x31_train,x32_train,'m');
scatter(x41_train,x42_train,'r');
hold off
