%x12 is 2nd column of 1st class.
[x11_train,x12_train]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_train.txt','%f %f');
[x11_test,x12_test]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_test.txt','%f %f');
[x11_val,x12_val]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_val.txt','%f %f');

[x21_train,x22_train]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_train.txt','%f %f');
[x21_test,x22_test]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_test.txt','%f %f');
[x21_val,x22_val]=textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_val.txt','%f %f');

 x11_test=x11_train; x12_test=x12_train;
 x21_test=x21_train; x22_test=x22_train;
 
 %x11_test=x11_val; x12_test=x12_val;
 %x21_test=x21_val; x22_test=x22_val;
 
 X1=[x11_train,x12_train];
 X2=[x21_train,x22_train];
 
 X1_test=[x11_test,x12_test];
 X2_test=[x21_test,x22_test];
 
[Ntrain1,nq]=size(x11_train);
[Ntest1,nq]=size(x11_test);
[Nval1,nq]=size(x11_val);

[Ntrain2,nq]=size(x21_train);
[Ntest2,nq]=size(x21_test);
[Nval2,nq]=size(x21_val);

mean11=sum(x11_train)/Ntrain1;
mean12=sum(x12_train)/Ntrain1;

mean21=sum(x21_train)/Ntrain2;
mean22=sum(x22_train)/Ntrain2;

C1=covariance(X1);
C2=covariance(X2);

C1(1,2)=0; C1(2,1)=0;
C2(1,2)=0; C2(2,1)=0;

mean1=[mean11,mean12];
mean2=[mean21,mean22];

p1=0.25;
p2=0.25;

predicted1=zeros(Ntest1,3);
predicted2=zeros(Ntest2,3);

count=0;

for i=1:Ntest1
  %class 1
  x=[x11_test(i),x12_test(i)];
  constant=1/((2*pi)*(sqrt(det(C1))));  
  constant1=(x-mean1)*inv(C1)*(x-mean1).';
  constant2=exp(-0.5*constant1);
  ans1=constant*constant2;
  predicted1(i,1)=ans1;
  
  %class 2
  constant=1/((2*pi)*(sqrt(det(C2))));  
  constant1=(x-mean2)*inv(C2)*(x-mean2).';
  constant2=exp(-0.5*constant1);
  ans2=constant*constant2;
  predicted1(i,2)=ans2;
    
  max=0;class=0;
  for j=1:2
    if(predicted1(i,j)>max)
        max=predicted1(i,j);
        class=j;
    end
  end
  predicted1(i,3)=class;
  if(class==1) count=count+1; end
end


for i=1:Ntest2
  %class 1
  x=[x21_test(i),x22_test(i)];
  constant=1/((2*pi)*(sqrt(det(C1))));  
  constant1=(x-mean1)*inv(C1)*(x-mean1).';
  constant2=exp(-0.5*constant1);
  ans1=constant*constant2;
  predicted2(i,1)=ans1;
  
  %class 2
  constant=1/((2*pi)*(sqrt(det(C2))));  
  constant1=(x-mean2)*inv(C2)*(x-mean2).';
  constant2=exp(-0.5*constant1);
  ans2=constant*constant2;
  predicted2(i,2)=ans2;
  
  max=0;class=0;
  for j=1:2
    if(predicted2(i,j)>max)
        max=predicted2(i,j);
        class=j;
    end
  end
  predicted2(i,3)=class;
  if(class==2) count=count+1; end
end

%Accuracy for test data

classification_accuracy=double(count)/(Ntest1+Ntest2);
disp('Classification accuracy is:');
disp(classification_accuracy);

%Computing confusion matrix.
confusion_matrix=zeros(2,2);
for i=1:Ntest1
  confusion_matrix(1,predicted1(i,3))=confusion_matrix(1,predicted1(i,3))+1;
end

for i=1:Ntest2
  confusion_matrix(2,predicted2(i,3))=confusion_matrix(2,predicted2(i,3))+1;
end

disp('Confusion matrix is:');
disp(confusion_matrix);

%Plotting

xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];

Ntest=length(xy);

N=Ntrain1+Ntrain2;

predicted=zeros(Ntest,1);
predicted5=zeros(Ntest,2);

for i=1:Ntest
  x=xy(i,:);
  %class 1
  constant=1/((2*pi)*(sqrt(det(C1))));  
  constant1=(x-mean1)*inv(C1)*(x-mean1).';
  constant2=exp(-0.5*constant1);
  ans1=constant*constant2;
  predicted5(i,1)=ans1;
  
  %class 2
  constant=1/((2*pi)*(sqrt(det(C2))));  
  constant1=(x-mean2)*inv(C2)*(x-mean2).';
  constant2=exp(-0.5*constant1);
  ans2=constant*constant2;
  predicted5(i,2)=ans2;
  
  max=0;class=0;
  for j=1:2
    if(predicted5(i,j)>max)
        max=predicted5(i,j);
        class=j;
    end
  end
  predicted(i)=class;
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

