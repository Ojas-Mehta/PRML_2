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

% x11_test=x11_train; x12_test=x12_train;
% x21_test=x21_train; x22_test=x22_train;
% x31_test=x31_train; x32_test=x32_train;
% x41_test=x41_train; x42_test=x42_train;
% 
% x11_test=x11_val; x12_test=x12_val;
% x21_test=x21_val; x22_test=x22_val;
% x31_test=x31_val; x32_test=x32_val;
% x41_test=x41_val; x42_test=x42_val;

 X1=[x11_train,x12_train];
 X2=[x21_train,x22_train];
 X3=[x31_train,x32_train];
 X4=[x41_train,x42_train];

 X1_test=[x11_test,x12_test];
 X2_test=[x21_test,x22_test];
 X3_test=[x31_test,x32_test];
 X4_test=[x41_test,x42_test];
 
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

mean11=sum(x11_train)/Ntrain1;
mean12=sum(x12_train)/Ntrain1;

mean21=sum(x21_train)/Ntrain2;
mean22=sum(x22_train)/Ntrain2;

mean31=sum(x31_train)/Ntrain3;
mean32=sum(x32_train)/Ntrain3;

mean41=sum(x41_train)/Ntrain4;
mean42=sum(x42_train)/Ntrain4;

C1=covariance(X1);
C2=covariance(X2);
C3=covariance(X3);
C4=covariance(X4);

C1(1,2)=0; C1(2,1)=0;
C2(1,2)=0; C2(2,1)=0;
C3(1,2)=0; C3(2,1)=0;
C4(1,2)=0; C4(2,1)=0;

mean1=[mean11,mean12];
mean2=[mean21,mean22];
mean3=[mean31,mean32];
mean4=[mean41,mean42];

p1=0.25;
p2=0.25;
p3=0.25;
p4=0.25;

predicted1=zeros(Ntest1,5);
predicted2=zeros(Ntest2,5);
predicted3=zeros(Ntest3,5);
predicted4=zeros(Ntest4,5);

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
  
  %class 3
  constant=1/((2*pi)*(sqrt(det(C3))));  
  constant1=(x-mean3)*inv(C3)*(x-mean3).';
  constant2=exp(-0.5*constant1);
  ans3=constant*constant2;
  predicted1(i,3)=ans3;
  
  %class 4
  constant=1/((2*pi)*(sqrt(det(C4))));  
  constant1=(x-mean4)*inv(C4)*(x-mean4).';
  constant2=exp(-0.5*constant1);
  ans4=constant*constant2;
  predicted1(i,4)=ans4;
  
  max=0;class=0;
  for j=1:4
    if(predicted1(i,j)>max)
        max=predicted1(i,j);
        class=j;
    end
  end
  predicted1(i,5)=class;
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
  
  %class 3
  constant=1/((2*pi)*(sqrt(det(C3))));  
  constant1=(x-mean3)*inv(C3)*(x-mean3).';
  constant2=exp(-0.5*constant1);
  ans3=constant*constant2;
  predicted2(i,3)=ans3;
  
  %class 4
  constant=1/((2*pi)*(sqrt(det(C4))));  
  constant1=(x-mean4)*inv(C4)*(x-mean4).';
  constant2=exp(-0.5*constant1);
  ans4=constant*constant2;
  predicted2(i,4)=ans4;
  
  max=0;class=0;
  for j=1:4
    if(predicted2(i,j)>max)
        max=predicted2(i,j);
        class=j;
    end
  end
  predicted2(i,5)=class;
  if(class==2) count=count+1; end
end



for i=1:Ntest3
  %class 1
  x=[x31_test(i),x32_test(i)];
  constant=1/((2*pi)*(sqrt(det(C1))));  
  constant1=(x-mean1)*inv(C1)*(x-mean1).';
  constant2=exp(-0.5*constant1);
  ans1=constant*constant2;
  predicted3(i,1)=ans1;
  
  %class 2
  constant=1/((2*pi)*(sqrt(det(C2))));  
  constant1=(x-mean2)*inv(C2)*(x-mean2).';
  constant2=exp(-0.5*constant1);
  ans2=constant*constant2;
  predicted3(i,2)=ans2;
  
  %class 3
  constant=1/((2*pi)*(sqrt(det(C3))));  
  constant1=(x-mean3)*inv(C3)*(x-mean3).';
  constant2=exp(-0.5*constant1);
  ans3=constant*constant2;
  predicted3(i,3)=ans3;
  
  %class 4
  constant=1/((2*pi)*(sqrt(det(C4))));  
  constant1=(x-mean4)*inv(C4)*(x-mean4).';
  constant2=exp(-0.5*constant1);
  ans4=constant*constant2;
  predicted3(i,4)=ans4;
  
  max=0;class=0;
  for j=1:4
    if(predicted3(i,j)>max)
        max=predicted3(i,j);
        class=j;
    end
  end
  predicted3(i,5)=class;
  
  if(class==3) count=count+1; end
end


for i=1:Ntest4
  %class 1
  x=[x41_test(i),x42_test(i)];
  constant=1/((2*pi)*(sqrt(det(C1))));  
  constant1=(x-mean1)*inv(C1)*(x-mean1).';
  constant2=exp(-0.5*constant1);
  ans1=constant*constant2;
  predicted4(i,1)=ans1;
  
  %class 2
  constant=1/((2*pi)*(sqrt(det(C2))));  
  constant1=(x-mean2)*inv(C2)*(x-mean2).';
  constant2=exp(-0.5*constant1);
  ans2=constant*constant2;
  predicted4(i,2)=ans2;
  
  %class 3
  constant=1/((2*pi)*(sqrt(det(C3))));  
  constant1=(x-mean3)*inv(C3)*(x-mean3).';
  constant2=exp(-0.5*constant1);
  ans3=constant*constant2;
  predicted4(i,3)=ans3;
  
  %class 4
  constant=1/((2*pi)*(sqrt(det(C4))));  
  constant1=(x-mean4)*inv(C4)*(x-mean4).';
  constant2=exp(-0.5*constant1);
  ans4=constant*constant2;
  predicted4(i,4)=ans4;
  
  max=0;class=0;
  for j=1:4
    if(predicted4(i,j)>max)
        max=predicted4(i,j);
        class=j;
    end
  end
  predicted4(i,5)=class;
  
  if(class==4) 
      count=count+1;
  end
end

%Accuracy for test data

classification_accuracy=double(count)/(Ntest1+Ntest2+Ntest3+Ntest4);
disp('Classification accuracy is:');
disp(classification_accuracy);

%Computing confusion matrix.
confusion_matrix=zeros(4,4);
for i=1:Ntest1
  confusion_matrix(1,predicted1(i,5))=confusion_matrix(1,predicted1(i,5))+1;
end

for i=1:Ntest2
  confusion_matrix(2,predicted2(i,5))=confusion_matrix(2,predicted2(i,5))+1;
end

for i=1:Ntest3
  confusion_matrix(3,predicted3(i,5))=confusion_matrix(3,predicted3(i,5))+1;
end

for i=1:Ntest4
  confusion_matrix(4,predicted4(i,5))=confusion_matrix(4,predicted4(i,5))+1;
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

N=Ntrain1+Ntrain2+Ntrain3+Ntrain4;

predicted=zeros(Ntest,1);
predicted5=zeros(Ntest,4);

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
  
  %class 3
  constant=1/((2*pi)*(sqrt(det(C3))));  
  constant1=(x-mean3)*inv(C3)*(x-mean3).';
  constant2=exp(-0.5*constant1);
  ans3=constant*constant2;
  predicted5(i,3)=ans3;
  
  %class 4
  constant=1/((2*pi)*(sqrt(det(C4))));  
  constant1=(x-mean4)*inv(C4)*(x-mean4).';
  constant2=exp(-0.5*constant1);
  ans4=constant*constant2;
  predicted5(i,4)=ans4;
  
  max=0;class=0;
  for j=1:4
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
cmap = [0.7 1 0.7;0.9 0.9 1;1 0.7 1;1 0.8 0.8];
colormap(cmap);

scatter(x11_train,x12_train,'g');
scatter(x21_train,x22_train,'b');
scatter(x31_train,x32_train,'m');
scatter(x41_train,x42_train,'r');
hold off

