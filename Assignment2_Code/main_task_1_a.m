clearvars;
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


x11_test=x11_train; x12_test=x12_train;
x21_test=x21_train; x22_test=x22_train;
x31_test=x31_train; x32_test=x32_train;
x41_test=x41_train; x42_test=x42_train;

% 
% x11_test=x11_val; x12_test=x12_val;
% x21_test=x21_val; x22_test=x22_val;
% x31_test=x31_val; x32_test=x32_val;
% x41_test=x41_val; x42_test=x42_val;

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

N=Ntrain1+Ntrain2+Ntrain3+Ntrain4;

class1_count=0;
class2_count=0;
class3_count=0;
class4_count=0;
count=0;

dist1=zeros(N,2);
freq1=zeros(4,1);
predicted1=zeros(Ntest1,1);

dist2=zeros(N,2);
freq2=zeros(4,1);
predicted2=zeros(Ntest2,1);

dist3=zeros(N,2);
freq3=zeros(4,1);
predicted3=zeros(Ntest3,1);

dist4=zeros(N,2);
freq4=zeros(4,1);
predicted4=zeros(Ntest4,1);

q=2; %Hyper-parameter.

%For class 1
k=1;
for i=1:Ntest1
    freq1=zeros(4,1);
    for j=1:Ntrain1
      dist1(k,1)=(x11_train(j)-x11_test(i))^2 + (x12_train(j)-x12_test(i))^2;
      dist1(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist1(k,1)=(x21_train(j)-x11_test(i))^2 + (x22_train(j)-x12_test(i))^2;
      dist1(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist1(k,1)=(x31_train(j)-x11_test(i))^2 + (x32_train(j)-x12_test(i))^2;
      dist1(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist1(k,1)=(x41_train(j)-x11_test(i))^2 + (x42_train(j)-x12_test(i))^2;
      dist1(k,2)=4;
      k=k+1;
    end
    
    dist1=sortrows(dist1,1);
    
    %q-nearest neighbours.
    for j=1:q
        freq1(dist1(j,2))=freq1(dist1(j,2))+1;
    end
    
    max=0;max1=0;
    for j=1:4
        if(freq1(j)>max1)
            max1=freq1(j);
            max=j; 
        end
    end
    
    predicted1(i)=max;
    if(max==1) count=count+1; end
        
    if(max==1) 
        class1_count=class1_count+1; 
    end
    
    if(max==2) 
        class2_count=class2_count+1; 
    end
    
    if(max==3) 
        class3_count=class3_count+1; 
    end
    
    if(max==4) 
        class4_count=class4_count+1; 
    end
    
    k=1;
end


%For class 2
k=1;
for i=1:Ntest2
    freq2=zeros(4,1);
    for j=1:Ntrain1
      dist2(k,1)=(x11_train(j)-x21_test(i))^2 + (x12_train(j)-x22_test(i))^2;
      dist2(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist2(k,1)=(x21_train(j)-x21_test(i))^2 + (x22_train(j)-x22_test(i))^2;
      dist2(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist2(k,1)=(x31_train(j)-x21_test(i))^2 + (x32_train(j)-x22_test(i))^2;
      dist2(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist2(k,1)=(x41_train(j)-x21_test(i))^2 + (x42_train(j)-x22_test(i))^2;
      dist2(k,2)=4;
      k=k+1;
    end
    
    dist2=sortrows(dist2,1);
   
    for j=1:q
        freq2(dist2(j,2))=freq2(dist2(j,2))+1;
    end
    
    max=0;max1=0;
    for j=1:4
        if(freq2(j)>max1) 
            max1=freq2(j);
            max=j; 
        end
    end
    
    predicted2(i)=max;
    
    if(max==2) count=count+1;end
    if(max==1) 
        class1_count=class1_count+1; 
    end
    
    if(max==2) 
        class2_count=class2_count+1; 
    end
    
    if(max==3) 
        class3_count=class3_count+1; 
    end
    
    if(max==4) 
        class4_count=class4_count+1; 
    end
    
    k=1;
end

k=1;
for i=1:Ntest3
    freq3=zeros(4,1);
    for j=1:Ntrain1
      dist3(k,1)=(x11_train(j)-x31_test(i))^2 + (x12_train(j)-x32_test(i))^2;
      dist3(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist3(k,1)=(x21_train(j)-x31_test(i))^2 + (x22_train(j)-x32_test(i))^2;
      dist3(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist3(k,1)=(x31_train(j)-x31_test(i))^2 + (x32_train(j)-x32_test(i))^2;
      dist3(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist3(k,1)=(x41_train(j)-x31_test(i))^2 + (x42_train(j)-x32_test(i))^2;
      dist3(k,2)=4;
      k=k+1;
    end
    
    dist3=sortrows(dist3,1);
    
    for j=1:q
        freq3(dist3(j,2))=freq3(dist3(j,2))+1;
    end
    
    max=0;max1=0;
    for j=1:4
        if(freq3(j)>max1)
            max1=freq3(j);
            max=j; 
        end
    end
    
    predicted3(i)=max;
    if(max==3) count=count+1; end
    
    if(max==1) 
        class1_count=class1_count+1; 
    end
    
    if(max==2) 
        class2_count=class2_count+1; 
    end
    
    if(max==3) 
        class3_count=class3_count+1; 
    end
    
    if(max==4) 
        class4_count=class4_count+1; 
    end
    
    k=1;
end

k=1;
for i=1:Ntest4
    freq4=zeros(4,1);
    for j=1:Ntrain1
      dist4(k,1)=(x11_train(j)-x41_test(i))^2 + (x12_train(j)-x42_test(i))^2;
      dist4(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist4(k,1)=(x21_train(j)-x41_test(i))^2 + (x22_train(j)-x42_test(i))^2;
      dist4(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist4(k,1)=(x31_train(j)-x41_test(i))^2 + (x32_train(j)-x42_test(i))^2;
      dist4(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist4(k,1)=(x41_train(j)-x41_test(i))^2 + (x42_train(j)-x42_test(i))^2;
      dist4(k,2)=4;
      k=k+1;
    end
    
    dist4=sortrows(dist4,1);
    
    for j=1:q
        freq4(dist4(j,2))=freq4(dist4(j,2))+1;
    end
    
    max=0;max1=0;
    for j=1:4
        if(freq4(j)>max1)
            max1=freq4(j);
            max=j; 
        end
    end
    
    predicted4(i)=max;
    if(max==4) count=count+1; end
    
    if(max==1) 
        class1_count=class1_count+1; 
    end
    
    if(max==2) 
        class2_count=class2_count+1; 
    end
    
    if(max==3) 
        class3_count=class3_count+1; 
    end
    
    if(max==4) 
        class4_count=class4_count+1; 
    end
    
    k=1;
end

p1=zeros(class1_count,2);
p2=zeros(class2_count,2);
p3=zeros(class3_count,2);
p4=zeros(class4_count,2);

c1=1; c2=1; c3=1; c4=1;

%For plots
for i=1:Ntest1
   if(predicted1(i)==1)
       p1(c1,1)=x11_test(i);
       p1(c1,2)=x12_test(i);
       c1=c1+1;
   end
   
   if(predicted1(i)==2)
       p2(c2,1)=x11_test(i);
       p2(c2,2)=x12_test(i);
       c2=c2+1;
   end
   
   if(predicted1(i)==3)
       p3(c3,1)=x11_test(i);
       p3(c3,2)=x12_test(i);
       c3=c3+1;
   end
   
   if(predicted1(i)==4)
       p4(c4,1)=x11_test(i);
       p4(c4,2)=x12_test(i);
       c4=c4+1;
   end
end

for i=1:Ntest2
   if(predicted2(i)==1)
       p1(c1,1)=x21_test(i);
       p1(c1,2)=x22_test(i);
       c1=c1+1;
   end
   
   if(predicted2(i)==2)
       p2(c2,1)=x21_test(i);
       p2(c2,2)=x22_test(i);
       c2=c2+1;
   end
   
   if(predicted2(i)==3)
       p3(c3,1)=x21_test(i);
       p3(c3,2)=x22_test(i);
       c3=c3+1;
   end
   
   if(predicted2(i)==4)
       p4(c4,1)=x21_test(i);
       p4(c4,2)=x22_test(i);
       c4=c4+1;
   end
end

for i=1:Ntest3
   if(predicted3(i)==1)
       p1(c1,1)=x31_test(i);
       p1(c1,2)=x32_test(i);
       c1=c1+1;
   end
   
   if(predicted3(i)==2)
       p2(c2,1)=x31_test(i);
       p2(c2,2)=x32_test(i);
       c2=c2+1;
   end
   
   if(predicted3(i)==3)
       p3(c3,1)=x31_test(i);
       p3(c3,2)=x32_test(i);
       c3=c3+1;
   end
   
   if(predicted3(i)==4)
       p4(c4,1)=x31_test(i);
       p4(c4,2)=x32_test(i);
       c4=c4+1;
   end
end

for i=1:Ntest4
   if(predicted4(i)==1)
       p1(c1,1)=x41_test(i);
       p1(c1,2)=x42_test(i);
       c1=c1+1;
   end
   
   if(predicted4(i)==2)
       p2(c2,1)=x41_test(i);
       p2(c2,2)=x42_test(i);
       c2=c2+1;
   end
   
   if(predicted4(i)==3)
       p3(c3,1)=x41_test(i);
       p3(c3,2)=x42_test(i);
       c3=c3+1;
   end
   
   if(predicted4(i)==4)
       p4(c4,1)=x41_test(i);
       p4(c4,2)=x42_test(i);
       c4=c4+1;
   end
end
% 
% hold on
% axis([-6 12 -6 12]);
% scatter(p1(:,1),p1(:,2),'g');
% scatter(p2(:,1),p2(:,2),'b');
% scatter(p3(:,1),p3(:,2),'m');
% scatter(p4(:,1),p4(:,2),'r');
% hold off

%Accuracy for test data

classification_accuracy=double(count)/(Ntest1+Ntest2+Ntest3+Ntest4);
disp('Classification accuracy is:');
disp(classification_accuracy);

%Computing confusion matrix.
confusion_matrix=zeros(4,4);
for i=1:Ntest1
  confusion_matrix(1,predicted1(i))=confusion_matrix(1,predicted1(i))+1;
end

for i=1:Ntest2
  confusion_matrix(2,predicted2(i))=confusion_matrix(2,predicted2(i))+1;
end

for i=1:Ntest3
  confusion_matrix(3,predicted3(i))=confusion_matrix(3,predicted3(i))+1;
end

for i=1:Ntest4
  confusion_matrix(4,predicted4(i))=confusion_matrix(4,predicted4(i))+1;
end

disp('Confusion matrix is:');
disp(confusion_matrix);

