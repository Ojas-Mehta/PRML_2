clearvars;

gorilla_size=221;
billiards_size=278;
grapes_size=201;
ladder_size=242;
faces_size=435;

gorilla_train=1;
gorilla_test=170;
gorilla_val=149;

billiards_train=1;
billiards_test=223;
billiards_val=195;

grapes_train=1;
grapes_test=162;
grapes_val=141;

ladder_train=1;
ladder_test=195;
ladder_val=170;

faces_train=1;
faces_test=195;
faces_val=305;

class1=[];
gorilla_string='..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_train\');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(gorilla_string,FileNames);
   temp1=importdata(temp);
   class1=[class1;temp1];
end

class2=[];
billiard_string='..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_train\');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(billiard_string,FileNames);
   temp2=importdata(temp);
   class2=[class2;temp2];
end

class3=[];
grapes_string='..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_train\');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(grapes_string,FileNames);
   temp1=importdata(temp);
   class3=[class3;temp1];
end

class4=[];
ladder_string='..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_train\');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(ladder_string,FileNames);
   temp1=importdata(temp);
   class4=[class4;temp1];
end

class5=[];
faces_string='..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_train\');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(faces_string,FileNames);
   temp1=importdata(temp);
   class5=[class5;temp1];
end

X1_train=class1;
X2_train=class2;
X3_train=class3;
X4_train=class4;
X5_train=class5;

Ntrain1=length(X1_train);
Ntrain2=length(X2_train);
Ntrain3=length(X3_train);
Ntrain4=length(X4_train);
Ntrain5=length(X5_train);

N=Ntrain1+Ntrain2+Ntrain3+Ntrain4+Ntrain5;

class1_count=0;
class2_count=0;
class3_count=0;
class4_count=0;
class5_count=0;
count2=0;

q=7; %Hyper-parameter.

%Class-1 Testing
gorilla_string='..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_train\');
ind=1;
predicted1=zeros(length(Files)-2,1);
Ntest1=length(Files);
for p=3:140
   Radius = ones(5,1);
   FileNames=Files(p).name;
   temp=strcat(gorilla_string,FileNames);
   X1_test=importdata(temp);
   
   %Class 1
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain1,2);
    m=1;
    for j=1:Ntrain1
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X1_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
    temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(1,1)=(Radius(1,1)*r);
  end
    
  %Class 2
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain2,2);
    m=1;
    for j=1:Ntrain2
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X2_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=2;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(2,1)=(Radius(2,1)*r);
  end
  
  %Class 3
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain3,2);
    m=1;
    for j=1:Ntrain3
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X3_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=3;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(3,1)=(Radius(3,1)*r);
  end
  
  %Class 4
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain4,2);
    m=1;
    for j=1:Ntrain4
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X4_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=4;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(4,1)=(Radius(4,1)*r);
  end
  
  %Class 5
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain5,2);
    m=1;
    for j=1:Ntrain5
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X5_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=5;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(5,1)=(Radius(5,1)*r);
  end
  
  [mvv,predicted1(ind,1)]=min(Radius);
  if(predicted1(ind,1)==1) count2=count2+1; end
  ind=ind+1;
end
disp('Class 1 over');

q=7;
%Class-2 Testing
billiards_string='..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_train\');
ind=1;
predicted2=zeros(length(Files)-2,1);
Ntest2=length(Files);

for p=3:100
   Radius = ones(5,1);
   FileNames=Files(p).name;
   temp=strcat(billiards_string,FileNames);
   X1_test=importdata(temp);
   
   %Class 1
  for i=1:size(X1_test,1)
   dist1=zeros(Ntrain1,2);
    m=1;
    for j=1:Ntrain1
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X1_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
    temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(1,1)=(Radius(1,1)*r);
  end
    
  %Class 2
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain2,2);
    m=1;
    for j=1:Ntrain2
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X2_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=2;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(2,1)=(Radius(2,1)*r);
  end
  
  %Class 3
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain3,2);
     m=1;
    for j=1:Ntrain3
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X3_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=3;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(3,1)=(Radius(3,1)*r);
  end
  
  %Class 4
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain4,2);
    m=1;
    for j=1:Ntrain4
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X4_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=4;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(4,1)=(Radius(4,1)*r);
  end
  
  %Class 5
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain5,2);
    m=1;
    for j=1:Ntrain5
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X5_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=5;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(5,1)=(Radius(5,1)*r);
  end
  
  [mvv,predicted2(ind,1)]=min(Radius);
  if(predicted2(ind,1)==2) count2=count2+1; 
  end
  ind=ind+1;
end
disp('Class 2 over');

q=7;
%Class-3 Testing
grapes_string='..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_train\');
ind=1;
predicted3=zeros(length(Files)-2,1);
Ntest3=length(Files);

for p=3:130
   Radius = ones(5,1);
   FileNames=Files(p).name;
   temp=strcat(grapes_string,FileNames);
   X1_test=importdata(temp);
   
   %Class 1   
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain1,2);
     m=1;
    for j=1:Ntrain1
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X1_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
    temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(1,1)=(Radius(1,1)*r);
  end
    
  %Class 2
  
  for i=1:size(X1_test,1)
  dist1=zeros(Ntrain2,2);     
   m=1;
    for j=1:Ntrain2
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X2_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(2,1)=(Radius(2,1)*r);
  end
  
  %Class 3
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain3,2);
     m=1;
    for j=1:Ntrain3
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X3_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(3,1)=(Radius(3,1)*r);
  end
  
  %Class 4
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain4,2);
     m=1;
    for j=1:Ntrain4
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X4_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(4,1)=(Radius(4,1)*r);
  end
  
  %Class 5
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain5,2);
     m=1;
    for j=1:Ntrain5
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X5_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(5,1)=(Radius(5,1)*r);
  end
  
  [mvv,predicted3(ind,1)]=min(Radius);
  if(predicted3(ind,1)==3) count2=count2+1; end
  ind=ind+1;
end
disp('Class 3 over');

q=7;
%Class-4 Testing
ladder_string='..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_train\');
ind=1;
predicted4=zeros(length(Files)-2,1);
Ntest4=length(Files);
for p=3:150
   Radius = ones(5,1);
   FileNames=Files(p).name;
   temp=strcat(ladder_string,FileNames);
   X1_test=importdata(temp);
   
   %Class 1
   
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain1,2);
    m=1;
    for j=1:Ntrain1
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X1_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
    temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(1,1)=(Radius(1,1)*r);
  end
    
  %Class 2
  
  for i=1:size(X1_test,1)
     dist1=zeros(Ntrain2,2);
    m=1;
    for j=1:Ntrain2
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X2_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(2,1)=(Radius(2,1)*r);
  end
  
  %Class 3
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain3,2);
    m=1;
    for j=1:Ntrain3
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X3_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(3,1)=(Radius(3,1)*r);
  end
  
  %Class 4
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain4,2);
    m=1;
    for j=1:Ntrain4
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X4_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(4,1)=(Radius(4,1)*r);
  end
  
  %Class 5
 
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain5,2);
    m=1;
    for j=1:Ntrain5
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X5_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(5,1)=(Radius(5,1)*r);
  end
  
  [mvv,predicted4(ind,1)]=min(Radius);
  if(predicted4(ind,1)==4) count2=count2+1; end
  ind=ind+1;
end
disp('Class 4 over');

q=7;
%Class-5 Testing
faces_string='..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_train\');
ind=1;
predicted5=zeros(length(Files)-2,1);
Ntest5=length(Files);
for p=3:160
   Radius = ones(5,1);
   FileNames=Files(p).name;
   temp=strcat(faces_string,FileNames);
   X1_test=importdata(temp);
   
   %Class 1
   
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain1,2);
    m=1;
    for j=1:Ntrain1
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X1_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
    temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(1,1)=(Radius(1,1)*r);
  end
    
  %Class 2
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain2,2);
     m=1;
    for j=1:Ntrain2
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X2_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(2,1)=(Radius(2,1)*r);
  end
  
  %Class 3
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain3,2);
    m=1;
    for j=1:Ntrain3
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X3_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(3,1)=(Radius(3,1)*r);
  end
  
  %Class 4
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain4,2);
    m=1;
    for j=1:Ntrain4
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X4_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(4,1)=(Radius(4,1)*r);
  end
  
  %Class 5
  
  for i=1:size(X1_test,1)
    dist1=zeros(Ntrain5,2);
    m=1;
    for j=1:Ntrain5
      for k=1:64
          dist1(m,1)= dist1(m,1)+(X5_train(j,k)-X1_test(i,k))^2;
      end
      dist1(m,2)=1;
      m=m+1;
    end
    dist1=sortrows(dist1,1);
    %%k-nearest neighbour.
     temp=dist1(1:q,1);
    [r,mvv]=max(temp);
    Radius(5,1)=(Radius(5,1)*r);
  end
  
  [mvv,predicted5(ind,1)]=min(Radius);
  if(predicted5(ind,1)==5) count2=count2+1; end;
  ind=ind+1;
end
disp('Class 5 over');

%Accuracy for test data

classification_accuracy=double(count2)/(Ntest1+Ntest2+Ntest3+Ntest4+Ntest5);
disp('Classification accuracy is:');
disp(classification_accuracy);

%Computing confusion matrix.
confusion_matrix=zeros(5,5);
Ntest1=Ntest1-2;
Ntest2=Ntest2-2;
Ntest3=Ntest3-2;
Ntest4=Ntest4-2;
Ntest5=Ntest5-2;

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

for i=1:Ntest5
  confusion_matrix(5,predicted5(i))=confusion_matrix(5,predicted5(i))+1;
end
disp('Confusion matrix is:');
disp(confusion_matrix);








% class4=cell(3000,64,ladder_size);
% ladder_string='..\data_assign2_group5\features_SURF\feats_surf\126.ladder\';
% Files=dir('..\data_assign2_group5\features_SURF\feats_surf\126.ladder\');
% for k=3:8
%    FileNames=Files(k).name;
%    temp=strcat(ladder_string,FileNames);
%    b=importdata(temp);
%    for i=1:size(b,1)
%      for j=1:size(b,2)
%        class4(i,j,k-2)=num2cell(b(i,j));l
%      end
%    end
% end
%disp(class4(2,2,2));
