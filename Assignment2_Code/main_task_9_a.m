clearvars;

class1_size=50;
class2_size=50;
class3_size=50;

fid=fopen('..\data_assign2_group5\group5\iris.txt');
X1 = textscan(fid,'%f %f %f %f %s', 150, 'Delimiter',',');
X = cell2mat(X1(:,1:4));

Ntrain1=0.7*50;
Nval1=0.1*50;
Ntest1=0.1*50;

Ntrain2=0.7*50;
Nval2=0.1*50;
Ntest2=0.1*50;

Ntrain3=0.7*50;
Nval3=0.1*50;
Ntest3=0.1*50;

X1_train=X(1:35,:);
x11_train=X1_train(:,1);
x12_train=X1_train(:,2);
x13_train=X1_train(:,3);
x14_train=X1_train(:,4);

X2_train=X(50:85,:);
x21_train=X2_train(:,1);
x22_train=X2_train(:,2);
x23_train=X2_train(:,3);
x24_train=X2_train(:,4);

X3_train=X(100:135,:);
x31_train=X3_train(:,1);
x32_train=X3_train(:,2);
x33_train=X3_train(:,3);
x34_train=X3_train(:,4);


X1_val=X(45:50,:);
xval11=X1_val(:,1);
xval12=X1_val(:,2);
xval13=X1_val(:,3);
xval14=X1_val(:,4);

X2_val=X(95:100,:);
xval21=X2_val(:,1);
xval22=X2_val(:,2);
xval23=X2_val(:,3);
xval24=X2_val(:,4);

X3_val=X(145:150,:);
xval31=X3_val(:,1);
xval32=X3_val(:,2);
xval33=X3_val(:,3);
xval34=X3_val(:,4);

X1_test=X(35:45,:);
X1_test=X1_val;
x11_test=X1_test(:,1);
x12_test=X1_test(:,2);
x13_test=X1_test(:,3);
x14_test=X1_test(:,4);

X2_test=X(85:95,:);
X2_test=X2_val;
x21_test=X2_test(:,1);
x22_test=X2_test(:,2);
x23_test=X2_test(:,3);
x24_test=X2_test(:,4);

X3_test=X(135:145,:);
X3_test=X3_val;
x31_test=X3_test(:,1);
x32_test=X3_test(:,2);
x33_test=X3_test(:,3);
x34_test=X3_test(:,4);

N=Ntrain1+Ntrain2+Ntrain3;

class1_count=0;
class2_count=0;
class3_count=0;
count=0;

predicted1=zeros(Ntest1,1);
predicted2=zeros(Ntest2,1);
predicted3=zeros(Ntest3,1);

q=3; %Hyper-parameter.
dist1=zeros(N,2);

%For class 1
dist1=zeros(N,2);
Radius = Inf(3,1);

for i=1:Ntest1
    k=1;
    freq1=zeros(4,1);
    for j=1:Ntrain1
      dist1(k,1)=(x11_train(j)-x11_test(i))^2 + (x12_train(j)-x12_test(i))^2+(x13_train(j)-x13_test(i))^2 + (x14_train(j)-x14_test(i))^2;
      dist1(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist1(k,1)=(x21_train(j)-x11_test(i))^2 + (x22_train(j)-x12_test(i))^2 + (x23_train(j)-x13_test(i))^2 + (x24_train(j)-x14_test(i))^2;
      dist1(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist1(k,1)=(x31_train(j)-x11_test(i))^2 + (x32_train(j)-x12_test(i))^2 + (x33_train(j)-x13_test(i))^2 + (x34_train(j)-x14_test(i))^2;
      dist1(k,2)=3;
      k=k+1;
    end
  
    dist1=sortrows(dist1,1);
    
     %class i q nearest neighbours
    for classi = 1:3
        count1 = 0;
        f_ind = -1;
        z = 1;
        while (count1 < q && z < N)
           
            if dist1(z, 2) == classi
                
                count1 = count1 + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count1 == q)
            Radius(classi) = dist1(f_ind, 1);
        end
    end
    [mvv, predicted1(i,1)] = min(Radius);
    
    max=predicted1(i,1);
    
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

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Radius = Inf(3,1);

for i=1:Ntest2
   k=1;
    for j=1:Ntrain1
      dist1(k,1)=(x11_train(j)-x21_test(i))^2 + (x12_train(j)-x22_test(i))^2+(x13_train(j)-x23_test(i))^2+(x14_train(j)-x24_test(i))^2;
      dist1(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist1(k,1)=(x21_train(j)-x21_test(i))^2 + (x22_train(j)-x22_test(i))^2+(x23_train(j)-x23_test(i))^2+(x24_train(j)-x24_test(i))^2;
      dist1(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist1(k,1)=(x31_train(j)-x21_test(i))^2 + (x32_train(j)-x22_test(i))^2+(x33_train(j)-x23_test(i))^2+(x34_train(j)-x24_test(i))^2;
      dist1(k,2)=3;
      k=k+1;
    end
  
    %disp(dist1);
    dist1=sortrows(dist1,1);
    
     %class i q nearest neighbours
    for classi = 1:3
        count1 = 0;
        f_ind = -1;
        z = 1;
        while (count1 < q && z < N)
           
            if dist1(z, 2) == classi
                
                count1 = count1 + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count1 == q)
            Radius(classi) = dist1(f_ind, 1);
        end
    end
    [mvv, predicted2(i,1)] = min(Radius);
    
    max=predicted2(i,1);
    
    if(max==2) count=count+1; end
        
    if(max==1) 
        class1_count=class1_count+1; 
    end
    
    if(max==2) 
        class2_count=class2_count+1; 
    end
    
    if(max==3) 
        class3_count=class3_count+1; 
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class 3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Radius = Inf(3,1);

for i=1:Ntest3
   k=1;
    for j=1:Ntrain1
      dist1(k,1)=(x11_train(j)-x31_test(i))^2 + (x12_train(j)-x32_test(i))^2 + (x13_train(j)-x33_test(i))^2 + (x14_train(j)-x34_test(i))^2;
      dist1(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist1(k,1)=(x21_train(j)-x31_test(i))^2 + (x22_train(j)-x32_test(i))^2 + (x23_train(j)-x33_test(i))^2 + (x24_train(j)-x34_test(i))^2;
      dist1(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist1(k,1)=(x31_train(j)-x31_test(i))^2 + (x32_train(j)-x32_test(i))^2 + (x33_train(j)-x33_test(i))^2 + (x34_train(j)-x34_test(i))^2;
      dist1(k,2)=3;
      k=k+1;
    end

    %disp(dist1);
    dist1=sortrows(dist1,1);
    
     %class i q nearest neighbours
    for classi = 1:3
        count1 = 0;
        f_ind = -1;
        z = 1;
        while (count1 < q && z < N)
           
            if dist1(z, 2) == classi
                
                count1 = count1 + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count1 == q)
            Radius(classi) = dist1(f_ind, 1);
        end
    end
    
    [mvv, predicted3(i,1)] = min(Radius);
     max=predicted3(i,1);
    
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

     
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class 4%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%disp(Radius);

c1=1; c2=1; c3=1;

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
end

hold on
scatter(p1(:,1),p1(:,2),'g');
scatter(p2(:,1),p2(:,2),'b');
scatter(p3(:,1),p3(:,2),'m');
hold off

 %Accuracy for test data
 
 classification_accuracy=double(count)/(Ntest1+Ntest2+Ntest3);
 disp('Classification accuracy is:');
 disp(classification_accuracy);
 
 %Computing confusion matrix.
confusion_matrix=zeros(3,3);
for i=1:Ntest1
  confusion_matrix(1,predicted1(i))=confusion_matrix(1,predicted1(i))+1;
end

for i=1:Ntest2
  confusion_matrix(2,predicted2(i))=confusion_matrix(2,predicted2(i))+1;
end

for i=1:Ntest3
  confusion_matrix(3,predicted3(i))=confusion_matrix(3,predicted3(i))+1;
end

disp('Confusion matrix is:');
disp(confusion_matrix);





