clearvars;
%x12 is 2nd column of 1st class.
[x11_train,x12_train]=textread('..\data_assign2_group5\group5\linearly_separable\class1_train.txt','%f %f');
[x11_test,x12_test]=textread('..\data_assign2_group5\group5\linearly_separable\class1_test.txt','%f %f');
[x11_val,x12_val]=textread('..\data_assign2_group5\group5\linearly_separable\class1_val.txt','%f %f');

[x21_train,x22_train]=textread('..\data_assign2_group5\group5\linearly_separable\class2_train.txt','%f %f');
[x21_test,x22_test]=textread('..\data_assign2_group5\group5\linearly_separable\class2_test.txt','%f %f');
[x21_val,x22_val]=textread('..\data_assign2_group5\group5\linearly_separable\class2_val.txt','%f %f');

[x31_train,x32_train]=textread('..\data_assign2_group5\group5\linearly_separable\class3_train.txt','%f %f');
[x31_test,x32_test]=textread('..\data_assign2_group5\group5\linearly_separable\class3_test.txt','%f %f');
[x31_val,x32_val]=textread('..\data_assign2_group5\group5\linearly_separable\class3_val.txt','%f %f');

[x41_train,x42_train]=textread('..\data_assign2_group5\group5\linearly_separable\class4_train.txt','%f %f');
[x41_test,x42_test]=textread('..\data_assign2_group5\group5\linearly_separable\class4_test.txt','%f %f');
[x41_val,x42_val]=textread('..\data_assign2_group5\group5\linearly_separable\class4_val.txt','%f %f');

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
count2=0;

predicted1=zeros(Ntest1,1);
predicted2=zeros(Ntest2,1);
predicted3=zeros(Ntest3,1);
predicted4=zeros(Ntest4,1);
q=3; %Hyper-parameter.
dist1=zeros(N,2);
%For class 1


dist1=zeros(N,2);
Radius = Inf(4,1);
for i=1:Ntest1
    k=1;
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
    
     %class i q nearest neighbours
    for classi = 1:4
        count = 0;
        f_ind = -1;
        z = 1;
        while (count < q && z < N)
           
            if dist1(z, 2) == classi
                
                count = count + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count == q)
            Radius(classi) = dist1(f_ind, 1);
        end
    end
    [mvv, predicted1(i,1)] = min(Radius);
    
    maxx=predicted1(i);
    if(maxx==1) count2=count2+1; end
        
    if(maxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(maxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(maxx==3) 
        class3_count=class3_count+1; 
    end
    
    if(maxx==4) 
        class4_count=class4_count+1; 
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Radius = Inf(4,1);

for i=1:Ntest2
   k=1;
    for j=1:Ntrain1
      dist1(k,1)=(x11_train(j)-x21_test(i))^2 + (x12_train(j)-x22_test(i))^2;
      dist1(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist1(k,1)=(x21_train(j)-x21_test(i))^2 + (x22_train(j)-x22_test(i))^2;
      dist1(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist1(k,1)=(x31_train(j)-x21_test(i))^2 + (x32_train(j)-x22_test(i))^2;
      dist1(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist1(k,1)=(x41_train(j)-x21_test(i))^2 + (x42_train(j)-x22_test(i))^2;
      dist1(k,2)=4;
      k=k+1;
    end
    %disp(dist1);
    dist1=sortrows(dist1,1);
    
     %class i q nearest neighbours
    for classi = 1:4
        count = 0;
        f_ind = -1;
        z = 1;
        while (count < q && z < N)
           
            if dist1(z, 2) == classi
                
                count = count + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count == q)
            Radius(classi) = dist1(f_ind, 1);
        end
    end
    [mvv, predicted2(i,1)] = min(Radius);
    
    maxx=predicted2(i);
    
    if(maxx==2) count2=count2+1; end
        
    if(maxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(maxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(maxx==3) 
        class3_count=class3_count+1; 
    end
    
    if(maxx==4) 
        class4_count=class4_count+1; 
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class 3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Radius = Inf(4,1);

for i=1:Ntest3
   k=1;
    for j=1:Ntrain1
      dist1(k,1)=(x11_train(j)-x31_test(i))^2 + (x12_train(j)-x32_test(i))^2;
      dist1(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist1(k,1)=(x21_train(j)-x31_test(i))^2 + (x22_train(j)-x32_test(i))^2;
      dist1(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist1(k,1)=(x31_train(j)-x31_test(i))^2 + (x32_train(j)-x32_test(i))^2;
      dist1(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist1(k,1)=(x41_train(j)-x31_test(i))^2 + (x42_train(j)-x32_test(i))^2;
      dist1(k,2)=4;
      k=k+1;
    end
    %disp(dist1);
    dist1=sortrows(dist1,1);
    
     %class i q nearest neighbours
    for classi = 1:4
        count = 0;
        f_ind = -1;
        z = 1;
        while (count < q && z < N)
           
            if dist1(z, 2) == classi
                
                count = count + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count == q)
            Radius(classi) = dist1(f_ind, 1);
        end
    end
    [mvv, predicted3(i,1)] = min(Radius);
    
    maxx=predicted3(i);
    if(maxx==3) count2=count2+1; end
        
    if(maxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(maxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(maxx==3) 
        class3_count=class3_count+1; 
    end
    
    if(maxx==4) 
        class4_count=class4_count+1; 
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class 4%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%disp(Radius);

for i=1:Ntest4
    k=1;
    Radius = Inf(4,1);
    for j=1:Ntrain1
      dist1(k,1)=(x11_train(j)-x41_test(i))^2 + (x12_train(j)-x42_test(i))^2;
      dist1(k,2)=1;
      k=k+1;
    end
    
    for j=1:Ntrain2
      dist1(k,1)=(x21_train(j)-x41_test(i))^2 + (x22_train(j)-x42_test(i))^2;
      dist1(k,2)=2;
      k=k+1;
    end
    
    for j=1:Ntrain3
      dist1(k,1)=(x31_train(j)-x41_test(i))^2 + (x32_train(j)-x42_test(i))^2;
      dist1(k,2)=3;
      k=k+1;
    end
  
    for j=1:Ntrain4
      dist1(k,1)=(x41_train(j)-x41_test(i))^2 + (x42_train(j)-x42_test(i))^2;
      dist1(k,2)=4;
      k=k+1;
    end
    %disp(dist1);
    dist1=sortrows(dist1,1);
    
     %class i q nearest neighbours
    for classi = 1:4
        count = 0;
        f_ind = -1;
        z = 1;
        while (count < q && z < N)
           
            if dist1(z, 2) == classi
                
                count = count + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count == q)
            Radius(classi) = dist1(f_ind, 1);
        end
    end
    %disp(Radius);
    [mvv, predicted4(i,1)] = min(Radius);
    maxx=predicted4(i);
    
    if(maxx==4) count2=count2+1; end
        
    if(maxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(maxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(maxx==3) 
        class3_count=class3_count+1; 
    end
    
    if(maxx==4) 
        class4_count=class4_count+1; 
    end
    
end








% % scatter(x41_test, x42_test , 'm');
% % hold on
% % scatter(x31_train, x32_train , 'b');
% % 
% % scatter(x31_test, x32_test , 'g');
% % scatter(x21_test, x22_test , 'r');
% % scatter(x11_test, x12_test , 'y');
% % hold off
% % hold off
% % hold off

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

% hold on
% scatter(p1(:,1),p1(:,2),'g');
% scatter(p2(:,1),p2(:,2),'b');
% scatter(p3(:,1),p3(:,2),'m');
% scatter(p4(:,1),p4(:,2),'r');
% hold off

%Accuracy for test data

classification_accuracy=double(count2)/(Ntest1+Ntest2+Ntest3+Ntest4);
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


%%%%%%%%%%%%%%%%%%%%%%%%%PLOTS for test
%%%%%%%%%%%%%%%%%%%%%%%%%data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

q=3; %Hyper-parameter.

for i=1:Ntest
    k=1;
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
    
     %class i q nearest neighbours
    for classi = 1:4
        count = 0;
        f_ind = -1;
        z = 1;
        while (count < q && z < N)
           
            if dist(z, 2) == classi
                
                count = count + 1;
                f_ind = z;
            end
            z = z + 1;
        end
        
        if(count == q)
            Radius(classi) = dist(f_ind, 1);
        end
    end
    [mvv, predicted(i,1)] = min(Radius);
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



