clearvars;

class1_size=50;
class2_size=50;
class3_size=50;

class1_count=0;
class2_count=0;
class3_count=0;

fid=fopen('..\data_assign2_group5\\group5\iris.txt');
X11 = textscan(fid,'%f %f %f %f %s', 150, 'Delimiter',',');
X = cell2mat(X11(:,1:4));

Ntrain1=0.7*50;
Nval1=0.1*50;
Ntest1=0.7*50;

Ntrain2=0.7*50;
Nval2=0.1*50;
Ntest2=0.7*50;

Ntrain3=0.7*50;
Nval3=0.1*50;
Ntest3=0.7*50;

X1_train=X(1:35,:);
x11=X1_train(:,1);
x12=X1_train(:,2);
x13=X1_train(:,3);
x14=X1_train(:,4);

X2_train=X(50:85,:);
x21=X2_train(:,1);
x22=X2_train(:,2);
x23=X2_train(:,3);
x24=X2_train(:,4);

X3_train=X(100:135,:);
x31=X3_train(:,1);
x32=X3_train(:,2);
x33=X3_train(:,3);
x34=X3_train(:,4);

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
X1_test=X1_train;
Xtest1=X1_test;
xtest11=X1_test(:,1);
xtest12=X1_test(:,2);
xtest13=X1_test(:,3);
xtest14=X1_test(:,4);

X2_test=X(85:95,:);
X2_test=X2_train;
Xtest2=X2_test;
xtest21=X2_test(:,1);
xtest22=X2_test(:,2);
xtest23=X2_test(:,3);
xtest24=X2_test(:,4);

X3_test=X(135:145,:);
X3_test=X3_train;
Xtest3=X3_test;
xtest31=X3_test(:,1);
xtest32=X3_test(:,2);
xtest33=X3_test(:,3);
xtest34=X3_test(:,4);

N=Ntrain1+Ntrain2+Ntrain3;

[Ntrain1, NotRequired] = size(x11);
Q1 = 2; % Number of clusters in class 1
Q2 = 1; % Number of clusters in class 2
Q3 = 2; % Number of clusters in class 3
X1 = [x11, x12,x13,x14];
[indices1, mu1] = kmeans(X1,Q1);
Ni1 = zeros(Q1, 1);
omega1 = zeros(Q1, 1);

for i = 1:Q1
    Ni1(i) = sum(indices1 == i);
end

for i = 1:Q1
    omega1(i) = Ni1(i)/Ntrain1;%summation of all omegas is found to be 1
end

clusterpoints1 = cell(1, Q1);
for i = 1:Q1
    %calculates points in cluster
    clusterpoints1(i) = {[x11(indices1 == i), x12(indices1 == i),x13(indices1==i),x14(indices1==i)]};
end

sigma1 = cell(1, Q1);

for i = 1:Q1
    %calculates covariances for each clusters
    points1 = clusterpoints1{1, i};
    temp1 = points1- mu1(i, :);
    sigma1(i) = { (1/(Ni1(i)-1))*(temp1' * temp1)};
end



L_new = 0;

for i = 1:Ntrain1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points1 = (X1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points1) * ((current_cov) \ (points1)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        L_new = L_new + log(temp_L);
        
end

L_old = -10;
%(abs(L_new - L_old)>0.05)
count = 0;
while (abs(L_new - L_old)>0.05)
    L_old = L_new;
    %To find gamma
    gamma1 = zeros(Ntrain1, Q1);
    gammasum1 = zeros(Q1, 1);
    % %disp(clusterpoints1{1,1}-mu1(1,:));

    for i = 1:Ntrain1
        %calculates gaussians for each clusters
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points1 = (X1(i,:))- current_mu;
            gamma1(i, j) = (omega1(j)/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points1) * ((current_cov) \(points1)'));
            
        end
        
    end
    gammasum1 = sum(gamma1, 2);
    for i = 1:Q1
        %normalizes gammas
        for j = 1:Ntrain1
            gamma1(j, i) = gamma1(j, i)/gammasum1(j);
        end
    end
    Ni1 = (sum(gamma1, 1))';
    for i = 1:Q1
        mu1(i,:) = (1/Ni1(i))*gamma1(:, i)' * X1;
        sigma1(i) = {  (1/Ni1(i))*((X1-mu1(i, :))' *  diag(gamma1(:, i)) * (X1-mu1(i, :))) };  
        omega1(i) = Ni1(i) / Ntrain1;
    end
    
    L_new = 0;

    for i = 1:Ntrain1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points1 = (X1(i,:))- current_mu;
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points1) * ((current_cov) \ (points1)'));
            temp_L = temp_L + omega1(j) * gauss;
        end
        L_new = L_new + log(temp_L);
    end
    
end
%disp(L_new);







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ntrain2, NotRequired] = size(x21);
X2 = [x21,x22,x23,x24];
[indices2, mu2] = kmeans(X2,Q2);
Ni2 = zeros(Q2, 1);
omega2 = zeros(Q2, 1);

for i = 1:Q2
    Ni2(i) = sum(indices2 == i);
end

for i = 1:Q2
    omega2(i) = Ni2(i)/Ntrain2;%summation of all omegas is found to be 1
end

clusterpoints2 = cell(1, Q2);
for i = 1:Q2
    %calculates points in cluster
    clusterpoints2(i) = {[x21(indices1 == i), x22(indices1 == i),x23(indices1==i),x24(indices1==i)]};
end

sigma2 = cell(1, Q2);

for i = 1:Q2
    %calculates covariances for each clusters
    points2 = clusterpoints2{1, i};
    temp2 = points2- mu2(i, :);
    sigma2(i) = { (1/(Ni2(i)-1))*(temp2' * temp2)};
end



L_new = 0;

for i = 1:Ntrain2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points2 = (X2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points2) * ((current_cov) \ (points2)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        L_new = L_new + log(temp_L);
        
end

L_old = -10;
%(abs(L_new - L_old)>0.05)
count = 0;
while (abs(L_new - L_old)>0.05)
    L_old = L_new;
    %To find gamma
    gamma2 = zeros(Ntrain2, Q2);
    gammasum2 = zeros(Q2, 1);
    % %disp(clusterpoints1{1,1}-mu1(1,:));

    for i = 1:Ntrain2
        %calculates gaussians for each clusters
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points2 = (X2(i,:))- current_mu;
            gamma2(i, j) = (omega2(j)/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points2) * ((current_cov) \(points2)'));
            
        end
        
    end
    gammasum2 = sum(gamma2, 2);
    for i = 1:Q2
        %normalizes gammas
        for j = 1:Ntrain2
            gamma2(j, i) = gamma2(j, i)/gammasum2(j);
        end
    end
    Ni2 = (sum(gamma2, 1))';
    for i = 1:Q2
        mu2(i,:) = (1/Ni2(i))*gamma2(:, i)' * X2;
        sigma2(i) = {  (1/Ni2(i))*((X2-mu2(i, :))' *  diag(gamma2(:, i)) * (X2-mu2(i, :))) };  
        omega2(i) = Ni2(i) / Ntrain2;
    end
    
    L_new = 0;

    for i = 1:Ntrain2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points2 = (X2(i,:))- current_mu;
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points2) * ((current_cov) \ (points2)'));
            temp_L = temp_L + omega2(j) * gauss;
        end
        L_new = L_new + log(temp_L);
    end
    
end
%disp(L_new);




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ntrain3, NotRequired] = size(x31);

X3 = [x31,x32,x33,x34];
[indices3, mu3] = kmeans(X3,Q3);
Ni3 = zeros(Q3, 1);
omega3 = zeros(Q3, 1);

for i = 1:Q3
    Ni3(i) = sum(indices3 == i);
end

for i = 1:Q3
    omega3(i) = Ni3(i)/Ntrain3;%summation of all omegas is found to be 1
end

clusterpoints3 = cell(1, Q3);
for i = 1:Q3
    %calculates points in cluster
    clusterpoints3(i) = {[x31(indices1 == i), x32(indices1 == i),x33(indices1==i),x34(indices1==i)]};
end

sigma3 = cell(1, Q3);

for i = 1:Q3
    %calculates covariances for each clusters
    points3 = clusterpoints3{1, i};
    temp3 = points3- mu3(i, :);
    sigma3(i) = { (1/(Ni3(i)-1))*(temp3' * temp3)};
end



L_new = 0;

for i = 1:Ntrain3
        temp_L=0;
        for j = 1:Q3
            current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points3 = (X3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points3) * ((current_cov) \ (points3)'));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        L_new = L_new + log(temp_L);
        
end

L_old = -10;
%(abs(L_new - L_old)>0.05)
count = 0;
while (abs(L_new - L_old)>0.05)
    L_old = L_new;
    %To find gamma
    gamma3 = zeros(Ntrain3, Q3);
    gammasum3 = zeros(Q3, 1);
    % %disp(clusterpoints1{1,1}-mu1(1,:));

    for i = 1:Ntrain3
        %calculates gaussians for each clusters
        for j = 1:Q3
            current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points3 = (X3(i,:))- current_mu;
            gamma3(i, j) = (omega3(j)/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points3) * ((current_cov) \(points3)'));
            
        end
        
    end
    gammasum3 = sum(gamma3, 2);
    for i = 1:Q3
        %normalizes gammas
        for j = 1:Ntrain3
            gamma3(j, i) = gamma3(j, i)/gammasum3(j);
        end
    end
    Ni3 = (sum(gamma3, 1))';
    for i = 1:Q3
        mu3(i,:) = (1/Ni3(i))*gamma3(:, i)' * X3;
        sigma3(i) = {  (1/Ni3(i))*((X3-mu3(i, :))' *  diag(gamma3(:, i)) * (X3-mu3(i, :))) };  
        omega3(i) = Ni3(i) / Ntrain3;
    end
    
    L_new = 0;

    for i = 1:Ntrain3
        temp_L=0;
        for j = 1:Q3
            current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points3 = (X3(i,:))- current_mu;
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points3) * ((current_cov) \ (points3)'));
            temp_L = temp_L + omega3(j) * gauss;
        end
        L_new = L_new + log(temp_L);
    end
    
end
%disp(L_new);


%validation to find parameters
[Nval1, NotRequired] = size(xval11);
Xval1 = [xval11, xval12,xval13,xval14];

Lval = 0;

for i = 1:Nval1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points1 = (Xval1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points1) * ((current_cov) \ (points1)')));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Lval = Lval + log(temp_L);
end
%disp(Lval);
% scatter(x11, x12);
% hold on
% scatter(xval11, xval12, 'r');
% hold off

%validation to find parameters
[Nval2, NotRequired] = size(xval21);
Xval2 = [xval21, xval22,xval23,xval24];

Lval = 0;

for i = 1:Nval2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points2 = (Xval2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points2) * ((current_cov) \ (points2)')));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Lval = Lval + log(temp_L);
end
%disp(Lval);
% scatter(x21, x22);
% hold on
% scatter(xval21, xval22, 'r');
% hold off

%validation to find parameters

[Nval3, NotRequired] = size(xval31);
Xval3 = [xval31, xval32,xval33,xval34];

Lval = 0;

for i = 1:Nval3
        temp_L=0;
        for j = 1:Q3
            current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points3 = (Xval3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points3) * ((current_cov) \ (points3)')));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Lval = Lval + log(temp_L);
end

%%%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TEST-DATA_ACCURACY%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%class1
Ldata = zeros(Ntest1, 3);
data_class = zeros(Ntest1, 1);
count =0;
for i = 1:Ntest1
        %for class 1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points = (Xtest1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points = (Xtest1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
       
        %find class corresponding to maximum value
        [dontcare, data_class(i)] = max(Ldata(i,:));
        predicted1(i,1)=data_class(i);
        
         mxx=predicted1(i,1);
    
    if(mxx==1) count=count+1; end
        
    if(mxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(mxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(mxx==3) 
        class3_count=class3_count+1; 
    end
end

%class2
Ldata = zeros(Ntest2, 3);
data_class = zeros(Ntest2, 1);
for i = 1:Ntest2
        %for class 1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
       
        %find class corresponding to maximum value
        [dontcare, data_class(i)] = max(Ldata(i,:));
        predicted2(i,1)=data_class(i);
        
         mxx=predicted2(i,1);
    
    if(mxx==2) count=count+1; end
        
    if(mxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(mxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(mxx==3) 
        class3_count=class3_count+1; 
    end
end

%class3
Ldata = zeros(Ntest3, 3);
data_class = zeros(Ntest3, 1);
for i = 1:Ntest3
        %for class 1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points = (Xtest3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points = (Xtest3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov)))* exp((-0.5)*(points) * ((current_cov) \ (points)')));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
       
        %find class corresponding to maximum value
        [dontcare, data_class(i)] = max(Ldata(i,:));
        predicted3(i,1)=data_class(i);
        
         mxx=predicted3(i,1);
    
    if(mxx==3) count=count+1; end
        
    if(mxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(mxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(mxx==3) 
        class3_count=class3_count+1; 
    end
end



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