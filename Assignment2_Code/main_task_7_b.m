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
%length(Files);
length_Files = 3;
class1=[];
gorilla_string='..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_train\');
for k=3:length_Files
   FileNames=Files(k).name;
   temp=strcat(gorilla_string,FileNames);
   temp111=importdata(temp);
   class1=[class1;temp111];
end

class2=[];
billiard_string='..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_train\');
for k=3:length_Files
   FileNames=Files(k).name;
   temp=strcat(billiard_string,FileNames);
   temp112=importdata(temp);
   class2=[class2;temp112];
end

class3=[];
grapes_string='..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_train\');
for k=3:length_Files
   FileNames=Files(k).name;
   temp=strcat(grapes_string,FileNames);
   temp113=importdata(temp);
   class3=[class3;temp113];
end

class4=[];
ladder_string='..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_train\');
for k=3:length_Files
   FileNames=Files(k).name;
   temp=strcat(ladder_string,FileNames);
   temp114=importdata(temp);
   class4=[class4;temp114];
end

class5=[];
faces_string='..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_train\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_train\');
for k=3:length_Files
   FileNames=Files(k).name;
   temp=strcat(faces_string,FileNames);
   temp115=importdata(temp);
   class5=[class5;temp115];
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

prior1 = Ntrain1 / N;
prior2 = Ntrain2 / N;
prior3 = Ntrain3 / N;
prior4 = Ntrain4 / N;
prior5 = Ntrain5 / N;

class1_count=0;
class2_count=0;
class3_count=0;
class4_count=0;
class5_count=0;
count2=0;



Q1 = 4; % Number of clusters in class 1
Q2 = 5; % Number of clusters in class 2
Q3 = 5; % Number of clusters in class 3
Q4 = 5; % Number of clusters in class 4
Q5 = 4; % Number of clusters in class 5
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class 1 training%%%%%%%%%%%%
X1 = X1_train;
%X1 = [x11, x12];
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
  
    clusterpoints1(i)={X1(indices1==i,:)};
end

sigma1 = cell(1, Q1);

for i = 1:Q1
    %calculates covariances for each clusters
    points1 = clusterpoints1{1, i};
    temp1 = points1- mu1(i, :);
    sigma1(i) = {diag(diag((1/(Ni1(i)-1))*(temp1' * temp1)))};
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
%(abs(L_new - L_old)>1)
count = 0;
while (abs(L_new - L_old)>1)
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
        sigma1(i) = { diag(diag((1/Ni1(i))*((X1-mu1(i, :))' *  diag(gamma1(:, i)) * (X1-mu1(i, :))))) };  
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


X2 = X2_train;
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
    clusterpoints2(i) = {X2(indices2 == i,:)};
end

sigma2 = cell(1, Q2);

for i = 1:Q2
    %calculates covariances for each clusters
    points2 = clusterpoints2{1, i};
    temp2 = points2- mu2(i, :);
    sigma2(i) = {diag(diag((1/(Ni2(i)-1))*(temp2' * temp2)))};
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
%(abs(L_new - L_old)>1)
count = 0;
while (abs(L_new - L_old)>1)
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
        sigma2(i) = {diag(diag( (1/Ni2(i))*((X2-mu2(i, :))' *  diag(gamma2(:, i)) * (X2-mu2(i, :))) ))};  
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


X3 = X3_train;
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
    clusterpoints3(i) = {X3(indices3 == i, :)};
end

sigma3 = cell(1, Q3);

for i = 1:Q3
    %calculates covariances for each clusters
    points3 = clusterpoints3{1, i};
    temp3 = points3- mu3(i, :);
    sigma3(i) = {diag(diag((1/(Ni3(i)-1))*(temp3' * temp3)))};
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
%(abs(L_new - L_old)>1)
count = 0;
while (abs(L_new - L_old)>10)
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
    gammasum3 = sum(gamma3, 3);
    for i = 1:Q3
        %normalizes gammas
        for j = 1:Ntrain3
            gamma3(j, i) = gamma3(j, i)/gammasum3(j);
        end
    end
    Ni3 = (sum(gamma3, 1))';
    for i = 1:Q3
        mu3(i,:) = (1/Ni3(i))*gamma3(:, i)' * X3;
        sigma3(i) = {diag(diag( (1/Ni3(i))*((X3-mu3(i, :))' *  diag(gamma3(:, i)) * (X3-mu3(i, :))))) };  
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












% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class4%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X4 = X4_train;
[indices4, mu4] = kmeans(X4,Q4);
Ni4 = zeros(Q4, 1);
omega4 = zeros(Q4, 1);
%disp(Ni4);
for i = 1:Q4
    Ni4(i) = sum(indices4 == i);
end
%disp(Ni4);
for i = 1:Q4
    omega4(i) = Ni4(i)/Ntrain4;%summation of all omegas is found to be 1
end

clusterpoints4 = cell(1, Q4);
for i = 1:Q4
    %calculates points in cluster
    clusterpoints4(i) = {X4(indices4 == i, :)};
end

sigma4 = cell(1, Q4);

for i = 1:Q4
    %calculates covariances for each clusters
    points4 = clusterpoints4{1, i};
    temp4 = points4- mu4(i, :);
    sigma4(i) = {diag(diag((1/(Ni4(i)-1))*(temp4' * temp4)))};
end



L_new = 0;

for i = 1:Ntrain4
        temp_L=0;
        for j = 1:Q4
            current_cov = sigma4{1, j};
            current_mu = mu4(j,:);
            points4 = (X4(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points4) * ((current_cov) \ (points4)'));
            
            temp_L = temp_L + omega4(j) * gauss;
        end
        L_new = L_new + log(temp_L);
        
end

L_old = -10;
%(abs(L_new - L_old)>1)
count = 0;
while (abs(L_new - L_old)>1)
    L_old = L_new;
    %To find gamma
    gamma4 = zeros(Ntrain4, Q4);
    gammasum4 = zeros(Q4, 1);
    % %disp(clusterpoints1{1,1}-mu1(1,:));

    for i = 1:Ntrain4
        %calculates gaussians for each clusters
        for j = 1:Q4
            current_cov = sigma4{1, j};
            current_mu = mu4(j,:);
            points4 = (X4(i,:))- current_mu;
            gamma4(i, j) = (omega4(j)/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points4) * ((current_cov) \(points4)'));
            
        end
        
    end
    gammasum4 = sum(gamma4, 4);
    for i = 1:Q4
        %normalizes gammas
        for j = 1:Ntrain4
            gamma4(j, i) = gamma4(j, i)/gammasum4(j);
        end
    end
    Ni4 = (sum(gamma4, 1))';
    for i = 1:Q4
        mu4(i,:) = (1/Ni4(i))*gamma4(:, i)' * X4;
        sigma4(i) = { diag(diag((1/Ni4(i))*((X4-mu4(i, :))' *  diag(gamma4(:, i)) * (X4-mu4(i, :))))) };  
        omega4(i) = Ni4(i) / Ntrain4;
    end
    
    L_new = 0;

    for i = 1:Ntrain4
        temp_L=0;
        for j = 1:Q4
            current_cov = sigma4{1, j};
            current_mu = mu4(j,:);
            points4 = (X4(i,:))- current_mu;
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points4) * ((current_cov) \ (points4)'));
            temp_L = temp_L + omega4(j) * gauss;
        end
        L_new = L_new + log(temp_L);
    end
    
end
%disp(L_new);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X5 = X5_train;
[indices5, mu5] = kmeans(X5,Q5);
Ni5 = zeros(Q5, 1);
omega5 = zeros(Q5, 1);
%disp(Ni5);
for i = 1:Q5
    Ni5(i) = sum(indices5 == i);
end
%disp(Ni5);
for i = 1:Q5
    omega5(i) = Ni5(i)/Ntrain5;%summation of all omegas is found to be 1
end

clusterpoints5 = cell(1, Q5);
for i = 1:Q5
    %calculates points in cluster
    clusterpoints5(i) = {X5(indices5 == i, :)};
end

sigma5 = cell(1, Q5);

for i = 1:Q5
    %calculates covariances for each clusters
    points5 = clusterpoints5{1, i};
    temp5 = points5- mu5(i, :);
    sigma5(i) = {diag(diag((1/(Ni5(i)-1))*(temp5' * temp5)))};
end



L_new = 0;

for i = 1:Ntrain5
        temp_L=0;
        for j = 1:Q5
            current_cov = sigma5{1, j};
            current_mu = mu5(j,:);
            points5 = (X5(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points5) * ((current_cov) \ (points5)'));
            
            temp_L = temp_L + omega5(j) * gauss;
        end
        L_new = L_new + log(temp_L);
        
end

L_old = -10;
%(abs(L_new - L_old)>1)
count = 0;
while (abs(L_new - L_old)>1)
    L_old = L_new;
    %To find gamma
    gamma5 = zeros(Ntrain5, Q5);
    gammasum5 = zeros(Q5, 1);
    % %disp(clusterpoints1{1,1}-mu1(1,:));

    for i = 1:Ntrain5
        %calculates gaussians for each clusters
        for j = 1:Q5
            current_cov = sigma5{1, j};
            current_mu = mu5(j,:);
            points5 = (X5(i,:))- current_mu;
            gamma5(i, j) = (omega5(j)/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points5) * ((current_cov) \(points5)'));
            
        end
        
    end
    gammasum5 = sum(gamma5, 5);
    for i = 1:Q5
        %normalizes gammas
        for j = 1:Ntrain5
            gamma5(j, i) = gamma5(j, i)/gammasum5(j);
        end
    end
    Ni5 = (sum(gamma5, 1))';
    for i = 1:Q5
        mu5(i,:) = (1/Ni5(i))*gamma5(:, i)' * X5;
        sigma5(i) = {diag(diag( (1/Ni5(i))*((X5-mu5(i, :))' *  diag(gamma5(:, i)) * (X5-mu5(i, :))))) };  
        omega5(i) = Ni5(i) / Ntrain5;
    end
    
    L_new = 0;

    for i = 1:Ntrain5
        temp_L=0;
        for j = 1:Q5
            current_cov = sigma5{1, j};
            current_mu = mu5(j,:);
            points5 = (X5(i,:))- current_mu;
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points5) * ((current_cov) \ (points5)'));
            temp_L = temp_L + omega5(j) * gauss;
        end
        L_new = L_new + log(temp_L);
    end
    
end
disp(L_new);


%%%%%%%%%%TEST_DATA ACCURACY%%%%%%%%%%%%%%%

%class1
gorilla_string='..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_test\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\090.gorilla\090.gorilla_test\');
predicted1=zeros(length(Files)-2,1);
indx=1;
for p=3:6
    Filenames=Files(p).name;
    temp_file=strcat(gorilla_string,Filenames);
    Xtest1=importdata(temp_file);
    
    Ntest1=size(Xtest1,1);
    
    Ldata = zeros(Ntest1, 5);
    data_class = zeros(Ntest1, 1);
    count =0;

    for i = 1:Ntest1
            %for class 1
            temp_L=0;
            for j = 1:Q1
                current_cov = sigma1{1, j};
                current_mu = mu1(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega1(j) * gauss;
            end
            Ldata(i, 1) = Ldata(i, 1) + log(temp_L);

            %for class 2
            temp_L=0;
            for j = 1:Q2
                current_cov = sigma2{1, j};
                current_mu = mu2(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega2(j) * gauss;
            end
            Ldata(i, 2) = Ldata(i, 2) + log(temp_L);

            %for class 3
            temp_L=0;
            for j = 1:Q3
                current_cov = sigma3{1, j};
                current_mu = mu3(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega3(j) * gauss;
            end
            Ldata(i, 3) = Ldata(i, 3) + log(temp_L);


            %for class 4
            temp_L=0;
            for j = 1:Q4
                current_cov = sigma4{1, j};
                current_mu = mu4(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega4(j) * gauss;
            end
            Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
            
            %for class 5
            temp_L=0;
            for j = 1:Q5
                current_cov = sigma5{1, j};
                current_mu = mu5(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega5(j) * gauss;
            end
            Ldata(i, 5) = Ldata(i, 5) + log(temp_L);          
    end
    %find class corresponding to maximum value
    prdct = ones(1, 5);
    for o=1:5
        for oo = 1:Ntest1
           prdct(1, o) = (prdct(1,o) * (Ldata(oo, o)))/170;
        end
    end
    [dontcare, data_class(i)] = max(prdct);
    predicted1(indx,1)=data_class(i);
    indx=indx+1;
end


%class2
billiard_string='..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_test\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\011.billiards\011.billiards_test\');
predicted2=zeros(length(Files)-2,1);
indx=1;
for p=3:5
    Filenames=Files(p).name;
    temp_file=strcat(billiard_string,Filenames);
    Xtest1=importdata(temp_file);
    
    Ntest1=size(Xtest1,1);
    
    Ldata = zeros(Ntest1, 5);
    data_class = zeros(Ntest1, 1);
    count =0;

    for i = 1:Ntest1
            %for class 1
            temp_L=0;
            for j = 1:Q1
                current_cov = sigma1{1, j};
                current_mu = mu1(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega1(j) * gauss;
            end
            Ldata(i, 1) = Ldata(i, 1) + log(temp_L);

            %for class 2
            temp_L=0;
            for j = 1:Q2
                current_cov = sigma2{1, j};
                current_mu = mu2(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega2(j) * gauss;
            end
            Ldata(i, 2) = Ldata(i, 2) + log(temp_L);

            %for class 3
            temp_L=0;
            for j = 1:Q3
                current_cov = sigma3{1, j};
                current_mu = mu3(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega3(j) * gauss;
            end
            Ldata(i, 3) = Ldata(i, 3) + log(temp_L);


            %for class 4
            temp_L=0;
            for j = 1:Q4
                current_cov = sigma4{1, j};
                current_mu = mu4(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega4(j) * gauss;
            end
            Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
            
            %for class 5
            temp_L=0;
            for j = 1:Q5
                current_cov = sigma5{1, j};
                current_mu = mu5(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega5(j) * gauss;
            end
            Ldata(i, 5) = Ldata(i, 5) + log(temp_L);          
    end
    %find class corresponding to maximum value
    prdct = ones(1, 5);
    for o=1:5
        for oo = 1:Ntest1
           prdct(1, o) = (prdct(1,o) * (Ldata(oo, o)))/170;
        end
    end
    [dontcare, data_class(i)] = max(prdct);
    predicted2(indx,1)=data_class(i);
    indx=indx+1;
end


%class3
grapes_string='..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_test\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\092.grapes\092.grapes_test\');
predicted3=zeros(length(Files)-2,1);
indx=1;
for p=3:5
    Filenames=Files(p).name;
    temp_file=strcat(grapes_string,Filenames);
    Xtest1=importdata(temp_file);
    
    Ntest1=size(Xtest1,1);
    
    Ldata = zeros(Ntest1, 5);
    data_class = zeros(Ntest1, 1);
    count =0;

    for i = 1:Ntest1
            %for class 1
            temp_L=0;
            for j = 1:Q1
                current_cov = sigma1{1, j};
                current_mu = mu1(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega1(j) * gauss;
            end
            Ldata(i, 1) = Ldata(i, 1) + log(temp_L);

            %for class 2
            temp_L=0;
            for j = 1:Q2
                current_cov = sigma2{1, j};
                current_mu = mu2(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega2(j) * gauss;
            end
            Ldata(i, 2) = Ldata(i, 2) + log(temp_L);

            %for class 3
            temp_L=0;
            for j = 1:Q3
                current_cov = sigma3{1, j};
                current_mu = mu3(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega3(j) * gauss;
            end
            Ldata(i, 3) = Ldata(i, 3) + log(temp_L);


            %for class 4
            temp_L=0;
            for j = 1:Q4
                current_cov = sigma4{1, j};
                current_mu = mu4(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega4(j) * gauss;
            end
            Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
            
            %for class 5
            temp_L=0;
            for j = 1:Q5
                current_cov = sigma5{1, j};
                current_mu = mu5(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega5(j) * gauss;
            end
            Ldata(i, 5) = Ldata(i, 5) + log(temp_L);          
    end
    %find class corresponding to maximum value
    prdct = ones(1, 5);
    for o=1:5
        for oo = 1:Ntest1
           prdct(1, o) = (prdct(1,o) * (Ldata(oo, o)))/170;
        end
    end
    [dontcare, data_class(i)] = max(prdct);
    predicted3(indx,1)=data_class(i);
    indx=indx+1;
end


%class4
ladder_string='..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_test\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\126.ladder\126.ladder_test\');
predicted4=zeros(length(Files)-2,1);
indx=1;
for p=3:5
    Filenames=Files(p).name;
    temp_file=strcat(ladder_string,Filenames);
    Xtest1=importdata(temp_file);
    
    Ntest1=size(Xtest1,1);
    
    Ldata = zeros(Ntest1, 5);
    data_class = zeros(Ntest1, 1);
    count =0;

    for i = 1:Ntest1
            %for class 1
            temp_L=0;
            for j = 1:Q1
                current_cov = sigma1{1, j};
                current_mu = mu1(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega1(j) * gauss;
            end
            Ldata(i, 1) = Ldata(i, 1) + log(temp_L);

            %for class 2
            temp_L=0;
            for j = 1:Q2
                current_cov = sigma2{1, j};
                current_mu = mu2(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega2(j) * gauss;
            end
            Ldata(i, 2) = Ldata(i, 2) + log(temp_L);

            %for class 3
            temp_L=0;
            for j = 1:Q3
                current_cov = sigma3{1, j};
                current_mu = mu3(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega3(j) * gauss;
            end
            Ldata(i, 3) = Ldata(i, 3) + log(temp_L);


            %for class 4
            temp_L=0;
            for j = 1:Q4
                current_cov = sigma4{1, j};
                current_mu = mu4(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega4(j) * gauss;
            end
            Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
            
            %for class 5
            temp_L=0;
            for j = 1:Q5
                current_cov = sigma5{1, j};
                current_mu = mu5(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega5(j) * gauss;
            end
            Ldata(i, 5) = Ldata(i, 5) + log(temp_L);          
    end
    %find class corresponding to maximum value
    prdct = ones(1, 5);
    for o=1:5
        for oo = 1:Ntest1
           prdct(1, o) = (prdct(1,o) * (Ldata(oo, o)))/170;
        end
    end
    [dontcare, data_class(i)] = max(prdct);
    predicted4(indx,1)=data_class(i);
    indx=indx+1;
end


%class5
faces_string='..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_test\';
Files=dir('..\data_assign2_group5\features_SURF\feats_surf\253.faces-easy-101\253.faces-easy-101_test\');
predicted5=zeros(length(Files)-2,1);
indx=1;
for p=3:5
    Filenames=Files(p).name;
    temp_file=strcat(faces_string,Filenames);
    Xtest1=importdata(temp_file);
    
    Ntest1=size(Xtest1,1);
    
    Ldata = zeros(Ntest1, 5);
    data_class = zeros(Ntest1, 1);
    count =0;

    for i = 1:Ntest1
            %for class 1
            temp_L=0;
            for j = 1:Q1
                current_cov = sigma1{1, j};
                current_mu = mu1(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega1(j) * gauss;
            end
            Ldata(i, 1) = Ldata(i, 1) + log(temp_L);

            %for class 2
            temp_L=0;
            for j = 1:Q2
                current_cov = sigma2{1, j};
                current_mu = mu2(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega2(j) * gauss;
            end
            Ldata(i, 2) = Ldata(i, 2) + log(temp_L);

            %for class 3
            temp_L=0;
            for j = 1:Q3
                current_cov = sigma3{1, j};
                current_mu = mu3(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega3(j) * gauss;
            end
            Ldata(i, 3) = Ldata(i, 3) + log(temp_L);


            %for class 4
            temp_L=0;
            for j = 1:Q4
                current_cov = sigma4{1, j};
                current_mu = mu4(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega4(j) * gauss;
            end
            Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
            
            %for class 5
            temp_L=0;
            for j = 1:Q5
                current_cov = sigma5{1, j};
                current_mu = mu5(j,:);
                points = (Xtest1(i,:))- current_mu;

                gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));

                temp_L = temp_L + omega5(j) * gauss;
            end
            Ldata(i, 5) = Ldata(i, 5) + log(temp_L);          
    end
    %find class corresponding to maximum value
    prdct = ones(1, 5);
    for o=1:5
        for oo = 1:Ntest1
           prdct(1, o) = (prdct(1,o) * (Ldata(oo, o)))/170;
        end
    end
    [dontcare, data_class(i)] = max(prdct);
    predicted5(indx,1)=data_class(i);
    indx=indx+1;
end