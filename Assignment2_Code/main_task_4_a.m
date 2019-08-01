clearvars;

[x11, x12] = textread('..\data_assign2_group5\group5\linearly_separable\class1_train.txt','%f %f');
[xval11, xval12] = textread('..\data_assign2_group5\group5\linearly_separable\class1_val.txt','%f %f');
[x21, x22] = textread('..\data_assign2_group5\group5\linearly_separable\class2_train.txt','%f %f');
[xval21, xval22] = textread('..\data_assign2_group5\group5\linearly_separable\class2_val.txt','%f %f');
[x31, x32] = textread('..\data_assign2_group5\group5\linearly_separable\class3_train.txt','%f %f');
[xval31, xval32] = textread('..\data_assign2_group5\group5\linearly_separable\class3_val.txt','%f %f');
[x41, x42] = textread('..\data_assign2_group5\group5\linearly_separable\class4_train.txt','%f %f');
[xval41, xval42] = textread('..\data_assign2_group5\group5\linearly_separable\class4_val.txt','%f %f');


[xtest11, xtest12] = textread('..\data_assign2_group5\group5\linearly_separable\class1_test.txt','%f %f');
[Ntest1, NotRequired] = size(xtest11);
Xtest1 = [xtest11, xtest12];
[xtest21, xtest22] = textread('..\data_assign2_group5\group5\linearly_separable\class2_test.txt','%f %f');
[Ntest2, NotRequired] = size(xtest21);
Xtest2 = [xtest21, xtest22];
[xtest31, xtest32] = textread('..\data_assign2_group5\group5\linearly_separable\class3_test.txt','%f %f');
[Ntest3, NotRequired] = size(xtest31);
Xtest3 = [xtest31, xtest32];
[xtest41, xtest42] = textread('..\data_assign2_group5\group5\linearly_separable\class4_test.txt','%f %f');
[Ntest4, NotRequired] = size(xtest41);
Xtest4 = [xtest41, xtest42];




[Ntrain1, NotRequired] = size(x11);
Q1 = 1; % Number of clusters in class 1
Q2 = 1; % Number of clusters in class 2
Q3 = 1; % Number of clusters in class 3
Q4 = 1; % Number of clusters in class 4
X1 = [x11, x12];
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
    clusterpoints1(i) = {[x11(indices1 == i), x12(indices1 == i)]};
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
%disp(L_new);







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ntrain2, NotRequired] = size(x21);
X2 = [x21, x22];
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
    clusterpoints2(i) = {[x21(indices2 == i), x22(indices2 == i)]};
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
        sigma2(i) = {diag(diag( (1/Ni2(i))*((X2-mu2(i, :))' *  diag(gamma2(:, i)) * (X2-mu2(i, :))))) };  
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

X3 = [x31, x32];
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
    clusterpoints3(i) = {[x31(indices3 == i), x32(indices3 == i)]};
end

sigma3 = cell(1, Q3);

for i = 1:Q3
    %calculates covariances for each clusters
    points3 = clusterpoints3{1, i};
    temp3 = points3- mu3(i, :);
    sigma3(i) = {(1/(Ni3(i)-1))*(temp3' * temp3)};
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


[Ntrain4, NotRequired] = size(x41);

X4 = [x41, x42];
[indices4, mu4] = kmeans(X4,Q4);
Ni4 = zeros(Q4, 1);
omega4 = zeros(Q4, 1);

for i = 1:Q4
    Ni4(i) = sum(indices4 == i);
end

for i = 1:Q4
    omega4(i) = Ni4(i)/Ntrain4;%summation of all omegas is found to be 1
end

clusterpoints4 = cell(1, Q4);
for i = 1:Q4
    %calculates points in cluster
    clusterpoints4(i) = {[x41(indices4 == i), x42(indices4 == i)]};
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
%(abs(L_new - L_old)>0.05)
count = 0;
while (abs(L_new - L_old)>0.05)
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
    gammasum4 = sum(gamma4, 2);
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


sumsigma1 = zeros(2, 2);
sumsigma2 = zeros(2, 2);
sumsigma3 = zeros(2, 2);
sumsigma4 = zeros(2, 2);
for i = 1:Q1
    sumsigma1 = sumsigma1 + sigma1{1, i};
end
sumsigma1 = sumsigma1/Q1;
for i = 1:Q2
    sumsigma2 = sumsigma2 + sigma2{1, i};
end
sumsigma2 = sumsigma2/Q2;
for i = 1:Q3
    sumsigma3 = sumsigma3 + sigma3{1, i}
end
sumsigma3 = sumsigma3/Q3;
for i = 1:Q4
    sumsigma4 = sumsigma4 + sigma4{1, i}
end
sumsigma4 = sumsigma4/Q4;

sumsigma = sumsigma1+sumsigma2+sumsigma3+sumsigma4;

current_cov = sumsigma;
det_currentcov = det(sumsigma);






%validation to find parameters
[Nval1, NotRequired] = size(xval11);
Xval1 = [xval11, xval12];

Lval = 0;

for i = 1:Nval1
        temp_L=0;
        for j = 1:Q1
            
            current_mu = mu1(j,:);
            points1 = (Xval1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points1) * ((current_cov) \ (points1)'));
            
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
Xval2 = [xval21, xval22];

Lval = 0;

for i = 1:Nval2
        temp_L=0;
        for j = 1:Q2
            %current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points2 = (Xval2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points2) * ((current_cov) \ (points2)'));
            
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
Xval3 = [xval31, xval32];

Lval = 0;

for i = 1:Nval3
        temp_L=0;
        for j = 1:Q3
            %current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points3 = (Xval3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points3) * ((current_cov) \ (points3)'));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Lval = Lval + log(temp_L);
end




%validation to find parameters
[Nval4, NotRequired] = size(xval41);
Xval4 = [xval41, xval42];

Lval = 0;

for i = 1:Nval4
        temp_L=0;
        for j = 1:Q4
           % current_cov = sigma4{1, j};
            current_mu = mu4(j,:);
            points4 = (Xval4(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points4) * ((current_cov) \ (points4)'));
            
            temp_L = temp_L + omega4(j) * gauss;
        end
        Lval = Lval + log(temp_L);
end
%disp(Lval);
% scatter(x11, x14);
% hold on
% scatter(xval11, xval14, 'r');
% hold off
%%%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];

class1_count=0;
class2_count=0;
class3_count=0;
class4_count=0;







Ltest = 0;
%Ntest = Ntest1 + Ntest2 + Ntest3 + Ntest4;
Ntest=length(xy);
predicted=zeros(Ntest,1);
%Xtest = [Xtest1; Xtest2; Xtest3; Xtest4];
Xtest=xy;

Ldata = zeros(Ntest, 4);
data_class = zeros(Ntest, 1);
for i = 1:Ntest
        %for class 1
        temp_L=0;
        for j = 1:Q1
            current_mu = mu1(j,:);
            points = (Xtest(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            %current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            %current_cov = sigma3{1, j};
            current_mu = mu3(j,:);
            points = (Xtest(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
        
        
        %for class 4
        temp_L=0;
        for j = 1:Q4
            %current_cov = sigma4{1, j};
            current_mu = mu4(j,:);
            points = (Xtest(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega4(j) * gauss;
        end
        Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
        
        %find class corresponding to maximum value
        [dontcare, data_class(i)] = max(Ldata(i,:)); 
        
end
N_data_class = zeros(4, 1);
for i = 1: 4
    N_data_class(i) = sum(data_class == i);
end
pred_class1 = zeros(N_data_class(1), 2);
pred_class2 = zeros(N_data_class(2), 2);
pred_class3 = zeros(N_data_class(3), 2);
pred_class4 = zeros(N_data_class(4), 2);
one = 1;
two = 1;
three = 1;
four = 1;
for i = 1: Ntest
    if data_class(i) ==1
        pred_class1(one, :) = Xtest(i,:);
        one = one + 1;
        predicted(i)=1;
    end
    if data_class(i) ==2
        pred_class2(two, :) = Xtest(i, :);
        two = two+1;
        predicted(i)=2;
    end
    if data_class(i) ==3
        pred_class3(three, :) =  Xtest(i, :);
        three = three + 1;
        predicted(i)=3;
    end
    if data_class(i) ==4
        pred_class4(four, :) = Xtest(i, :);
        four = four + 1;
        predicted(i)=4;
    end
end 
%disp(Ldata);
%scatter(pred_class1(:,1), pred_class1(:,2));

decisionmap = reshape(predicted, image_size);
figure;
imagesc(xrange,yrange,decisionmap);

hold on

set(gca,'ydir','normal');
cmap = [0.7 1 0.7;0.9 0.9 1;1 0.7 1;1 0.6 0.6];
colormap(cmap);

scatter(x11,x12,'g');
scatter(x21,x22,'b');
scatter(x31,x32,'m');
scatter(x41,x42,'r');
hold off




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TEST-DATA_ACCURACY%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%class1
Ldata = zeros(Ntest1, 4);
data_class = zeros(Ntest1, 1);
count =0;
for i = 1:Ntest1
        %for class 1
        temp_L=0;
        for j = 1:Q1
            
            current_mu = mu1(j,:);
            points = (Xtest1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
           
            current_mu = mu2(j,:);
            points = (Xtest1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            
            current_mu = mu3(j,:);
            points = (Xtest1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
        
        
        %for class 4
        temp_L=0;
        for j = 1:Q4
           
            current_mu = mu4(j,:);
            points = (Xtest1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega4(j) * gauss;
        end
        Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
        
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

    
    if(mxx==4)
        class4_count=class4_count+1;
    end
end

%class2
Ldata = zeros(Ntest2, 4);
data_class = zeros(Ntest2, 1);
for i = 1:Ntest2
        %for class 1
        temp_L=0;
        for j = 1:Q1
            
            current_mu = mu1(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            
            current_mu = mu3(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
        
        
        %for class 4
        temp_L=0;
        for j = 1:Q4
           
            current_mu = mu4(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega4(j) * gauss;
        end
        Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
        
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

    
    if(mxx==4)
        class4_count=class4_count+1;
    end
end

%class3
Ldata = zeros(Ntest3, 4);
data_class = zeros(Ntest3, 1);
for i = 1:Ntest3
        %for class 1
        temp_L=0;
        for j = 1:Q1
            
            current_mu = mu1(j,:);
            points = (Xtest3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            
            current_mu = mu3(j,:);
            points = (Xtest3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
        
        
        %for class 4
        temp_L=0;
        for j = 1:Q4
            
            current_mu = mu4(j,:);
            points = (Xtest3(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega4(j) * gauss;
        end
        Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
        
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

    if(mxx==4)
        class4_count=class4_count+1;
    end
end

%class4
Ldata = zeros(Ntest4, 4);
data_class = zeros(Ntest4, 1);
for i = 1:Ntest4
        %for class 1
        temp_L=0;
        for j = 1:Q1
           
            current_mu = mu1(j,:);
            points = (Xtest4(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
           
            current_mu = mu2(j,:);
            points = (Xtest4(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        %for class 3
        temp_L=0;
        for j = 1:Q3
            
            current_mu = mu3(j,:);
            points = (Xtest4(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega3(j) * gauss;
        end
        Ldata(i, 3) = Ldata(i, 3) + log(temp_L);
        
        
        %for class 4
        temp_L=0;
        for j = 1:Q4
            
            current_mu = mu4(j,:);
            points = (Xtest4(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det_currentcov)))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega4(j) * gauss;
        end
        Ldata(i, 4) = Ldata(i, 4) + log(temp_L); 
        
        %find class corresponding to maximum value
        [dontcare, data_class(i)] = max(Ldata(i,:));
        predicted4(i,1)=data_class(i);
        
         mxx=predicted4(i,1);
    
    if(mxx==4) count=count+1; end
        
    if(mxx==1) 
        class1_count=class1_count+1; 
    end
    
    if(mxx==2) 
        class2_count=class2_count+1; 
    end
    
    if(mxx==3) 
        class3_count=class3_count+1; 
    end
   
    if(mxx==4)
        class4_count=class4_count+1;
    end
end


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