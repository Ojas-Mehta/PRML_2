clearvars;
[x11, x12] = textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_train.txt','%f %f');
[Ntrain1, NotRequired] = size(x11);
Q1 = 15; % Number of clusters in class 1
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
    sigma1(i) = {(1/(Ni1(i)-1))*(temp1' * temp1)};
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
        sigma1(i) = { (1/Ni1(i))*((X1-mu1(i, :))' *  diag(gamma1(:, i)) * (X1-mu1(i, :))) };  
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


%validation to find parameters
[xval11, xval12] = textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_val.txt','%f %f');
[Nval1, NotRequired] = size(xval11);
Xval1 = [xval11, xval12];

Lval = 0;

for i = 1:Nval1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points1 = (Xval1(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points1) * ((current_cov) \ (points1)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Lval = Lval + log(temp_L);
end
%disp(Lval);
% scatter(x11, x12);
% hold on
% scatter(xval11, xval12, 'r');
% hold off




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%class2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x21, x22] = textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_train.txt','%f %f');
[Ntrain2, NotRequired] = size(x21);
Q2 = 15; % Number of clusters in class 2
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
    sigma2(i) = {(1/(Ni2(i)-1))*(temp2' * temp2)};
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
        sigma2(i) = { (1/Ni2(i))*((X2-mu2(i, :))' *  diag(gamma2(:, i)) * (X2-mu2(i, :))) };  
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


%validation to find parameters
[xval21, xval22] = textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_val.txt','%f %f');
[Nval2, NotRequired] = size(xval21);
Xval2 = [xval21, xval22];

Lval = 0;

for i = 1:Nval2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points2 = (Xval2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points2) * ((current_cov) \ (points2)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Lval = Lval + log(temp_L);
end
%disp(Lval);
% scatter(x21, x22);
% hold on
% scatter(xval21, xval22, 'r');
% hold off



%%%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[xtest11, xtest12] = textread('..\data_assign2_group5\group5\nonlinearly_separable\class1_test.txt','%f %f');
[Ntest1, NotRequired] = size(xtest11);
Xtest1 = [xtest11, xtest12];
[xtest21, xtest22] = textread('..\data_assign2_group5\group5\nonlinearly_separable\class2_test.txt','%f %f');
[Ntest2, NotRequired] = size(xtest21);
Xtest2 = [xtest21, xtest22];

Ltest = 0;

xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];


%Ntest = Ntest1 + Ntest2;
Ntest=length(xy);
predicted=zeros(Ntest,1);
Xtest = xy;

Ldata = zeros(Ntest, 2);
data_class = zeros(Ntest, 1);
for i = 1:Ntest
        %for class 1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points = (Xtest(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        %find class corresponding to maximum value
        [dontcare, data_class(i)] = max(Ldata(i,:)); 
        predicted(i)=data_class(i);
end


%%%%%%%%%%%%TEST-DATA ACCURACY%%%%%%%%%%%%%%


%class1
Ldata = zeros(Ntest1, 2);
data_class = zeros(Ntest1, 1);
count =0;
class1_count = 0;
class2_count = 0;
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
    
end

%class2
Ldata = zeros(Ntest2, 2);
data_class = zeros(Ntest2, 1);
for i = 1:Ntest2
        %for class 1
        temp_L=0;
        for j = 1:Q1
            current_cov = sigma1{1, j};
            current_mu = mu1(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega1(j) * gauss;
        end
        Ldata(i, 1) = Ldata(i, 1) + log(temp_L);
        
        %for class 2
        temp_L=0;
        for j = 1:Q2
            current_cov = sigma2{1, j};
            current_mu = mu2(j,:);
            points = (Xtest2(i,:))- current_mu;
            
            gauss = (1/(2*pi*sqrt(det(current_cov))))* exp((-0.5)*(points) * ((current_cov) \ (points)'));
            
            temp_L = temp_L + omega2(j) * gauss;
        end
        Ldata(i, 2) = Ldata(i, 2) + log(temp_L);
        
        
        
        
        
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
    

end



 %Accuracy for test data
 
 classification_accuracy=double(count)/(Ntest1+Ntest2);
 disp('Classification accuracy is:');
 disp(classification_accuracy);
 
 
 %Computing confusion matrix.
confusion_matrix=zeros(2,2);
for i=1:Ntest1
  confusion_matrix(1,predicted1(i))=confusion_matrix(1,predicted1(i))+1;
end

for i=1:Ntest2
  confusion_matrix(2,predicted2(i))=confusion_matrix(2,predicted2(i))+1;
end


disp('Confusion matrix is:');
disp(confusion_matrix);
% N_data_class = zeros(2, 1);
% 
% for i = 1: 2
%     N_data_class(i) = sum(data_class == i);
% end
% pred_class1 = zeros(N_data_class(1), 2);
% pred_class2 = zeros(N_data_class(2), 2);
% one = 1;
% two = 1;
% three = 1;
% four = 1;
%  for i = 1: Ntest
%      if data_class(i) ==1
%          pred_class1(one, :) = Xtest(i,:);
%          one = one + 1;
%          predicted(i)=1;
%      end
%      if data_class(i) ==2
%          pred_class2(two, :) = Xtest(i, :);
%          two = two+1;
%          predicted(i)=2;
%      end
%  end 
%%disp(Ldata);


decisionmap = reshape(predicted, image_size);
figure;
imagesc(xrange,yrange,decisionmap);

hold on
%scatter(pred_class1(:,1), pred_class1(:,2), 10, 'y', 'filled');
%scatter(pred_class2(:,1), pred_class2(:,2),10, 'r','filled');
set(gca,'ydir','normal');
cmap = [0.7 1 0.7;0.9 0.9 1];
colormap(cmap);
scatter(x11, x12,'g');
scatter(x21, x22,'b');
hold off