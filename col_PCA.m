function col_PCA(k,k_all,k_train,k_test)
%需要自行改动数据集位置
% k:KNN的k
% k_all:所有人类别数
% k_train:每一类的训练集个数
% k_test:每一类的测试集个数
X_train=[];%存放训练图像的矩阵
label_train=[];%存放训练集的类别

%读入训练数据----------------------------------------------------------------------------------------------------------------------
cnt=1;
for i=1:k_all
    for j=1:k_train
        if(i<10)
           temp=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject0',num2str(i),'_',num2str(j),'.bmp'));     
        else
           temp=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject',num2str(i),'_',num2str(j),'.bmp'));  
        end  
        [high,wide]=size(temp);
        single_sample=reshape(temp,high*wide,1);
        single_sample=double(single_sample);
        X_train=[X_train single_sample];
        label_train(1,cnt)=i;
        cnt=cnt+1;
    end
end

%中心化训练样本-----------------------------------------------------------------------------------------------------------------------
%1.计算平均脸(也就是计算训练矩阵每一行的均值并存于一个列向量中）
train_mean=mean(X_train,2);
%展示平均脸
% imshow(reshape(train_mean,high,wide));
%2.将平均脸拓展为矩阵，并中心化训练样本
train_mean=repmat(train_mean,1,k_train*k_all);
%3.中心化操作
A=X_train-train_mean;

%PCA核心-------------------------------------------------------------------------------------------------------------------------------

%首先计算特征向量矩阵与特征值矩阵
S=A'*A;%协方差矩阵
[V,D]=eig(S);
d1=diag(D);

%对特征值进行降序排序
[D_sort,index]=sort(diag(D),'descend');
D_sort=diag(D_sort,0);

%选择98%的能量
%显示能量占比图
[m,~]=size(D_sort);
dsum=trace(D);
dsort=reshape(diag(D_sort),1,m);
x=1:1:k_all*k_train;
p=0;
for i=1:1:(k_all*k_train)
    y(i)=sum(dsort(x(1:i)));
    %记录占98%能量的特征值个数
    if(y(i)/dsum<0.98)
        p=p+1;
    end
end
figure
y1=ones(1,k_all*k_train);
plot(x,y/dsum,x,y1*0.98,'linewidth',2);
grid
title('前n个特征值占总能量百分比');
xlabel('前n个特征值');
ylabel('所占百分比');

%对特征向量进行排序
for i=1:(k_all*k_train)
    V_sort(i,:)=V(index(i),:);
    d_sort(i)=d1(index(i));
end

%取出前p个特征向量组成一个特征矩阵
i=1;
while(i<=p&&d_sort(i)>0)
    V_end(i,:)=V_sort(i,:);
    i=i+1;
end
base=V_end*A';%将特征向量矩阵转化为原始特征向量矩阵

%训练结束--------------------------------------------------------------------------------------------------------------------------
%训练结果为特征矩阵V_done

%测试过程--------------------------------------------------------------------------------------------------------------------------
Y_train=base*X_train;%对训练集进行投影（也就是降维）
X_test=[];%存放测试图像数据
label_test=[];%存放测试图像标签

%读入测试数据-----------------------------------------------------------------------------------------------------------------------
cnt=1;
for i=1:k_all
    for j=(k_train+1):(k_train+k_test)
        if(i<10)
           temp=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject0',num2str(i),'_',num2str(j),'.bmp'));     
        else
           temp=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject',num2str(i),'_',num2str(j),'.bmp'));  
        end 
        single_sample=reshape(temp,high*wide,1);
        single_sample=double(single_sample);
        X_test=[X_test single_sample];
        label_test(1,cnt)=i;
        cnt=cnt+1;
    end
end

%将测试数据投影到特征空间（降维）
Y_test=base*X_test;

%KNN分类器识别Y_test--------------------------------------------------------------------------------------------------------------------
class_test=[];%存放测试结果
class_test=col_KNN(k,k_all,label_train,Y_train,Y_test);

% %计算识别率--------------------------------------------------------------------------------------------------------------------
accu=0;%统计正确数量
for i=1:k_all*k_test
    if class_test(i)==label_test(i)
        accu=accu+1;
    end
end
accuracy=accu/(k_all*k_test)
end