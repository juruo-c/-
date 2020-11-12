function class_test=row_KNN(k,label_num,label_train,Y_train,Y_test)
%每一类按行排列
%k为k个最近邻
%label_num为类别数
%label_train为训练集类别
%Y_train,Y_test分别为训练集和测试集
[row1,col]=size(Y_train);
[row2,~]=size(Y_test);
class_test=[];%存放测试结果
for i=1:row2
    dis=[];%存放该测试样本与每一个训练样本的距离
    for j=1:row1
        %计算欧几里得距离
        distance=0;
        for m=1:col
            distance=distance+(Y_train(j,m)-Y_test(i,m)).^2;
        end
        dis(1,j)=distance.^0.5;
    end
    %找到k个最近邻
    [~,index]=sort(dis);
    label_cnt=zeros(1,label_num);
    for n=1:k
        label_cnt(label_train(index(n)))=label_cnt(label_train(index(n)))+1;
    end
    max=0;
    for n=1:label_num
        if label_cnt(n)>max
            max=label_cnt(n);
            class_test(i)=n;
        end
    end
end