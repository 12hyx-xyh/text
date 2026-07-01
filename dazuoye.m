%% 单数字OCR 
clear; clc; close all;

targetSize = [28,28];
templateNum = 10;

%% 加载模板，统一预处理标准
templateSet = cell(1,templateNum);
for num = 0:9
    path = sprintf('./template/%d.png',num);
    tpl = imread(path);
    if(size(tpl,3)==3)
        tpl = rgb2gray(tpl);
    end
    tpl = imresize(tpl,[24,24]);
    tplBin = imbinarize(tpl);
    tplBin = ~tplBin; % 和测试图黑白统一
    % 模板居中放到28画布
    fullTpl = zeros(28,28);
    off = 2;
    fullTpl(off+1:off+24,off+1:off+24) = tplBin;
    templateSet{num+1} = fullTpl;
end

%% 读取测试图片
imgRaw = imread('test_num9.png');
figure('Name','1 原图'); imshow(imgRaw); title('原始图像');
recNum = singleDigitOCR(imgRaw, targetSize, templateSet);
fprintf('最终识别数字：%d\n', recNum);

%% 子函数（新增数字居中处理，全局二值，优化匹配）
function resNum = singleDigitOCR(img, stdSize, templates)
    % 1 灰度
    if(size(img,3)==3)
        gray = rgb2gray(img);
    else
        gray = img;
    end
    figure('Name','2 灰度图'); imshow(gray);
    
    % 2 高斯降噪
    blur = imgaussfilt(gray, 1.2);
    figure('Name','3 高斯降噪'); imshow(blur);
    
    % 3 全局二值（印刷数字比自适应稳定）
    level = graythresh(blur);
    bw = imbinarize(blur, level);
    bw = ~bw; % 统一：数字白色，背景黑色
    figure('Name','4 二值图'); imshow(bw);
    
    % 4 形态学
    seSmall = strel('square',1);
    bwOpen = imopen(bw, seSmall);
    figure('Name','5 开运算去噪'); imshow(bwOpen);
    
    seHole = strel('square',2);
    bwFinal = imclose(bwOpen, seHole);
    figure('Name','6 闭运算补孔洞'); imshow(bwFinal);
    
    % 5 连通域过滤噪点
    stats = regionprops(bwFinal, 'BoundingBox','Area');
    areaList = [stats.Area];
    validIdx = find(areaList > 30);
    if isempty(validIdx)
        error('未检测到有效数字，请更换图片');
    end
    [~, tmp] = max(areaList(validIdx));
    idx = validIdx(tmp);
    bbox = stats(idx).BoundingBox;
    
    % 绘制红色框
    figure('Name','7 数字定位框');
    imshow(bwFinal);
    rectangle('Position',bbox,'EdgeColor','r','LineWidth',1);
    
    % 裁剪数字区域
    x1 = round(bbox(1)); y1 = round(bbox(2));
    w = round(bbox(3)); h = round(bbox(4));
    digitCrop = bwFinal(y1:y1+h-1, x1:x1+w-1);
    figure('Name','8 裁剪原始数字'); imshow(digitCrop);
    
    % ==========核心修复：数字居中填充，消除偏移误差==========
    digitResize = imresize(digitCrop, [24,24]);
    digitStd = zeros(28,28);
    offset = 2;
    digitStd(offset+1:offset+24, offset+1:offset+24) = digitResize;
    figure('Name','9 居中归一化28×28'); imshow(digitStd);
    
    % 6 归一化互相关匹配
    score = zeros(1,10);
    for i = 1:10
        tpl = templates{i};
        corrMap = normxcorr2(digitStd, tpl);
        score(i) = max(corrMap(:));
    end
    [maxScore, matchIdx] = max(score);
    resNum = matchIdx - 1;
    
    % 匹配可信度判断
    figure('Name','10 最优匹配模板');
    imshow(templates{matchIdx});
    if maxScore < 0.3
        title(sprintf('警告：匹配可信度极低！识别：%d，相关系数：%.3f',resNum,maxScore));
    else
        title(sprintf('匹配数字：%d，相关系数：%.3f',resNum,maxScore));
    end
end