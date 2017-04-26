function [stimuli,diagnostics]=createStimuli(image_dir,filter,imageIs,normaliseArea,stimFormat,objectF_x_offset,monitor)

%% preparation
close all;
path='/imaging/mm07/occludedObjReprDynamics';
cd(path);
addpath(genpath('/imaging/mm07/^programs/matlab'));

disp('hello');

%% control variables
if ~exist('normaliseArea','var'); normaliseArea=0; end
if ~exist('stimFormat','var'); stimFormat=0; end % 0 = colour, 1 = grayscale, 2 = silhouette grayscales, 3 = silhouette black, 4 = contour 
if ~exist('objectF_x_offset','var'); objectF_x_offset=0; end % front object centered on back object
if ~exist('monitor','var'); monitor=1; end
    
if normaliseArea==0, norm_str='areaOrig'; elseif normaliseArea==1, norm_str='areaNorm'; end
if stimFormat==0, stimFormat_str='colour'; elseif stimFormat==1; stimFormat_str='grayscale'; elseif stimFormat==2; stimFormat_str='silh_colour'; elseif stimFormat==3; stimFormat_str='silh_grayscale'; elseif stimFormat==4; stimFormat_str='contours'; end
if objectF_x_offset==0, offset_str='FctrOnB'; elseif objectF_x_offset<0, offset_str=['F2left_',num2str(abs(objectF_x_offset)),'pix']; offset_direction_str='left'; 
elseif objectF_x_offset>0, offset_str=['F2right_',num2str(objectF_x_offset),'pix']; offset_direction_str='right'; end

%imageIs=[1:3,14:17,25:28,39:42,49:52,57:60,76:80,89:92];


%% save information
save('createStimuli_vars');


%% load images
files=dir(fullfile(image_dir,filter));   
if ~exist('imageIs','var'); imageIs=1:numel(files); end
nImages=numel(imageIs);
for imageI=imageIs
    image=imread(fullfile(image_dir,files(imageI).name));    
    images_col(:,:,:,imageI)=image;
    images_gs(:,:,imageI)=rgb2gray(image);
    if monitor
        figure(101); clf;
        subplot(1,2,1); imshow(image);
        subplot(1,2,2); imshow(rgb2gray(image));
    end
end


%% are the images centered? (yes!)
image_x_ctr=round(size(images_gs(:,:,1),2)/2);
image_y_ctr=round(size(images_gs(:,:,1),1)/2);
imageRow=[];
for imageI=imageIs
    % get grayscale image
    image_gs=images_gs(:,:,imageI);
    image_gs=repmat(image_gs,[1 1 3]);
    % draw cross at image center (red)
    image_gs(image_y_ctr,image_x_ctr-2:image_x_ctr+2,:)=repmat([255 0 0],[5 1]);
    image_gs(image_y_ctr-2:image_y_ctr+2,image_x_ctr,:)=repmat([255 0 0],[5 1]);    
    % create silhouette image
    image_col=images_col(:,:,:,imageI);
    image_sh=colour2silhouette(image_col);
    images_sh(:,:,imageI)=image_sh;
    % get basic image stats
    image_labeled=double(~logical(image_sh));
    if monitor, 
        figure(102); clf; 
        subplot(1,2,1); imshow(image_sh); title('silhouette image');
        subplot(1,2,2); imshow(image_labeled); title('labeled image');
    end
    stats=regionprops(image_labeled,'basic');
    imageStats(imageI)=stats;
    % draw cross at center of object bounding box (green)
    object_x_ctr_inImage=round(stats.BoundingBox(1)+(stats.BoundingBox(3))/2);
    object_y_ctr_inImage=round(stats.BoundingBox(2)+(stats.BoundingBox(4))/2);
    image_gs(object_y_ctr_inImage,object_x_ctr_inImage-2:object_x_ctr_inImage+2,:)=repmat([0 255 0],[5 1]);
    image_gs(object_y_ctr_inImage-2:object_y_ctr_inImage+2,object_x_ctr_inImage,:)=repmat([0 255 0],[5 1]);
    % draw cross at object center of mass (blue)
    x_centerMass=round(stats.Centroid(1));
    y_centerMass=round(stats.Centroid(2));
    image_gs(y_centerMass,x_centerMass-2:x_centerMass+2,:)=repmat([0 0 255],[5 1]);
    image_gs(y_centerMass-2:y_centerMass+2,x_centerMass,:)=repmat([0 0 255],[5 1]);
    % show image
    if monitor
        figure(103); clf; imshow(image_gs);
        text(0,182,'image center','Color','r');
        text(0,194,'object bounding-box center','Color','g');
        text(0,206,'object center-of-mass','Color','b');
    end
    imageRow=[imageRow,image_gs];
end
imwrite(imageRow,'imageRow_grayscale.bmp','bmp');
save('imagesAndStats','images_col','images_gs','images_sh','imageStats');


%% normalise object area (scale to mean by iterative adjustment)
if normaliseArea
    imageRow=[];
    for imageI=imageIs, objectArea(imageI)=imageStats(imageI).Area; end 
    objectArea_avg=mean(objectArea(imageIs));
    for imageI=imageIs
        scaleFactor=sqrt(objectArea_avg/imageStats(imageI).Area);
        image_col_resized=imresize(images_col(:,:,:,imageI),scaleFactor);
        image_gs_resized=rgb2gray(image_col_resized);
        image_sh_resized=colour2silhouette(image_col_resized);
        image_labeled_resized=double(~logical(image_sh_resized));
        stats=regionprops(image_labeled_resized,'basic');
        % area optimization - iterative loop
        area_diff=stats.Area-objectArea_avg;
        iterationI=1;
        while abs(area_diff)>2 && iterationI<100
            c=0.00001*abs(area_diff);
            if area_diff>0
                scaleFactor=scaleFactor*(1-c);                 
            else
                scaleFactor=scaleFactor*(1+c);
            end
            image_col_resized=imresize(images_col(:,:,:,imageI),scaleFactor);
            image_gs_resized=rgb2gray(image_col_resized);
            image_sh_resized=colour2silhouette(image_col_resized);
            image_labeled_resized=double(~logical(image_sh_resized));
            stats=regionprops(image_labeled_resized,'basic');
            area_diff=stats.Area-objectArea_avg;
            iterationI=iterationI+1;
        end            
        if monitor
            figure(104); clf;
            subplot(2,3,1); imshow(images_col(:,:,:,imageI)); title('original');
            subplot(2,3,2); imshow(image_col_resized); title('resized'); xlabel(['scale factor = ',num2str(scaleFactor,'%1.2f')]);
            subplot(2,3,3); imshow(image_gs_resized); title('resized grayscale');
            subplot(2,3,4); imshow(image_sh_resized); title('resized silhouette');
            subplot(2,3,5); imshow(image_labeled_resized); title('resized labeled');
        end    
        images_col_resized{imageI}=image_col_resized;
        imageStats_resized(imageI)=stats;
        % crop the resized image
        object_topL_x=ceil(stats.BoundingBox(1)); object_bottomR_x=object_topL_x+floor(stats.BoundingBox(3))-1; 
        object_topL_y=ceil(stats.BoundingBox(2)); object_bottomR_y=object_topL_y+floor(stats.BoundingBox(4))-1;         
        image_col_resized_cropped=image_col_resized(object_topL_y:object_bottomR_y,object_topL_x:object_bottomR_x,:);        
        images_col_resized_cropped{imageI}=image_col_resized_cropped;
        images_col_resized_cropped_size(:,imageI)=size(image_col_resized_cropped);        
    end % imageI
    if monitor
        for imageI=imageIs, area(imageI)=imageStats_resized(imageI).Area; end
        figure(105); clf; plot(area(imageIs)); hold on;
        plot(objectArea_avg*ones(1,nImages),'r');
        xlabel('images'); ylabel('nr of object pixels');
        title('object area after resizing');
        legend('resized images','average before resizing');
    end
    % center the resized objects
    max_height=max(images_col_resized_cropped_size(1,:));
    max_width=max(images_col_resized_cropped_size(2,:));
    for imageI=imageIs
        image_col_resized_centered=repmat(uint8(128*ones(max_height,max_width)),[1 1 3]);
        image_col_resized_centered_x_ctr=round(size(image_col_resized_centered,2)/2);
        image_col_resized_centered_y_ctr=round(size(image_col_resized_centered,1)/2);
        image_col_resized_cropped=images_col_resized_cropped{imageI};
        image_col_resized_cropped_x_ctr=round(size(image_col_resized_cropped,2)/2);
        image_col_resized_cropped_y_ctr=round(size(image_col_resized_cropped,1)/2);
        ctr_diff_x=abs(image_col_resized_centered_x_ctr-image_col_resized_cropped_x_ctr);
        ctr_diff_y=abs(image_col_resized_centered_y_ctr-image_col_resized_cropped_y_ctr);
        image_col_resized_centered(ctr_diff_y+1:ctr_diff_y+size(image_col_resized_cropped,1),ctr_diff_x+1:ctr_diff_x+size(image_col_resized_cropped,2),:)=image_col_resized_cropped;
        image_gs_resized_centered=rgb2gray(image_col_resized_centered); 
        image_sh_resized_centered=colour2silhouette(image_col_resized_centered);
        if monitor
            figure(106); clf;
            subplot(3,1,1); imshow(image_col_resized_centered); title('resized and centered');
            subplot(3,1,2); imshow(image_gs_resized_centered); 
            subplot(3,1,3); imshow(image_sh_resized_centered); 
        end
        images_col_resized_centered(:,:,:,imageI)=image_col_resized_centered;
        images_gs_resized_centered(:,:,imageI)=image_gs_resized_centered;
        images_sh_resized_centered(:,:,imageI)=image_sh_resized_centered;
        if stimFormat==0
            imageRow=[imageRow,image_col_resized_centered];
        elseif stimFormat==1
            imageRow=[imageRow,image_gs_resized_centered];
        elseif stimFormat>1
            imageRow=[imageRow,image_sh_resized_centered];
        end
    end % imageI
    imwrite(imageRow,['imageRow_areaNorm_',stimFormat_str,'.bmp'],'bmp');
    save('imagesAndStats_areaNorm',...
        'objectArea_avg','images_col_resized','imageStats_resized',...
        'images_col_resized_cropped','images_col_resized_centered','images_gs_resized_centered','images_sh_resized_centered');
end


%% create stimuli
% all possible combinations of 2 overlapping objects (2 depth orders each)
% assume that the original objects are centered
stimulusI=1;
for imageI_back=imageIs
    stimulusRow=[];
    for imageI_front=imageIs 
        if normaliseArea==1 
            images_col=images_col_resized_centered;
        end
        % back image
        imageB_col=images_col(:,:,:,imageI_back); imageB_gs=rgb2gray(imageB_col); imageB_sh=colour2silhouette(imageB_col);
        imageB_sh_col_R=imageB_sh; imageB_sh_col_G=imageB_sh; imageB_sh_col_B=imageB_sh; imageB_sh_col_R(imageB_sh==0)=0.7; imageB_sh_col_G(imageB_sh==0)=1; imageB_sh_col_B(imageB_sh==0)=0.7; imageB_sh_col=cat(3,imageB_sh_col_R,imageB_sh_col_G,imageB_sh_col_B);    
        imageB_sh_gs=imageB_sh; imageB_sh_gs(imageB_sh==0)=0.3;
        % front image
        imageF_col=images_col(:,:,:,imageI_front); imageF_gs=rgb2gray(imageF_col); imageF_sh=colour2silhouette(imageF_col); 
        imageF_sh_col_R=imageF_sh; imageF_sh_col_G=imageF_sh; imageF_sh_col_B=imageF_sh; imageF_sh_col_R(imageF_sh==0)=0.7; imageF_sh_col_G(imageF_sh==0)=0.7; imageF_sh_col_B(imageF_sh==0)=1; imageF_sh_col=cat(3,imageF_sh_col_R,imageF_sh_col_G,imageF_sh_col_B);    
        imageF_sh_gs=imageF_sh; 
        if stimFormat==0, imageB=imageB_col; imageF=imageF_col; elseif stimFormat==1, imageB=imageB_gs; imageF=imageF_gs; 
        elseif stimFormat==2, imageB=imageB_sh_col; imageF=imageF_sh_col; elseif stimFormat>2, imageB=imageB_sh_gs; imageF=imageF_sh_gs; end
        if monitor
            figure(107); clf;
            subplot(3,2,1); imshow(imageB); ylabel('original'); subplot(3,2,2); imshow(imageF);
        end
        % move front image relative to back image (in the x direction)
        image_height=size(imageB,1); % the two images are assumed to be of equal size
        image_width=size(imageB,2); 
        add2image=uint(128*ones(image_height,abs(objectF_x_offset))); if stimFormat>1, add2image=0.5*ones(image_height,abs(objectF_x_offset)); end
        if stimFormat==0 || stimFormat==2, add2image=repmat(add2image,[1 1 3]); end
        if objectF_x_offset>0
            imageB=[imageB,add2image];
            imageF=[add2image,imageF];
            imageB_sh=[imageB_sh,ones(image_height,objectF_x_offset)];
            imageF_sh=[ones(image_height,objectF_x_offset),imageF_sh];
        elseif objectF_x_offset<0 
            imageB=[add2image,imageB];
            imageF=[imageF,add2image];
            imageB_sh=[ones(image_height,abs(objectF_x_offset)),imageB_sh];
            imageF_sh=[imageF_sh,ones(image_height,abs(objectF_x_offset))];
        end
        % nonlinearly combine front and back image to create stimulus
        if stimFormat==0 || stimFormat==2, imageF_sh_orig=imageF_sh; imageF_sh=repmat(imageF_sh,[1 1 3]); end
        Is_objectF=find(imageF_sh==0);
        imageB_overlap2zero=imageB;
        imageB_overlap2zero(Is_objectF)=0;
        BGIs_imageF=find(imageF_sh);
        imageF_blackBG=imageF; imageF_blackBG(BGIs_imageF)=0;
        stimulus=imageB_overlap2zero+imageF_blackBG; 
        if stimFormat==2, stimulus_sumAlongD3=sum(stimulus,3); stimulus_sumAlongD3=repmat(stimulus_sumAlongD3,[1 1 3]); stimulus(stimulus_sumAlongD3==3)=0.5; elseif stimFormat>2, stimulus(stimulus==1)=0.5; end
        if monitor
            figure(107);
            subplot(3,2,3); imshow(imageB); ylabel('moved'); subplot(3,2,4); imshow(imageF);
            subplot(3,2,5); imshow(imageB_overlap2zero); ylabel('prepared for addition'); subplot(3,2,6); imshow(imageF_blackBG);
        end
        if stimFormat==0 || stimFormat==2, imageF_sh=imageF_sh_orig; Is_objectF=find(imageF_sh==0); end
        % compute diagnostics (e.g. proportion occluded)
        Is_objectB=find(imageB_sh==0);
        Is_objectOverlap=intersect(Is_objectB,Is_objectF);
        propOccluded_objectsB(stimulusI)=numel(Is_objectOverlap)/numel(Is_objectB);       
        Is_objectsB{stimulusI}=Is_objectB; Is_objectsF{stimulusI}=Is_objectF; Is_objectOverlaps{stimulusI}=Is_objectOverlap;
        % center and save the stimulus
        stimulus_sh=imageB_sh+imageF_sh; stimulus_sh=stimulus_sh./2; stimulus_sh(stimulus_sh<1)=0;       
        stimulus_labeled=double(~logical(stimulus_sh));
        stats=regionprops(stimulus_labeled,'basic');
        stimulus_topL_x=ceil(stats.BoundingBox(1)); stimulus_bottomR_x=stimulus_topL_x+floor(stats.BoundingBox(3))-1;
        if stimFormat==0 || stimFormat==2, stimulus_cropped=stimulus(:,stimulus_topL_x:stimulus_bottomR_x,:);
        else stimulus_cropped=stimulus(:,stimulus_topL_x:stimulus_bottomR_x); end        
        stimulus_x_ctr=round(size(stimulus,2)/2);
        stimulus_cropped_x_ctr=round(size(stimulus_cropped,2)/2);
        ctr_diff=abs(stimulus_x_ctr-stimulus_cropped_x_ctr);
        stimulus_centered=uint8(128*ones(size(stimulus))); if stimFormat>1, stimulus_centered=0.5*ones(size(stimulus)); end
        if stimFormat==0 || stimFormat==2, stimulus_centered(:,ctr_diff+1:ctr_diff+size(stimulus_cropped,2),:)=stimulus_cropped;
        else stimulus_centered(:,ctr_diff+1:ctr_diff+size(stimulus_cropped,2))=stimulus_cropped; end
        if monitor
            figure(108); clf;
            subplot(1,3,1); imshow(stimulus);
            if objectF_x_offset==0,
                xlabel('front object centered on back object'); 
            else
                xlabel(['front object ',num2str(abs(objectF_x_offset)),' pix to the ',offset_direction_str]);
            end
            ylabel([num2str(propOccluded_objectsB(stimulusI)*100,'%3.1f'),'% occluded']);
            title('original');
            subplot(1,3,2); imshow(stimulus_sh);
            title('silhouette');
            subplot(1,3,3); imshow(stimulus_centered);
            title('centered');
        end
        saveStr=[norm_str,'_',stimFormat_str,'_',offset_str];
        stim_dir=fullfile(path,'stimuli',norm_str,stimFormat_str,offset_str); try mkdir(stim_dir); end
        if stimFormat==4 
            figure(109); clf;
            contour(1:size(stimulus_centered,2),1:size(stimulus_centered,1),flipud(stimulus_centered),2,'Color','k','LineWidth',2); axis off;
            print('-dbmpmono',fullfile(stim_dir,[saveStr,'_stimulus',num2str(stimulusI),'.bmp']));
            stimulus_centered=double(imread(fullfile(stim_dir,[saveStr,'_stimulus',num2str(stimulusI),'.bmp'])));            
            stimulus_centered(stimulus_centered==0)=0.5; stimulus_centered(stimulus_centered==1)=0;
        else
            imwrite(stimulus_centered,fullfile(stim_dir,[saveStr,'_stimulus',num2str(stimulusI),'.bmp']),'bmp');
        end
        if stimFormat==0 || stimFormat==2, stimuli(:,:,:,stimulusI)=stimulus_centered; else stimuli(:,:,stimulusI)=stimulus_centered; end 
        stimulusRow=[stimulusRow,stimulus_centered];
        stimulusI=stimulusI+1;
    end % imageI_front 
    imwrite(stimulusRow,fullfile(stim_dir,['stimulusRow_',saveStr,'_backImage',num2str(imageI_back),'.bmp']),'bmp');
end % imageI_back 
save(['stimuli_',saveStr],'stimuli','-v7.3');
save(['stimuli_',saveStr,'_diagnostics'],'Is_objectsB','Is_objectsF','Is_objectOverlaps','propOccluded_objectsB');

if monitor 
    figure(110); clf;
    subplot(2,1,1); hist(propOccluded_objectsB);
    xlabel('proportion of background object occluded'); ylabel('frequency');
    title('how much of background object is occluded?');
    subplot(2,1,2); 
    [propOccluded_objectsB_sorted,Is]=sort(propOccluded_objectsB);
    plot(propOccluded_objectsB_sorted);
    xlabel('stimuli'); ylabel('proportion of background object occluded'); 
    title('stimuli ranked by prop occl');
    addHeadingAndPrint('proportion of background object occluded',fullfile(stim_dir,['stimulusDiagnostics_',saveStr,'.ps']),110);
    
    % plot all stimuli, sorted from least to most overlap (multiple pages, save each as bmp)
    nRows=40;
    nCols=15;
    nStimuliPerPage=nRows*nCols;
    nStimuli=numel(Is_objectsB);
    nPages=ceil(nStimuli/nStimuliPerPage);
    if stimFormat==0 || stimFormat==2, stimuli_sorted=stimuli(:,:,:,Is); else stimuli_sorted=stimuli(:,:,Is); end
    for pageI=1:nPages        
        mosaic=[];        
        if pageI<nPages
            if stimFormat==0 || stimFormat==2, stimuli_sorted_cPage=stimuli_sorted(:,:,:,(pageI-1)*nStimuliPerPage+1:pageI*nStimuliPerPage);
            else stimuli_sorted_cPage=stimuli_sorted(:,:,(pageI-1)*nStimuliPerPage+1:pageI*nStimuliPerPage); end
        elseif pageI==nPages
            if stimFormat==0 || stimFormat==2
                stimuli_sorted_cPage=stimuli_sorted(:,:,:,(pageI-1)*nStimuliPerPage+1:end);
                nStimuli_lastPage=size(stimuli_sorted_cPage,4);
            else
                stimuli_sorted_cPage=stimuli_sorted(:,:,(pageI-1)*nStimuliPerPage+1:end);
                nStimuli_lastPage=size(stimuli_sorted_cPage,3);
            end
            nRows=ceil(nStimuli_lastPage/nCols);
            nSlots_lastPage=nRows*nCols;
            if stimFormat<2, grayVal=128; else grayVal=0.5; end
            if stimFormat==0 || stimFormat==2
                stimuli_sorted_cPage=cat(4,stimuli_sorted_cPage,grayVal*ones(size(stimuli_sorted_cPage,1),size(stimuli_sorted_cPage,2),size(stimuli_sorted_cPage,3),nSlots_lastPage-nStimuli_lastPage));                
            else
                stimuli_sorted_cPage=cat(3,stimuli_sorted_cPage,grayVal*ones(size(stimuli_sorted_cPage,1),size(stimuli_sorted_cPage,2),nSlots_lastPage-nStimuli_lastPage));
            end
        end
        for rowI=1:nRows
            if stimFormat==0 || stimFormat==2
                stimuli_cRow=stimuli_sorted_cPage(:,:,:,(rowI-1)*nCols+1:rowI*nCols);
                stimulusRow=[];
                for stimulusI=1:size(stimuli_cRow,4)
                    stimulusRow=cat(2,stimulusRow,stimuli_cRow(:,:,:,stimulusI));
                end
            else
                stimuli_cRow=stimuli_sorted_cPage(:,:,(rowI-1)*nCols+1:rowI*nCols);
                stimulusRow=reshape(stimuli_cRow,[size(stimuli_cRow,1),size(stimuli_cRow,2)*size(stimuli_cRow,3)]);
            end
            mosaic=[mosaic;stimulusRow]; 
        end
        figure(111); clf; imshow(mosaic);
        imwrite(mosaic,fullfile(stim_dir,['stimulusMosaic_',saveStr,'_page',num2str(pageI),'.bmp']),'bmp');
    end
    
    % scatterplot of proportion overlap for each object pair (A in front, B occluded on one axis; B in front, A occluded on the other axis)
    stimulusIs=1:nStimuli;
    stimulusIs_mat=reshape(stimulusIs,[nImages nImages]);
    stimulusIs_mat_inv=stimulusIs_mat';
    mask_ltv=logical(tril(ones(nImages,nImages),-1));
    stimulusIs_ltv=stimulusIs_mat(mask_ltv);    
    stimulusIs_utv=stimulusIs_mat_inv(mask_ltv);
    pageFigure(112); clf; scatter(propOccluded_objectsB(stimulusIs_ltv),propOccluded_objectsB(stimulusIs_utv));
    title('proportion occluded for both depth orders (each dot refers to one image pair)'); ylabel('proportion of object A occluded (e.g. image 1 in front of 2)'); xlabel('proportion of object B occluded (e.g. image 2 in front of 1)');      
    [r_ltv,c_ltv]=ind2sub([nImages nImages],stimulusIs_ltv); % indices to the pairs of images making up the stimuli (rowI = front image, colI = back image) 
    [r_utv,c_utv]=ind2sub([nImages nImages],stimulusIs_utv); % indices to the pairs of images making up the stimuli (rowI = front image, colI = back image) - complement of previous line (occluder now the occluded)
    addHeadingAndPrint('proportion occluded for each object pair',fullfile(stim_dir,['stimulusDiagnostics_',saveStr,'.ps']),112);
    save(['stimuli_',saveStr,'_diagnostics'],'stimulusIs_mat','stimulusIs_ltv','stimulusIs_utv','r_ltv','c_ltv','r_utv','c_utv','-append');  
    % same scatterplot with actual stimuli (only show one depth order, in this case the one shown on the y axis)
    for stimulusI=stimulusIs
        if stimFormat==0 || stimFormat==2, stimulusData(stimulusI).image=stimuli(:,:,:,stimulusI);
        else stimulusData(stimulusI).image=repmat(stimuli(:,:,stimulusI),[1 1 3]); end
    end
    pageFigure(113); clf;
    if stimFormat<2, transparentCol=[128 128 128]; else transparentCol=[.5 .5 .5]; end
    drawImageArrangement_mm(stimulusData(stimulusIs_utv),[propOccluded_objectsB(stimulusIs_ltv)',propOccluded_objectsB(stimulusIs_utv)'],1,transparentCol);
    title('proportion occluded for both depth orders (each dot refers to one image pair)'); ylabel('\bfproportion of object A occluded (e.g. image 1 in front of 2)'); xlabel('proportion of object B occluded (e.g. image 2 in front of 1)');
    addHeadingAndPrint('proportion occluded for each object pair',fullfile(stim_dir,['stimulusDiagnostics_',saveStr,'.ps']),113);
end


%% control occluder
saveStr=[norm_str,'_',stimFormat_str,'_',offset_str];
stimulusRow_controlOccl1=[];
stimulusRow_controlOccl2=[];
for imageI=imageIs
    if stimFormat==0, image=images_col(:,:,:,imageI); elseif stimFormat==1, image=images_gs(:,:,imageI);
    elseif stimFormat>1, image=images_sh(:,:,imageI); end
    controlOccluder=0.5*ones([size(image,1) size(image,2)]);
    controlOccluder_hor_1bar=controlOccluder; controlOccluder_hor_2bars=controlOccluder;
    controlOccluder_hor_1bar(round(size(image,1)/3):2*round(size(image,1)/3),:)=0;
    controlOccluder_hor_2bars(round(size(image,1)/5):2*round(size(image,1)/5),:)=0; controlOccluder_hor_2bars(3*round(size(image,1)/5):4*round(size(image,1)/5),:)=0;
    if stimFormat==0 || stimFormat==2, controlOccluder_hor_1bar=repmat(controlOccluder_hor_1bar,[1 1 3]); controlOccluder_hor_2bars=repmat(controlOccluder_hor_2bars,[1 1 3]); end
    stimulus_contrOccl1=image; stimulus_contrOccl2=image;
    stimulus_contrOccl1(controlOccluder_hor_1bar==0)=0; stimulus_contrOccl2(controlOccluder_hor_2bars==0)=0;
    if stimFormat==0 || stimFormat==2, stimuli_contrOccl1(:,:,:,imageI)=stimulus_contrOccl1;  stimuli_contrOccl2(:,:,:,imageI)=stimulus_contrOccl2;
    else stimuli_contrOccl1(:,:,imageI)=stimulus_contrOccl1; stimuli_contrOccl2(:,:,imageI)=stimulus_contrOccl2; end
    save(['stimuli_controlOccluders',saveStr],'stimuli_contrOccl1','stimuli_contrOccl2');
    if monitor
        pageFigure (114); clf;
        subplot(2,2,1); imshow(controlOccluder_hor_1bar);
        subplot(2,2,2); imshow(controlOccluder_hor_2bars);
        subplot(2,2,3); imshow(stimulus_contrOccl1);
        subplot(2,2,4); imshow(stimulus_contrOccl2);
    end
    stimulusRow_controlOccl1=[stimulusRow_controlOccl1,stimulus_contrOccl1];
    stimulusRow_controlOccl2=[stimulusRow_controlOccl2,stimulus_contrOccl2];   
end
imwrite(stimulusRow_controlOccl1,fullfile(stim_dir,['stimulusRow_controlOccluder_1bar',saveStr,'.bmp']),'bmp');
imwrite(stimulusRow_controlOccl2,fullfile(stim_dir,['stimulusRow_controlOccluder_2bars',saveStr,'.bmp']),'bmp');


% % obsolete
% row_marginals=sum(image_sh,2);
% object_rowIs=find(row_marginals<size(image_sh,2)); object_rowIstart=object_rowIs(1); object_rowIend=object_rowIs(end)+1;
% object_height=object_rowIend-object_rowIstart; object_y_ctr=ceil(object_height/2);
% col_marginals=sum(image_sh);
% object_colIs=find(col_marginals<size(image_sh,1)); object_colIstart=object_colIs(1); object_colIend=object_colIs(end)+1;
% object_width=object_colIend-object_colIstart; object_x_ctr=ceil(object_width/2);
% object_y_ctr_inImage1=object_rowIstart+object_y_ctr;
% object_x_ctr_inImage1=object_colIstart+object_x_ctr;





%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% convert colour image to silhouette image 
function image_sh=colour2silhouette(image_col)

for rgbChannelI=1:3
    image_col_cChan=image_col(:,:,rgbChannelI);
    image_sh_cChan=ones(size(image_col_cChan));
    image_sh_cChan(find(image_col_cChan~=128))=0;
    image_sh(:,:,rgbChannelI)=image_sh_cChan;
end
image_sh=sum(image_sh,3);
image_sh=image_sh./3;
image_sh(image_sh<1)=0;

