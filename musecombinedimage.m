numoftrials = 5;
upperfilelimit=20
startfile=16
color="blue"
image_num=12
museData = cell(numoftrials,1);
museElements=cell(numoftrials,1);
for n = startfile:upperfilelimit
   [ museData{n}, museElements{n} ]= mmImport(sprintf('/Users/mahima/research/df1%s%d.csv',color,n-1));
   ![ museDatared{n}, museElementsred{n} ]= mmImport(sprintf('/Users/mahima/Desktop/research/df1red%d.csv',n-1));
   ![ museDatablue{n}, museElementsblue{n} ]= mmImport(sprintf('/Users/mahima/Desktop/research/df1blue%d.csv',n-1));
end
k=cell(1)
n=cell(1)
j=startfile
for i = 1:numoftrials
    
    [k{i},n{i}]=size(museData{j});
    j=j+1
    ![k{i},n{i}]=size(museDatagreen{i});
    ![k{i},n{i}]=size(museDatagreen{i});
end
m=k{1};
j=startfile-1
for i = 1:numoftrials
    
    m=min(m,k{i});
    j=j+1
    if m==k{i}
        
        disp(j)
        mtime=transpose(museData{j}.TimeStamp);
        
    end
end

for i = startfile:upperfilelimit
    museData{i}=museData{i}(1:m,:);
end

data_TP10=zeros(numoftrials,m);
data_AF7=zeros(numoftrials,m);
data_AF8=zeros(numoftrials,m);
data_TP9=zeros(numoftrials,m);
j=startfile
for i = 1:numoftrials
    data_TP10(i,:)=[transpose(museData{j}.RAW_TP10)];
    data_AF7(i,:)=[transpose(museData{j}.RAW_AF7)];
    data_AF8(i,:)=[transpose(museData{j}.RAW_AF8)];
    data_TP9(i,:)=[transpose(museData{j}.RAW_TP9)];
    j=j+1
end

newdata= zeros(4,m,numoftrials);
newdata(1,:,:)=transpose(data_TP10);
newdata(2,:,:)=transpose(data_AF7);
newdata(3,:,:)=transpose(data_AF8);
newdata(4,:,:)=transpose(data_TP9);
    
mtime1= datevec(mtime);
mtime=mtime1(:,6);
tt=diff(mtime);
tt(tt<0)=60+tt(tt<0);
tt=cumsum(tt);
mtime=tt;
[u,I,J] = unique(mtime, 'rows', 'first');
ixDupRows = setdiff(1:size(mtime,1), I);
!ixDupRows= ixDupRows(:,1:length(ixDupRows)-1);
for i=1:length(ixDupRows)
    mtime(ixDupRows(:,i),:)= mtime(ixDupRows(:,i),:)+(i*0.0000001);
end
[C,ia] = unique(mtime);
mtime = mtime(ia,:);
newdata=newdata(:,ia,:);

mtime=transpose(mtime);
mtime(:, 1)=0;
mtime(:, length(mtime)+1)= mtime(:, length(mtime))+0.01;

name=['1234' ]

![m,n,k]=size(data_TP10);
![m,n,k]=size(data)


% set a few different wavelet widths ("number of cycles" parameter)
num_cycles = [ 2 6 8 15 ];
frex = 6.5;
srate=256
time = -2:1/srate:2;



%% comparing fixed number of wavelet cycles

% wavelet parameters
num_frex = 30;
min_freq =  7;
max_freq = 22;


baseline_window = [ 0.5 1];



% other wavelet parameters
frex = linspace(min_freq,max_freq,num_frex);
time = -1.5:1/srate:1.5;
half_wave = (length(time)-1)/2;

% FFT parameters
nKern = length(time);
nData = m*numoftrials;
nConv = nKern+nData-1;

% initialize output time-frequency data
tf = zeros(4,length(frex),m);

% convert baseline time into indices
[~,baseidx(1)] = min(abs(mtime-baseline_window(1)))
[~,baseidx(2)] = min(abs(mtime-baseline_window(2)))


% FFT of data (doesn't change on frequency iteration)
for cyclei=1:4
  dataX = fft(reshape(newdata(cyclei,:,:),1,[]),nConv);



% loop over cycles

    
    for fi=1:length(frex)
        
        
        % create wavelet and get its FFT
        s = 8/(2*pi*frex(fi));
        %dataX = fft((data(cyclei,:)),nConv);
        %dataX=dataX./max(dataX);
        cmw  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
        cmwX = fft(cmw,nConv);
        cmwX = cmwX./max(cmwX);
        
        % run convolution, trim edges, and reshape to 2D (time X trials)
        as =ifft(cmwX.*dataX,nConv);
        as1 = as(half_wave+1:end-half_wave);
        as=reshape(as1,m,numoftrials);
        %as=reshape(as, EEG.pnts, EEG.trials);
        
        % put power data into big matrix
        tf(cyclei,fi,:) = mean(abs(as).^2,2);
    end
    
    % db conversion
    tf(cyclei,:,:) = 10*log10( bsxfun(@rdivide, squeeze(tf(cyclei,:,:)), mean(tf(cyclei,:,baseidx(1):baseidx(2)),3)' ) );
    
    
end
tmax=max(mtime)
tmin=min(mtime)
map=cell(3,1);
map{1} = [0.2 0 0
    0.1 0 0
    0.2 0 0
    0.8 0 0
    0.9 0 0
    0.8 0 0
    0.55 0 0
    0.2 0 0];

map{2}= [0 0.2 0
    0 0.1 0
    0 0.2 0
    0 0.8 0
    0 0.9 0
    0 0.8 0
    0 0.55 0
    0 0.2 0];

map{3}= [0 0 0.2
    0 0 0.1
    0 0 0.2
    0 0 0.8
    0 0 0.9
    0 0 0.8
    0 0 0.55
    0 0 0.2];
    


% plot results
filename=["TP_10","AF_7","AF_8","TP_9"]
!figure(1), clf;
for cyclei=1:4
    !subplot(2,2,cyclei)
    tmax=max(mtime)
    tmin=min(mtime)
    tfmin=min(tf(cyclei,:,:),[],'all');
    tfmax=max(tf(cyclei,:,:),[],'all');
    f=figure(cyclei), clf;
    contourf(mtime,frex,squeeze(tf(cyclei,:,:)),40,'linecolor','none');
    !colormap(gray)
    !colormap( flipud(gray(256)) )

    set(gca,'clim',[-6 6],'xlim',[1.5 3.5],'XTick',[], 'YTick', []);
    
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 0.425 0.425]);
    set(gca,'LooseInset',get(gca,'TightInset'));
    file=sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(cyclei),image_num)
    !file=sprintf('/Users/mahima/Desktop/%d.png',cyclei)
    saveas(gcf,file)

end


rgbImage1=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(1),image_num));
rgbImage2=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(2),image_num));
rgbImage3=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(3),image_num));
redChannel = rgbImage1(:, :, 1);
greenChannel = rgbImage2(:, :, 2);
blueChannel = rgbImage3(:, :, 3);
combo=cat(3, redChannel,greenChannel,blueChannel);

!file1='/Users/mahima/Desktop/test.png'
 imwrite(combo,sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s%s%s/%d.png',color,filename(1),filename(2),filename(3),image_num));

imshow(combo);

rgbImage1=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(1),image_num));
rgbImage2=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(2),image_num));
rgbImage3=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(4),image_num));
redChannel = rgbImage1(:, :, 1);
greenChannel = rgbImage2(:, :, 2);
blueChannel = rgbImage3(:, :, 3);
combo=cat(3, redChannel,greenChannel,blueChannel);

!file1='/Users/mahima/Desktop/test.png'
 imwrite(combo,sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s%s%s/%d.png',color,filename(1),filename(2),filename(4),image_num));

imshow(combo);

rgbImage1=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(4),image_num));
rgbImage2=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(2),image_num));
rgbImage3=imread(sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s/%d.png',color,filename(3),image_num));
redChannel = rgbImage1(:, :, 1);
greenChannel = rgbImage2(:, :, 2);
blueChannel = rgbImage3(:, :, 3);
combo=cat(3, redChannel,greenChannel,blueChannel);

!file1='/Users/mahima/Desktop/test.png'
 imwrite(combo,sprintf('/Users/mahima/research/museappdata/museimagedata/%s/%s%s%s/%d.png',color,filename(2),filename(3),filename(4),image_num));

imshow(combo);