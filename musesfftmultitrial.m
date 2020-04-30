numoftrials = 20;
museData = cell(numoftrials,1);
museElements=cell(numoftrials,1);
for n = 1:numoftrials
   [ museData{n}, museElements{n} ]= mmImport(sprintf('/Users/mahima/Desktop/research/df1red%d.csv',n-1));
   ![ museDatared{n}, museElementsred{n} ]= mmImport(sprintf('/Users/mahima/Desktop/research/df1red%d.csv',n-1));
   ![ museDatablue{n}, museElementsblue{n} ]= mmImport(sprintf('/Users/mahima/Desktop/research/df1blue%d.csv',n-1));
end
k=cell(1)
n=cell(1)
for i = 1:numoftrials
    [k{i},n{i}]=size(museData{i});
    ![k{i},n{i}]=size(museDatagreen{i});
    ![k{i},n{i}]=size(museDatagreen{i});
end
m=k{1};
for i = 1:numoftrials
    
    m=min(m,k{i});
    if m==k{i}
        mtime=transpose(museData{i}.TimeStamp);
    end
end

for i = 1:numoftrials
    museData{i}=museData{i}(1:m,:);
end

data_TP10=zeros(numoftrials,m);
data_AF7=zeros(numoftrials,m);
data_AF8=zeros(numoftrials,m);
data_TP9=zeros(numoftrials,m);
for i = 1:numoftrials
    data_TP10(i,:)=[transpose(museData{i}.RAW_TP10)];
    data_AF7(i,:)=[transpose(museData{i}.RAW_AF7)];
    data_AF8(i,:)=[transpose(museData{i}.RAW_AF8)];
    data_TP9(i,:)=[transpose(museData{i}.RAW_TP9)];
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
srate=256
time = -2:1/srate:2;
tmax=max(mtime)
tmin=min(mtime)

frex       = logspace(log10(2),log10(srate/10),20);
times2save = tmin+2:0.125:tmax-1;
basetime   = [1 2];
timewin    = 1.5; % in ms

% convert time points to indices
times2saveidx = dsearchn(mtime',times2save');
% convert time window to points
timewinpnts   = round(timewin*srate);

% find baselinetimepoints
baseidx = dsearchn(times2save',basetime');

% define frequencies for FFT
hz = linspace(0,srate/2,timewinpnts/2+1);

% hanning window for tapering
hannwin = .5 - .5*cos(2*pi.*linspace(0,1,timewinpnts))';


shortFFT_tf = zeros(length(frex),length(times2save));

db_shortFFT_tf = zeros(4,length(frex),length(times2save));
for cyclei=1:4
 for ti=1:length(times2saveidx)
        !data_TP10=transpose(data_TP10);
       % data_TP10=reshape(data_TP10,1,n,m);
        !newdata(1,:,:)=(data(:,times2saveidx(ti)-floor(timewinpnts/2)+1:times2saveidx(ti)+ceil(timewinpnts/2)));
    % window and taper data, and get power spectrum
        bdata = bsxfun(@times, squeeze(newdata(cyclei,times2saveidx(ti)-floor(timewinpnts/2)+1:times2saveidx(ti)+ceil(timewinpnts/2),:)), hannwin);
    
    % uncomment the next line to use non-tapegreen data
        %bdata = squeeze(data(:,times2saveidx(ti)-floor(timewinpnts/2)+1:times2saveidx(ti)+ceil(timewinpnts/2)));
    
        y    = fft(bdata,timewinpnts)/timewinpnts;
        pow  = mean(abs(y).^2,2);
    
    % finally, get power from closest frequency
        closestfreq = dsearchn(hz',frex');
        shortFFT_tf(:,ti) = pow(closestfreq);    
    end % end time loop
db_shortFFT_tf(cyclei,:,:) = 10*log10( bsxfun(@rdivide,shortFFT_tf,mean(shortFFT_tf(:,baseidx(1):baseidx(2)),2)) );
end
%dbmin=min(db_shortFFT_tf,[],'all');
%dbmax=max(db_shortFFT_tf,[],'all');
% plot!
figure(1), clf
for cyclei=1:4
    subplot(2,2,cyclei)
    tmax=max(mtime)
    tmin=min(mtime)
    dbmin=min(db_shortFFT_tf(cyclei,:,:),[],'all');
    dbmax=max(db_shortFFT_tf(cyclei,:,:),[],'all');

   
    contourf(times2save,frex,squeeze(db_shortFFT_tf(cyclei,:,:)),40,'linecolor','none');
%caxis([dbmin dbmax])
    colorbar
    
    set(gca,'clim',[-3 3])
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
    title([ 'Power via short-window FFT (window=' num2str(timewin) ') from channel '  ])
end
