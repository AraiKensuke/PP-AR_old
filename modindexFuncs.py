import scipy.signal as _ssig

def generatePoissonTrain(totalTime, rateFunction, refractoryPeriod):
    # Generate poisson spike train using a rate function

    #spikeTrain=_N.zeros(totalTime, dtype=_N.int)
    spikeTrain=_N.zeros(totalTime)
    if (refractoryPeriod == 0):
        sts = _N.where(_N.random.rand(totalTime) < rateFunction)[0]
        spikeTrain[sts] = 1
    else:
        rands = _N.random.rand(totalTime)
        for i in xrange(totalTime):
            if rands[i] < rateFunction[i]:
                spikeTrain[i]=1
                maxJ = refractoryPeriod if i+refractoryPeriod < totalTime else (totalTime - (i + refractoryPeriod))
                for j in xrange(maxJ): # set ref period rate to 0
                    rateFunction[i+j]= 0

    return spikeTrain

def powerSpectrum(spikeTrain, fs, startBand, endBand, winLen):
    # This function estimates the power spectrum of a spike train using Welch's
    # method. It return  the peak in a specific searched band.
    #  Input parameters:
    #   spikeTrain: point process spike train
    #   fs: the sampling rate of the spike train, in Hz.
    #   stratBand: the low limit of the searched band, in Hz.
    #   endBand: the high limit of the searched band, in Hz.
    #   winLen: the window length to use in pwelch
    #
    # Output parameters:  
    #   spectrum : The spectrum of the spike train
    #   freqRange: corresponding vector of frequncies, in Hz.
    #   snr: the SNR of the peak relative to the 100-500Hz baseline.
    #   peakPower: the power of the highest peak in the searched band
    #   peakFreq: the frequency of the highest peak in the searched band

    overlap=0 
    nfft = winLen
    freqRange, ps = _ssig.welch(spikeTrain,nperseg=winLen,noverlap=overlap,nfft=nfft,fs=fs)

    startBand = _N.where(freqRange <= startBand)[0][-1]
    endBand = _N.where(freqRange >=endBand)[0][0]

    # check for significance
    meanFreqStart = 100
    meanFreqStart = _N.where(freqRange>=meanFreqStart)[0][0]
    meanFreqEnd = 500
    meanFreqEnd = _N.where(freqRange>=meanFreqEnd)[0][0]
    meanPower = _N.mean(ps[meanFreqStart:meanFreqEnd])
    stdPower = _N.std(ps[meanFreqStart:meanFreqEnd])
    mI = _N.argmax(ps[startBand:endBand])     # look for peak in specific band
    peakPower = ps[mI + startBand]
    peakFreq = freqRange[startBand+mI-1]
    snr = (peakPower-meanPower) / stdPower

    return ps, freqRange, snr, peakPower, peakFreq


def getModulationIndex(peakPower, r0, T , winLen, fs):
    # [m] = getModulationIndex(peak, r0, T , winLen, fs)
    # This function takes as input the parameters of a spike train, and its
    # spectrum (estimated using welch's method), and returns the modualtion index of the spike train.
    # Input parameters:
    #   peakPower: the spectral peak's power.
    #   r0: base firing rate.
    #   T: total recording time (length of the spike train).
    #   winLen: the window length used in the calculation of the spectrum.
    #   fs: the sampling rate of the spike train, in Hz.
    #
    # Output parameters:  
    #   m : The modulation index of the spike train

    # First correct the peak for comparison with Welch's estimator
    peak = correctWelchPeak(peakPower, winLen, T, r0, fs)

    if (peak<r0):
        #print "welch peak  %.3f" % peak
        #print "r0          %.3f" % r0
        m =0;
    else:
        m = 2*(_N.sqrt(peak-r0))
        m = m/(r0*_N.sqrt(T))
        m = abs(m)

    return m

def correctWelchPeak(peak, winLen, T, r0,fs):
    # This function scales the power estimated by Welch's method to the
    # analytically calculated power
    # Input parameters:
    #   peak: the spectral peak's power.
    #   winLen: the window length used in the calculation of the spectrum.
    #   T: total recording time (length of the spike train).
    #   r0: base firing rate.
    #   fs: the sampling rate of the spike train, in Hz.
    #
    # Output parameters:  
    #   correctedPeak : the corrected peak's power.

    w= _N.hamming(winLen)
    U = _N.dot(w, w)
    welchFactor = (_N.mean(w)*winLen)/U
    nWins = _N.ceil(T/winLen)

    peak = peak*fs/2
    peak = (nWins*peak - ((nWins-1)*r0))
    correctedPeak = peak*welchFactor
    return correctedPeak


def getModIndexWithRefPer(spikeTrain, refPeriod, fs, freq, winLen):
    # This function includes a correction procedure that corrects the
    # modulation index to accomodate for the refractory period.
    # Input parameters:
    #   spikeTrain: point process spike train
    #   refPeriod: length of the absolute refractory period, in ms
    #   fs: the sampling rate of the spike train, in Hz.
    #   freq: the frequncy of the spectral peak.
    #   winLen: the window length to use in the estimation of the spectrum
    #
    # Output parameters:  
    #   modulation : The corrected modulation index of the spike train
    #   originalModulation : The original modulation index of the spike train, without correction.

    startBand = freq-3
    endBand = freq+3
    totalTime = len(spikeTrain)
    firingRate = _N.sum(spikeTrain)/(totalTime/1000)
    print "fr  %.3f" % firingRate
    ps, freqRange,snr, peakPower, peakFreq = powerSpectrum(spikeTrain, fs, startBand, endBand, winLen);
    originalModulation=  getModulationIndex(peakPower, firingRate/1000 ,totalTime, 1000, fs);

    modulation = originalModulation
    if (originalModulation==0):
        print "oM is 0"
        return 0, 0;
    if (originalModulation > 1):
        print "oM > 1"
        return 0, 0

    # calculate the firing rate without refractory period
    totalTimeWithoutRef = totalTime - refPeriod*_N.sum(spikeTrain)
    meanRateWithoutRef  = (_N.sum(spikeTrain)/(totalTimeWithoutRef /fs))/fs

    time=_N.arange(0, totalTime/fs, 1./fs)

    ex=1
    M= 100
    while (ex > 0):
        rateFunc = (modulation*meanRateWithoutRef) *(_N.cos(2*_N.pi*freq*time))+meanRateWithoutRef
        amps = _N.zeros(M)
        for j in xrange(M):
            st= generatePoissonTrain(totalTime, rateFunc, refPeriod)
            firingRate = _N.sum(st)/(totalTime/1000)
            ps, freqRange,snr, peakPower, peakFreq = powerSpectrum(st, fs, startBand, endBand, winLen)

            amps[j] =  getModulationIndex(peakPower, firingRate/1000 ,totalTime  , 1000, fs);
        reconstructedMods = _N.mean(amps);
        if (reconstructedMods >=originalModulation):
            ex=0
        else:
            modulation = modulation+0.01

    print "%(m).3f   %(om).3f" % {"m" : modulation, "om" : originalModulation}
    return modulation, originalModulation



