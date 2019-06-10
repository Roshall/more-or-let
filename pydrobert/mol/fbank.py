import numpy
import scipy.io.wavfile


def get_filterbanks(nfilt=40, nfft=512, samplerate=16000, lowfreq=20, highfreq=8000):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


def delta(feat):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """

    NUMFRAMES = len(feat)
    denominator = 5
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((2, 2), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-2, 3), padded[t : t+5]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def main():
    sample_rate, signal = scipy.io.wavfile.read('61-70970-0032.wav')  # File assumed to be in the same directory
    signal = signal[0:int(2 * sample_rate)]  # Keep the first 2 seconds

    pre_emphasis = 0.97
    frames_num = 320
    frame_length = int(sample_rate * 0.025)
    frame_stride = int(sample_rate * 0.01)

    signal_length = len(signal)
    signal = numpy.concatenate((signal[:1], signal[1:] - pre_emphasis * signal[:-1], numpy.zeros(frames_num, dtype=numpy.float32)), 0)

    windowed = numpy.stack([signal[i:i + frame_length] for i in range(0, signal_length - frames_num, frame_stride)], 1)

    NFFT = 512
    ffted = numpy.fft.rfft(windowed, 512)
    pow_frames = 1.0 / NFFT * numpy.square(numpy.abs(ffted))

    fbank = get_filterbanks()
    filter_banks = numpy.dot(pow_frames, fbank.T) + 1e-30

    engergy = numpy.sum(pow_frames,1)
    fbank_plus_energy = numpy.concatenate((engergy.reshape(-1,1),filter_banks),1)
    fbank_plus_energy = numpy.log(fbank_plus_energy)

    deltas = delta(fbank_plus_energy)
    delta_deltas = delta(deltas)

    features = numpy.concatenate((fbank_plus_energy, deltas, delta_deltas),1)

    return features


if __name__ == "__main__":
    main()
