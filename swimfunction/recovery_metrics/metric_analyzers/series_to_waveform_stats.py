''' Calculates waveform stats from series of measurements.
'''

from typing import List
from tqdm import tqdm
import numpy
from scipy import signal
from collections import namedtuple

from swimfunction.global_config.config import config

FPS = config.getint('VIDEO', 'fps')

WaveformStats = namedtuple(
    'WaveformStats',
    ['periods', 'frequencies', 'amplitudes', 'start_end_frames']
)

def get_alternating_argextrema(one_series: numpy.ndarray) -> numpy.ndarray:
    ''' Locates extrema in series. Ensures alternates crest-trough-crest-trough-...

    Parameters
    ----------
    one_series : numpy.ndarray
        1D array

    Returns
    -------
    peaks : numpy.ndarray
    '''
    argpeaks = signal.argrelextrema(one_series, numpy.greater)[0]
    argtroughs = signal.argrelextrema(one_series, numpy.less)[0]
    argextrema = []
    # Using len because it may be list or numpy.ndarray
    # pylint: disable-next=use-implicit-booleaness-not-len
    if not len(argpeaks) or not len(argtroughs):
        return numpy.asarray(argextrema, dtype=int)
    ti = 0
    pi = 0
    collecting_peak = argpeaks[0] < argtroughs[0]
    def collect_extrema(argvals, i):
        while argextrema and i < len(argvals) and argvals[i] < argextrema[-1]:
            # ignore any peaks that got skipped (if two peaks in a row, for example)
            i += 1
        return i
    while pi < len(argpeaks) or ti < len(argtroughs):
        if collecting_peak:
            pi = collect_extrema(argpeaks, pi)
            if pi < len(argpeaks):
                argextrema.append(argpeaks[pi])
            pi += 1
        else:
            ti = collect_extrema(argtroughs, ti)
            if ti < len(argtroughs):
                argextrema.append(argtroughs[ti])
            ti += 1
        collecting_peak = not collecting_peak
    return numpy.asarray(argextrema, dtype=int)

def remove_false_waves(
        argpeaks: numpy.ndarray, one_series: numpy.ndarray) -> numpy.ndarray:
    ''' If the distance between two extrema is less than a threshold,
    then it skips the next two extrema
    (to make sure the peak-trough-peak-trough-... pattern is kept)
    '''
    if argpeaks.size < 3:
        return argpeaks
    approx_amplitudes = numpy.abs(one_series[argpeaks[:-1]] - one_series[argpeaks[1:]])
    threshold = numpy.median(approx_amplitudes) * 0.1
    if numpy.all(approx_amplitudes > threshold):
        return argpeaks
    new_peaks = []
    i = 0
    while i < argpeaks.size:
        if i < approx_amplitudes.size and approx_amplitudes[i] < threshold:
            i += 2
        else:
            new_peaks.append(argpeaks[i])
            i += 1
    return numpy.asarray(new_peaks, dtype=int)

def peaks_to_amplitudes2d(peaks2d) -> numpy.ndarray:
    ''' Estimates amplitudes for end-to-end windows of three extrema
    (peak-trough-peak; peak-trough-peak; ...)
    Note: amplitude is half the peak-to-peak distance, or the distance
    from a peak to the midline.

    Note on Implementation
    ----------------------
    A wave with the following peak-trough pattern
        p-t-p-t-p-t-p
    has three measurable periods (peak to peak)
    and three measurable amplitudes (peak to trough to peak).

    Midline is estimated by the line through
    midpoint(peak1->peak2) and midpoint(peak2->peak3),
    not assumed to be at y=0

    A measureable amplitude is the distance between three successive extrema divided by 4.
    Basically it is the average of peak to trough divided by 2 and trough to next peak divided by 2.
    The value is calculated from the original series, but the limits
    (beginning and end of waveform) are calculated from the series minus the trend.

    https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
    '''
    peak1 = peaks2d[:-2:2]
    peak2 = peaks2d[1:-1:2]
    peak3 = peaks2d[2::2]
    midp1 = (peak2 + peak1) / 2
    midp2 = (peak3 + peak2) / 2
    def dist_to_midline(point):
        return numpy.abs(
            numpy.cross(midp2 - midp1, midp1 - point, axis=1)
            / numpy.linalg.norm(midp2 - midp1, axis=1)
        )
    amps = (
        dist_to_midline(peak1) + (2 * dist_to_midline(peak2)) + dist_to_midline(peak3)
    ) / 4
    return amps

def argpeaks_to_periods(argpeaks: numpy.ndarray) -> numpy.ndarray:
    ''' Estimates extreme-to-extreme periods (unit: frames) for end-to-end windows of three extrema
        (peak-trough-peak; peak-trough-peak; ...)

    Note on Implementation
    ----------------------
    A wave with the following peak-trough pattern
        p-t-p-t-p-t-p
    has three measurable periods (peak to peak)
    and three measurable amplitudes (peak to trough to peak).

    A measurable period is the time it takes for the time series to
    go from one extrema to the next extrema of the same type (peak to peak or trough to trough).
    The extrema type is the first extrema observed, peak or trough.
    Shouldn't matter which one is chosen.
    This is calculated from the series minus the trend.
    '''
    return argpeaks[2::2] - argpeaks[:-2:2]

def periods_to_frequencies(periods: numpy.ndarray) -> numpy.ndarray:
    ''' Converts periods to frequencies
    '''
    # Using len because it may be list or numpy.ndarray
    # pylint: disable-next=use-implicit-booleaness-not-len
    if periods is None or not len(periods):
        return []
    return FPS / periods

def get_waveform_stats_from_series(
        one_series: numpy.ndarray,
        series_idx: list,
        min_waveforms_per_cruise: int) -> WaveformStats:
    ''' Gets all measurable periods and amplitudes for each dimension in series.

    Parameters
    ----------
    one_series : numpy.ndarray
        NxD array where N is number of observations and D is number of measured dimensions
    min_waveforms_per_cruise : int
        If there is no dimension that has at least this number of waveforms, None is returned.

    Returns
    -------
    WaveformStats : namedtuple, None
        (periods, frequencies, amplitudes, start_end_frames) or None if does not have min_waveforms_per_cruise
    '''
    series = numpy.asarray(one_series)
    if len(series.shape) == 1:
        series = series[:, numpy.newaxis]
    periods = [[] for _ in range(series.shape[1])]
    amplitudes = [[] for _ in range(series.shape[1])]
    for d in range(series.shape[1]):
        argpeaks = remove_false_waves(get_alternating_argextrema(
            series[:, d]).astype(int), series[:, d])
        if len(argpeaks) < 3:
            continue
        periods[d] = argpeaks_to_periods(argpeaks)
        peaks2d = numpy.asarray([argpeaks, numpy.take(series[:, d], argpeaks)]).T
        amplitudes[d] = peaks_to_amplitudes2d(peaks2d)
    if max([len(x) for x in periods]) < min_waveforms_per_cruise:
        return None
    frequencies = list(map(periods_to_frequencies, periods))
    return WaveformStats(periods, frequencies, amplitudes, [series_idx[0], series_idx[-1]])

def get_waveform_stats_from_many_series(
        episodes, episodes_idx, min_waveforms_per_cruise) -> List[WaveformStats]:
    ''' Get waveform features from a list of episodes
    '''
    return list(filter(
        lambda x: x is not None,
        [
            get_waveform_stats_from_series(e, e_idx, min_waveforms_per_cruise)
            for e, e_idx in tqdm(
                zip(episodes, episodes_idx),
                total=len(episodes),
                leave=False)
        ]))
