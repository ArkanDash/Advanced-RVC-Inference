import numpy as np

from sortedcontainers import SortedList

class Timeline:
    @classmethod
    def from_df(cls, df, uri = None):
        return cls(segments=list(df['segment']), uri=uri)

    def __init__(self, segments = None, uri = None):
        if segments is None: segments = ()
        segments_set = set([segment for segment in segments if segment])

        self.segments_set_ = segments_set
        self.segments_list_ = SortedList(segments_set)
        self.segments_boundaries_ = SortedList((boundary for segment in segments_set for boundary in segment))
        self.uri = uri

    def __len__(self):
        return len(self.segments_set_)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        return len(self.segments_set_) > 0

    def __iter__(self):
        return iter(self.segments_list_)

    def __getitem__(self, k):
        return self.segments_list_[k]

    def __eq__(self, other):
        return self.segments_set_ == other.segments_set_

    def __ne__(self, other):
        return self.segments_set_ != other.segments_set_

    def index(self, segment):
        return self.segments_list_.index(segment)

    def add(self, segment):
        segments_set_ = self.segments_set_
        if segment in segments_set_ or not segment: return self

        segments_set_.add(segment)
        self.segments_list_.add(segment)

        segments_boundaries_ = self.segments_boundaries_
        segments_boundaries_.add(segment.start)
        segments_boundaries_.add(segment.end)

        return self

    def remove(self, segment):
        segments_set_ = self.segments_set_
        if segment not in segments_set_: return self

        segments_set_.remove(segment)
        self.segments_list_.remove(segment)

        segments_boundaries_ = self.segments_boundaries_
        segments_boundaries_.remove(segment.start)
        segments_boundaries_.remove(segment.end)

        return self

    def discard(self, segment):
        return self.remove(segment)

    def __ior__(self, timeline):
        return self.update(timeline)

    def update(self, timeline):
        segments_set = self.segments_set_
        segments_set |= timeline.segments_set_

        self.segments_list_ = SortedList(segments_set)
        self.segments_boundaries_ = SortedList((boundary for segment in segments_set for boundary in segment))

        return self

    def __or__(self, timeline):
        return self.union(timeline)

    def union(self, timeline):
        return Timeline(segments=self.segments_set_ | timeline.segments_set_, uri=self.uri)

    def co_iter(self, other):
        for segment in self.segments_list_:
            temp = Segment(start=segment.end, end=segment.end)

            for other_segment in other.segments_list_.irange(maximum=temp):
                if segment.intersects(other_segment): yield segment, other_segment

    def crop_iter(self, support, mode = 'intersection', returns_mapping = False):
        if mode not in {'loose', 'strict', 'intersection'}: raise ValueError
        if not isinstance(support, (Segment, Timeline)): raise TypeError

        if isinstance(support, Segment):
            support = Timeline(segments=([support] if support else []), uri=self.uri)

            for yielded in self.crop_iter(support, mode=mode, returns_mapping=returns_mapping):
                yield yielded

            return

        support = support.support()

        if mode == 'loose':
            for segment, _ in self.co_iter(support):
                yield segment

            return

        if mode == 'strict':
            for segment, other_segment in self.co_iter(support):
                if segment in other_segment: yield segment

            return

        for segment, other_segment in self.co_iter(support):
            mapped_to = segment & other_segment
            if not mapped_to: continue

            if returns_mapping: yield segment, mapped_to
            else: yield mapped_to

    def crop(self, support, mode = 'intersection', returns_mapping = False):
        if mode == 'intersection' and returns_mapping:
            segments, mapping = [], {}
            
            for segment, mapped_to in self.crop_iter(support, mode='intersection', returns_mapping=True):
                segments.append(mapped_to)
                mapping[mapped_to] = mapping.get(mapped_to, list()) + [segment]

            return Timeline(segments=segments, uri=self.uri), mapping

        return Timeline(segments=self.crop_iter(support, mode=mode), uri=self.uri)

    def overlapping(self, t):
        return list(self.overlapping_iter(t))

    def overlapping_iter(self, t):
        for segment in self.segments_list_.irange(maximum=Segment(start=t, end=t)):
            if segment.overlaps(t): yield segment

    def get_overlap(self):
        overlaps_tl = Timeline(uri=self.uri)

        for s1, s2 in self.co_iter(self):
            if s1 == s2: continue

            overlaps_tl.add(s1 & s2)

        return overlaps_tl.support()

    def extrude(self, removed, mode = 'intersection'):
        if isinstance(removed, Segment): removed = Timeline([removed])

        if mode == "loose": mode = "strict"
        elif mode == "strict": mode = "loose"

        return self.crop(removed.gaps(support=Timeline([self.extent()], uri=self.uri)), mode=mode)

    def __str__(self):
        n = len(self.segments_list_)
        string = "["

        for i, segment in enumerate(self.segments_list_):
            string += str(segment)
            string += "\n " if i + 1 < n else ""

        string += "]"
        return string

    def __repr__(self):
        return "<Timeline(uri=%s, segments=%s)>" % (self.uri, list(self.segments_list_))

    def __contains__(self, included):
        if isinstance(included, Segment): return included in self.segments_set_
        elif isinstance(included, Timeline): return self.segments_set_.issuperset(included.segments_set_)
        else: raise TypeError

    def empty(self):
        return Timeline(uri=self.uri)

    def covers(self, other):
        gaps = self.gaps(support=other.extent())

        for _ in gaps.co_iter(other):
            return False

        return True

    def copy(self, segment_func = None):
        if segment_func is None: return Timeline(segments=self.segments_list_, uri=self.uri)
        return Timeline(segments=[segment_func(s) for s in self.segments_list_], uri=self.uri)

    def extent(self):
        if self.segments_set_:
            segments_boundaries_ = self.segments_boundaries_
            return Segment(start=segments_boundaries_[0], end=segments_boundaries_[-1])

        return Segment(start=0.0, end=0.0)

    def support_iter(self, collar = 0.0):
        if not self: return

        new_segment = self.segments_list_[0]

        for segment in self:
            possible_gap = segment ^ new_segment

            if not possible_gap or possible_gap.duration < collar: new_segment |= segment
            else:
                yield new_segment
                new_segment = segment

        yield new_segment

    def support(self, collar = 0.):
        return Timeline(segments=self.support_iter(collar), uri=self.uri)

    def duration(self):
        return sum(s.duration for s in self.support_iter())

    def gaps_iter(self, support = None):
        if support is None: support = self.extent()
        if not isinstance(support, (Segment, Timeline)): raise TypeError

        if isinstance(support, Segment):
            end = support.start

            for segment in self.crop(support, mode='intersection').support():
                gap = Segment(start=end, end=segment.start)
                if gap: yield gap

                end = segment.end

            gap = Segment(start=end, end=support.end)
            if gap: yield gap
        elif isinstance(support, Timeline):
            for segment in support.support():
                for gap in self.gaps_iter(support=segment):
                    yield gap

    def gaps(self, support = None):
        return Timeline(segments=self.gaps_iter(support=support), uri=self.uri)

    def segmentation(self):
        support = self.support()
        timestamps = set([])

        for (start, end) in self:
            timestamps.add(start)
            timestamps.add(end)

        timestamps = sorted(timestamps)
        if len(timestamps) == 0: return Timeline(uri=self.uri)

        segments = []
        start = timestamps[0]

        for end in timestamps[1:]:
            segment = Segment(start=start, end=end)

            if segment and support.overlapping(segment.middle): segments.append(segment)
            start = end

        return Timeline(segments=segments, uri=self.uri)

    def _iter_uem(self):
        uri = self.uri if self.uri else "<NA>"

        for segment in self:
            yield f"{uri} 1 {segment.start:.3f} {segment.end:.3f}\n"

    def to_uem(self):
        return "".join([line for line in self._iter_uem()])

    def write_uem(self, file):
        for line in self._iter_uem():
            file.write(line)

    def _repr_png_(self):
        return None

class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @staticmethod
    def set_precision(ndigits = None):
        global AUTO_ROUND_TIME, SEGMENT_PRECISION

        if ndigits is None:
            AUTO_ROUND_TIME = False
            SEGMENT_PRECISION = 1e-6
        else:
            AUTO_ROUND_TIME = True
            SEGMENT_PRECISION = 10 ** (-ndigits)

    def __bool__(self):
        return bool((self.end - self.start) > SEGMENT_PRECISION)

    def __post_init__(self):
        if AUTO_ROUND_TIME:
            object.__setattr__(self, 'start', int(self.start / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)
            object.__setattr__(self, 'end', int(self.end / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)

    @property
    def duration(self):
        return self.end - self.start if self else 0.

    @property
    def middle(self):
        return .5 * (self.start + self.end)

    def __iter__(self):
        yield self.start
        yield self.end

    def copy(self):
        return Segment(start=self.start, end=self.end)

    def __contains__(self, other):
        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        return Segment(start=max(self.start, other.start), end=min(self.end, other.end))

    def intersects(self, other):
        return (self.start < other.start and other.start < self.end - SEGMENT_PRECISION) or (self.start > other.start and self.start < other.end - SEGMENT_PRECISION) or (self.start == other.start)

    def overlaps(self, t):
        return self.start <= t and self.end >= t

    def __or__(self, other):
        if not self: return other
        if not other: return self

        return Segment(start=min(self.start, other.start), end=max(self.end, other.end))

    def __xor__(self, other):
        if (not self) or (not other): raise ValueError

        return Segment(start=min(self.end, other.end), end=max(self.start, other.start))

    def _str_helper(self, seconds):
        from datetime import timedelta

        negative = seconds < 0
        td = timedelta(seconds=abs(seconds))

        hours, remainder = divmod(td.seconds + 86400 * td.days, 3600)
        minutes, seconds = divmod(remainder, 60)

        return '%s%02d:%02d:%02d.%03d' % ('-' if negative else ' ', hours, minutes, seconds, td.microseconds / 1000)

    def __str__(self):
        if self: return '[%s --> %s]' % (self._str_helper(self.start), self._str_helper(self.end))
        return '[]'

    def __repr__(self):
        return '<Segment(%g, %g)>' % (self.start, self.end)

    def _repr_png_(self):
        return None

class SlidingWindow:
    def __init__(self, duration=0.030, step=0.010, start=0.000, end=None):
        if duration <= 0: raise ValueError
        self.__duration = duration
        if step <= 0: raise ValueError
        
        self.__step = step
        self.__start = start

        if end is None: self.__end = np.inf
        else:
            if end <= start: raise ValueError
            self.__end = end

        self.__i = -1

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def step(self):
        return self.__step

    @property
    def duration(self):
        return self.__duration

    def closest_frame(self, t):
        return int(np.rint((t - self.__start - .5 * self.__duration) / self.__step))

    def samples(self, from_duration, mode = 'strict'):
        if mode == 'strict': return int(np.floor((from_duration - self.duration) / self.step)) + 1
        elif mode == 'loose': return int(np.floor((from_duration + self.duration) / self.step))
        elif mode == 'center': return int(np.rint((from_duration / self.step)))

    def crop(self, focus, mode = 'loose', fixed = None, return_ranges = False):
        if not isinstance(focus, (Segment, Timeline)): raise TypeError

        if isinstance(focus, Timeline):
            if fixed is not None: raise ValueError

            if return_ranges:
                ranges = []

                for i, s in enumerate(focus.support()):
                    rng = self.crop(s, mode=mode, fixed=fixed, return_ranges=True)

                    if i == 0 or rng[0][0] > ranges[-1][1]: ranges += rng
                    else: ranges[-1][1] = rng[0][1]

                return ranges

            return np.unique(np.hstack([self.crop(s, mode=mode, fixed=fixed, return_ranges=False) for s in focus.support()]))

        if mode == 'loose':
            i = int(np.ceil((focus.start - self.duration - self.start) / self.step))

            if fixed is None:
                j = int(np.floor((focus.end - self.start) / self.step))
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode='loose')
                rng = (i, i + n)
        elif mode == 'strict':
            i = int(np.ceil((focus.start - self.start) / self.step))

            if fixed is None:
                j = int(np.floor((focus.end - self.duration - self.start) / self.step))
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode='strict')
                rng = (i, i + n)
        elif mode == 'center':
            i = self.closest_frame(focus.start)

            if fixed is None:
                j = self.closest_frame(focus.end)
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode='center')
                rng = (i, i + n)
        else: raise ValueError

        if return_ranges: return [list(rng)]
        return np.array(range(*rng), dtype=np.int64)

    def segmentToRange(self, segment):
        return self.segment_to_range(segment)

    def segment_to_range(self, segment):
        return self.closest_frame(segment.start), int(segment.duration / self.step) + 1

    def rangeToSegment(self, i0, n):
        return self.range_to_segment(i0, n)

    def range_to_segment(self, i0, n):
        start = self.__start + (i0 - .5) * self.__step + .5 * self.__duration

        if i0 == 0: start = self.start
        return Segment(start, start + (n * self.__step))

    def samplesToDuration(self, nSamples):
        return self.samples_to_duration(nSamples)

    def samples_to_duration(self, n_samples):
        return self.range_to_segment(0, n_samples).duration

    def durationToSamples(self, duration):
        return self.duration_to_samples(duration)

    def duration_to_samples(self, duration):
        return self.segment_to_range(Segment(0, duration))[1]

    def __getitem__(self, i):
        start = self.__start + i * self.__step
        if start >= self.__end: return None

        return Segment(start=start, end=start + self.__duration)

    def next(self):
        return self.__next__()

    def __next__(self):
        self.__i += 1
        window = self[self.__i]

        if window: return window
        else: raise StopIteration()

    def __iter__(self):
        self.__i = -1
        return self

    def __len__(self):
        if np.isinf(self.__end): raise ValueError
        i = self.closest_frame(self.__end)

        while (self[i]):
            i += 1

        length = i
        return length

    def copy(self):
        return self.__class__(duration=self.duration, step=self.step, start=self.start, end=self.end)

    def __call__(self, support, align_last = False):
        if isinstance(support, Timeline): segments = support
        elif isinstance(support, Segment): segments = Timeline(segments=[support])
        else: raise TypeError

        for segment in segments:
            if segment.duration < self.duration: continue

            for s in SlidingWindow(duration=self.duration, step=self.step, start=segment.start, end=segment.end):
                if s in segment:
                    yield s
                    last = s

            if align_last and last.end < segment.end: yield Segment(start=segment.end - self.duration, end=segment.end)