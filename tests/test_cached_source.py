from stockanalysis.data import CachedYFinanceSource, SyntheticSource


class _CountingSource(SyntheticSource):
    """SyntheticSource that tracks how many times fetch() actually ran,
    so tests can check whether the cache was used."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fetch_count = 0

    def fetch(self, ticker, start, end=None):
        self.fetch_count += 1
        return super().fetch(ticker, start, end)


def test_second_fetch_of_same_range_hits_cache_not_live_source(tmp_path):
    live = _CountingSource(seed=1)
    cached = CachedYFinanceSource(tmp_path, live_source=live)

    cached.fetch("AAA", "2020-01-01", "2020-06-01")
    cached.fetch("AAA", "2020-01-01", "2020-06-01")

    assert live.fetch_count == 1


def test_wider_range_triggers_a_refetch(tmp_path):
    live = _CountingSource(seed=1)
    cached = CachedYFinanceSource(tmp_path, live_source=live)

    cached.fetch("AAA", "2020-01-01", "2020-06-01")
    cached.fetch("AAA", "2019-01-01", "2020-06-01")

    assert live.fetch_count == 2


def test_refresh_true_always_refetches(tmp_path):
    live = _CountingSource(seed=1)
    cached = CachedYFinanceSource(tmp_path, live_source=live, refresh=True)

    cached.fetch("AAA", "2020-01-01", "2020-06-01")
    cached.fetch("AAA", "2020-01-01", "2020-06-01")

    assert live.fetch_count == 2


def test_cached_fetch_returns_equivalent_data(tmp_path):
    live = _CountingSource(seed=1)
    cached = CachedYFinanceSource(tmp_path, live_source=live)

    original = cached.fetch("AAA", "2020-01-01", "2020-06-01")
    from_cache = cached.fetch("AAA", "2020-01-01", "2020-06-01")

    assert list(original.columns) == list(from_cache.columns)
    assert len(original) == len(from_cache)
