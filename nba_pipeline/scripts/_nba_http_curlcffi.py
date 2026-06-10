"""
Route nba_api's HTTP layer through curl_cffi with a Chrome TLS fingerprint.

stats.nba.com sits behind Akamai bot-mitigation that silently stalls requests
whose TLS/JA3 fingerprint doesn't look like a real browser. Python's `requests`
(and plain libcurl) get hung indefinitely; curl_cffi's `impersonate="chrome"`
presents a genuine Chrome fingerprint and sails through.

Importing this module monkeypatches `nba_api.library.http.requests` so every
nba_api endpoint call uses curl_cffi. It is a no-op (silent) if curl_cffi is
unavailable, so the daily pipeline degrades gracefully back to plain requests.

Usage: just `import _nba_http_curlcffi` once, before any nba_api request is made.
"""
import os

_IMPERSONATE = os.environ.get("NBA_IMPERSONATE", "chrome120")


def _install():
    try:
        from curl_cffi import requests as _creq
    except Exception:
        return False
    try:
        import nba_api.library.http as _h
    except Exception:
        return False

    class _CurlCffiShim:
        """Drop-in for the subset of `requests` that nba_api uses (.get)."""

        @staticmethod
        def get(url=None, params=None, headers=None, proxies=None, timeout=None, **kwargs):
            return _creq.get(
                url,
                params=params,
                headers=headers,
                proxies=proxies,
                timeout=timeout or 30,
                impersonate=_IMPERSONATE,
            )

    # Only patch once.
    if getattr(_h.requests, "__curlcffi_shim__", False):
        return True
    _CurlCffiShim.__curlcffi_shim__ = True
    _h.requests = _CurlCffiShim
    return True


INSTALLED = _install()
