"""Imported at startup by every Python test that ctest runs on Windows.

CTest starts tests with the default Windows error mode. In that mode a child process that fails
to load (a corrupt executable) or crashes opens a modal system dialog, and the test stalls until
its timeout instead of seeing the failure. Set the error mode a console shell uses, so the
failure comes back to the caller as an error or exit code. Child processes inherit it.
"""

import sys

if sys.platform == "win32":
    import ctypes

    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
