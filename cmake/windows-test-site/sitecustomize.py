"""Imported at startup by every Python test that ctest runs on Windows.

Sets SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX, which child processes inherit. Under
CTest's default error mode a child that fails to load or crashes opens a modal system dialog
and the test stalls until its timeout; with these flags the failure returns as an exit code.
"""

import sys

if sys.platform == "win32":
    import ctypes

    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
