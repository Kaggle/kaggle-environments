import ctypes
import os
import platform


class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


def _lib_path(lib_dir: str) -> str:
    os_name = platform.system()
    if os_name == "Windows":
        return os.path.join(lib_dir, "cg.dll")
    elif os_name == "Darwin":
        return os.path.join(lib_dir, "libcg.dylib")
    elif platform.machine() in ("arm64", "aarch64"):
        return os.path.join(lib_dir, "libcg-arm64.so")
    return os.path.join(lib_dir, "libcg.so")


def load_lib(lib_dir: str) -> ctypes.CDLL:
    lib = ctypes.cdll.LoadLibrary(_lib_path(lib_dir))

    lib.GameInitialize()

    lib.BattleStart.restype = StartData
    lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

    # Absent in v1.
    if hasattr(lib, "BattleStartReverse"):
        lib.BattleStartReverse.restype = StartData
        lib.BattleStartReverse.argtypes = [ctypes.POINTER(ctypes.c_int)]

    lib.BattleFinish.argtypes = [ctypes.c_void_p]

    lib.GetBattleData.restype = SerialData
    lib.GetBattleData.argtypes = [ctypes.c_void_p]

    lib.Select.restype = ctypes.c_int
    lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

    lib.VisualizeData.restype = ctypes.c_char_p
    lib.VisualizeData.argtypes = [ctypes.c_void_p]
    return lib


_dir = os.path.dirname(os.path.abspath(__file__))
LATEST_VERSION = 2
lib = load_lib(_dir)
_libs = {LATEST_VERSION: lib}


def get_lib(version: int = LATEST_VERSION) -> ctypes.CDLL:
    """Return the engine library for `version`, loading it on first use."""
    if version not in _libs:
        if version != 1:
            raise ValueError(f"Unknown cabt version: {version}")
        _libs[version] = load_lib(os.path.join(_dir, "v1"))
    return _libs[version]


class Battle:
    lib = lib
    battle_ptr = None
    obs = None
    decks = None
    result = [0, 0, 0]
    vis = []
    last_step = 0
