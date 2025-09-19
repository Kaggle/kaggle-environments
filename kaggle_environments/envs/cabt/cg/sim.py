import ctypes
import os


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


if os.name == "nt":
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cg.dll")
else:
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg.so")
lib = ctypes.cdll.LoadLibrary(lib_path)

lib.GameInitialize()

lib.BattleStart.restype = StartData
lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

lib.BattleFinish.argtypes = [ctypes.c_void_p]

lib.GetBattleData.restype = SerialData
lib.GetBattleData.argtypes = [ctypes.c_void_p]

lib.Select.restype = ctypes.c_int
lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.VisualizeData.restype = ctypes.c_char_p
lib.VisualizeData.argtypes = [ctypes.c_void_p]


class Battle:
    battle_ptr = None
    obs = None
    raminingTime = [[], []]
