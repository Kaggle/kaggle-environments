# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Type

import jsonschema

from .errors import InvalidArgument, NotFound

# Path Utilities.
root_path = Path(__file__).parent.resolve()
envs_path = Path.joinpath(root_path, "envs")


# Primitive Utilities.
def get(
    o: Any,
    classinfo: Type | None = None,
    default: Any = None,
    path: list[str] | None = None,
    is_callable: bool | None = None,
    fallback: Any = None,
) -> Any:
    if path is None:
        path = []
    if o is None and default is not None:
        o = default
    if has(o, classinfo, default, path, is_callable):
        cur = o
        for p in path:
            cur = cur[p]
        return cur
    else:
        if default is not None:
            return default
        return fallback


def has(
    o: Any,
    classinfo: Type | None = None,
    default: Any = None,
    path: list[str] | None = None,
    is_callable: bool | None = None,
) -> bool:
    if path is None:
        path = []
    try:
        cur = o
        for p in path:
            cur = cur[p]
        if classinfo is not None and not isinstance(cur, classinfo):
            raise ValueError("Not a match")
        if is_callable and not callable(cur):
            raise ValueError("Not callable")
        if not is_callable and callable(cur):
            raise ValueError("Is callable")
        return True
    except Exception:
        if default is not None and o is not None and len(path) > 0:
            cur = o
            for p in path[:-1]:
                if not has(cur, dict, path=[p]):
                    cur[p] = {}
                cur = cur[p]
            cur[path[-1]] = default
        return False


def call(
    o: Any,
    default: Any = None,
    path: list[str] | None = None,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    if path is None:
        path = []
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    o = get(o, default=False, path=path, is_callable=True)
    if o is not False:
        return o(*args, **kwargs)
    else:
        return default


class Struct(dict):
    def __init__(self, **entries: Any) -> None:
        entries = {k: v for k, v in entries.items() if k != "items"}
        dict.__init__(self, entries)
        self.__dict__.update(entries)

    def __setattr__(self, attr: str, value: Any) -> None:
        self.__dict__[attr] = value
        self[attr] = value


# Added benefit of cloning lists and dicts.
def structify(o: Any) -> Any:
    if isinstance(o, list):
        return [structify(o[i]) for i in range(len(o))]
    elif isinstance(o, dict):
        return Struct(**{k: structify(v) for k, v in o.items()})
    return o


# File Utilities.
def read_file(path: str | Path, fallback: str | None = None) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        if fallback is not None:
            return fallback
        raise NotFound(f"{path} not found")


def get_file_json(path: str | Path, fallback: Any | None = None) -> Any:
    try:
        with open(path, "r") as json_file:
            return json.load(json_file)
    except Exception:
        if fallback is not None:
            return fallback
        raise InvalidArgument(f"{path} does not contain valid JSON")


# Schema Utilities.
schemas = structify(get_file_json(Path.joinpath(root_path, "schemas.json")))


def default_schema(schema: Any, data: Any) -> Any:
    default = get(schema, path=["default"])
    if default is None and data is None:
        return

    if get(schema, path=["type"]) == "object":
        default = deepcopy(get(default, dict, {}))
        if data is None or not has(data, dict):
            obj = default
        else:
            obj = data
            for k, v in default.items():
                if k not in obj:
                    obj[k] = v
        properties = get(schema, dict, {}, ["properties"])
        for key, prop_schema in properties.items():
            new_value = default_schema(prop_schema, get(obj, path=[key]))
            if new_value is not None:
                obj[key] = new_value
        return obj

    if get(schema, path=["type"]) == "array":
        default = deepcopy(get(default, list, []))
        arr = get(data, list, default)
        item_schema = get(schema, dict, {}, ["items"])
        for index, value in enumerate(arr):
            if value is None and len(default) > index:
                new_value = default[index]
            else:
                new_value = default_schema(item_schema, value)
            if new_value is not None:
                arr[index] = new_value
        return arr

    return data if data is not None else default


def process_schema(schema: Any, data: Any, use_default: bool = True) -> tuple[str | None, Any]:
    error = None
    if use_default is True:
        data = default_schema(schema, deepcopy(data))
    try:
        jsonschema.validate(data, schema)
    except Exception as err:
        error = str(err)
    return error, data


# Player utilities
def get_player(window_kaggle: dict[str, Any], renderer: tuple[str, str] | str) -> str:
    # TODO: resolve type alert here. Renderer is clearly not a string, but gets .strip() called.
    if renderer[0] == "html_path":
        key = "/*window.kaggle*/"
        value = f"""
window.kaggle = {json.dumps(window_kaggle, indent=2)};\n\n
        """
        return read_file(renderer[1]).replace(key, value)

    key = "/*window.kaggle*/"
    value = f"""
window.kaggle = {json.dumps(window_kaggle, indent=2)};\n\n
window.kaggle.renderer = {renderer.strip()};\n\n
    """
    return read_file(Path.joinpath(root_path, "static", "player.html")).replace(key, value)
