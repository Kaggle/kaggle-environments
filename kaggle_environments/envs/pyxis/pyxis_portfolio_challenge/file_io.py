import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import upath


def _azure_like(uri: upath.UPath) -> bool:
    """Check if the URI is Azure-like (starts with 'az://')."""
    return str(uri).startswith("az://")


def _load_bytes(uri: upath.UPath):
    """Main loader API. Loads from local or Azure, if Azure support is available."""
    if _azure_like(uri):
        try:
            from app.azure import _load_bytes_azure
        except ImportError:
            raise ImportError(
                "Azure Blob Storage loading is not available in this installation. "
            )
        return _load_bytes_azure(uri)
    else:
        return _load_bytes_local(uri)


def _load_bytes_bulk(uris: list[upath.UPath], max_workers: int = 16) -> list[bytes]:
    """
    Bulk-load bytes from a list of URIs.

    Routing each through _load_bytes (Azure or local) in parallel.
    Returns a list of bytes in the same order as requested.
    """
    uri_to_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        uri_to_future = {uri: executor.submit(_load_bytes, uri) for uri in uris}
        for uri, future in uri_to_future.items():
            uri_to_results[uri] = future.result()

    return [uri_to_results[uri] for uri in uris]  # Preserve original order


def list_files(uri: upath.UPath) -> list[str]:
    """List files given a uri. Supports local and Azure if available."""
    if _azure_like(uri):
        try:
            from app.azure import _list_files_azure
        except ImportError:
            raise ImportError(
                "Azure Blob Storage listing is not available in this installation. "
            )
        return _list_files_azure(uri)
    else:
        return _list_files_local(uri)


def _load_bytes_local(path: upath.UPath) -> bytes:
    """Load bytes from a local file."""
    with open(path, "rb") as f:
        return f.read()


def _list_files_local(path: str) -> list[str]:
    """List files in a local directory."""
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]


def load_json(uri: upath.UPath) -> dict:
    """Load JSON data from a file from a uri."""
    data_bytes = _load_bytes(uri)
    return json.loads(data_bytes.decode("utf-8"))


def load_json_bulk(uris: list[upath.UPath], max_workers: int = 16) -> list[dict]:
    """Bulk-load JSON data from a list of URIs using multi-threading."""
    raw_bytes = _load_bytes_bulk(uris, max_workers=max_workers)
    result = [json.loads(data.decode("utf-8")) for data in raw_bytes]
    return result


def download_file(uri: upath.UPath):
    """Download file from uri to a temporary local file and return the local path."""
    data_bytes = _load_bytes(uri)
    suffix = upath.UPath(uri).suffix.lstrip(".")
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as temp_file:
        temp_file.write(data_bytes)
        temp_file_name = temp_file.name
    return temp_file_name


def write_bytes(data: bytes, path: upath.UPath, overwrite: bool = False):
    """Main writer API. Writes to local or Azure, if Azure support is available."""
    if _azure_like(path):
        try:
            from app.azure import upload_bytes
        except ImportError:
            raise ImportError(
                "Azure Blob Storage writing is not available in this installation. "
            )
        upload_bytes(data, uri=upath.UPath(path), overwrite=overwrite)
    else:
        _write_bytes_local(data, path, overwrite=overwrite)


def _write_bytes_local(data: bytes, path: upath.UPath, overwrite: bool):
    """Write bytes to a local file."""
    if not overwrite and path.exists():
        raise FileExistsError(f"File {path} already exists and overwrite is False.")

    with open(str(path), "wb") as f:
        f.write(data)


def write_json(data: dict, path: upath.UPath, overwrite: bool = False):
    """Write JSON data to a file at the given path."""
    data_bytes = json.dumps(data, indent=4).encode("utf-8")
    write_bytes(data_bytes, path, overwrite=overwrite)
