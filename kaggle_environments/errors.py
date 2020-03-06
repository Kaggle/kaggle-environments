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
from pathlib import Path

# Load the Error Codes
status_path = Path.joinpath(Path(__file__).parent, "status_codes.json")
with open(status_path) as json_file:
    codes = json.load(json_file)

for status in codes:
    codes[status]["name"] = "".join(
        "".join([word[0].upper(), word[1:].lower()]) for word in status.split("_")
    )


class CanonicalError(Exception):
    def __init__(self, error="", status="UNKNOWN"):
        super().__init__(error)
        if status not in codes:
            status = "UNKNOWN"
        self.status = status
        self.message = error
        self.name = codes[status]["name"]
        self.code = codes[status]["code"]
        self.http_status = codes[status]["status"]

    def toJSON(self):
        return {"code": self.code, "message": self.message, "status": self.status}


class Cancelled(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "CANCELLED")


class Unknown(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "UNKNOWN")


class InvalidArgument(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "INVALID_ARGUMENT")


class DeadlineExceeded(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "DEADLINE_EXCEEDED")


class NotFound(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "NOT_FOUND")


class AlreadyExists(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "ALREADY_EXISTS")


class PermissionDenied(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "PERMISSION_DENIED")


class Unauthenticated(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "UNAUTHENTICATED")


class ResourceExhausted(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "RESOURCE_EXHAUSTED")


class FailedPrecondition(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "FAILED_PRECONDITION")


class Aborted(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "ABORTED")


class OutOfRange(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "OUT_OF_RANGE")


class Unimplemented(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "UNIMPLEMENTED")


class Internal(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "INTERNAL")


class Unavailable(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "UNAVAILABLE")


class DataLoss(CanonicalError):
    def __init__(self, error=""):
        super().__init__(error, "DATA_LOSS")
