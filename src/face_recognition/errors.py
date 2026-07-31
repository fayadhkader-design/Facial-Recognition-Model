"""Application-specific exceptions surfaced as friendly CLI errors."""


class FaceRecognitionError(Exception):
    """Base class for expected application errors."""


class ModelError(FaceRecognitionError):
    """A model file is unavailable, invalid, or cannot be loaded."""


class ImageError(FaceRecognitionError):
    """An input image cannot be read or does not meet requirements."""


class EnrollmentError(FaceRecognitionError):
    """Reference photos cannot be converted into an identity database."""


class DatabaseError(FaceRecognitionError):
    """An embedding database is absent, unsafe, or malformed."""

