"""The shared strict-validation base for all mixup data models.

``strict=True`` disables pydantic's implicit coercion (``"5"`` -> ``5``, ``1`` -> ``True``,
``5`` -> ``5.0``), so a value arriving in the wrong type fails loudly instead of silently
becoming something else. Enum fields override this with ``Field(strict=False)`` because the
JSON config carries enum values as plain strings.
"""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for all mixup models: strict validation, no implicit type coercion."""

    model_config = ConfigDict(strict=True)
