# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import abc
import datetime
from enum import Enum
from typing import *

import pydantic
from pydantic.functional_validators import BeforeValidator

# --------------------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------------------

class DatabaseSpaceName(str, Enum):
    PRODUCTION = 'production'
    TESTING = 'testing'


class WebhookName(str, Enum):
    PIPELINE = 'pipeline'
    JOB = 'job'


class BuildStatus(str, Enum):
    CREATED = 'created'
    PENDING = 'pending'
    RUNNING = 'running'
    FAILED = 'failed'
    SUCCESS = 'success'
    CANCELING = 'canceling'
    CANCELED = 'canceled'
    SKIPPED = 'skipped'
    MANUAL = 'manual'
    SCHEDULED = 'scheduled'
    WAITING_FOR_RESOURCE = 'waiting_for_resource'
    PREPARING = 'preparing'


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------

class _BaseModel(
    pydantic.BaseModel, abc.ABC,
    extra='allow',
):
    pass


class Project(_BaseModel):
    id: int
    name: str


class JobEvent(_BaseModel):
    pipeline_id: int
    build_id: int
    build_status: BuildStatus
    project: Project


class PipelineCommit(_BaseModel):
    id: str


class ObjectAttributes(_BaseModel):
    id: int
    status: BuildStatus


class PipelineEvent(_BaseModel):
    object_attributes: ObjectAttributes
    project: Project
