import asyncio
import logging
import os
import time
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Dict, Any, Annotated
from uuid import UUID, uuid4

import faster_whisper
import torch
import uvicorn
import whisperx
from fastapi import BackgroundTasks, FastAPI, status, Body, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, AnyHttpUrl

from diarization import DiarizationWithEmbeddingsPipeline

# setup loggers
logging.config.fileConfig(Path(__file__).parent / 'logging.conf', disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"


class InputOptions(BaseModel):
    audio_url: AnyHttpUrl = Field(
        description="Audio file url"),
    language: str | None = Field(
        description="Optional language of the audio file",
        default=None),
    initial_prompt: str | None = Field(
        description="Optional text to provide as a prompt for the first window",
        default=None),
    temperature: float = Field(
        description="Temperature to use for sampling",
        default=0),
    batch_size: int = Field(
        description="Parallelization of input audio transcription",
        default=64),
    align_output: bool = Field(
        description="Aligns whisper output to get accurate word-level timestamps",
        default=True),
    diarization: bool = Field(
        description="Assign speaker ID labels",
        default=True),
    group_adjacent_speakers: bool = Field(
        description="Group adjacent segments with same speakers",
        default=True),
    min_speakers: int = Field(
        description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
        default=None),
    max_speakers: int | None = Field(
        description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
        default=None),
    return_embeddings: bool = Field(
        description="Return representative speaker embeddings",
        default=False),


class Output(BaseModel):
    segments: Any | None = None
    embeddings: Any | None = None
    detected_language: str | None = None
    start_time: float | None = None
    elapsed_time: float | None = None

class JobStatus(str, Enum):
    STARTED = 'STARTED'
    FAILED = 'FAILED'
    COMPLETED = 'COMPLETED'


class Job(BaseModel):
    uid: UUID = Field(default_factory=uuid4)
    status: JobStatus = JobStatus.STARTED
    result: Output | None = None


jobs: Dict[UUID, Job] = {}

templates = Jinja2Templates(directory=".")
app = FastAPI()


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(app.state.executor, fn, *args)  # wait and return result


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def task(options: InputOptions):
    debug = str2bool(os.getenv("DEBUG", "false"))
    logger.info(f"Task starting: {options.audio_url}")
    output = Output(start_time=time.time_ns() / 1e6)
    if debug:
        time.sleep(5)
        logger.info("Task done")
        return output
    else:
        with torch.inference_mode():
            start_time = time.time_ns() / 1e6
            new_asr_options = app.model.options._asdict()
            if (options.initial_prompt and new_asr_options[
                "initial_prompt"] != options.initial_prompt) or options.temperature not in \
                    new_asr_options[
                        "temperatures"]:
                new_asr_options["initial_prompt"] = options.initial_prompt
                new_asr_options["temperatures"] = [options.temperature]
                new_options = faster_whisper.transcribe.TranscriptionOptions(**new_asr_options)
                app.model.options = new_options
            audio = whisperx.load_audio(str(options.audio_url))
            result = app.model.transcribe(audio, task="transcribe", language=options.language, batch_size=options.batch_size)
            detected_language = result["language"]
            elapsed_time = time.time_ns() / 1e6 - start_time
            logger.info(f"Duration to transcribe audio: {elapsed_time:.2f} ms")

            if options.align_output:
                if detected_language in app.alignment_model:
                    result = align(audio, detected_language, result)
                else:
                    logger.warning(
                        f"Cannot align output as language {detected_language} is not supported for alignment")

            if options.diarization:
                result = diarize(audio, result, options.min_speakers, options.max_speakers,
                                 options.return_embeddings)
                if options.group_adjacent_speakers:
                    result = group_speakers(result)
            logger.info(
                f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        segments = result["segments"]
        for segment in segments:
            segment.pop("words", None)
    logger.info("Task done")
    output.elapsed_time = time.time_ns() / 1e6 - output.start_time
    output.segments = result["segments"]
    output.embeddings = result.get("embeddings", None)
    output.detected_language = detected_language
    return output


def group_speakers(result):
    start_time = time.time_ns() / 1e6
    segments = result["segments"]
    grouped_segments = []
    for key, group in groupby(segments, lambda seg: seg.get('speaker', 'Unknown')):
        segs = list(group)
        grouped_segment = segs[0]
        grouped_segment['end'] = segs[-1]['end']
        grouped_segment['text'] = "\n".join(seg['text'] for seg in segs)
        grouped_segment.pop('words', None)
        grouped_segments.append(grouped_segment)

    elapsed_time = time.time_ns() / 1e6 - start_time
    logger.info(f"Duration to group speakers: {elapsed_time:.2f} ms")

    result['segments'] = grouped_segments
    return result


def align(audio, language, result):
    start_time = time.time_ns() / 1e6
    result = whisperx.align(result["segments"], app.alignment_model[language], app.alignment_metadata[language], audio,
                            device,
                            return_char_alignments=False)

    elapsed_time = time.time_ns() / 1e6 - start_time
    logger.info(f"Duration to align output: {elapsed_time:.2f} ms")

    return result


def diarize(audio, result, min_speakers, max_speakers, return_embeddings):
    start_time = time.time_ns() / 1e6

    diarize_segments, embeddings = app.diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers,
                                                     return_embeddings=return_embeddings)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    if return_embeddings:
        result['embeddings'] = embeddings
        print(embeddings)

    elapsed_time = time.time_ns() / 1e6 - start_time
    logger.info(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    return result


async def start_cpu_bound_task(uid: UUID, options: InputOptions) -> None:
    try:
        jobs[uid].result = await run_in_process(task, options)
        jobs[uid].status = JobStatus.COMPLETED
    except BaseException as err:
        logger.exception(f"Failed to process {options.audio_url}",
                         exc_info=err)
        jobs[uid].status = JobStatus.FAILED


@app.post("/submit", status_code=status.HTTP_202_ACCEPTED)
async def task_handler(background_tasks: BackgroundTasks, options: Annotated[InputOptions, Body(embed=True)]):
    new_task = Job()
    jobs[new_task.uid] = new_task
    background_tasks.add_task(start_cpu_bound_task, new_task.uid, options)
    return new_task


@app.get("/status/{uid}")
async def status_handler(uid: UUID):
    if uid in jobs:
        job = jobs[uid]
        if job.status != JobStatus.STARTED:
            job = jobs.pop(uid)
        return job
    else:
        logger.warning(f"Unknown job uid: {uid}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unkown jobs uid: {uid}")


@app.on_event("startup")
async def startup_event():
    langs = os.getenv("LANG", "fr").split()
    # app.state.executor = ProcessPoolExecutor()
    app.state.executor = ThreadPoolExecutor()
    asr_options = {
        "temperatures": [int(os.getenv("TEMPERATURE", "0"))],
        "initial_prompt": os.getenv("INITIAL_PROMPT", None)
    }

    vad_options = {
        "vad_onset": float(os.getenv("VAD_ONSET", "0.500")),
        "vad_offset": float(os.getenv("VAD_OFFSET", "0.363"))
    }

    app.model = whisperx.load_model(os.getenv("WHISPER_MODEL", "tiny"), device,
                                    language=os.getenv("LANG", "fr"), compute_type=compute_type,
                                    asr_options=asr_options, vad_options=vad_options)
    app.alignment_model = {}
    app.alignment_metadata = {}
    for lang in langs:
        app.alignment_model[lang], app.alignment_metadata[lang] = whisperx.load_align_model(language_code=os.getenv("LANG", "fr"),
                                                                                device=device)
    app.diarize_model = DiarizationWithEmbeddingsPipeline(model_name='pyannote/speaker-diarization-3.1',
                                                          use_auth_token=os.getenv("HF_TOKEN"), device=device)


@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()
    app.model = None
    app.alignment_model, app.alignment_metadata = None, None
    app.diarize_model = None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "15555")))
