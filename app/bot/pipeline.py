"""Builds and runs one Pipecat pipeline per voice call.

    browser mic
        |
    transport.input()      WebRTC audio in (SmallWebRTC, peer to peer, no Daily account)
    stt                    Deepgram streaming speech-to-text
    user_aggregator        Silero VAD + turn detection -> emits one LLMContextFrame per user turn
    echo                   EchoProcessor: says back what it heard (replaced by the game in the next step)
    tts                    Deepgram text-to-speech
    transport.output()     WebRTC audio out (also emits BotStarted/StoppedSpeaking)
    assistant_aggregator   records what the bot actually said (only the heard part if interrupted)
        |
    browser speaker
"""

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMContextFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from app.config import Settings
from app.service import GameService


class EchoProcessor(FrameProcessor):
    """Sits where an LLM would go. When the user finishes a turn, says it back."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            # Consume the frame (don't pass it on) and answer with a TTS frame instead.
            last = frame.context.messages[-1] if frame.context.messages else {}
            text = last.get("content", "") if isinstance(last, dict) else ""
            if isinstance(text, str) and text.strip():
                await self.push_frame(TTSSpeakFrame(f"I heard: {text}"))
            return
        await self.push_frame(frame, direction)


@dataclass
class BotRegistry:
    """Bots running in this process, by session id."""

    bots: dict[int, PipelineWorker] = field(default_factory=dict)

    def is_running(self, session_id: int) -> bool:
        return session_id in self.bots


async def run_bot(
    connection: SmallWebRTCConnection,
    session_id: int,
    service: GameService,
    settings: Settings,
    registry: BotRegistry,
) -> None:
    logger.info(f"starting bot for session {session_id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt = DeepgramSTTService(api_key=settings.deepgram_api_key)
    tts = DeepgramTTSService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramTTSService.Settings(voice=settings.deepgram_voice),
    )

    # The context is not sent to an LLM. The aggregators still earn their place:
    # the user side runs turn detection, the assistant side keeps an honest
    # transcript of what the user actually heard.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            EchoProcessor(),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        # Metrics include TTFB per service, which is how we watch speech-to-speech latency.
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    registry.bots[session_id] = worker

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info(f"client ready for session {session_id}")
        await worker.queue_frames([TTSSpeakFrame("Hi! I'm your memory game host. Say anything and I'll repeat it.")])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"client disconnected from session {session_id}")
        await worker.cancel()

    try:
        await runner.run()
    except asyncio.CancelledError:
        pass
    finally:
        registry.bots.pop(session_id, None)
        logger.info(f"bot for session {session_id} stopped")
