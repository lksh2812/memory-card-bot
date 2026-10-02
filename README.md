# Memory Lane: a voice memory card game on Pipecat

A voice bot that hosts a memory game. It reads out a sequence of cards, the
player repeats them back, and the bot checks the answer. Each cleared round adds
one card. Three misses and the game is over.

Built with Pipecat 1.12 (SmallWebRTC transport, Deepgram STT and TTS, Groq for
host lines), FastAPI, PostgreSQL and Redis.

## Quick start

Requirements: Python 3.11+, Docker, a microphone, and a free
[Deepgram](https://console.deepgram.com) API key. A [Groq](https://console.groq.com)
key is optional (without it the host uses template lines).

```bash
# 1. Postgres and Redis
docker compose up -d

# 2. Python deps
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Config
cp .env.example .env        # then add DEEPGRAM_API_KEY (and GROQ_API_KEY)

# 4. Run
uvicorn app.main:app --port 7860
```

Open http://localhost:7860, enter a name, allow the microphone, and play.
Headphones help: the bot hears itself less, which means fewer false interruptions.

### Or run everything in Docker

```bash
cp .env.example .env        # add the keys
docker compose --profile full up --build
```

The app container uses host networking, because WebRTC sends audio over UDP on
random ports that a normal port mapping can't cover. This works as is on Linux.
On Docker Desktop (Mac/Windows), turn on Settings > Resources > Network >
"Enable host networking" (Docker Desktop 4.34+); if that option isn't there, use
the uvicorn steps above. Plain `docker compose up -d` still starts only Postgres
and Redis.

If your network blocks Daily's CDN (the WebRTC client loads a small helper from
it), open http://localhost:7860/?media=native instead, which uses the browser's
own audio APIs.

The frontend is prebuilt (`frontend/app.js`). Node is only needed if you change
`frontend/src/main.js`: `cd frontend && npm install && npm run build`.

## Tests

```bash
pytest                                  # SQLite + in-memory Redis, no keys needed
TEST_DATABASE_URL=postgresql+asyncpg://memory:memory@localhost:5432/memory_game_test pytest   # real Postgres
```

`TEST_DATABASE_URL` must point at a throwaway database: the fixtures drop and
recreate every table. Create it once with
`docker compose exec postgres createdb -U memory memory_game_test`.

77 tests cover card matching, game rules, the repository (including ten
concurrent attempts to score the same round), the Redis cache (including Redis
being down), the REST API, the turn-end strategy, and the game processor driven
with real Pipecat frames (correct and wrong answers, interruptions, repeats,
quitting, game over).

## How it works

```
 Browser (mic + speaker + scoreboard)
   |  WebRTC audio + RTVI data channel          REST (create / state / end / scores)
   v                                            v
 FastAPI ── POST /api/offer starts one Pipecat bot per session
   |
   |  transport.input()     audio from the browser
   |  DeepgramSTT           streaming transcripts
   |  user aggregator       Silero VAD + turn strategies, one LLMContextFrame per user turn
   |  MemoryGameProcessor   validates, scores, decides what to say      ── Postgres (truth)
   |  DeepgramTTS                                                        ── Redis (live state, leaderboard)
   |  transport.output()    audio to the browser, bot started/stopped speaking events
   |  assistant aggregator  records what the user actually heard
```

### The custom frame processor

`app/bot/game_processor.py` sits where an LLM would normally go. When the user
aggregator decides a user turn is over it pushes an `LLMContextFrame`; the game
processor consumes it, validates the answer in Python, writes the result, and
pushes a `TTSSpeakFrame` with exactly what the bot should say. Cards are spoken
verbatim, never through the LLM, so they can't be paraphrased or reordered.

The LLM (Groq, called out of band from `app/bot/host.py`) only writes short host
lines like "Ooh, so close!". It never sees the cards or decides correctness, has
a 2 second timeout with template fallback, and any line that mentions a card
word is rejected.

Game phases: `connecting -> presenting -> listening -> evaluating -> presenting ... -> finished`.

### Validation (no LLM involved)

`app/game/matching.py` turns a transcript into the list of cards said:
lowercase and tokenize, map tokens to cards (exact, plural, a word STT split in
two like "pen guin", or a close fuzzy match), drop fillers, collapse stutters.
The answer is correct only if that list equals the expected sequence exactly.
Extra cards are wrong, otherwise reciting the whole deck would pass. The deck was
chosen to avoid homophones ("pear/pair").

### Turn-taking

A memory answer has natural pauses ("apple... river... um... tiger"), and a fixed
silence timeout either cuts people off or makes every reply slow. Since the bot
knows how many cards to expect, `SequenceAwareTurnStopStrategy`
(`app/bot/turn_strategy.py`) extends Pipecat's speech-timeout strategy and adapts
the silence window: 2.0s while the answer is incomplete, 0.35s once all expected
cards have been heard, 0.6s otherwise.

### Interruptions

* While the bot is speaking, a turn needs at least two words to start
  (`MinWordsUserTurnStartStrategy`), so "okay" or "hmm" doesn't cut it off.
* If the user talks over the cards, Pipecat stops the audio and the processor
  marks the presentation as interrupted. Nothing is evaluated; once the user's
  turn ends, the bot says "No worries, here they are again from the top" and
  replays the same cards. The replay is counted in `rounds.times_presented`.
* Interrupting anything else (feedback, banter) just stops the audio.
* The assistant aggregator records only the part of an utterance that was played.

### No double scoring

Two layers:

1. The processor moves to `evaluating` before its first `await`, so the silence
   timer and a real answer can't both start scoring the same round.
2. The database only lets a round be scored once:
   `UPDATE rounds SET status='correct' WHERE id=:id AND status='awaiting' RETURNING ...`
   Only the first caller gets a row back; the response insert and the score
   update happen in the same transaction. `responses.round_id` is also `UNIQUE`
   as a backstop. A round can't be scored before it reaches `awaiting`, which only
   happens once all the cards were heard.

### Caching (Redis)

| What | Pattern | Key |
| --- | --- | --- |
| Live session state | write-through by the bot, read by `GET /api/sessions/{id}` | `mcb:session:{id}` (2h TTL) |
| Leaderboard | sorted set, `ZADD GT` keeps each player's best; rebuilt from Postgres if cold | `mcb:leaderboard` |
| Recently used sequences | capped list per player, new rounds avoid them | `mcb:player:{id}:recent` (50 items, 7d) |
| Recent scores | short-TTL read cache, invalidated when a game ends | `mcb:recent_scores` (30s) |

Postgres is always the source of truth. Every cache call catches Redis errors
and falls back, so the game keeps working if Redis goes down. Cached endpoints
return an `X-Cache: hit|miss` header.

### Data model

`players`, `game_sessions` (status, score, lives, rounds cleared, best length,
end reason), `rounds` (cards, status, times presented; `UNIQUE(session_id, round_number)`),
`responses` (transcript, cards heard, correct, points; `UNIQUE(round_id)`).

## API

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/api/sessions` | `{"player_name": "..."}` creates a session |
| GET | `/api/sessions/{id}` | live state from Redis; the current round's cards are never exposed |
| GET | `/api/sessions/{id}/rounds` | history; cards revealed once a round is evaluated |
| POST | `/api/sessions/{id}/end` | ends the game; a live bot says goodbye and hangs up |
| GET | `/api/leaderboard?limit=10` | best score per player |
| GET | `/api/scores/recent?limit=10` | latest finished games |
| POST / PATCH | `/api/offer?session_id=...` | WebRTC signaling (used by the client library) |

Interactive docs at http://localhost:7860/docs.

```bash
curl -X POST localhost:7860/api/sessions -H 'content-type: application/json' -d '{"player_name":"lokesh"}'
curl -i localhost:7860/api/sessions/<id>
curl localhost:7860/api/leaderboard
```

## Project layout

```
app/
  game/          pure game logic, no I/O: cards.py, matching.py, engine.py
  bot/           pipeline.py, game_processor.py, turn_strategy.py, host.py, frames.py
  models.py      SQLAlchemy tables
  repository.py  SQL, including the scoring transaction
  cache.py       Redis
  service.py     use cases shared by the API and the bot
  main.py        FastAPI routes and WebRTC signaling
frontend/        index.html, styles.css, src/main.js (bundled to app.js)
tests/
```

## Design choices and trade-offs

* **LLM out of the pipeline.** The game needs exact control over what is spoken
  and when, so the processor drives TTS directly and calls the LLM only for
  flavor. This also removes LLM latency from the critical path when it's slow:
  the timeout falls back to a template.
* **One bot per call, in the API process.** Simple for an assignment. To scale,
  bots would run as separate workers behind the API (each call is independent),
  and the in-process `BotRegistry` would become a message to the worker
  (Redis pub/sub or NATS).
* **Tables created on startup.** Production would use Alembic migrations.
* **SmallWebRTC** is peer to peer, so no Daily account is needed. Behind strict
  NATs it would need a TURN server.

## Troubleshooting

* *No bot audio*: click anywhere on the page once (browser autoplay rules), and
  check the terminal for Deepgram errors (usually a missing or wrong key).
* *Bot keeps getting interrupted*: use headphones; speakers can feed the bot's
  voice back into the mic.
* *Groq errors*: check `GROQ_MODEL`. Groq retired the Llama 3.x models in
  August 2026; `openai/gpt-oss-20b` is the default.
