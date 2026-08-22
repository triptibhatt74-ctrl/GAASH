# Voice transcription contract

`POST /voice/transcribe` accepts an authenticated `multipart/form-data` request.

- File field: `audio`
- Authentication: `Authorization: Bearer <access token>`
- Maximum upload: `MAX_AUDIO_BYTES` (25 MiB by default)
- Accepted media types: MPEG/MP3, MP4/M4A, WAV, WebM, Ogg/Oga, FLAC, AAC, and 3GP.
- Success response: `{ "transcript": "<spoken words>" }`

The client must not set the multipart `Content-Type` header manually. It sends the currently recorded blob and its matching filename extension. The transcript is placed in the composer for review; it is not sent as a chat message automatically.

Possible responses:

- `400`: empty recording
- `401`: missing or invalid bearer authentication
- `413`: recording exceeds the configured byte limit
- `415`: unsupported audio media type or filename suffix
- `422`: no intelligible speech could be transcribed
- `502`: transcription provider unavailable or malformed upstream response

No client-side sample, fallback, or generated transcript is used.
