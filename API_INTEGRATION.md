# API integration

This frontend reads its public authentication API base URL from `VITE_AUTH_API_BASE_URL`. All authentication requests are centralized in `src/services/authService.js`; UI components do not embed API paths or credentials.

## Active account flows

The following frontend requests use the current canonical backend paths:

- `POST /auth/register` with `username`, `email`, and `password`
- `POST /auth/verify-registration` with `email` and `otp`
- `POST /auth/resend-otp` with `email`
- `POST /auth/forgot-password` with `email`
- `POST /auth/verify-reset-otp` with `email` and `otp`
- `POST /auth/reset-password` with `new_password` and the temporary Bearer credential returned by the reset-verification response
- `POST /auth/login` with `email` and `password`
- `POST /auth/logout` with the current Bearer credential; this revokes that credential until its expiry
- `GET /me` with the existing Bearer credential for session restoration

The registration-verification response may establish a session only when it provides the same `access_token` field used by the existing login architecture. Otherwise, the frontend completes local onboarding and asks the person to sign in. The reset-verification response returns a short-lived `reset_token`; it is held only in memory for the current password-reset flow.

No reset token, access token, password, or verification code is placed in a URL, logged, or persisted beyond the session credential. A reset token is held only in memory for the active password-reset flow.

## CORS

The backend must allow the deployed frontend origin and permit `Content-Type` and `Authorization` headers for account requests. This frontend does not add a proxy or browser-side CORS workaround.

## Wellbeing data boundaries

`src/services/chatService.js`, `src/services/assessmentService.js`, `src/services/profileService.js`, and `src/services/wellbeingService.js` are the browser-to-wellbeing API boundaries. They use the configured `VITE_CHAT_API_BASE_URL` and the existing Bearer credential for persisted conversations, screening sessions, profile preferences, analytics, and reports. The resource centre remains deliberately static and reads its verified source data from `src/data/resources.json`; it does not call the backend.
