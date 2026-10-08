-- YogaAlign schema — run once:
--   psql -U postgres -d yoga_app -f schema.sql

CREATE TABLE IF NOT EXISTS users (
  id            SERIAL PRIMARY KEY,
  email         VARCHAR(255) UNIQUE NOT NULL,
  name          VARCHAR(255),
  age           INTEGER,
  height_cm     NUMERIC(5,1),
  weight_kg     NUMERIC(5,1),
  gender        VARCHAR(50),
  experience    VARCHAR(50),
  password_hash VARCHAR(255) NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
  id             SERIAL PRIMARY KEY,
  user_id        INTEGER REFERENCES users(id) ON DELETE CASCADE,
  participant_id VARCHAR(100),
  participant_name VARCHAR(255),
  session_number INTEGER,
  video_fps      INTEGER DEFAULT 30,
  t_zero         BIGINT,
  started_at     TIMESTAMPTZ DEFAULT NOW(),
  ended_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS pose_recordings (
  id          SERIAL PRIMARY KEY,
  session_id  INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
  pose_id     VARCHAR(50),
  pose_name   VARCHAR(255),
  started_at  TIMESTAMPTZ DEFAULT NOW(),
  ended_at    TIMESTAMPTZ,
  metadata    JSONB
);

CREATE TABLE IF NOT EXISTS practices (
  id               SERIAL PRIMARY KEY,
  user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
  session_id       INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
  pose_name        VARCHAR(255),
  duration_seconds NUMERIC(8,2),
  completed        BOOLEAN DEFAULT FALSE,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_practices_user   ON practices(user_id);
CREATE INDEX IF NOT EXISTS idx_practices_session ON practices(session_id);
