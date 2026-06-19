# 🧘 YogaAlign — AI Pose Correction Coach

An AI-powered yoga pose correction application that watches your practice in real time via a local TensorFlow.js model and provides personalized coaching instructions using a local LLM (Ollama) — no cloud AI dependency.

## Features

- **13 Yoga Poses** from the Common Yoga Protocol (Standing, Sitting, Prone, Supine)
- **Real-Time Step Verification** — a local TensorFlow.js model classifies each pose step from MediaPipe landmarks, frame by frame
- **Local LLM Coaching** — when stuck on a step for 3+ seconds, a local Ollama model gives rich, specific coaching tips (fully offline, no API key needed)
- **Prioritized Corrections** — High/Medium/Low priority coaching cues with body-part specificity
- **Safety Warnings** — Automatic detection of potential injury risks
- **Premium UI** — Dark mode with glassmorphism, gradients, and smooth micro-animations

## Quick Start

### 1. Install Ollama (local LLM)

Install [Ollama](https://ollama.com), start the server, and pull a model:

```bash
ollama serve
ollama pull llama3.2
```

### 2. Configure Environment

Copy the example environment file (defaults already point at the local Ollama server):

```bash
cp .env.example .env
```

You can change `OLLAMA_MODEL` in `.env` to any model you've pulled.

### 3. Install Dependencies

```bash
npm install
```

### 4. Start the Server

```bash
npm run dev
```

The app will be available at **http://localhost:3000**

## Usage

1. **Select a pose** from the dropdown (grouped by category)
2. **Upload a video** of yourself performing the pose
3. **Click "Analyze My Pose"** and wait ~15-30 seconds
4. **Review your results** — score, positives, corrections, and safety warnings

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Node.js + Express |
| Step Verification Model | TensorFlow.js (local MLP, trained on MediaPipe landmarks) |
| Rich Coaching | Local LLM via Ollama |
| Frontend | Vanilla HTML/CSS/JS |
| File Upload | Multer |
| Design | Dark mode, glassmorphism, Inter + Outfit fonts |

## Supported Poses

| Category | Poses |
|----------|-------|
| **Standing** | Tadasana, Vrkasana, Pada-hastasana, Ardha Cakrasana, Ardha Katichakrasana, Trikonasana, Parivritta Trikonasana |
| **Sitting** | Ardha Ustrasana, Vakrasana |
| **Prone** | Bhujangasana, Makarasana |
| **Supine** | Ardha Halasana, Savasana |

## Project Structure

```
├── server.js          # Express backend + local TF.js model + Ollama integration
├── package.json       # Dependencies
├── .env               # Local config (Ollama host/model, port)
├── .env.example       # Environment template
├── train_model.js      # Trains tfjs_model/ from training_data.json
├── prepare_dataset.py  # Builds training_data.json from recorded sessions
└── public/
    ├── index.html     # Main page
    ├── index.css      # Design system
    └── app.js         # Frontend logic
```
