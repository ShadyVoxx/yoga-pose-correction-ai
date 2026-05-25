# 🧘 YogaAlign — AI Pose Correction Coach

An AI-powered yoga pose correction application that analyzes your practice videos and provides personalized coaching instructions using Google Gemini's multimodal capabilities.

## Features

- **13 Yoga Poses** from the Common Yoga Protocol (Standing, Sitting, Prone, Supine)
- **Video Upload** — Drag & drop or click to browse (WebM, MP4, MOV, AVI, MKV)
- **AI-Powered Analysis** — Google Gemini 2.5 Flash analyzes joint angles, alignment, weight distribution, and more
- **Prioritized Corrections** — High/Medium/Low priority coaching cues with body-part specificity
- **Safety Warnings** — Automatic detection of potential injury risks
- **Premium UI** — Dark mode with glassmorphism, gradients, and smooth micro-animations

## Quick Start

### 1. Get a Gemini API Key

Visit [Google AI Studio](https://aistudio.google.com/apikey) and create a free API key.

### 2. Configure Environment

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_gemini_api_key_here` with your actual API key:

```
GEMINI_API_KEY=AIzaSy...your-key-here
```

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
| AI Model | Google Gemini 2.5 Flash (multimodal) |
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
├── server.js          # Express backend + Gemini API integration
├── package.json       # Dependencies
├── .env               # API key (not committed)
├── .env.example       # Environment template
└── public/
    ├── index.html     # Main page
    ├── index.css      # Design system
    └── app.js         # Frontend logic
```
